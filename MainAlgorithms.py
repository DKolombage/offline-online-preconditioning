import os, time, tempfile
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

import world as wrld
import fem, util, Reference_solver, error_metrics
import build_coefficient, Direct_DD_util
import coarse_data, patch_data, offline_data, online_data, coef


def _save_npz_atomic(out_path, **arrays):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_dir or ".", suffix=".npz", delete=False) as tf:
        np.savez_compressed(tf, **arrays)
        tf.flush()
        try:
            os.fsync(tf.fileno())
        except OSError:
            pass
        tmp_name = tf.name
    os.replace(tmp_name, out_path)


def _embed_full(free_idx, x_free, NWorldFine):
    """Embed reduced free vector into full fine grid (zero on boundary nodes)."""
    NpFine = int(np.prod(NWorldFine + 1))
    E = sp.csr_matrix(
        (np.ones(len(free_idx)), (free_idx, np.arange(len(free_idx)))),
        shape=(NpFine, len(free_idx))
    )
    return E @ x_free


# ==============================================================
# DIRECT - DD Preconditioners
# ==============================================================
def run_direct_Btilde(cfg, out_path="data/B/direct_run.npz", use_elementwise=False):
    NCoarse = np.array(cfg["NCoarse"])
    Nepsilon = np.array(cfg["Nepsilon"])
    NFine = np.array(cfg["NFine"])
    NSamples = int(cfg["NSamples"])
    k = int(cfg["k"])
    alpha, beta, p = float(cfg["alpha"]), float(cfg["beta"]), float(cfg["p"])
    rtol = float(cfg.get("rtol", 1e-8))
    atol = float(cfg.get("atol", 1e-12))
    maxiter = int(cfg.get("maxiter", 200))
    boundaryConditions = cfg.get("boundaryConditions", None)
    model_type = cfg.get("model_type", "incl").lower()

    world = wrld.World(NCoarse, NFine // NCoarse, boundaryConditions=boundaryConditions)
    Pmat = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)

    left = np.array([0.25, 0.25])
    right = np.array([0.75, 0.75])

    if model_type == "incl":
        model = {
            'name': 'incl',
            'alpha': alpha, 'beta': beta,
            'bgval': alpha, 'inclval': beta,
            'left': np.array([0.25, 0.25]),
            'right': np.array([0.75, 0.75])
        }
    elif model_type == "inclfill":
        model = {'name': 'inclfill', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right}
    elif model_type == "inclshift":
        model = {'name': 'inclshift', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right,
            'def_bl': np.array([0.75, 0.75]), 'def_tr': np.array([1., 1.])}
    elif model_type == "incllshape":
        model = {'name': 'incllshape', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right,
            'def_bl': np.array([0.5, 0.5]), 'def_tr': np.array([0.75, 0.75])}
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")


    def f_rhs(x):
        x = np.atleast_2d(x)
        if x.shape[1] == 1:
            return np.sin(np.pi * x[:, 0])
        return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

    L2_list, H1_list, EN_list = [], [], []
    it_list, conv_list, stop_res_list, pc_res_list = [], [], [], []
    res_hist_list, l2_hist_list, h1_hist_list, en_hist_list = [], [], [], []  

    t_coarse, t_patches, t_solver = [], [], []
    t_offline = 0.0

    for s in range(NSamples):
        # --- random coefficient
        if model_type == "incl":
            aPert = build_coefficient.build_inclusions_defect_2d(
                NFine=NFine, Nepsilon=Nepsilon,
                bg=alpha, val=beta,
                incl_bl=np.array([0.25, 0.25]), incl_tr=np.array([0.75, 0.75]),
                p_defect=p, def_val=alpha)
        else:
            aPert =build_coefficient.build_inclusions_change_2d(NFine, Nepsilon, alpha, beta, 
                np.array([0.25, 0.25]), np.array([0.75, 0.75]), p, model)

        # --- fine reference
        (u_ref_full, u_ref_free, KFineFull, KFineFree,
         MFineFull, MFineFree, bFineFull, bFineFree, freeFine) = Reference_solver.solve_fem_standard(
            world, aPert, f_rhs, boundaryMarker=None)

        freeFine_, _, freeCoarse, _, _, _ = coarse_data._free_fixed_masks_from_world(world)

        t1 = time.perf_counter()
        if not use_elementwise:
            B0_free = Direct_DD_util.build_B0_direct_global(
                world, KFineFree, MFineFree, Pmat, freeFine_, freeCoarse
            )
        else:
            B0_free = Direct_DD_util.build_B0_direct_element(
                world, aPert, KFineFree, MFineFree, Pmat, freeFine_, freeCoarse)
            B0_free_global = Direct_DD_util.build_B0_direct_global(
                world, KFineFree, MFineFree, Pmat, freeFine_, freeCoarse)

            #diff = B0_free - B0_free_global
            #fnorm_g = sp.linalg.norm(B0_free_global)
            #fnorm_diff = sp.linalg.norm(diff)
            #rel_err = fnorm_diff / (fnorm_g + 1e-16)

        t2 = time.perf_counter()

        patch_list, B_i_loc_list, g_nodes_list, rows_free_list, free_dir, fixed_dir = \
            Direct_DD_util.build_direct_patch_Btilde_contributions(world, aPert, k)
        B_tilde_free = patch_data.assemble_Btilde_reduced_from_online(
            world, B_i_loc_list, g_nodes_list, rows_free_list, freeFine
        )

        t3 = time.perf_counter()
        B_total_free = B0_free + B_tilde_free

        # --- residual & error history callback
        res_hist, l2_hist, h1_hist, en_hist = [], [], [], []
        b_norm = np.linalg.norm(bFineFree)
        k_iters = 0

        def _cb(xk):
            nonlocal k_iters
            k_iters += 1
            r = bFineFree - KFineFree @ xk
            relres = np.linalg.norm(r) / (b_norm if b_norm != 0 else 1.0)
            res_hist.append(relres)

            x_full_iter = np.zeros(world.NpFine)
            x_full_iter[freeFine] = xk
            relL2 = error_metrics.computeErrorL2(u_ref_full, x_full_iter, MFineFull)
            relH1 = error_metrics.computeErrorH1(u_ref_full, x_full_iter, MFineFull, KFineFull)
            relEN = error_metrics.computeErrorEnergyNorm(u_ref_full, x_full_iter, KFineFull)
            l2_hist.append(relL2)
            h1_hist.append(relH1)
            en_hist.append(relEN)

        # --- solve
        try:
            x_free, info = cg(KFineFree, bFineFree, M=B_total_free,
                              rtol=rtol, atol=atol, maxiter=maxiter, callback=_cb)
        except TypeError:
            x_free, info = cg(KFineFree, bFineFree, M=B_total_free,
                              tol=rtol, maxiter=maxiter, callback=_cb)
        t4 = time.perf_counter()

        res_hist_list.append(np.array(res_hist))
        l2_hist_list.append(np.array(l2_hist))
        h1_hist_list.append(np.array(h1_hist))
        en_hist_list.append(np.array(en_hist))

        # --- final diagnostics
        x_full = _embed_full(freeFine_, x_free, world.NWorldFine)
        u_full = _embed_full(freeFine_, u_ref_free, world.NWorldFine)
        L2 = error_metrics.computeErrorL2(u_full, x_full, MFineFull)
        H1 = error_metrics.computeErrorH1(u_full, x_full, MFineFull, KFineFull)
        EN = error_metrics.computeErrorEnergyNorm(u_full, x_full, KFineFull)

        r_true = bFineFree - KFineFree @ x_free
        stop_res = np.linalg.norm(r_true) / (np.linalg.norm(bFineFree) or 1.0)
        z = B_total_free @ r_true
        pc_res = np.sqrt(max(0.0, r_true @ z)) / np.sqrt(max(1e-300, bFineFree @ (B_total_free @ bFineFree)))

        stop_res_list.append(stop_res)
        pc_res_list.append(pc_res)
        L2_list.append(L2)
        H1_list.append(H1)
        EN_list.append(EN)
        it_list.append(len(res_hist))
        conv_list.append(info == 0)

        t_coarse.append(t2 - t1); t_patches.append(t3 - t2); t_solver.append(t4 - t3)

    t_total = np.array(t_coarse) + np.array(t_patches) + np.array(t_solver)   

    _save_npz_atomic(
        out_path,
        method="Direct-Btilde",
        NCoarse=NCoarse, NFine=NFine, Nepsilon=Nepsilon,
        NSamples=NSamples, k=k, alpha=alpha, beta=beta, p=p,

        errors_L2=np.array(L2_list), 
        errors_H1=np.array(H1_list), 
        errors_Energy=np.array(EN_list),

        iters=np.array(it_list), 
        converged=np.array(conv_list),
        stop_residual=np.array(stop_res_list), 
        pc_residual=np.array(pc_res_list),
        residual_history=np.array(res_hist_list, dtype=object),

        l2_history=np.array(l2_hist_list, dtype=object),
        h1_history=np.array(h1_hist_list, dtype=object),
        en_history=np.array(en_hist_list, dtype=object),

        t_offline=float(t_offline),
        t_coarse=np.array(t_coarse), t_patches=np.array(t_patches),
        t_solver=np.array(t_solver), t_total=t_total
    )


# ==============================================================
# OFFLINE–ONLINE DD Preconditioner
# ==============================================================
def run_offlineonline_Btilde(cfg, out_path="data/Btilde/offline_online_run.npz", use_elementwise=True):
    NCoarse = np.array(cfg["NCoarse"])
    Nepsilon = np.array(cfg["Nepsilon"])
    NFine = np.array(cfg["NFine"])
    NSamples = int(cfg["NSamples"])
    k = int(cfg["k"])
    alpha, beta, p = float(cfg["alpha"]), float(cfg["beta"]), float(cfg["p"])
    rtol = float(cfg.get("rtol", 1e-8))
    atol = float(cfg.get("atol", 1e-12))
    maxiter = int(cfg.get("maxiter", 200))
    boundaryConditions = cfg.get("boundaryConditions", None)
    model_type = cfg.get("model_type", "incl").lower()

    left = np.array([0.25, 0.25])
    right = np.array([0.75, 0.75])

    # --- model setup
    if model_type == "incl":
        model = {
            'name': 'incl',
            'alpha': alpha, 'beta': beta,
            'bgval': alpha, 'inclval': beta,
            'left': np.array([0.25, 0.25]),
            'right': np.array([0.75, 0.75])
        }
    elif model_type == "inclfill":
        model = {'name': 'inclfill', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right}
    elif model_type == "inclshift":
        model = {'name': 'inclshift', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right,
              'def_bl': np.array([0.75, 0.75]), 'def_tr': np.array([1., 1.])}
    elif model_type == "incllshape":
        model = {'name': 'incllshape', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right,
              'def_bl': np.array([0.5, 0.5]), 'def_tr': np.array([0.75, 0.75])}
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    world = wrld.World(NCoarse, NFine // NCoarse, boundaryConditions=boundaryConditions)
    Pmat = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    NepsilonElement = Nepsilon // NCoarse

    def f_rhs(x):
        x = np.atleast_2d(x)
        return np.sin(np.pi * x[:, 0]) * (np.sin(np.pi * x[:, 1]) if x.shape[1] == 2 else 1)

    # --- offline
    t0 = time.perf_counter()
    patch_ref = wrld.construct_center_node_patch(world, k)
    aRefList = offline_data.compute_offline_coefficients(world, NepsilonElement, k, boundaryConditions, model)
    B_loc_list, free_patch = offline_data.compute_offline_patch_Bloc(patch_ref, aRefList)

    aRefList_B0 = offline_data.compute_offline_coarse_coefficients_B0(world, NepsilonElement, model)
    K_T_ref_list = offline_data.compute_offline_coarse_B0loc(world, aRefList_B0)

    t_offline = time.perf_counter() - t0

    L2_list, H1_list, EN_list = [], [], []
    it_list, conv_list, stop_res_list, pc_res_list = [], [], [], []
    res_hist_list, l2_hist_list, h1_hist_list, en_hist_list = [], [], [], []

    t_coarse, t_patches, t_solver = [], [], []

    # --- samples
    for s in range(NSamples):
        if model_type == "incl":
            aPert = build_coefficient.build_inclusions_defect_2d(
                NFine=NFine, Nepsilon=Nepsilon,
                bg=alpha, val=beta,
                incl_bl=np.array([0.25, 0.25]), incl_tr=np.array([0.75, 0.75]),
                p_defect=p, def_val=alpha)
        else:
            aPert =build_coefficient.build_inclusions_change_2d(NFine, Nepsilon, 
                alpha, beta, np.array([0.25, 0.25]), np.array([0.75, 0.75]), p, model)

        (u_ref_full, uFineFree, KFineFull, KFineFree,
         MFineFull, MFineFree, bFineFull, bFineFree, freeFine) = Reference_solver.solve_fem_standard(
             world, aPert, f_rhs, boundaryMarker=None)

        freeFine_, _, freeCoarse, _, _, _ = coarse_data._free_fixed_masks_from_world(world)

        t1 = time.perf_counter()
        if not use_elementwise:
            B0_free = Direct_DD_util.build_B0_direct_global(
                world, KFineFree, MFineFree, Pmat, freeFine, freeCoarse
            )
        else:
            B0_direct = Direct_DD_util.build_B0_direct_global(world, KFineFree, MFineFree, Pmat, freeFine, freeCoarse)
            #B0_free = Direct_DD_util.build_B0_direct_element(world, aPert, KFineFree, MFineFree, Pmat, freeFine_, freeCoarse)
            B0_free_OO = coarse_data.build_B0_offlineonline_element(world, aPert, Pmat,freeFine_, freeCoarse,aRefList_B0, K_T_ref_list, model,materialize=True)
            #print("||B0_offlineonline - B0_direct||_F =", np.linalg.norm((B0_free_OO - B0_direct).toarray(), 'fro'))
            
        t2 = time.perf_counter()

        patches, B_i_loc_list, g_nodes_list, rows_free_list, FREEFine, _ = patch_data.build_online_patch_Btilde_contributions(
            world, aPert, aRefList, B_loc_list, free_patch, k, online_data.compute_mu_for_patch, model
        ) 

        B_tilde_free = patch_data.assemble_Btilde_reduced_from_online(
            world, B_i_loc_list, g_nodes_list, rows_free_list, FREEFine
        )

        t3 = time.perf_counter()
        B_total_free = B0_free_OO + B_tilde_free

        res_hist, l2_hist, h1_hist, en_hist = [], [], [], []
        b_norm = np.linalg.norm(bFineFree)
        k_iters = 0

        def _cb(xk):
            nonlocal k_iters
            k_iters += 1
            r = bFineFree - KFineFree @ xk
            relres = np.linalg.norm(r) / (b_norm if b_norm != 0 else 1.0)
            res_hist.append(relres)

            x_full_iter = np.zeros(world.NpFine)
            x_full_iter[freeFine] = xk
            relL2 = error_metrics.computeErrorL2(u_ref_full, x_full_iter, MFineFull)
            relH1 = error_metrics.computeErrorH1(u_ref_full, x_full_iter, MFineFull, KFineFull)
            relEN = error_metrics.computeErrorEnergyNorm(u_ref_full, x_full_iter, KFineFull)
            l2_hist.append(relL2)
            h1_hist.append(relH1)
            en_hist.append(relEN)

        try:
            x_free, info = cg(KFineFree, bFineFree, M=B_total_free,
                              rtol=rtol, atol=atol, maxiter=maxiter, callback=_cb)
        except TypeError:
            x_free, info = cg(KFineFree, bFineFree, M=B_total_free,
                              tol=rtol, maxiter=maxiter, callback=_cb)

        t4 = time.perf_counter()

        res_hist_list.append(np.array(res_hist))
        l2_hist_list.append(np.array(l2_hist))
        h1_hist_list.append(np.array(h1_hist))
        en_hist_list.append(np.array(en_hist))

        x_full = _embed_full(freeFine_, x_free, world.NWorldFine)
        u_full = _embed_full(freeFine_, uFineFree, world.NWorldFine)
        L2 = error_metrics.computeErrorL2(u_full, x_full, MFineFull)
        H1 = error_metrics.computeErrorH1(u_full, x_full, MFineFull, KFineFull)
        EN = error_metrics.computeErrorEnergyNorm(u_full, x_full, KFineFull)

        r_true = bFineFree - KFineFree @ x_free
        stop_res = np.linalg.norm(r_true) / (np.linalg.norm(bFineFree) or 1.0)
        z = B_total_free @ r_true
        pc_res = np.sqrt(abs(r_true @ z)) / np.sqrt(max(1e-300, bFineFree @ (B_total_free @ bFineFree)))

        stop_res_list.append(stop_res)
        pc_res_list.append(pc_res)
        it_list.append(len(res_hist))
        conv_list.append(info == 0)

        t_coarse.append(t2 - t1); t_patches.append(t3 - t2); t_solver.append(t4 - t3)

    t_total = np.array(t_coarse) + np.array(t_patches) + np.array(t_solver)

    _save_npz_atomic(
        out_path,
        method="Offline-Online B-form",
        NCoarse=NCoarse, 
        NFine=NFine, 
        Nepsilon=Nepsilon,
        NSamples=NSamples, 
        k=k, alpha=alpha, beta=beta, p=p,

        errors_L2=np.array(L2_list), 
        errors_H1=np.array(H1_list), 
        errors_Energy=np.array(EN_list),

        iters=np.array(it_list), 
        converged=np.array(conv_list),
        stop_residual=np.array(stop_res_list), 
        pc_residual=np.array(pc_res_list),
        residual_history=np.array(res_hist_list, dtype=object),

        l2_history=np.array(l2_hist_list, dtype=object),
        h1_history=np.array(h1_hist_list, dtype=object),
        en_history=np.array(en_hist_list, dtype=object),
        t_offline=float(t_offline),
        t_coarse=np.array(t_coarse), t_patches=np.array(t_patches),
        t_solver=np.array(t_solver), t_total=t_total
    )



# ==============================================================
# FIXED - DD Preconditioner on ZERO-DEFECT-ideal coefficients
# ==============================================================
def run_direct_Btilde_A0(cfg, out_path="data/BA0/direct_run_A0.npz", use_elementwise=False):
    """
    Direct-DD using a *fixed* preconditioner built entirely on the zero-defect
    coefficient A0. The same A0-based B0 and patch blocks are reused for all samples.

    """

    NCoarse = np.array(cfg["NCoarse"])
    Nepsilon = np.array(cfg["Nepsilon"])
    NFine = np.array(cfg["NFine"])
    NSamples = int(cfg["NSamples"])
    k = int(cfg["k"])
    alpha, beta, p = float(cfg["alpha"]), float(cfg["beta"]), float(cfg["p"])
    rtol = float(cfg.get("rtol", 1e-8))
    atol = float(cfg.get("atol", 1e-12))
    maxiter = int(cfg.get("maxiter", 200))
    boundaryConditions = cfg.get("boundaryConditions", None)
    model_type = cfg.get("model_type", "incl").lower()

    world = wrld.World(NCoarse, NFine // NCoarse, boundaryConditions=boundaryConditions)
    Pmat  = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)

    left = np.array([0.25, 0.25])
    right = np.array([0.75, 0.75])

    if model_type == "incl":
        model = {
            'name': 'incl',
            'alpha': alpha, 'beta': beta,
            'bgval': alpha, 'inclval': beta,
            'left': np.array([0.25, 0.25]),
            'right': np.array([0.75, 0.75])
        }
    elif model_type == "inclfill":
        model = {'name': 'inclfill', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right}
    elif model_type == "inclshift":
        model = {'name': 'inclshift', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right,
              'def_bl': np.array([0.75, 0.75]), 'def_tr': np.array([1., 1.])}
    elif model_type == "incllshape":
        model = {'name': 'incllshape', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right,
              'def_bl': np.array([0.5, 0.5]), 'def_tr': np.array([0.75, 0.75])}
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    def f_rhs(x):
        x = np.atleast_2d(x)
        if x.shape[1] == 1:
            return np.sin(np.pi * x[:,0])
        return np.sin(np.pi * x[:,0]) * np.sin(np.pi * x[:,1])

    # Build A0 preconditioner ONCE (coarse part + patch part)
    #print("[A0] Building zero-defect preconditioner...")
    if model_type == "check":
        a0 = alpha * np.ones(np.prod(NFine))
    elif model_type == "incl":
        a0 = build_coefficient.build_inclusions_defect_2d(
                NFine=NFine, Nepsilon=Nepsilon,
                bg=alpha, val=beta,incl_bl=np.array([0.25, 0.25]), incl_tr=np.array([0.75, 0.75]),
                p_defect=0, def_val=alpha)
    else:
        a0 = build_coefficient.build_inclusions_change_2d(NFine, Nepsilon, 
                alpha, beta, np.array([0.25, 0.25]), np.array([0.75, 0.75]), 0, model)

    (_u0_full, _u0_free, K0Full, K0Free,
     M0Full, M0Free, _b0Full, _b0Free, freeFineA0) = \
        Reference_solver.solve_fem_standard(world, a0, f_rhs, boundaryMarker=None)

    freeFine_all, _, freeCoarse, _, _, _ = coarse_data._free_fixed_masks_from_world(world)

    t0 = time.perf_counter()

    # --- Coarse block B0(A0)
    B0_free = Direct_DD_util.build_B0_direct_global(
        world, K0Free, M0Free, Pmat, freeFine_all, freeCoarse
    )
    print("[A0] B0 coarse part built.")

    # --- Patch blocks Btilde(A0)
    patch_list_A0, B_i_loc_list_A0, g_nodes_list_A0, rows_free_list_A0, _, _ = \
        Direct_DD_util.build_direct_patch_Btilde_contributions(world, a0, k)
    #print("[A0] Patch blocks built.")

    B_tilde_free_A0 = patch_data.assemble_Btilde_reduced_from_online(
        world, B_i_loc_list_A0, g_nodes_list_A0, rows_free_list_A0, freeFine_all)

    B_total_free = B0_free + B_tilde_free_A0

    t_offline = time.perf_counter() - t0

    L2_list, H1_list, EN_list = [], [], []
    it_list, conv_list = [], []
    stop_res_list, pc_res_list = [], []

    res_hist_list = []
    l2_hist_list  = []
    h1_hist_list  = []
    en_hist_list  = []

    t_solver = []
    for s in range(NSamples):

        # random coefficient
        if model_type == "incl":
            aPert = build_coefficient.build_inclusions_defect_2d(
                NFine=NFine, Nepsilon=Nepsilon,
                bg=alpha, val=beta,
                incl_bl=np.array([0.25, 0.25]), incl_tr=np.array([0.75, 0.75]),
                p_defect=p, def_val=alpha
            )
        else:
            aPert = build_coefficient.build_inclusions_change_2d(NFine, Nepsilon, 
                alpha, beta, np.array([0.25, 0.25]), np.array([0.75, 0.75]), p, model) 
            #raise ValueError("Include model not implemented in A0 version yet.")

        # fine reference for THIS sample
        (u_ref_full, u_ref_free, KFineFull, KFineFree,
         MFineFull, MFineFree, bFineFull, bFineFree, freeFine) = \
            Reference_solver.solve_fem_standard(world, aPert, f_rhs, boundaryMarker=None)

        # Iteration histories
        res_hist, l2_hist, h1_hist, en_hist = [], [], [], []
        b_norm = np.linalg.norm(bFineFree)

        def _cb(xk):
            r = bFineFree - KFineFree @ xk
            relres = np.linalg.norm(r) / (b_norm if b_norm > 0 else 1.0)
            res_hist.append(relres)

            x_full_iter = np.zeros(world.NpFine)
            x_full_iter[freeFine] = xk

            l2_hist.append(error_metrics.computeErrorL2(u_ref_full, x_full_iter, MFineFull))
            h1_hist.append(error_metrics.computeErrorH1(u_ref_full, x_full_iter, MFineFull, KFineFull))
            en_hist.append(error_metrics.computeErrorEnergyNorm(u_ref_full, x_full_iter, KFineFull))

        # Solve
        t3 = time.perf_counter()
        try:
            x_free, info = cg(
                KFineFree, bFineFree,
                M=B_total_free,
                rtol=rtol, atol=atol,
                maxiter=maxiter,
                callback=_cb
            )
        except TypeError:
            x_free, info = cg(
                KFineFree, bFineFree,
                M=B_total_free,
                tol=rtol,
                maxiter=maxiter,
                callback=_cb
            )

        t4 = time.perf_counter()
        
        # save histories
        res_hist_list.append(np.array(res_hist))
        l2_hist_list.append(np.array(l2_hist))
        h1_hist_list.append(np.array(h1_hist))
        en_hist_list.append(np.array(en_hist))

        r_true = bFineFree - KFineFree @ x_free
        stop_res = np.linalg.norm(r_true) / (np.linalg.norm(bFineFree) or 1.0)
        z = B_total_free @ r_true
        pc_res = np.sqrt(max(0.0, r_true @ z)) / \
                 np.sqrt(max(1e-300, bFineFree @ (B_total_free @ bFineFree)))

        stop_res_list.append(stop_res)
        pc_res_list.append(pc_res)
        it_list.append(len(res_hist))
        conv_list.append(info == 0)

        # final L2/H1/EN error
        x_full = np.zeros(world.NpFine)
        x_full[freeFine] = x_free
        L2_list.append(error_metrics.computeErrorL2(u_ref_full, x_full, MFineFull))
        H1_list.append(error_metrics.computeErrorH1(u_ref_full, x_full, MFineFull, KFineFull))
        EN_list.append(error_metrics.computeErrorEnergyNorm(u_ref_full, x_full, KFineFull))

        print(f"[A0-run {s}] iters={len(res_hist)}, stop_res={stop_res:.2e}, pc_res={pc_res:.2e}")

        t_solver.append(t4 - t3)
    t_total = np.array(t_solver)

    _save_npz_atomic(
        out_path,
        method="Direct-Btilde-A0",
        NCoarse=NCoarse, NFine=NFine, Nepsilon=Nepsilon,
        NSamples=NSamples, k=k, alpha=alpha, beta=beta, p=p,

        errors_L2=np.array(L2_list),
        errors_H1=np.array(H1_list),
        errors_Energy=np.array(EN_list),

        iters=np.array(it_list),
        converged=np.array(conv_list),
        stop_residual=np.array(stop_res_list),
        pc_residual=np.array(pc_res_list),
        residual_history=np.array(res_hist_list, dtype=object),

        l2_history=np.array(l2_hist_list, dtype=object),
        h1_history=np.array(h1_hist_list, dtype=object),
        en_history=np.array(en_hist_list, dtype=object),

        t_offline=float(t_offline),
        t_solver=np.array(t_solver), t_total=t_total
    )

