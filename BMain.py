import os
import time
import tempfile
import numpy as np
import scipy.sparse as sp

import world as wrld
import fem
import build_coefficient
import Direct_DD_util
import coarse_data
import patch_data
import offline_data
import online_data
import Reference_solver


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


def _build_model(cfg):
    alpha, beta, p = float(cfg["alpha"]), float(cfg["beta"]), float(cfg["p"])
    model_type = cfg.get("model_type", "incl").lower()

    left = np.array([0.25, 0.25])
    right = np.array([0.75, 0.75])

    if model_type == "incl":
        return {
            "name": "incl",
            "alpha": alpha, "beta": beta,
            "bgval": alpha, "inclval": beta,
            "left": left, "right": right
        }
  
    if model_type == "inclshift":
        return {
            "name": "inclshift", "bgval": alpha, "inclval": beta,
            "left": left, "right": right,
            "def_bl": np.array([0.75, 0.75]), "def_tr": np.array([1.0, 1.0])
        }

    if model_type == "incllshape":
        return {
            "name": "incllshape", "bgval": alpha, "inclval": beta,
            "left": left, "right": right,
            "def_bl": np.array([0.5, 0.5]), "def_tr": np.array([0.75, 0.75])
        }

    raise ValueError(f"Unknown model_type '{model_type}'.")


def _build_random_coefficient(cfg, model):
    NCoarse  = np.array(cfg["NCoarse"])
    NFine    = np.array(cfg["NFine"])
    Nepsilon = np.array(cfg["Nepsilon"])
    alpha, beta, p = float(cfg["alpha"]), float(cfg["beta"]), float(cfg["p"])
    model_type = cfg.get("model_type", "incl").lower()

    left = np.array([0.25, 0.25])
    right = np.array([0.75, 0.75])

    if model_type == "incl":
        return build_coefficient.build_inclusions_defect_2d(
            NFine=NFine, Nepsilon=Nepsilon,
            bg=alpha, val=beta,
            incl_bl=left, incl_tr=right,
            p_defect=p, def_val=alpha
        )

    # all other incl* variants
    return build_coefficient.build_inclusions_change_2d(
        NFine, Nepsilon, alpha, beta, left, right, p, model
    )


def _build_A0_coefficient(cfg, model):
    NFine    = np.array(cfg["NFine"])
    Nepsilon = np.array(cfg["Nepsilon"])
    alpha, beta = float(cfg["alpha"]), float(cfg["beta"])
    model_type = cfg.get("model_type", "incl").lower()

    left = np.array([0.25, 0.25])
    right = np.array([0.75, 0.75])

    if model_type == "incl":
        return build_coefficient.build_inclusions_defect_2d(
            NFine=NFine, Nepsilon=Nepsilon,
            bg=alpha, val=beta,
            incl_bl=left, incl_tr=right,
            p_defect=0.0, def_val=alpha
        )

    # inclshift / incllshape etc: same generator, but p=0
    return build_coefficient.build_inclusions_change_2d(
        NFine, Nepsilon, alpha, beta, left, right, 0.0, model
    )


def _solve_mats(world, aFine):
    zero_rhs = lambda x: np.zeros(x.shape[0])
    (u_full, u_free, KFull, KFree, MFull, MFree, bFull, bFree, freeFine) = \
        Reference_solver.solve_fem_standard(world, aFine, zero_rhs, boundaryMarker=None)
    return KFull, KFree, MFull, MFree, freeFine


def _build_B_direct(world, Pmat, aFine, k, use_elementwise_B0=False):
    KFull, KFree, MFull, MFree, freeFine = _solve_mats(world, aFine)
    freeFine_, _, freeCoarse, _, _, _ = coarse_data._free_fixed_masks_from_world(world)

    if not use_elementwise_B0:
        B0_free = Direct_DD_util.build_B0_direct_global(
            world, KFree, MFree, Pmat, freeFine_, freeCoarse
        )
    else:
        B0_free = Direct_DD_util.build_B0_direct_element(
            world, aFine, KFree, MFree, Pmat, freeFine_, freeCoarse
        )

    patch_list, B_i_loc_list, g_nodes_list, rows_free_list, free_dir, fixed_dir = \
        Direct_DD_util.build_direct_patch_Btilde_contributions(world, aFine, k)

    Bpatch_free = patch_data.assemble_Btilde_reduced_from_online(
        world, B_i_loc_list, g_nodes_list, rows_free_list, freeFine
    )

    return (B0_free + Bpatch_free), freeFine


def _build_offline_library(world, cfg, model):
    NCoarse  = np.array(cfg["NCoarse"])
    Nepsilon = np.array(cfg["Nepsilon"])
    k = int(cfg["k"])
    boundaryConditions = cfg.get("boundaryConditions", None)

    NepsilonElement = Nepsilon // NCoarse
    patch_ref = wrld.construct_center_node_patch(world, k)

    aRefList = offline_data.compute_offline_coefficients(
        world, NepsilonElement, k, boundaryConditions, model
    )
    B_loc_list, free_patch = offline_data.compute_offline_patch_Bloc(
        patch_ref, aRefList
    )
    return aRefList, B_loc_list, free_patch


def _build_B_offlineonline(world, Pmat, aFine, k, model, offline_lib):
    aRefList, B_loc_list, free_patch = offline_lib

    KFull, KFree, MFull, MFree, freeFine = _solve_mats(world, aFine)
    freeFine_, _, freeCoarse, _, _, _ = coarse_data._free_fixed_masks_from_world(world)

    B0_free = Direct_DD_util.build_B0_direct_global(
        world, KFree, MFree, Pmat, freeFine_, freeCoarse
    )

    patch_list, B_i_loc_list, g_nodes_list, rows_free_list, free, fixed = \
        patch_data.build_online_patch_Btilde_contributions(
            world,
            aFine,
            aRefList,
            B_loc_list,
            free_patch,
            k,
            online_data.compute_mu_for_patch,
            model
        )

    Bpatch_free = patch_data.assemble_Btilde_reduced_from_online(
        world, B_i_loc_list, g_nodes_list, rows_free_list, free
    )

    return (B0_free + Bpatch_free), free


def save_B_per_sample(cfg, out_dir="Bdata/B_mats", use_elementwise_B0=False):
    """
    Save B matrices per sample for:
      - direct
      - offline-online
      - A0
    """
    NCoarse  = np.array(cfg["NCoarse"])
    NFine    = np.array(cfg["NFine"])
    NSamples = int(cfg["NSamples"])
    k        = int(cfg["k"])
    boundaryConditions = cfg.get("boundaryConditions", None)

    world = wrld.World(NCoarse, NFine // NCoarse, boundaryConditions=boundaryConditions)
    Pmat  = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)

    model = _build_model(cfg)

    t0 = time.perf_counter()
    offline_lib = _build_offline_library(world, cfg, model)
    t1 = time.perf_counter()
    t_offline = float(t1 - t0)

    a0 = _build_A0_coefficient(cfg, model)
    B_A0_free, freeA0 = _build_B_direct(world, Pmat, a0, k, use_elementwise_B0=use_elementwise_B0)

    os.makedirs(out_dir, exist_ok=True)

    for s in range(NSamples):
        aPert = _build_random_coefficient(cfg, model)

        # DIRECT
        B_dir_free, free_dir = _build_B_direct(
            world, Pmat, aPert, k, use_elementwise_B0=use_elementwise_B0
        )

        _save_npz_atomic(
            os.path.join(out_dir, f"B_direct_s{s:04d}.npz"),
            method="direct",
            sample=int(s),
            B_total_free=B_dir_free,
            freeFine=free_dir,
            NCoarse=NCoarse, NFine=NFine, k=int(k),
            model_type=str(cfg.get("model_type", "incl")),
        )

        # OFFLINE–ONLINE
        B_oo_free, free_oo = _build_B_offlineonline(
            world, Pmat, aPert, k, model, offline_lib
        )

        _save_npz_atomic(
            os.path.join(out_dir, f"B_OO_s{s:04d}.npz"),
            method="offline-online",
            sample=int(s),
            B_total_free=B_oo_free,
            freeFine=free_oo,
            t_offline=t_offline,
            NCoarse=NCoarse, NFine=NFine, k=int(k),
            model_type=str(cfg.get("model_type", "incl")),
        )

        # A0 
        _save_npz_atomic(
            os.path.join(out_dir, f"B_A0_s{s:04d}.npz"),
            method="A0",
            sample=int(s),
            B_total_free=B_A0_free,
            freeFine=freeA0,
            NCoarse=NCoarse, NFine=NFine, k=int(k),
            model_type=str(cfg.get("model_type", "incl")),
        )

        print(f"[save_B_per_sample] s={s:4d} saved: direct, OO, A0  (offline={t_offline:.2f}s)")


if __name__ == "__main__":
    cfg = dict(
        NCoarse=[16,16],
        NFine=[128,128],
        Nepsilon=[32,32],
        NSamples=100,
        k=1,
        alpha=0.1,
        beta=50.0,
        p=0.1,
        model_type="incl",          # "incl", "inclshift", "incllshape"
        boundaryConditions=None
    )
    save_B_per_sample(cfg, out_dir="Bdata/B_mats", use_elementwise_B0=False)