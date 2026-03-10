import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import splu, factorized, LinearOperator

import build_coefficient, util, fem, online_data, offline_data

def compute_lambda(aFineLocal, aRefList, tol=1e-14):
    """
    lambda in {0,1} and detects which defect cells are active.
    """
    aLoc = np.asarray(aFineLocal).ravel()
    aRefs = [np.asarray(a).ravel() for a in aRefList]

    L = len(aRefs)
    bg_idx = L - 1   # background last
    lam = np.zeros(L)

    a_bg = aRefs[bg_idx]

    for l in range(L):
        if l == bg_idx:
            continue
        mask = np.abs(aRefs[l] - a_bg) > tol
        if np.any(mask):
            # defect present if ANY fine cell matches this defect
            if np.allclose(aLoc[mask], aRefs[l][mask]):
                lam[l] = 1.0

    lam[bg_idx] = 1.0
    lam[bg_idx] -= lam[:-1].sum()
    return lam

# ==============================================================
# ONLINE B0
# ==============================================================
def compute_online_coarse_B0loc(aFineLocal, aRefList, K0_ref_list, model):
    """
    Given coefficient aFineLocal on a coarse element,
    compute:
        K0_T ~ Sum_l mu[l] K0_ref_list[l]
        M0_T ~ Sum_l mu[l] M0_ref_list[l]
    """
    alpha=model['alpha']
    beta=model['beta']
    mu = coarse_data.compute_lambda(aFineLocal, aRefList)
    L = len(mu)

    K0_T = K0_ref_list[0].multiply(mu[0])
    #M0_T = M0_ref_list[0].multiply(mu[0])

    for l in range(1, L):
        if mu[l] != 0.0:
            K0_T = K0_T + K0_ref_list[l].multiply(mu[l])
            #M0_T = M0_T + M0_ref_list[l].multiply(mu[l])

    return K0_T.tocsr()

def _assemble_elementwise_K0_offlineonline(world, aPert, freeFine, freeCoarse,
                                           P_free, aRefList_B0, K_T_ref_list, model):
    """
    Assemble the coarse stiffness matrix K0 elementwise:
            K0 = sum_T P_T^T K_T P_T,
    where K_T is now obtained via the offline-online combination:
        K_T ~ sum_l mu_l(a_loc) K_T_ref_list[l],
    with a_loc the local coefficient on coarse element T.

    Parameters
    ----------
    world : World
    aPert : ndarray
        Global fine coefficient for the sample.
    freeFine : ndarray
        Free fine DOF indices.
    freeCoarse : ndarray
        Interior coarse DOF indices.
    P_free : csr_matrix
        Prolongation matrix restricted to free fine rows and interior coarse cols.
    aRefList_B0 : list of ndarray
        Reference coefficients on one coarse element.
    K_T_ref_list : list of csr_matrix
        Reference local stiffness matrices on one coarse element.
    model : dict
        
    Returns
    -------
    K0 : csr_matrix
        Coarse stiffness matrix assembled in coarse interior space.
    """
    NFine = np.asarray(world.NWorldFine, dtype=int)
    dim   = NFine.size

    nff = len(freeFine)
    ncf = len(freeCoarse)

    # Accumulator in coarse space
    K0_acc = sp.csr_matrix((ncf, ncf), dtype=float)

    # Map global fine index --> position in freeFine
    pos_in_free = -np.ones(world.NpFine, dtype=int)
    pos_in_free[freeFine] = np.arange(nff, dtype=int)

    if dim == 1:
        Ncx = int(world.NWorldCoarse[0])
        for ex in range(Ncx):
            elem_idx = (ex,)

            # All fine nodes of this coarse element (global numbering)
            fine_idx_all = np.array(
                util.getFineElementIndices(world.NWorldFine,
                                           world.NCoarseElement,
                                           elem_idx), dtype=int)

            # Free fine DOFs belonging to this element (positions in freeFine)
            fine_idx_free_pos = [
                pos_in_free[g] for g in fine_idx_all if pos_in_free[g] >= 0
            ]
            if not fine_idx_free_pos:
                continue

            fine_idx_free_pos = np.array(fine_idx_free_pos, dtype=int)
            fine_idx_free_glob = freeFine[fine_idx_free_pos]

            # Local coefficient on this coarse element
            a_local = util.extract_aFineLocal_for_coarse_element(
                aPert, world, elem_idx
            )

            # OFFLINE-ONLINE local stiffness on this coarse element
            K_T_full = compute_online_coarse_B0loc(
                a_local, aRefList_B0, K_T_ref_list, model
            )

            # Map global fine node -->  local node index on this element
            local_id_map = {g: i for i, g in enumerate(fine_idx_all)}
            rows_free_loc = np.fromiter(
                (local_id_map[g] for g in fine_idx_free_glob),
                dtype=int
            )

            # Restrict to free-local nodes
            K_T = K_T_full[rows_free_loc, :][:, rows_free_loc]  # (nloc_free x nloc_free)

            # Local prolongation rows for this element
            P_T = P_free[fine_idx_free_pos, :]                  # (nloc_free x ncf)

            # Element contribution in coarse space
            K0_acc += P_T.T @ (K_T @ P_T)

    elif dim == 2:
        Ncx, Ncy = map(int, world.NWorldCoarse)
        for ey in range(Ncy):
            for ex in range(Ncx):
                elem_idx = (ex, ey)

                fine_idx_all = np.array(
                    util.getFineElementIndices(world.NWorldFine,
                                               world.NCoarseElement,
                                               elem_idx),
                    dtype=int)
                fine_idx_free_pos = [
                    pos_in_free[g] for g in fine_idx_all if pos_in_free[g] >= 0]
                if not fine_idx_free_pos:
                    continue

                fine_idx_free_pos = np.array(fine_idx_free_pos, dtype=int)
                fine_idx_free_glob = freeFine[fine_idx_free_pos]

                # Local coefficient on this coarse element
                a_local = util.extract_aFineLocal_for_coarse_element(
                    aPert, world, elem_idx)

                # OFFLINE-ONLINE local stiffness
                K_T_full = online_data.compute_online_coarse_B0loc(
                    a_local, aRefList_B0, K_T_ref_list, model )

                local_id_map = {g: i for i, g in enumerate(fine_idx_all)}
                rows_free_loc = np.fromiter(
                    (local_id_map[g] for g in fine_idx_free_glob),
                    dtype=int)

                K_T = K_T_full[rows_free_loc, :][:, rows_free_loc]

                P_T = P_free[fine_idx_free_pos, :]
                K0_acc += P_T.T @ (K_T @ P_T)
    else:
        raise NotImplementedError("_assemble_elementwise_K0_offlineonline only supports 1D/2D.")

    return K0_acc.tocsc()


def build_B0_offlineonline_element(world, aPert, Pmat,
                                   freeFine, freeCoarse,
                                   aRefList_B0, K_T_ref_list, model,
                                   materialize=True):
    """
    OFFLINE-ONLINE elementwise B0:
        B0 = P_free * K0^{-1} * P_free^T,where K0 is assembled elementwise using the offline-online local
    stiffness matrices (see _assemble_elementwise_K0_offlineonline).

    This is equal to the direct elementwise B0 constructed from the true coefficient aPert.

    Parameters
    ----------
    world : World
    aPert : ndarray
        Global fine coefficient for this sample.
    Pmat : csr_matrix
        Global prolongation (fine <- coarse), unreduced.
    freeFine : ndarray
        Free fine DOF indices.
    freeCoarse : ndarray
        Interior coarse DOF indices.
    aRefList_B0 : list of ndarray
        Reference coefficients on one coarse element.
    K_T_ref_list : list of csr_matrix
        Reference local stiffness matrices on one coarse element.
    model : dict
        Model parameters, including 'alpha','beta','name'.
    materialize : bool
        If True, return explicit csr_matrix; otherwise a LinearOperator.

    Returns
    -------
    B0_free : csr_matrix or LinearOperator
        Coarse B0 correction acting on fine-free space.
    """
    freeFine   = np.asarray(freeFine, dtype=int)
    freeCoarse = np.asarray(freeCoarse, dtype=int)

    n_free = len(freeFine)
    n_cf   = len(freeCoarse)

    # Prolongation restricted to free fine + interior coarse
    P_free = Pmat[freeFine, :][:, freeCoarse]   # (n_free x n_cf)

    # Coarse stiffness via offline–online elementwise assembly
    K0 = _assemble_elementwise_K0_offlineonline(
        world, aPert, freeFine, freeCoarse, P_free,
        aRefList_B0, K_T_ref_list, model
    )

    # Restriction operator fine->coarse
    R0 = P_free.T                                # (n_cf x n_free)

    # Factorize once in coarse space
    K0_lu = splu(K0)

    if not materialize:
        def _matvec(x):
            rhs_c = R0 @ x
            y_c   = K0_lu.solve(rhs_c)
            return P_free @ y_c

        return LinearOperator((n_free, n_free), _matvec)

    # Explicit matrix: B0_free = P_free K0^{-1} R0
    X = K0_lu.solve(R0.toarray())               # (n_cf x n_free)
    B0_free = P_free @ sp.csr_matrix(X)         # (n_free x n_free)
    return B0_free


# ==============================================================
# ASSEMBLE GLOBAL B0 FROM ONLINE COARSE ELEMENT CONTRIBUTIONS
# ==============================================================

def assemble_B0_from_online(world, freeFine, freeCoarse,
                            K0_T_list, M0_T_list,
                            Pmat):
    """
    Given K0_T[element], M0_T[element] for each coarse element,
    assemble:
        K0 = sum_T K0_T
        M0 = sum_T M0_T
        B0 = P_free * K0^{-1} * M0
    exactly mirroring the Direct-DD elementwise construction.
    """
    ncf = len(freeCoarse)
    nff = len(freeFine)

    # global accumulators
    K0 = sp.lil_matrix((ncf, ncf))
    M0 = sp.lil_matrix((ncf, nff))

    # mapping to coarse-free index
    coarse_map = {c:i for i,c in enumerate(freeCoarse)}

    # mapping fine->fine-free
    pos_in_free = -np.ones(world.NpFine, int)
    pos_in_free[freeFine] = np.arange(nff)

    # unfold prolongation
    P_free = Pmat[freeFine][:, freeCoarse]

    NCoarse = world.NWorldCoarse
    d = len(NCoarse)

    elem_idx = 0
    for idx in np.ndindex(*tuple(NCoarse)):
        K0_T = K0_T_list[elem_idx]
        M0_T = M0_T_list[elem_idx]

        # coarse nodes of this element
        cNodes = util.getCoarseElementNodes(idx, NCoarse)
        cNodes = np.array(cNodes)
        cFree  = [coarse_map[c] for c in cNodes if c in coarse_map]

        if len(cFree) == 0:
            elem_idx += 1
            continue

        # assign K0_T
        # K0_T : (len(cFree) × len(cFree))
        K0[cFree[:, None], cFree] += K0_T

        # M0_T : (len(cFree) × n_free_fine)
        M0[cFree, :] += M0_T
        elem_idx += 1

    K0 = K0.tocsc()
    M0 = M0.tocsc()

    # solve K0^{-1} M0
    K0_lu = splu(K0)
    X = K0_lu.solve(M0.toarray())
    B0_free = P_free @ sp.csr_matrix(X)
    return B0_free

# ==============================================================
# Coarse RHS from fine RHS 
# ==============================================================
def assemble_coarse_rhs_from_fine(world, f_rhs):
    NFine   = world.NWorldFine
    NCoarse = world.NWorldCoarse
    Ne      = world.NCoarseElement

    NpFine   = np.prod(NFine + 1)
    NpCoarse = np.prod(NCoarse + 1)

    boundaryMapFine   = np.ones((len(NFine),   2), dtype=bool)
    boundaryMapCoarse = np.ones((len(NCoarse), 2), dtype=bool)

    fixedFine   = util.boundarypIndexMap(NFine,   boundaryMap=boundaryMapFine)
    fixedCoarse = util.boundarypIndexMap(NCoarse, boundaryMap=boundaryMapCoarse)

    freeFine   = np.setdiff1d(np.arange(NpFine),   fixedFine)
    freeCoarse = np.setdiff1d(np.arange(NpCoarse), fixedCoarse)

    P_full = fem.assembleProlongationMatrix(NCoarse, Ne)          # (NpFine, NpCoarse)
    P_free = P_full[freeFine][:, freeCoarse]                       # (nff, ncf)

    coords   = util.pCoordinates(NFine)
    f_vals   = f_rhs(coords).ravel()
    M_fine   = fem.assemblePatchMatrix(NFine, world.MLocFine)
    M_free   = M_fine[freeFine][:, freeFine]

    f_h_free = M_free @ f_vals[freeFine]
    f_H_free = P_free.T @ f_h_free

    E_H = sp.csr_matrix(
        (np.ones(len(freeCoarse)), (freeCoarse, np.arange(len(freeCoarse)))),
        shape=(NpCoarse, len(freeCoarse))
    )
    F_H_full = E_H @ f_H_free
    return f_H_free, F_H_full

# ==============================================================
# Coarse stiffness
# ==============================================================
def coarse_stiffness_matrix(KFineFree, Pmat, NCoarse, NFine):
    NCoarse = np.asarray(NCoarse, dtype=int)
    NFine   = np.asarray(NFine,   dtype=int)

    NpCoarse = int(np.prod(NCoarse + 1))
    NpFine   = int(np.prod(NFine   + 1))

    bmap_c = np.ones((len(NCoarse), 2), dtype=bool)
    bmap_f = np.ones((len(NFine),   2), dtype=bool)

    fixedCoarse = util.boundarypIndexMap(NCoarse, boundaryMap=bmap_c)
    fixedFine   = util.boundarypIndexMap(NFine,   boundaryMap=bmap_f)

    freeCoarse  = np.setdiff1d(np.arange(NpCoarse), fixedCoarse)
    freeFine    = np.setdiff1d(np.arange(NpFine),   fixedFine)

    P_free = Pmat[freeFine][:, freeCoarse]

    nff, ncf = len(freeFine), len(freeCoarse)
    if KFineFree.shape != (nff, nff):
        raise ValueError(f"KFineFree shape {KFineFree.shape} != ({nff},{nff})")
    if P_free.shape != (nff, ncf):
        raise ValueError(f"P_free shape {P_free.shape} != ({nff},{ncf})")

    AH = (P_free.T @ (KFineFree @ P_free)).tocsr()

    E = sp.csr_matrix((np.ones(ncf), (freeCoarse, np.arange(ncf))), shape=(NpCoarse, ncf))
    AHFull = (E @ AH @ E.T).tocsr()
    return AH, AHFull

# ==============================================================
# utility functions 
# ==============================================================
def _free_fixed_masks_from_world(world):
    NFine   = np.asarray(world.NWorldFine,   dtype=int)
    NCoarse = np.asarray(world.NWorldCoarse, dtype=int)

    NpFine   = int(np.prod(NFine + 1))
    NpCoarse = int(np.prod(NCoarse + 1))

    bmap_f = np.ones((len(NFine),   2), dtype=bool)
    bmap_c = np.ones((len(NCoarse), 2), dtype=bool)

    fixedFine   = util.boundarypIndexMap(NFine,   boundaryMap=bmap_f)
    fixedCoarse = util.boundarypIndexMap(NCoarse, boundaryMap=bmap_c)

    freeFine   = np.setdiff1d(np.arange(NpFine),   fixedFine)
    freeCoarse = np.setdiff1d(np.arange(NpCoarse), fixedCoarse)
    return (freeFine, fixedFine, freeCoarse, fixedCoarse, NpFine, NpCoarse)

# Background index utility
def _find_background_index(aRefList, alpha, atol=1e-12):
    a = float(alpha)
    for i, ref in enumerate(aRefList):
        r = np.asarray(ref, dtype=float).ravel()
        if np.allclose(r, a, atol=atol):
            return i
    return None