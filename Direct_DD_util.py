import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import LinearOperator

import fem, util, coef
import coarse_data

def build_B0_direct_global(world, KFineFree, MFineFree, Pmat,
                           freeFine, freeCoarse, materialize=True):
    """
    Direct coarse contribution B0 using the global product route:
        B0 = P_free * (P_free^T K_ff P_free)^{-1} * P_free^T.

    Parameters
    ----------
    world : World
        Global mesh object.
    KFineFree : csr_matrix
        Fine stiffness K_ff on free fine DOFs (same indexing as freeFine).
    Pmat : csr_matrix or ndarray
        Global prolongation.
    freeFine : ndarray of int
        Indices of free fine DOFs.
    freeCoarse : ndarray of int
        Indices of free coarse DOFs (interior coarse nodes).
    materialize : bool
        If True: return explicit csr_matrix.
        If False: return LinearOperator with the same action.

    Returns
    -------
    B0_free : csr_matrix or LinearOperator
        Coarse correction on the fine-free space, shape (n_free, n_free).
    """
    return build_B0_global(world, KFineFree, MFineFree, Pmat,
                           freeFine=freeFine,
                           freeCoarse=freeCoarse,
                           materialize=materialize)


def build_B0_global(world, KFineFree, MFineFree, Pmat,
                    freeFine, freeCoarse, materialize=True):
    """
    Exact global B0 on true coefficient, assembled on FREE fine DOFs:
        B0 = P_free * (P_free^T K_ff P_free)^{-1} * P_free^T.
    Here:
        - P_free : restriction of the global prolongation P to
                   free fine rows and interior coarse columns;
        - R0     : P_free^T is the *restriction* operator fine -> coarse.
    Parameters
    ----------
    Pmat : csr_matrix or ndarray
        Global prolongation matrix 
    freeFine : ndarray of int
        Indices of free fine DOFs.
    freeCoarse : ndarray of int
        Indices of interior coarse DOFs.

    Returns
    -------
    B0_free : csr_matrix or LinearOperator
        Coarse B0 correction on the fine-free space, shape (n_free, n_free).
    """
    # Restrict P to free fine rows and interior coarse columns
    if Pmat is None:
        Pmat = fem.assembleProlongationMatrix(world.NWorldCoarse,
                                              world.NCoarseElement)
    if not sp.isspmatrix(Pmat):
        Pmat = sp.csr_matrix(Pmat)
    else:
        Pmat = Pmat.tocsr()

    # Prolongation from coarse interior DOFs to free fine DOFs
    P_free = Pmat[freeFine, :][:, freeCoarse]        # (n_free x n_cint)
    n_free = KFineFree.shape[0]

    # Coarse stiffness (SPD) in coarse space
    Kc = (P_free.T @ KFineFree @ P_free).tocsc()     # (n_cint x n_cint)

    # Restriction (fine -> coarse) is the transpose of prolongation
    R0 = P_free.T                                    # (n_cint x n_free)

    # Factorize once in coarse space
    Kc_lu = splu(Kc)

    if not materialize:
        # Return as a LinearOperator acting on fine-free vectors
        def _matvec(x):
            # Restrict fine vector to coarse space
            rhs_c = R0 @ x              # (n_cint,)
            y_c   = Kc_lu.solve(rhs_c)  # (n_cint,)
            # Prolongate back to fine-free
            return P_free @ y_c

        return LinearOperator((n_free, n_free), matvec=_matvec)

    R0_csc = R0.tocsc()
    n_c, n_f = R0_csc.shape
    block = max(1, 131072 // max(1, n_c))
    cols = []

    for j0 in range(0, n_f, block):
        j1  = min(n_f, j0 + block)
        rhs = R0_csc[:, j0:j1].toarray()     # (n_c x nb) restriction of basis
        X   = Kc_lu.solve(rhs)              # (n_c x nb)
        cols.append(P_free @ sp.csr_matrix(X))  # (n_free x nb)

    B0_free = sp.hstack(cols, format="csr")
    B0_free.eliminate_zeros()
    return B0_free


# Elementwise coarse stiffness K0 
def _assemble_elementwise_K0(world, aPert, freeFine, freeCoarse, P_free):
    """
    Assemble the coarse stiffness matrix K0 elementwise:
        K0 = sum_T P_T^T K_T P_T,

    where K_T is the fine-scale stiffness on one coarse element with
    heterogeneous coefficient aPert, and P_T are the rows of P_free
    corresponding to fine-free DOFs owned by that element.
    """
    dim = len(world.NWorldCoarse)
    ncf = len(freeCoarse)
    nff = len(freeFine)

    K0_acc = sp.lil_matrix((ncf, ncf))

    # Map fine-global index -> position in freeFine vector
    pos_in_free = -np.ones(world.NpFine, dtype=int)
    pos_in_free[freeFine] = np.arange(nff, dtype=int)

    ALocElem = world.ALocFine

    if dim == 1:
        # 1D: loop over coarse elements ex = 0,...,Ncx-1
        Ncx = int(world.NWorldCoarse[0])
        for ex in range(Ncx):
            elem_idx = (ex,)

            # All fine nodes of this coarse element (global numbering)
            fine_idx_all = np.array(
                util.getFineElementIndices(world.NWorldFine,
                                           world.NCoarseElement,
                                           elem_idx),
                dtype=int
            )

            # Positions inside reduced freeFine
            fine_idx_free_pos = [
                pos_in_free[g] for g in fine_idx_all if pos_in_free[g] >= 0
            ]
            if not fine_idx_free_pos:
                continue

            fine_idx_free_pos = np.array(fine_idx_free_pos, dtype=int)
            fine_idx_free_glob = freeFine[fine_idx_free_pos]

            # Local fine coefficient on this coarse element
            a_local = util.extract_aFineLocal_for_coarse_element(
                aPert, world, elem_idx
            )

            # Local stiffness on all element nodes (using fine-grid operator)
            K_T_full = fem.assemblePatchMatrix(
                world.NCoarseElement, ALocElem, a_local
            ).tocsr()

            # Map global fine node -> local node index in this element
            local_id_map = {g: i for i, g in enumerate(fine_idx_all)}
            rows_free_loc = np.fromiter(
                (local_id_map[g] for g in fine_idx_free_glob),
                dtype=int
            )

            # Restrict to free-local nodes
            K_T = K_T_full[rows_free_loc, :][:, rows_free_loc]  # (nloc_free x nloc_free)

            # Local prolongation rows
            P_T = P_free[fine_idx_free_pos, :]                  # (nloc_free x ncf)

            # Element contribution
            K0_acc += P_T.T @ (K_T @ P_T)

    elif dim == 2:
        # 2D: loop over all coarse elements (ex,ey)
        Ncx, Ncy = map(int, world.NWorldCoarse)
        for ey in range(Ncy):
            for ex in range(Ncx):
                elem_idx = (ex, ey)

                fine_idx_all = np.array(
                    util.getFineElementIndices(world.NWorldFine,
                                               world.NCoarseElement,
                                               elem_idx),dtype=int)

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
                # Local stiffness on this coarse element (fine-grid operator)
                K_T_full = fem.assemblePatchMatrix(
                    world.NCoarseElement, ALocElem, a_local
                ).tocsr()

                local_id_map = {g: i for i, g in enumerate(fine_idx_all)}
                rows_free_loc = np.fromiter(
                    (local_id_map[g] for g in fine_idx_free_glob),
                    dtype=int
                )

                K_T = K_T_full[rows_free_loc, :][:, rows_free_loc]
                P_T = P_free[fine_idx_free_pos, :]
                K0_acc += P_T.T @ (K_T @ P_T)
    else:
        raise NotImplementedError(
            f"_assemble_elementwise_K0 currently supports only 1D / 2D, got dim={dim}."
        )

    return K0_acc.tocsc()


def build_B0_direct_element(world, aPert, KFineFree, MFineFree, Pmat,
                            freeFine, freeCoarse, materialize=True):
    """
    Fully elementwise construction of B0:
        B0 = P_free * K0^{-1} * P_free^T,
    where
        K0 = sum_T P_T^T K_T P_T is assembled per coarse element using the true coefficient aPert.

    Returns: B0_free 
    """
    NFine = np.asarray(world.NWorldFine, dtype=int)
    dim = NFine.size

    nff = len(freeFine)
    ncf = len(freeCoarse)

    # Ensure sparse matrix for prolongation
    if Pmat is None:
        Pmat = fem.assembleProlongationMatrix(world.NWorldCoarse,
                                              world.NCoarseElement)
    if not sp.isspmatrix(Pmat):
        Pmat = sp.csr_matrix(Pmat)
    else:
        Pmat = Pmat.tocsr()

    # Restrict prolongation to free fine rows / interior coarse cols
    P_free = Pmat[freeFine, :][:, freeCoarse]   # (nff x ncf)

    # Elementwise coarse stiffness K0
    K0 = _assemble_elementwise_K0(world, aPert, freeFine, freeCoarse, P_free)

    # Restriction operator R0 = P_free^T (fine -> coarse)
    R0 = P_free.T                                # (ncf x nff)

    # Solve in coarse space and lift to fine-free space
    K0_lu = splu(K0)

    if not materialize:
        # Return as a LinearOperator
        def _matvec(x):
            rhs_H = R0 @ x              # restrict fine vector to coarse space
            uH_free = K0_lu.solve(rhs_H)
            return P_free @ uH_free     # prolongate back

        return LinearOperator((nff, nff), matvec=_matvec)
=
    R0_csc = R0.tocsc()
    n_c, n_f = R0_csc.shape
    block = max(1, 131072 // max(1, n_c))
    cols = []

    for j0 in range(0, n_f, block):
        j1 = min(n_f, j0 + block)
        rhs = R0_csc[:, j0:j1].toarray()     # (n_c x nb)
        X   = K0_lu.solve(rhs)              # (n_c x nb)
        cols.append(P_free @ sp.csr_matrix(X))

    B0_free = sp.hstack(cols, format="csr")
    B0_free.eliminate_zeros()

    print(f"[B0-element] Built elementwise coarse correction: "
          f"shape={B0_free.shape}, nnz={B0_free.nnz}")
    return B0_free


# ==============================================================
# Direct patch-wise B contributions 
# ==============================================================
def build_direct_patch_Btilde_contributions(world, aPert, k):
    """
    Direct (no offline-online) construction of patch-wise operators for B.
    For each interior node patch w_i of radius k we build the local operator
        B_i^loc = A_ff^{-1} R_fS      (n_free_patch x n_patch),
    where
        - A_ff is the stiffness block on interior (patch-free) nodes, with
          homogeneous Dirichlet on the *patch boundary*,
        - R_fS is the identity-based restriction operator selecting all patch
          DOFs and mapping them to the interior DOFs.
    Returns
    -------
    patch_list : list[PatchFromNode]
    B_i_loc_list : list[csr_matrix]
        Local operators acting on full patch DOFs (columns) and producing
        values on patch-free DOFs (rows).
    g_nodes_list : list[np.ndarray]
        For each patch, global fine-node indices (length = n_patch).
    rows_free_list : list[np.ndarray]
        For each patch, indices into the global FREE fine vector corresponding
        to the patch's free nodes (same order as rows of B_i^loc).
    free : np.ndarray
        Global free fine-node indices.
    fixed : np.ndarray
        Global fixed fine-node indices.
    """
    # All valid (interior) patches fully inside the domain
    patch_list = util.create_all_valid_patches(world, k)

    # Global free/fixed pattern on the fine grid 
    Nf = np.asarray(world.NWorldFine, dtype=int)
    NpFine = int(np.prod(Nf + 1))
    bmap_f = np.ones((len(Nf), 2), dtype=bool)
    fixed  = util.boundarypIndexMap(Nf, boundaryMap=bmap_f)
    free   = np.setdiff1d(np.arange(NpFine), fixed)

    # Map fine global index --> position in reduced freeFine vector
    inv_free = -np.ones(NpFine, dtype=int)
    inv_free[free] = np.arange(free.size)

    B_i_loc_list = []
    g_nodes_list = []
    rows_free_list = []

    for patch in patch_list:
        # Global fine nodes belonging to this patch (all patch DOFs)
        g = util.LocalToGlobalFine_PatchFromNode(patch)

        # Coefficient restricted to this patch (consistent ordering)
        aFineLocal = coef.localizeCoefficientFromNodePatch(patch, aPert)

        # Assemble local patch stiffness
        A_patch = fem.assemblePatchMatrix(
            patch.NPatchFine, patch.world.ALocFine, aFineLocal
        )

        # Patch boundary Dirichlet --> determine free vs fixed patch nodes
        n_patch = int(np.prod(patch.NPatchFine + 1))
        fixed_patch = util.boundarypIndexMap(
            patch.NPatchFine,
            boundaryMap=np.ones((len(patch.NPatchFine), 2), dtype=bool)
        )
        free_patch = np.setdiff1d(np.arange(n_patch), fixed_patch)

        # Stiffness block: rows/cols on patch-free nodes
        A_ff = A_patch[free_patch][:, free_patch].tocsc()

        # Identity-based restriction R_fS: (n_free_patch x n_patch)
        # row i picks column free_patch[i]
        R_fS = sp.csr_matrix(
            (np.ones(len(free_patch)),
             (np.arange(len(free_patch)), free_patch)),
            shape=(len(free_patch), n_patch)
        )

        # Factorize A_ff once; compute B_loc = A_ff^{-1} R_fS
        LU = spla.splu(A_ff)
        B_loc_arr = LU.solve(R_fS.toarray())   # (n_free_patch x n_patch)
        B_loc = sp.csr_matrix(B_loc_arr)

        # Map patch-free local nodes to global FREE fine indices
        rows_free = inv_free[g[free_patch]]
        # Interior patches must be fully inside global free region
        assert np.all(rows_free >= 0), "Patch touches global Dirichlet boundary."

        B_i_loc_list.append(B_loc)
        g_nodes_list.append(g)
        rows_free_list.append(rows_free)

    return patch_list, B_i_loc_list, g_nodes_list, rows_free_list, free, fixed

    
