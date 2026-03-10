import numpy as np
import scipy.sparse as sp
import coef, util

def build_online_patch_Btilde_contributions(world, aPert, aRefList,
                                            B_loc_list, free_patch,
                                            k, compute_mu, model):
    """
    Build local online B_i_loc operators for all valid interior patches
    using the offline reference operators and precomputed mu.

    Implements the offline-online combination:
        B_i_loc = sum_l mu[l] * B_loc_list[l].

    Parameters
    ----------
    aPert : ndarray
        Global fine-scale coefficient.
    aRefList : list of ndarray
        List of reference coefficient patches from the offline phase.The last entry must be the background coefficient.
    B_loc_list : list of scipy.sparse.csr_matrix
        Offline reference patch operators (A_ff^-1 M_fS).
    free_patch : ndarray
        Local indices of free nodes in the reference patch.
    k : int
        Patch radius (number of coarse layers).
    compute_mu : callable
        Function to compute mu for a local coefficient:
            compute_mu(aFineLocal, aRefList, model) --> ndarray of shape (L,)
    model : dict

    Returns
    -------
    patch_list : list of PatchFromNode
        List of all valid interior patches.
    B_i_loc_list : list of csr_matrix
        Online-computed local operators B_i_loc.
    g_nodes_list : list of ndarray
        Global fine-node indices per patch.
    rows_free_list : list of ndarray
        Global indices of free nodes corresponding to patch-free nodes.
    free : ndarray
        Global indices of all free fine nodes.
    fixed : ndarray
        Global indices of Dirichlet boundary nodes.
    """


    # Construct all valid interior patches
    patch_list = util.create_all_valid_patches(world, k)
    NpFine = int(np.prod(world.NWorldFine + 1))
    d = len(world.NWorldFine)

    # Identify Dirichlet boundary nodes
    bmap_f = np.ones((d, 2), dtype=bool)
    fixed = util.boundarypIndexMap(world.NWorldFine, boundaryMap=bmap_f)
    free = np.setdiff1d(np.arange(NpFine), fixed)

    # Map fine-node index --> free-node index
    inv_free = -np.ones(NpFine, dtype=int)
    inv_free[free] = np.arange(free.size)

    # check
    L = len(aRefList)
    assert L == len(B_loc_list), "Mismatch: aRefList and B_loc_list lengths differ."

    B_i_loc_list = []
    g_nodes_list = []
    rows_free_list = []

    #  Loop over patches and build online operators
    for patch in patch_list:
        g = util.LocalToGlobalFine_PatchFromNode(patch) # Global fine indices for this patch
        aFineLocal = coef.localizeCoefficientFromNodePatch(patch, aPert) # Local fine-scale coefficient restriction

        # Compute mu 
        mu = np.asarray(compute_mu(aFineLocal, aRefList, model), dtype=float)

        # Combine reference patch operators: B_i_loc = sum mu[l] * B_loc_list[l]
        B_comb = sp.csr_matrix(B_loc_list[0].shape)
        for l in range(L):
            if abs(mu[l]) > 1e-15:
                B_comb += mu[l] * B_loc_list[l]

        # Map local free nodes to global free indices
        rows_free = inv_free[g[free_patch]]
        if np.any(rows_free < 0):
            raise ValueError("Patch touches boundary — only interior patches allowed.")

        B_i_loc_list.append(B_comb.tocsr())
        g_nodes_list.append(g)
        rows_free_list.append(rows_free)

    return patch_list, B_i_loc_list, g_nodes_list, rows_free_list, free, fixed


def assemble_Btilde_reduced_from_online(
    world,
    B_i_loc_list,
    g_nodes_list,
    rows_free_list,
    free
):
    """
    Assemble the global free-by-free matrix B_tilde from local patch operators.

    Inputs
    ------
    B_i_loc_list : list[csr_matrix]
        Each B_i is (n_free_patch x n_patch).
    g_nodes_list : list[np.ndarray]
        For patch i, g_nodes_list[i] gives the global fine node indices (length = n_patch).
    rows_free_list : list[np.ndarray]
        For patch i, rows_free_list[i] gives the global FREE row indices (length = n_free_patch),
        aligned with B_i rows.
    free : np.ndarray
        Global FREE node indices (size n_free).

    Returns
    -------
    B_tilde_free : csr_matrix, shape (n_free, n_free)
        The global preconditioner matrix assembled on FREE-DOFs.
    """

    n_free = free.size
    data = []
    rows = []
    cols = []

    # Build fast lookup: global fine node -> global free index (or -1 if fixed)
    inv_free = -np.ones(world.NpFine, dtype=int)
    inv_free[free] = np.arange(n_free)

    for B_i, g_nodes, rows_free in zip(B_i_loc_list, g_nodes_list, rows_free_list):
        # Global FREE columns corresponding to patch columns; -1 for fixed
        cols_free_full = inv_free[g_nodes]             # length = n_patch
        keep = cols_free_full >= 0                     # mask to keep FREE columns only
        if not np.any(keep):
            continue

        # Restrict local matrix to FREE columns
        # B_i: (n_free_patch x n_patch) -> (n_free_patch x n_free_cols_kept)
        B_i_sub = B_i[:, keep].tocoo(copy=False)
        sub_cols = cols_free_full[keep]                # global FREE column ids

        # Scatter-add triplets
        # For each nonzero, map row via rows_free and col via sub_cols
        rr = rows_free[B_i_sub.row]
        cc = sub_cols[B_i_sub.col]
        vv = B_i_sub.data

        rows.append(rr)
        cols.append(cc)
        data.append(vv)

    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        B_tilde_free = sp.coo_matrix((data, (rows, cols)), shape=(n_free, n_free)).tocsr()
    else:
        B_tilde_free = sp.csr_matrix((n_free, n_free))

    return B_tilde_free
