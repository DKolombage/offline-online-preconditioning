import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import world as wrld
import build_coefficient, util, fem

# ==============================================================
# Reference coefficients
# ==============================================================
def compute_offline_coefficients(world, NepsilonElement, k, boundaryConditions, model):
    """
    Build the list of reference fine-scale coefficients on a reference patch.
    """
    patch = wrld.construct_center_node_patch(world, k)

    if model["name"] == "incl":
        aRefList = build_coefficient.build_inclusionbasis_2d(
            patch.NPatchCoarse,
            NepsilonElement,
            world.NCoarseElement,
            model["bgval"],
            model["inclval"],
            model["left"],
            model["right"],
        )
    elif model['name'] in ['inclfill', 'inclshift', 'incllshape']:
        aRefList = build_coefficient.build_inclusionbasis_change_2d(patch.NPatchCoarse, NepsilonElement, world.NCoarseElement,
                                                             model['bgval'], model['inclval'], model['left'],
                                                             model['right'], model)
    else:
        raise NotImplementedError("Unsupported model type for offline coefficients.")

    return aRefList


# ==============================================================
# Local B_loc = A_ff^{-1}  on reference patch
# ==============================================================
def compute_offline_patch_Bloc(patch, aRefList_patchlocal):
    """
    For each reference coefficient A^{(l)} on the reference patch, assemble
        B_loc[l] = A_ff^{-1} R_fS
    where:
        - A_ff   = local stiffness restricted to free interior nodes
        - R_fS   = identity-based restriction operator selecting ALL patch nodes
                   and mapping them to the free interior DOFs.
    Returns
    -------
    B_loc_list : list of csr_matrix
        Each entry has shape (n_free_patch x n_patch) and maps patch nodal values -> interior-patch values.
    free_patch : ndarray
        Local patch indices of interior DOFs.
    """

    # Patch dimensions
    NPF = patch.NPatchFine                        
    n_patch = int(np.prod(NPF + 1))   # number of fine nodes in patch
    d = len(NPF)

    # Determine patch boundary nodes and free (interior) nodes
    # boundaryMap = True on ALL patch boundaries -> Dirichlet BC on patch boundary
    bmap = np.ones((d, 2), dtype=bool)
    fixed_patch = util.boundarypIndexMap(NPF, boundaryMap=bmap)
    free_patch  = np.setdiff1d(np.arange(n_patch), fixed_patch)

    #  the restriction operator R_fS (identity rows)
    # Shape: (n_free_patch, n_patch)
    # Entry (i,j) = 1 if j = free_patch[i], else 0
    R_fS = sp.csr_matrix(
        (np.ones(len(free_patch)), (np.arange(len(free_patch)), free_patch)),
        shape=(len(free_patch), n_patch)
    )

    B_loc_list = []

    for aLoc in aRefList_patchlocal:

        # Assemble stiffness matrix for this coefficient
        A_patch = fem.assemblePatchMatrix(NPF, patch.world.ALocFine, aLoc)

        # Extract A_ff block -> free interior rows & columns
        A_ff = A_patch[free_patch][:, free_patch].tocsc()

        # Factorization for all RHS solves
        LU = spla.splu(A_ff)

        # Apply solve to identity-restriction matrix
        # B_loc = A_ff^{-1} R_fS
        B_loc = LU.solve(R_fS.toarray())   # shape: (n_free_patch, n_patch)

        B_loc_list.append(sp.csr_matrix(B_loc))

    return B_loc_list, free_patch
    

def build_offline_data_for_Btilde(world, NepsilonElement, k, boundaryConditions, model):
    """
    Returns
    -------
    aRefList : list of ndarray
        Reference coefficients on patch.
    B_loc_list : list of csr_matrix
        Reference local B_loc operators (A_ff^{-1} M_fS).
    free_patch : ndarray
        Local free-node indices within the patch.
    """
    t0 = time.perf_counter()
    aRefList = compute_offline_coefficients(world, NepsilonElement, k, boundaryConditions, model)
    patch = wrld.construct_center_node_patch(world, k)
    B_loc_list, free_patch = compute_offline_patch_Bloc(patch, aRefList)
    t1 = time.perf_counter()

    print(f"[offline_data] Built {len(aRefList)} reference B_loc operators in {t1 - t0:.2f}s")
    return aRefList, B_loc_list, free_patch


# ==============================================================
# OFFLINE B0 — coarse-element reference data
# ==============================================================
def compute_offline_coarse_coefficients_B0(world, NepsilonElement, model):
    """
    Build reference coefficients aRefList on ONE coarse element.
    """
    #alpha = model["alpha"]
    #beta  = model["beta"]

    # One coarse element only
    NPatch = np.ones_like(world.NWorldCoarse, dtype=int)

    if model["name"] == "incl":
        aRefList = build_coefficient.build_inclusionbasis_2d(
            NPatch,
            np.asarray(NepsilonElement, int),
            np.asarray(world.NCoarseElement, int),
            model["bgval"],
            model["inclval"],
            model["left"],
            model["right"],
        )
    elif model['name'] in ['inclfill', 'inclshift', 'incllshape']:
        aRefList = build_coefficient.build_inclusionbasis_change_2d(NPatch, np.asarray(NepsilonElement, int), np.asarray(world.NCoarseElement, int),
                                                             model['bgval'], model['inclval'], model['left'],
                                                             model['right'], model)
    else:
        raise NotImplementedError("Unsupported model type for offline coefficients.")
    return aRefList


def compute_offline_coarse_B0loc(world, aRefList_B0):
    """
    OFFLINE B0: reference element stiffness matrices K_T^{(l)}.

    For each reference coefficient aLoc in aRefList_B0 (defined on ONE coarse
    element), assemble the fine-grid stiffness matrix on that coarse element:
        K_T_ref[l] = A_T(aLoc)
    where A_T is the standard element-wise FEM stiffness assembled on the
    fine grid inside a single coarse element.

    These K_T_ref[l] are later combined online via mu to form
    the local stiffness K_T for an arbitrary local coefficient a_loc:
        K_T(a_loc) ≈ sum_l mu_l(a_loc) K_T_ref[l].

    Returns
    -------
    K_T_ref_list : list of csr_matrix
        Each entry has shape (Np_elem x Np_elem), where Np_elem is the number
        of fine nodes on one coarse element ((NCoarseElement + 1)).
    """
    import scipy.sparse as sp

    NCE = np.asarray(world.NCoarseElement, dtype=int)   # fine elems per coarse elem
    d   = len(NCE)
    if d not in (1, 2):
        raise NotImplementedError("compute_offline_coarse_B0loc currently supports 1D/2D.")

    # Number of fine nodes on ONE coarse element
    Np_elem = int(np.prod(NCE + 1))

    # We use the fine-grid local operator for one coarse element.
    ALocElem = world.ALocFine

    K_T_ref_list = []

    for aLoc in aRefList_B0:
        # aLoc is the fine-element coefficient on THIS coarse element
        # Assemble stiffness on this coarse element
        K_T_full = fem.assemblePatchMatrix(NCE, ALocElem, aLoc).tocsr()
        if K_T_full.shape != (Np_elem, Np_elem):
            raise ValueError(
                f"K_T_full shape {K_T_full.shape} != ({Np_elem},{Np_elem}) "
                "in compute_offline_coarse_B0loc."
            )
        K_T_ref_list.append(K_T_full)

    return K_T_ref_list