import numpy as np
import coarse_data

# ================================================
# COMPUTE online mu (sum constraint 1) for Patch 
# ================================================
def compute_mu_for_patch(aFineLocal, aRefList, model):
    """
    mu computation for all inclusion-type models:
      - incl
      - inclfill
      - inclshift
      - inclLshape

    Assumptions:
      - aRefList[-1] is the background (no defect)
      - aRefList[l], l < bg_idx, each encode exactly one defect
      - coefficients take values in {alpha, beta}

    Returns:
      mu : ndarray, shape (len(aRefList),), sum(mu) = 1
    """
    tol = 1e-12
    name = model["name"].lower()

    # Coefficient values
    if name in ["incl", "inclfill", "inclshift", "incllshape"]:
        alpha = float(model["bgval"])
        beta  = float(model["inclval"])
    else:
        raise ValueError(f"Unsupported model type '{name}'")

    L = len(aRefList)
    bg_idx = L - 1

    a_loc = np.asarray(aFineLocal, dtype=float)
    a_bg  = np.asarray(aRefList[bg_idx], dtype=float)

    mu = np.zeros(L, dtype=float)

    # Identify defect region in local coefficient
    if name in ["incl", "incllshape"]:
        mask_def = (np.abs(a_bg - beta) < tol) & (np.abs(a_loc - alpha) < tol) # beta disappears

    elif name in ["inclfill", "inclshift"]:
        mask_def = (np.abs(a_loc - beta) < tol) & (np.abs(a_bg - alpha) < tol) # beta appears where background had alpha

    else:
        raise ValueError(f"Unhandled model '{name}'")

    num_def = int(mask_def.sum())

    if num_def == 0:           # No defect --> pure background
        mu[bg_idx] = 1.0
        return mu

    # Overlap with each reference defect
    for l in range(L - 1):
        a_ref = np.asarray(aRefList[l], dtype=float)

        if name in ["incl", "incllshape"]:
            mask_ref = (np.abs(a_bg - beta) < tol) & (np.abs(a_ref - alpha) < tol)

        elif name in ["inclfill", "inclshift"]:
            mask_ref = (np.abs(a_ref - beta) < tol) & (np.abs(a_bg - alpha) < tol)

        overlap = int((mask_def & mask_ref).sum())
        if overlap > 0:
            mu[l] = overlap / num_def

    # Background 
    mu[bg_idx] = max(0.0, 1.0 - mu[:-1].sum())

    return mu

# ==============================================================
# ONLINE B0 — combine reference element matrices using mu
# ==============================================================
def compute_online_coarse_B0loc(aFineLocal, aRefList, K0_ref_list, model):
    """
    Given coefficient aFineLocal on a coarse element,
    compute:
        K0_T ≈ sum_l mu[l] K0_ref_list[l]
        M0_T ≈ sum_l mu[l] M0_ref_list[l]
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

    return K0_T.tocsr() #, M0_T.tocsr()

