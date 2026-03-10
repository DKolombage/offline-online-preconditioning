import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

from BMain import save_B_per_sample

def sparse_to_npz_dict(A: sp.spmatrix):
    A = A.tocsr()
    return dict(
        data=A.data,
        indices=A.indices,
        indptr=A.indptr,
        shape=np.array(A.shape, dtype=np.int64),
    )

def sparse_from_npz(npz):
    shape = tuple(npz["shape"].tolist())
    return sp.csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=shape)

def load_B_total(path):
    npz = np.load(path, allow_pickle=True)
    if "B_total_free" in npz.files:
        Bobj = npz["B_total_free"]
        if hasattr(Bobj, "item"):
            B = Bobj.item()
            if sp.issparse(B):
                return B.tocsr(), npz["freeFine"].astype(int)

    needed = {"data", "indices", "indptr", "shape"}
    if needed.issubset(set(npz.files)):
        return sparse_from_npz(npz), npz["freeFine"].astype(int)

    raise RuntimeError(f"Sparse matrix from {path} loading error. "
                       f"Keys found: {npz.files}")


def rmse_over_samples(folder, NSamples, seed=0, relative=False):
    """
    Compute RMSE over samples for:
      e_OO  = || (B - Btilde) v ||   (Btilde = OO)
      e_A0  = || (B - B(A0)) v ||

    v is a random vector on free DOFs (length = n_free).
    """
    rng = np.random.default_rng(seed)

    B0, free0 = load_B_total(os.path.join(folder, "B_direct_s0000.npz"))
    n_free = B0.shape[0]
    v = rng.standard_normal(n_free)

    errs_oo = []
    errs_a0 = []

    BA0, freeA0 = load_B_total(os.path.join(folder, "B_A0_s0000.npz"))
    if BA0.shape[0] != n_free:
        raise ValueError("A0 matrix size mismatch vs direct size.")

    for s in range(NSamples):
        Bd, freed = load_B_total(os.path.join(folder, f"B_direct_s{s:04d}.npz"))
        Boo, freeoo = load_B_total(os.path.join(folder, f"B_OO_s{s:04d}.npz"))

        yB  = Bd @ v
        yOO = Boo @ v
        yA0 = BA0 @ v

        eoo = np.linalg.norm(yB - yOO)
        ea0 = np.linalg.norm(yB - yA0)

        if relative:
            denom = np.linalg.norm(yB)
            denom = denom if denom > 0 else 1.0
            eoo /= denom
            ea0 /= denom

        errs_oo.append(eoo)
        errs_a0.append(ea0)

    # RMSE over samples (scalar errors):
    # RMSE = sqrt(mean(e_s^2))
    rmse_oo = float(np.sqrt(np.mean(np.square(errs_oo))))
    rmse_a0 = float(np.sqrt(np.mean(np.square(errs_a0))))
    return rmse_oo, rmse_a0

# Full sweep: run + postprocess + plot
def sweep_p_and_plot(cfg_base, p_list, out_root, seed_v=0, relative=False):
    """
    For each p:
      - set cfg["p"]=p
      - run save_B_per_sample -> save matrices into out_root/p_XXX/
      - compute RMSE metrics and plot vs p
    """
    os.makedirs(out_root, exist_ok=True)

    rmses_oo = []
    rmses_a0 = []

    for p in p_list:
        cfg = dict(cfg_base)
        cfg["p"] = float(p)

        folder = os.path.join(out_root, f"p_{p:.3f}")
        print(f"\n=== p={p:.3f} -> {folder} ===")

        save_B_per_sample(cfg, out_dir=folder, use_elementwise_B0=False)

        rmse_oo, rmse_a0 = rmse_over_samples(
            folder=folder,
            NSamples=int(cfg["NSamples"]),
            seed=seed_v,          
            relative=relative
        )
        rmses_oo.append(rmse_oo)
        rmses_a0.append(rmse_a0)

        print(f"RMSE  (B-Btilde)v : {rmse_oo:.6e}")
        print(f"RMSE  (B-B(A0))v  : {rmse_a0:.6e}")

    # plot
    plt.figure()
    plt.plot(p_list, rmses_oo, marker="o", label=r"OO-DD")
    plt.plot(p_list, rmses_a0, marker="s", label=r"ND-DD")
    plt.xlabel("probability p")
    plt.ylabel("relative RMSE" if relative else "RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(out_root, "rmse_vs_p.png")
    plt.savefig(fig_path, dpi=200)
    print(f"\nSaved plot: {fig_path}")
    plt.show()

    return np.array(rmses_oo), np.array(rmses_a0)


if __name__ == "__main__":
    cfg_base = dict(
        NCoarse=[16,16],
        NFine=[128,128],
        Nepsilon=[32,32],
        NSamples=100,
        k=1,
        alpha=0.1,
        beta=50.0,
        p=0.1,              
        model_type="incl",
        boundaryConditions=None
    )

    p_list = [0.00, 0.02, 0.04,0.06,0.08,0.1]
    out_root = "data/Operator_data/Bdata/"

    sweep_p_and_plot(cfg_base, p_list, out_root, seed_v=123, relative=True)
