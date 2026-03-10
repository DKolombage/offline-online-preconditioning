import os
import time
import numpy as np
import matplotlib.pyplot as plt
from MainAlgorithms import (
    run_direct_Btilde,
    run_direct_Btilde_A0,
    run_offlineonline_Btilde,
)

# ==============================================================
#  CONFIGURATION 
# ==============================================================
cfg_base = dict(
    NCoarse=[16,16],
    Nepsilon=[32,32],
    NFine=[128,128],

    k=1,
    alpha=0.1,
    beta=10,

    boundaryConditions=None,

    rtol=1e-6,
    atol=1e-7,
    maxiter=200,

    model_type='inclshift'
)

np.random.seed(5)

# ==============================================================
#  USER INPUT 
# ==============================================================
pList = [0.00,0.02, 0.04,0.06, 0.08, 0.10] 
NSamples = 150

OUT_DIR = "data/Test/"
# data/Inclshift/cnt_10 -->  alpha =0.1, beta=1.0, Nsamples = 150
# data/Inclshift/cnt_100 -->  alpha =0.1, beta=10, Nsamples = 150
os.makedirs(OUT_DIR, exist_ok=True)


avg_direct = []
avg_A0     = []
avg_OO     = []

def average_iters(Z):
    iters = np.asarray(Z["iters"]).ravel()          # number of iterations per sample
    conv  = np.asarray(Z["converged"]).ravel()      # boolean array

    converged_iters = iters[conv == True]

    if converged_iters.size == 0:
        return np.nan   # no convergence at all -> return NaN

    return float(np.mean(converged_iters))


# ==============================================================
# RUN ALL METHODS
# ==============================================================
for p in pList:

    print("\n=====================================================")
    print(f"                     p = {p:.3f}")
    print("=====================================================\n")

    cfg = dict(cfg_base)
    cfg["p"] = float(p)
    cfg["NSamples"] = int(NSamples)

    # Output files
    out_direct = os.path.join(OUT_DIR, f"direct_p{p:.3f}.npz")
    out_A0     = os.path.join(OUT_DIR, f"A0_p{p:.3f}.npz")
    out_OO     = os.path.join(OUT_DIR, f"OO_p{p:.3f}.npz")

    # ======================================================
    # Direct Btilde
    # ======================================================
    print(f"[Run] Direct Btilde, p={p}")
    run_direct_Btilde(cfg, out_path=out_direct)

    Z1 = np.load(out_direct, allow_pickle=True)
    avg_direct.append( average_iters(Z1) )

    # ======================================================
    # Direct A0 (zero-defect preconditioner)
    # ======================================================
    print(f"[Run] Direct A0 Btilde, p={p}")
    run_direct_Btilde_A0(cfg, out_path=out_A0)

    Z2 = np.load(out_A0, allow_pickle=True)
    avg_A0.append( average_iters(Z2) )

    # ======================================================
    # Offline–Online Btilde
    # ======================================================
    print(f"[Run] Offline-Online Btilde, p={p}")
    run_offlineonline_Btilde(cfg, out_path=out_OO)

    Z3 = np.load(out_OO, allow_pickle=True)
    avg_OO.append( average_iters(Z3) )

np.savez(
    os.path.join(OUT_DIR, "summary_iters_3methods.npz"),
    pList=np.array(pList),
    iters_direct=np.array(avg_direct),
    iters_A0=np.array(avg_A0),
    iters_OO=np.array(avg_OO),
)

print("\nSaved summary_iters_3methods.npz")

plt.figure(figsize=(8,6))
plt.plot(pList, avg_direct, 'o-', label="Direct-DD")
plt.plot(pList, avg_A0, 's-', label="ND-DD")
plt.plot(pList, avg_OO, '^-', label="OO-DD")

plt.xlabel("Defect probability p")
plt.ylabel("Average iterations")
plt.title("Average PCG Iterations vs p")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iters_3methods.png"), dpi=200)
plt.show()
