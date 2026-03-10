import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER INPUT
# ============================================================

EXP_DIR = "data/Test/"   # <-- experiment folder
# Incl         --- "data/Incl/cnt_500/"
# Incl-L-shape --- "data/InclLshape/cnt_500/"
# Incl-Shift   --- "data/InclShift/cnt_500/"
pList   = [0.02, 0.06, 0.10]

METHODS = {
    "Direct-DD": "direct_p{p:.3f}.npz",
    "ND-DD":     "A0_p{p:.3f}.npz",
    "OO-DD": "OO_p{p:.3f}.npz",
}

ERROR_KEYS = {
    "Residual": "residual_history",
    "L2":       "l2_history",
    "Energy":   "en_history",
}


def compute_rmse_curve(histories):
    if len(histories) == 0:
        return np.array([]), np.array([])

    maxlen = max(len(h) for h in histories)
    rmse = np.zeros(maxlen)

    for k in range(maxlen):
        vals = [h[k] for h in histories if len(h) > k]
        rmse[k] = np.sqrt(np.mean(np.square(vals))) if vals else np.nan

    return np.arange(1, maxlen + 1), rmse

def plot_error_type(error_name, error_key):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for j, p in enumerate(pList):
        ax = axes[j]

        for method, pattern in METHODS.items():
            file_path = os.path.join(EXP_DIR, pattern.format(p=p))
            if not os.path.exists(file_path):
                continue

            Z = np.load(file_path, allow_pickle=True)
            histories = Z[error_key]

            iters, rmse = compute_rmse_curve(histories)
            ax.semilogy(iters, rmse, lw=2, label=method)

        ax.set_title(f"$p={p:.2f}$")
        ax.set_xlabel("Iteration $k$")
        ax.grid(True, which="both", alpha=0.3)

        if j == 0:
            ax.set_ylabel(f"RMSE ({error_name})")

    axes[0].legend()
    fig.tight_layout()

    out_file = os.path.join(EXP_DIR, f"RMSE_{error_name}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"[saved] {out_file}")

for name, key in ERROR_KEYS.items():
    plot_error_type(name, key)

print("\nAll RMSE figures generated successfully.")
