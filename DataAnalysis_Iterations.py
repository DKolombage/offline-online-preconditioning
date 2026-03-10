import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER INPUT
# ============================================================

EXP_DIRS = [
    "data/Incl/cnt_10/",
    "data/Incl/cnt_100/",
    "data/Incl/cnt_500/",
]

#Incl       --->  "data/Incl/cnt_10/", "data/Incl/cnt_100/", "data/Incl/cont500/",
#InclLshape --->  "data/InclLshape/cnt_10/","data/InclLshape/cnt_100/","data/InclLshape/cont500/",
#Incl_Shift --->  "data/Inclshift/cnt_10/", "data/Inclshift/cnt_100/",
TITLES = [
    r"$\alpha=0.1,\ \beta=1$",
    r"$\alpha=0.1,\ \beta=10$",
    r"$\alpha=0.1,\ \beta=50$",
]

METHODS = ["direct", "A0", "OO"]

OUTFIG = "data/Incl/Incl_contrast_side_by_side_with_variation.png"
# InclLshape --->   OUTFIG = "data/InclLshape/Incl_contrast_side_by_side_with_variation.png"
# Incl_Shift --->  OUTFIG = "data/Inclshift/Incl_contrast_side_by_side_with_variation.png"
os.makedirs(os.path.dirname(OUTFIG), exist_ok=True)

def remove_outliers_iqr(x):
    x = np.asarray(x)
    if x.size < 4:
        return x, np.zeros_like(x, dtype=bool)

    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (x >= lower) & (x <= upper)
    return x[mask], ~mask

def compute_mean_std_from_folder(folder):
    results = {m: {} for m in METHODS}

    files = sorted(f for f in os.listdir(folder) if f.endswith(".npz"))

    for fname in files:
        for method in METHODS:
            if fname.startswith(method + "_p"):
                p = float(fname.replace(method + "_p", "").replace(".npz", ""))
                Z = np.load(os.path.join(folder, fname), allow_pickle=True)

                iters = np.asarray(Z["iters"]).ravel()
                conv  = np.asarray(Z["converged"]).ravel()
                valid = iters[conv == True]

                if valid.size == 0:
                    results[method][p] = (np.nan, np.nan)
                    continue

                filtered, _ = remove_outliers_iqr(valid)
                results[method][p] = (
                    float(np.mean(filtered)),
                    float(np.std(filtered))
                )
    return results

all_data = []
ymin= np.inf
ymax =  -np.inf  

for folder in EXP_DIRS:
    stats = compute_mean_std_from_folder(folder)
    all_data.append(stats)

    for method in METHODS:
        for p, (mu, sig) in stats[method].items():
            if np.isfinite(mu):
                ymax = max(ymax, mu + sig)

fig, axes = plt.subplots(1, len(EXP_DIRS), figsize=(5*len(EXP_DIRS), 4), sharey=True)

if len(EXP_DIRS) == 1:
    axes = [axes]

MARKERS = {
    "direct": ("o", "Direct-DD"),
    "A0":     ("s", "ND-DD"),
    "OO":     ("^", "OO-DD")
} #Direct $\\tilde B$

for ax, stats, title in zip(axes, all_data, TITLES):

    for method in METHODS:
        ps = sorted(stats[method].keys())
        means = [stats[method][p][0] for p in ps]
        stds  = [stats[method][p][1] for p in ps]

        marker, label = MARKERS[method]

        ax.errorbar(
            ps, means, yerr=stds,
            marker=marker,
            linestyle='-',
            capsize=4,
            linewidth=2,
            label=label
        )

    ax.set_title(title)
    ax.set_xlabel("Defect probability $p$")
    ax.grid(True)

axes[0].set_ylabel("Average PCG iterations (mean ± std)")

for ax in axes:
    ax.set_ylim(0, 1.02*ymax)  # change here to ax.set_ylim(0, 1.02*ymax) ax.set_ylim(0.98*ymin, 1.02*ymax) 

axes[-1].legend(loc="upper left")

plt.tight_layout()
plt.savefig(OUTFIG, dpi=200)
plt.close()

print("\nSaved figure to:")
print(f"  {OUTFIG}")


