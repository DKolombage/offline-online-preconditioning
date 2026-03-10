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
# Incl-L-shape --- "data/InclLshape/cnt_10", "data/InclLshape/cnt_100", "data/InclLshape/cnt_500"
# Incl-Shift   --- "data/InclShift/cnt_10", "data/InclShift/cnt_100"

TITLES = [
    r"$\alpha=0.1,\ \beta=1$",
    r"$\alpha=0.1,\ \beta=10$",
    r"$\alpha=0.1,\ \beta=50$",
]

OUTFIG = "data/Incl/Incl_contrast_side_by_side.png"
# Incl-L-shape --- "data/InclLshape/InclLshape_contrast_side_by_side.png"
# Incl-Shift   --- "data/InclShift/InclShift_contrast_side_by_side.png"
# ============================================================
os.makedirs(os.path.dirname(OUTFIG), exist_ok=True)

data = []
ymin = np.inf
ymax = -np.inf

for d in EXP_DIRS:
    Z = np.load(os.path.join(d, "summary_iters_3methods.npz"))

    p  = Z["pList"]
    yD = Z["iters_direct"]
    yA = Z["iters_A0"]
    yO = Z["iters_OO"]

    data.append((p, yD, yA, yO))

    ymin = min(ymin, np.nanmin(yD), np.nanmin(yA), np.nanmin(yO))
    ymax = max(ymax, np.nanmax(yD), np.nanmax(yA), np.nanmax(yO))

n = len(data)
fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=True)

if n == 1:
    axes = [axes]

for ax, (p, yD, yA, yO), title in zip(axes, data, TITLES):

    ax.plot(p, yD, "o-", label="Direct-DD")
    ax.plot(p, yA, "s-", label="$ND-DD")
    ax.plot(p, yO, "^-", label="OO-DD$")

    ax.set_title(title)
    ax.set_xlabel("Defect probability $p$")
    ax.grid(True)

axes[0].set_ylabel("Average PCG iterations")

for ax in axes:
    ax.set_ylim(bottom=0.98*ymin, top=1.02*ymax)

axes[-1].legend(loc="upper left")

plt.tight_layout()
plt.savefig(OUTFIG, dpi=200)
plt.show()
