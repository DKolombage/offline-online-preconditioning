import os
import numpy as np
import matplotlib.pyplot as plt
import csv


# ============================================================
# USER INPUT
# ============================================================
FOLDER = "data/InclLshape/cnt_10"   #
# Incl         --- "data/Incl/cnt_10", "data/Incl/cnt_100", "data/Incl/cnt_500"
# Incl-L-shape --- "data/InclLshape/cnt_10", "data/InclLshape/cnt_100", "data/InclLshape/cnt_500"
# Incl-Shift   --- "data/InclShift/cnt_10", "data/InclShift/cnt_100"

METHODS = ["direct", "A0", "OO"]    # filename prefixes

print("g")
# ============================================================
# OUTLIER REMOVAL (IQR METHOD)
# ============================================================
def remove_outliers_iqr(x):
    """
    Removes outliers using the Interquartile Range (IQR) rule.
    Returns: filtered_data, outlier_mask
    """
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


# ============================================================
# COLLECT FILES AND GROUP BY METHOD + p
# ============================================================
files = sorted([f for f in os.listdir(FOLDER) if f.endswith(".npz")])

results = {m: {} for m in METHODS}

for filename in files:
    for method in METHODS:
        if filename.startswith(method + "_p"):
            p_str = filename.replace(method + "_p", "").replace(".npz", "")
            p_val = float(p_str)

            full_path = os.path.join(FOLDER, filename)
            results[method][p_val] = full_path


# ============================================================
# ANALYSE ITERATIONS + OUTLIERS
# ============================================================
raw_avg      = {m: {} for m in METHODS}
filtered_avg = {m: {} for m in METHODS}
num_removed  = {m: {} for m in METHODS}
std_filtered = {m: {} for m in METHODS}

for method in METHODS:
    for p in sorted(results[method].keys()):

        Z = np.load(results[method][p], allow_pickle=True)

        iters = np.asarray(Z["iters"]).ravel()
        conv  = np.asarray(Z["converged"]).ravel()

        # Only converged samples are considered
        valid = iters[conv == True]

        if valid.size == 0:
            raw_avg[method][p] = np.nan
            filtered_avg[method][p] = np.nan
            num_removed[method][p] = 0
            std_filtered[method][p] = np.nan
            continue

        # Raw average
        raw_avg[method][p] = float(np.mean(valid))

        # Outlier removal
        filtered, removed_mask = remove_outliers_iqr(valid)

        filtered_avg[method][p] = float(np.mean(filtered))
        std_filtered[method][p] = float(np.std(filtered))
        num_removed[method][p] = int(np.sum(removed_mask))


# ============================================================
# SAVE RESULTS TO CSV
# ============================================================
csv_file = os.path.join(FOLDER, "convergence_summary_outlier_filtered.csv")

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "method",
        "p",
        "raw_avg_iterations",
        "filtered_avg_iterations",
        "filtered_std",
        "num_outliers_removed"
    ])

    for method in METHODS:
        for p in sorted(filtered_avg[method].keys()):
            writer.writerow([
                method,
                p,
                raw_avg[method][p],
                filtered_avg[method][p],
                std_filtered[method][p],
                num_removed[method][p]
            ])

print("\n Outlier-filtered convergence summary saved to:")
print(f"   {csv_file}")


# ============================================================
# PLOT FILTERED AVERAGE ITERATIONS 
# ============================================================
plt.figure(figsize=(8, 6))

for method in METHODS:
    ps = sorted(filtered_avg[method].keys())
    means = [filtered_avg[method][p] for p in ps]
    stds  = [std_filtered[method][p] for p in ps]

    plt.errorbar(
        ps,
        means,
        yerr=stds,
        marker='o',
        capsize=4,
        linewidth=2,
        label=method
    )

plt.xlabel("Defect probability p")
plt.ylabel("Average iterations (outliers removed)")
plt.title("Average PCG Iterations vs p")
plt.grid(True)
plt.legend()
plt.tight_layout()

plot_file = os.path.join(FOLDER, "iters_3methods_outlier_filtered.png")
plt.savefig(plot_file, dpi=200)
plt.close()

