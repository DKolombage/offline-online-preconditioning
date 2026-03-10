import os
import numpy as np
import re
import csv

# ============================================================
# USER INPUT
# ============================================================
FOLDER = 'data/InclShift/cnt_100' 
# Incl         --- "data/Incl/cnt_10", "data/Incl/cnt_100", "data/Incl/cnt_500"
# Incl-L-shape --- "data/InclLshape/cnt_10", "data/InclLshape/cnt_100", "data/InclLshape/cnt_500"
# Incl-Shift   --- "data/InclShift/cnt_10", "data/InclShift/cnt_100"

METHODS = ["direct", "A0", "OO"]           # prefixes in filenames


def extract_p(fname):
    """
    Extracts p from filenames like:
      direct_p0.040.npz  -->  0.04
      A0_p0.10.npz      -->  0.10
      OO_p0.000.npz      -->  0.00
    """
    match = re.search(r"p(\d+\.\d+)", fname)
    return float(match.group(1)) if match else None

# ============================================================
# GATHER FILES & PROCESS
# ============================================================
all_files = sorted(
    [f for f in os.listdir(FOLDER) if f.endswith(".npz") and "summary" not in f]
)

# Map data: results[method][p] = dict(...)
results = {m: {} for m in METHODS}

for fname in all_files:
    p = extract_p(fname)
    if p is None:
        continue

    if fname.startswith("direct"):
        method = "direct"
    elif fname.startswith("A0"):
        method = "A0"
    elif fname.startswith("OO"):
        method = "OO"
    else:
        continue

    # Load npz
    Z = np.load(os.path.join(FOLDER, fname), allow_pickle=True)
    iters = np.asarray(Z["iters"]).ravel()
    conv  = np.asarray(Z["converged"]).ravel()

    total = conv.size
    num_converged = int(np.sum(conv))
    num_not_converged = total - num_converged

    if num_converged > 0:
        avg_iters = float(np.mean(iters[conv == True]))
    else:
        avg_iters = np.nan

    # Store results
    results[method][p] = {
        "total": total,
        "converged": num_converged,
        "not_converged": num_not_converged,
        "conv_rate": num_converged / total * 100.0,
        "avg_iters": avg_iters,
    }

p_values = sorted({extract_p(f) for f in all_files if extract_p(f) is not None})

print("\n======================= SUMMARY =======================\n")
for p in p_values:
    print(f"\n*** p = {p:.3f} ***")
    for method in METHODS:
        if p not in results[method]:
            print(f"  {method:>8}:   (no data)")
            continue
        R = results[method][p]
        print(
            f"  {method:>8}: converged {R['converged']:3d}/{R['total']:3d}"
            f"  | rate = {R['conv_rate']:6.2f}%"
            f"  | avg iters = {R['avg_iters']:.2f}"
        )

# IF SAVE TO CSV:
save_csv = True
if save_csv:
    csv_file = os.path.join(FOLDER, "convergence_summary.csv")

    with open(csv_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "p", "total", "converged", "not_converged",
                    "conv_rate(%)", "avg_iters"])

        for method in METHODS:
            for p in sorted(results[method].keys()):
                R = results[method][p]
                w.writerow([
                    method, p, R["total"], R["converged"],
                    R["not_converged"], R["conv_rate"], R["avg_iters"]
                ])

    print(f"\nCSV summary saved to: {csv_file}")
