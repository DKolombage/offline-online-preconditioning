import os
import glob
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_all_rmse_summaries(summary_dir):
    files = sorted(glob.glob(os.path.join(summary_dir, "rmse_summary_p_*.npz")))
    if not files:
        raise FileNotFoundError(f"No rmse_summary_p_*.npz files found in {summary_dir}")

    rows = []
    for path in files:
        data = np.load(path, allow_pickle=True)

        row = {
            "file": path,
            "p": float(data["p"]),
            "n_samples": int(data["n_samples"]),
            "rmse_abs_oo": float(data["rmse_abs_oo"]),
            "rmse_abs_a0": float(data["rmse_abs_a0"]),
            "rmse_rel_oo": float(data["rmse_rel_oo"]),
            "rmse_rel_a0": float(data["rmse_rel_a0"]),
        }

        if "seed" in data.files:
            row["seed"] = int(data["seed"])
        else:
            row["seed"] = None

        if "p_folder" in data.files:
            row["p_folder"] = str(data["p_folder"])
        else:
            row["p_folder"] = ""

        rows.append(row)

    rows.sort(key=lambda r: r["p"])
    return rows


def save_combined_summary(rows, summary_dir):
    p = np.array([r["p"] for r in rows], dtype=float)
    n_samples = np.array([r["n_samples"] for r in rows], dtype=int)
    rmse_abs_oo = np.array([r["rmse_abs_oo"] for r in rows], dtype=float)
    rmse_abs_a0 = np.array([r["rmse_abs_a0"] for r in rows], dtype=float)
    rmse_rel_oo = np.array([r["rmse_rel_oo"] for r in rows], dtype=float)
    rmse_rel_a0 = np.array([r["rmse_rel_a0"] for r in rows], dtype=float)

    npz_path = os.path.join(summary_dir, "rmse_summary_abs_rel_combined.npz")
    csv_path = os.path.join(summary_dir, "rmse_summary_abs_rel_combined.csv")

    np.savez_compressed(
        npz_path,
        p=p,
        n_samples=n_samples,
        rmse_abs_oo=rmse_abs_oo,
        rmse_abs_a0=rmse_abs_a0,
        rmse_rel_oo=rmse_rel_oo,
        rmse_rel_a0=rmse_rel_a0,
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "p",
            "n_samples",
            "rmse_abs_oo",
            "rmse_abs_a0",
            "rmse_rel_oo",
            "rmse_rel_a0",
        ])
        for r in rows:
            writer.writerow([
                r["p"],
                r["n_samples"],
                r["rmse_abs_oo"],
                r["rmse_abs_a0"],
                r["rmse_rel_oo"],
                r["rmse_rel_a0"],
            ])

    return npz_path, csv_path


def make_plots(rows, summary_dir, show=False):
    p = np.array([r["p"] for r in rows], dtype=float)
    rmse_abs_oo = np.array([r["rmse_abs_oo"] for r in rows], dtype=float)
    rmse_abs_a0 = np.array([r["rmse_abs_a0"] for r in rows], dtype=float)
    rmse_rel_oo = np.array([r["rmse_rel_oo"] for r in rows], dtype=float)
    rmse_rel_a0 = np.array([r["rmse_rel_a0"] for r in rows], dtype=float)

    abs_png = os.path.join(summary_dir, "rmse_abs_vs_p.png")
    rel_png = os.path.join(summary_dir, "rmse_rel_vs_p.png")

    plt.figure()
    plt.plot(p, rmse_abs_oo, marker="o", label=r"OO-DD")
    plt.plot(p, rmse_abs_a0, marker="s", label=r"ND-DD")
    plt.xlabel("Defect probability p")
    plt.ylabel("Absolute RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(abs_png, dpi=250)

    plt.figure()
    plt.plot(p, rmse_rel_oo, marker="o", label=r"OO-DD")
    plt.plot(p, rmse_rel_a0, marker="s", label=r"ND-DD")
    plt.xlabel("Defect probability p")
    plt.ylabel("Relative RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rel_png, dpi=250)

    if show:
        plt.show()
    else:
        plt.close("all")

    return abs_png, rel_png


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-p RMSE summary files and plot RMSE vs p."
    )
    parser.add_argument("summary_dir", help="Folder containing rmse_summary_p_*.npz files")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures"
    )
    args = parser.parse_args()

    rows = load_all_rmse_summaries(args.summary_dir)
    npz_path, csv_path = save_combined_summary(rows, args.summary_dir)
    abs_png, rel_png = make_plots(rows, args.summary_dir, show=args.show)

    print(f"Loaded {len(rows)} per-p summary files from: {args.summary_dir}")
    print(f"Saved combined NPZ : {npz_path}")
    print(f"Saved combined CSV : {csv_path}")
    print(f"Saved abs plot     : {abs_png}")
    print(f"Saved rel plot     : {rel_png}")

    print("\nValues used for plotting:")
    for r in rows:
        print(
            f"p={r['p']:.3f}  "
            f"AbsOO={r['rmse_abs_oo']:.4e}  "
            f"AbsA0={r['rmse_abs_a0']:.4e}  "
            f"RelOO={r['rmse_rel_oo']:.4e}  "
            f"RelA0={r['rmse_rel_a0']:.4e}"
        )


if __name__ == "__main__":
    main()