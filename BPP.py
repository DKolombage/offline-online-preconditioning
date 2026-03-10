import os
import re
import csv
import glob
import argparse
import numpy as np
import scipy.sparse as sp


# Load sparse matrix 
def load_B_total(path):
    npz = np.load(path, allow_pickle=True)

    if "B_total_free" in npz.files:
        Bobj = npz["B_total_free"]
        B = Bobj.item() if hasattr(Bobj, "item") else Bobj
        if sp.issparse(B):
            return B.tocsr(), npz["freeFine"].astype(int)

    needed = {"data", "indices", "indptr", "shape"}
    if needed.issubset(set(npz.files)):
        shape = tuple(npz["shape"].tolist())
        B = sp.csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=shape)
        return B, npz["freeFine"].astype(int)

    raise RuntimeError(f"Could not load sparse matrix from {path}. Keys found: {npz.files}")


# Parse p from folder name
# e.g.  p_00    -> 0.00, p_02    -> 0.02
def parse_p_from_folder(folder_name: str) -> float:
    base = os.path.basename(os.path.normpath(folder_name))

    m = re.match(r"^p_([0-9]+(?:\.[0-9]+)?)$", base)
    if m:
        token = m.group(1)
        if "." in token:
            return float(token)
        return int(token) / 100.0

    raise ValueError(f"Cannot parse p from folder name: {base}")

def infer_sample_ids(folder):
    files = sorted(glob.glob(os.path.join(folder, "B_direct_s*.npz")))
    ids = []
    for path in files:
        m = re.search(r"B_direct_s(\d+)\.npz$", os.path.basename(path))
        if m:
            ids.append(int(m.group(1)))
    if not ids:
        raise FileNotFoundError(f"No B_direct_sXXXX.npz files found in {folder}")
    return ids

# Compute RMSE for one p-folder
def compute_rmse_one_folder(folder, sample_ids=None, seed=123):
    if sample_ids is None:
        sample_ids = infer_sample_ids(folder)

    sample_ids = sorted(sample_ids)
    if len(sample_ids) == 0:
        raise ValueError("No sample ids found.")

    rng = np.random.default_rng(seed)

    first_id = sample_ids[0]
    Bd0, _ = load_B_total(os.path.join(folder, f"B_direct_s{first_id:04d}.npz"))
    n = Bd0.shape[0]
    v = rng.standard_normal(n)

    BA0, _ = load_B_total(os.path.join(folder, f"B_A0_s{first_id:04d}.npz"))
    if BA0.shape[0] != n:
        raise ValueError("A0 size mismatch with direct matrix size.")

    abs_oo = []
    abs_a0 = []
    rel_oo = []
    rel_a0 = []

    for s in sample_ids:
        path_Bd  = os.path.join(folder, f"B_direct_s{s:04d}.npz")
        path_Boo = os.path.join(folder, f"B_OO_s{s:04d}.npz")

        Bd, _  = load_B_total(path_Bd)
        Boo, _ = load_B_total(path_Boo)

        if Bd.shape[0] != n or Boo.shape[0] != n:
            raise ValueError(f"Size mismatch at sample {s}")

        yB  = Bd @ v
        yOO = Boo @ v
        yA0 = BA0 @ v

        err_oo = np.linalg.norm(yB - yOO)
        err_a0 = np.linalg.norm(yB - yA0)

        denom = np.linalg.norm(yB)
        denom = denom if denom > 0 else 1.0

        abs_oo.append(err_oo)
        abs_a0.append(err_a0)
        rel_oo.append(err_oo / denom)
        rel_a0.append(err_a0 / denom)

    abs_oo = np.asarray(abs_oo, dtype=float)
    abs_a0 = np.asarray(abs_a0, dtype=float)
    rel_oo = np.asarray(rel_oo, dtype=float)
    rel_a0 = np.asarray(rel_a0, dtype=float)

    rmse_abs_oo = float(np.sqrt(np.mean(abs_oo**2)))
    rmse_abs_a0 = float(np.sqrt(np.mean(abs_a0**2)))
    rmse_rel_oo = float(np.sqrt(np.mean(rel_oo**2)))
    rmse_rel_a0 = float(np.sqrt(np.mean(rel_a0**2)))

    return {
        "n_samples": int(len(sample_ids)),
        "sample_ids": np.asarray(sample_ids, dtype=int),
        "rmse_abs_oo": rmse_abs_oo,
        "rmse_abs_a0": rmse_abs_a0,
        "rmse_rel_oo": rmse_rel_oo,
        "rmse_rel_a0": rmse_rel_a0,
    }


def save_summary(result, p_value, p_folder, output_dir, seed):
    os.makedirs(output_dir, exist_ok=True)

    base_tag = f"p_{p_value:.3f}"
    npz_path = os.path.join(output_dir, f"rmse_summary_{base_tag}.npz")
    csv_path = os.path.join(output_dir, f"rmse_summary_{base_tag}.csv")

    np.savez_compressed(
        npz_path,
        p=float(p_value),
        p_folder=str(os.path.abspath(p_folder)),
        seed=int(seed),
        n_samples=int(result["n_samples"]),
        sample_ids=result["sample_ids"],
        rmse_abs_oo=float(result["rmse_abs_oo"]),
        rmse_abs_a0=float(result["rmse_abs_a0"]),
        rmse_rel_oo=float(result["rmse_rel_oo"]),
        rmse_rel_a0=float(result["rmse_rel_a0"]),
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "p",
            "p_folder",
            "seed",
            "n_samples",
            "rmse_abs_oo",
            "rmse_abs_a0",
            "rmse_rel_oo",
            "rmse_rel_a0",
        ])
        writer.writerow([
            float(p_value),
            os.path.abspath(p_folder),
            int(seed),
            int(result["n_samples"]),
            float(result["rmse_abs_oo"]),
            float(result["rmse_abs_a0"]),
            float(result["rmse_rel_oo"]),
            float(result["rmse_rel_a0"]),
        ])

    return npz_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute RMSE for one p-folder and save one summary file."
    )
    parser.add_argument("p_folder", help="Path to one p-folder, e.g. Bdata/BpN_folder/p_0.020")
    parser.add_argument("output_dir", help="Folder where RMSE summary files will be saved")
    parser.add_argument("--seed", type=int, default=123, help="Seed for vector v")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Optional number of samples to use from s=0000,... If omitted, infer from files."
    )
    args = parser.parse_args()

    p_folder = args.p_folder
    output_dir = args.output_dir
    seed = args.seed

    p_value = parse_p_from_folder(p_folder)

    if args.samples is None:
        sample_ids = infer_sample_ids(p_folder)
    else:
        sample_ids = list(range(args.samples))

    result = compute_rmse_one_folder(
        folder=p_folder,
        sample_ids=sample_ids,
        seed=seed,
    )

    npz_path, csv_path = save_summary(
        result=result,
        p_value=p_value,
        p_folder=p_folder,
        output_dir=output_dir,
        seed=seed,
    )

    print(f"Processed folder : {p_folder}")
    print(f"p                : {p_value:.6f}")
    print(f"n_samples        : {result['n_samples']}")
    print(f"Abs RMSE OO      : {result['rmse_abs_oo']:.6e}")
    print(f"Abs RMSE A0      : {result['rmse_abs_a0']:.6e}")
    print(f"Rel RMSE OO      : {result['rmse_rel_oo']:.6e}")
    print(f"Rel RMSE A0      : {result['rmse_rel_a0']:.6e}")
    print(f"Saved NPZ        : {npz_path}")
    print(f"Saved CSV        : {csv_path}")


if __name__ == "__main__":
    main()


