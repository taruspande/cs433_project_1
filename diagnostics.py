import argparse
import json

import numpy as np

from helpers import load_csv_data
from preprocessing import (
    add_feature_args,
    preprocess,
    read_feature_names,
    validate_feature_args,
)


def constant_columns(tx, cols, tol=1e-12):
    """Columns whose variance is below `tol`."""
    var = tx.var(axis=0)
    return [cols[i] for i in range(1, tx.shape[1]) if var[i] < tol]


def identical_column_groups(tx, cols, seed=0, decimals=6):
    """Groups of columns with the same values. Every column is projected on one random
    vector; identical columns get identical projections (a hash, so a collision between
    different columns is possible in principle but extremely unlikely)."""
    rng = np.random.default_rng(seed)
    h = np.round(tx.T @ rng.standard_normal(tx.shape[0]), decimals)
    groups = {}
    for i, v in enumerate(h):
        groups.setdefault(v, []).append(i)
    return [[cols[i] for i in g] for g in groups.values() if len(g) > 1]


def rank_deficiency(tx, cols, rel_tol=1e-12, n_vectors=8, n_top=6):
    """Eigen-decompose X^T X. Eigenvalues below rel_tol * max_eigenvalue count as zero.

    Returns (number of near-zero eigenvalues, smallest eigenvalues, largest eigenvalue,
    list of null-space descriptions). Each description lists the columns with the largest
    weight in one null-space eigenvector. The null space is only defined up to a basis
    change, so with several dependencies these lists can mix them."""
    w, v = np.linalg.eigh(tx.T @ tx)
    tol = w.max() * rel_tol
    null = np.where(w < tol)[0]
    descriptions = []
    for k in null[:n_vectors]:  # n_vectors=None keeps all
        top = np.argsort(-np.abs(v[:, k]))[:n_top]
        descriptions.append([(cols[i], round(float(v[i, k]), 3)) for i in top])
    return len(null), w[:5], w.max(), tol, descriptions


def low_support_columns(tx, cols, min_support):
    """Columns (other than the bias) with fewer than `min_support` non-zero rows.

    Rare dummies and dummy products can be all-zero inside one cross-validation fold's training
    rows even though they are non-zero in the full data; such a fold has a singular X^T X that the
    full-data checks cannot see. Dropping them removes that failure."""
    nnz = (tx != 0).sum(axis=0)
    return [i for i in range(1, tx.shape[1]) if nnz[i] < min_support]


def redundant_columns(G, cols, rel_tol=1e-9):
    """Columns that are (numerically) linear combinations of earlier columns.

    `G` is the Gram matrix X^T X of the columns named in `cols`. Gram-Schmidt in the inner
    product given by G (so on the d x d matrix, not on the N x d data): every kept column gets a
    coefficient vector q_k with q_k^T G q_l = delta_kl. A new column e_j is orthogonalised against
    them (twice, to limit round-off) and its residual r = ||X v||^2 = v^T G v is the part of
    column j not explained by the kept columns. The column is dropped if r <= rel_tol * G_jj.
    Earlier columns win: the bias is always kept, and of two identical columns the first one
    stays. Returns (dropped names, kept indices into `cols`)."""
    d = G.shape[0]
    Q = np.zeros((d, d))
    kept, dropped = [], []
    for j in range(d):
        v = np.zeros(d)
        v[j] = 1.0
        s = len(kept)
        if s:
            for _ in range(2):  # re-orthogonalise for numerical stability
                v -= Q[:, :s] @ (Q[:, :s].T @ (G @ v))
        r = v @ G @ v
        if G[j, j] > 0 and r > rel_tol * G[j, j]:
            Q[:, s] = v / np.sqrt(r)
            kept.append(j)
        else:
            dropped.append(cols[j])
    return dropped, kept


def check_full_rank(tx, kept):
    """Check that the kept columns give an invertible X^T X: the smallest/largest eigenvalue
    ratio, and whether np.linalg.solve (what least_squares uses) runs without LinAlgError."""
    X = tx[:, kept]
    G = X.T @ X
    w = np.linalg.eigvalsh(G)
    try:
        np.linalg.solve(G, X.T @ np.ones(X.shape[0]))
        solve_ok = True
    except np.linalg.LinAlgError:
        solve_ok = False
    return float(w.min() / w.max()), solve_ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--sub-sample", action="store_true", help="use every 50th training row")
    p.add_argument("--max-groups", type=int, default=None, help="print only this many identical groups")
    p.add_argument("--rel-tol", type=float, default=1e-12, help="eigenvalue tolerance relative to the largest")
    p.add_argument("--save-json", metavar="PATH", help="write all categories and columns_to_drop to this json")
    p.add_argument(
        "--min-support",
        type=int,
        default=100,
        help="columns with fewer non-zero rows than this are dropped (all-zero inside a CV fold otherwise)",
    )
    p.add_argument(
        "--redundancy-tol",
        type=float,
        default=1e-6,
        help="a column is redundant if its residual variance after the kept columns is below this fraction",
    )
    add_feature_args(p)
    args = p.parse_args()
    validate_feature_args(p, args)

    x_train, x_test, _, _, _ = load_csv_data(args.data_dir, args.sub_sample)
    names = read_feature_names(args.data_dir)
    tx, _, info = preprocess(x_train, x_test, names, args)
    cols = info["columns"]
    assert len(cols) == tx.shape[1], (len(cols), tx.shape)
    print(f"features (incl. bias): {tx.shape[1]}, rows: {tx.shape[0]}")

    const = constant_columns(tx, cols)
    print(f"\n[1] constant columns (besides bias): {len(const)}")
    if const:
        print("   ", const)

    groups = identical_column_groups(tx, cols)
    print(f"\n[2] groups of identical columns: {len(groups)}")
    for g in groups[: args.max_groups]:
        print("   ", g)

    n_vectors = None if args.save_json else 8
    n_null, smallest, largest, tol, descriptions = rank_deficiency(
        tx, cols, args.rel_tol, n_vectors=n_vectors
    )
    print(f"\n[3] null-space dimension (eigenvalue < {tol:.2e}): {n_null} of {tx.shape[1]}")
    print(f"    largest eigenvalue {largest:.3e}, five smallest {smallest}")
    print("    (tiny negative eigenvalues are round-off; the condition number is meaningless here)")
    for d in descriptions[:8]:
        print("    dependency among:", d)

    if args.save_json:
        low = low_support_columns(tx, cols, args.min_support)
        low_names = [cols[i] for i in low]
        print(f"\n[4] low-support columns (< {args.min_support} non-zero rows): {len(low)}")
        low_set = set(low)
        remaining = [i for i in range(tx.shape[1]) if i not in low_set]
        sub = tx[:, remaining]
        dropped, kept_sub = redundant_columns(sub.T @ sub, [cols[i] for i in remaining], args.redundancy_tol)
        kept = [remaining[i] for i in kept_sub]
        print(f"[5] redundant columns (linear combination of earlier ones): {len(dropped)}")
        ratio, solve_ok = check_full_rank(tx, kept)
        print(f"    {len(kept)} columns remain; min/max eigenvalue of X^T X on them: {ratio:.2e}")
        print(f"    np.linalg.solve on the kept columns: {'OK' if solve_ok else 'STILL SINGULAR'}")
        report = {
            "settings": {
                "recode": args.recode,
                "bin": args.bin,
                "bins": args.bins,
                "interactions": args.interactions,
                "one_hot": args.one_hot,
                "max_levels": args.max_levels,
                "min_level_frac": args.min_level_frac,
                "dummy_interactions": args.dummy_interactions,
                "drop_json": args.drop_json,
                "sub_sample": args.sub_sample,
                "n_rows": int(tx.shape[0]),
                "n_features": int(tx.shape[1]),
                "eigenvalue_rel_tol": args.rel_tol,
                "redundancy_tol": args.redundancy_tol,
                "min_support": args.min_support,
            },
            "constant_columns": const,
            # in each group the first column is kept and the others can be dropped
            "identical_groups": [{"keep": g[0], "drop": g[1:]} for g in groups],
            # hints only: the columns with the largest weight in each null-space vector
            "null_space_hints": [[{"column": c, "weight": w} for c, w in d] for d in descriptions],
            "low_support_columns": low_names,
            "redundant_columns": dropped,
            # the list to remove: low-support + redundant, the rest is linearly independent
            "columns_to_drop": low_names + dropped,
            "columns_kept": [cols[i] for i in kept],
            # sanity check of the kept columns (should be well above 0 and solve_ok true)
            "kept_min_over_max_eigenvalue": ratio,
            "kept_solve_ok": solve_ok,
        }
        with open(args.save_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {args.save_json}")


if __name__ == "__main__":
    main()
