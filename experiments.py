import argparse
import csv
import datetime
import gc
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

import run
from helpers import create_csv_submission, load_csv_data
from preprocessing import preprocess, read_feature_names

### FEATURE SETS ###
FEATURE_SETS = {
    "plain": {},
    "inter": {"interactions": True},
    "oh": {"one_hot": True},
    "oh_recode": {"one_hot": True, "recode": True},
    "oh_inter": {"one_hot": True, "interactions": True},
    "oh_inter_recode": {"one_hot": True, "interactions": True, "recode": True},
    "oh_inter_recode_bin": {"one_hot": True, "interactions": True, "recode": True, "bin": True},
    "oh_dummy": {"one_hot": True, "dummy_interactions": True},
    "final": {"one_hot": True, "recode": True, "dummy_interactions": True},
    "final_bin": {"one_hot": True, "recode": True, "bin": True, "dummy_interactions": True},
    "both": {"one_hot": True, "recode": True, "interactions": True, "dummy_interactions": True},
    # for gradient descent / logistic regression: all columns standardized
    "oh_std": {"one_hot": True, "standardize_dummies": True},
    "final_std": {"one_hot": True, "recode": True, "dummy_interactions": True, "standardize_dummies": True},
    "both_std": {
        "one_hot": True, "recode": True, "interactions": True, "dummy_interactions": True,
        "standardize_dummies": True,
    },
}
FEATURE_ORDER = list(FEATURE_SETS)  # sets are built in this order (each only once)
FLAG_DEFAULTS = dict(
    recode=False, bin=False, bins=8, interactions=False, one_hot=False, max_levels=10,
    min_level_frac=0.005, dummy_interactions=False, standardize_dummies=False, drop_json=None,
)

ITERATIVE = {
    "mean_squared_error_gd", "mean_squared_error_sgd", "logistic_regression", "reg_logistic_regression",
}
SHORT = {
    "mean_squared_error_gd": "gd", "mean_squared_error_sgd": "sgd", "least_squares": "ls",
    "ridge_regression": "ridge", "logistic_regression": "logreg", "reg_logistic_regression": "reglogreg",
}
STAGE_NAMES = {1: "s1_base", 2: "s2_sgd", 3: "s3_iterative", 4: "s4_features", 5: "s5_final"}

CSV_FIELDS = [
    "run_id", "stage", "label", "algorithm", "featureset", "n_features", "gamma", "lambda", "max_iters",
    "seed", "folds", "fold", "threshold", "accuracy", "balanced_accuracy", "precision", "recall",
    "specificity", "f1", "mcc", "auc", "train_f1", "tp", "fp", "fn", "tn", "time_s", "diverged",
]
METRIC_KEYS = [
    "threshold", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc",
    "auc", "train_f1", "time_s",
]


### PLAN ###
def job(stage, label, algo, featureset, gamma=None, lam=None, iters=None, seed=0, refit=False, submit=None):
    """One experiment. iters is a list of checkpoints for the iterative methods (one run each,
    all from the same training run), None for the exact solvers."""
    return {
        "stage": STAGE_NAMES[stage], "stage_no": stage, "label": label, "algo": algo,
        "featureset": featureset, "gamma": gamma, "lam": lam,
        "iters": iters if algo in ITERATIVE else None, "seed": seed, "refit": refit, "submit": submit,
    }


def run_id(j, c):
    part = lambda v: "-" if v is None else str(v)
    return "_".join(
        [j["stage"], SHORT[j["algo"]], j["featureset"], "g" + part(j["gamma"]), "l" + part(j["lam"]),
         "it" + part(c), "s" + str(j["seed"])]
    )


def job_run_ids(j):
    return [run_id(j, c) for c in (j["iters"] or [None])]


def build_jobs():
    J = []
    # 1. base comparison: plain features, run.py defaults (gamma 0.01, 300 iterations, lambda 0.01)
    for algo in run.ALGORITHMS:
        J.append(job(1, "base comparison", algo, "plain", gamma=0.01,
                     lam=0.01 if algo in ("ridge_regression", "reg_logistic_regression") else None,
                     iters=[300]))
    # 2. SGD tuning (one training run per step size, checkpoints = number of iterations)
    for g in [1e-2, 1e-3, 1e-4, 1e-5]:
        J.append(job(2, "SGD step size x iterations", "mean_squared_error_sgd", "plain", gamma=g,
                     iters=[1000, 10000, 100000, 500000]))
    # 3. iterative methods: converge with a proper step size, plus the under-trained contrast
    for algo in ["mean_squared_error_gd", "logistic_regression", "reg_logistic_regression"]:
        lam = 0.01 if algo == "reg_logistic_regression" else None
        J.append(job(3, "iterative methods", algo, "plain", gamma=0.02, lam=lam, iters=[100, 300, 1000]))
        J.append(job(3, "iterative, small step", algo, "plain", gamma=0.001, lam=lam, iters=[400]))
    # 4. feature ablation with ridge (lambda 1e-4)
    for fs in ["plain", "inter", "oh", "oh_recode", "oh_inter", "oh_inter_recode", "oh_inter_recode_bin",
               "oh_dummy", "final", "final_bin"]:
        J.append(job(4, "feature ablation", "ridge_regression", fs, lam=1e-4))
    for fs in ["plain", "oh", "oh_inter", "oh_dummy", "final"]:
        for seed in [1, 2]:
            J.append(job(4, "feature ablation, other seed", "ridge_regression", fs, lam=1e-4, seed=seed))
    for fs in ["plain", "final"]:
        for lam in [1e-7, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:  # 1e-4: ablation run above
            J.append(job(4, "ridge lambda sweep", "ridge_regression", fs, lam=lam))
    for fs in ["plain", "oh", "final"]:
        J.append(job(4, "least squares", "least_squares", fs))
    # 5. final models
    J.append(job(5, "final logistic", "reg_logistic_regression", "final_std", gamma=0.1, lam=0.001,
                 iters=[100, 250, 500], refit=True, submit="reg_logistic_final"))
    J.append(job(5, "final logistic, unregularized", "logistic_regression", "final_std", gamma=0.1,
                 iters=[500], refit=True))
    J.append(job(5, "logistic, raw + dummy interactions", "reg_logistic_regression", "both_std", gamma=0.1,
                 lam=0.001, iters=[100, 250, 500], refit=True))
    J.append(job(5, "logistic, one-hot only", "reg_logistic_regression", "oh_std", gamma=0.1, lam=0.001,
                 iters=[100, 250, 500]))
    J.append(job(5, "final ridge", "ridge_regression", "final", lam=1e-4, refit=True, submit="ridge_final"))
    J.append(job(5, "ridge, raw + dummy interactions", "ridge_regression", "both", lam=1e-4, refit=True))
    J.sort(key=lambda j: FEATURE_ORDER.index(j["featureset"]))  # stable: stage order kept per set
    return J


### TRAINING AND EVALUATION ###
def train_path(j, y, X):
    """Yield (checkpoint, weights, cumulative fit seconds). Exact solvers yield once.
    The iterative methods are trained in chunks from the previous weights, which is the same
    run as one long training because they are deterministic (SGD: one random stream)."""
    args = SimpleNamespace(gamma=j["gamma"], lambda_=j["lam"], max_iters=0)
    if j["iters"] is None:
        t = time.time()
        w = run.fit(j["algo"], y, X, args)
        yield None, w, time.time() - t
        return
    np.random.seed(j["seed"])
    w, done, total = np.zeros(X.shape[1]), 0, 0.0
    for c in j["iters"]:
        t = time.time()
        w = run.fit(j["algo"], y, X, args, w0=w, max_iters=c - done)
        total += time.time() - t
        done = c
        yield c, w, total


def evaluate(algo, w, Xtr, ytr, Xva, yva):
    """Scores, F1-optimal threshold, metrics, AUC and train F1 of one fitted model."""
    sc = run.scores(algo, Xva, w)
    diverged = not (np.isfinite(w).all() and np.isfinite(sc).all())
    sc = np.nan_to_num(sc, nan=0.0, posinf=1e30, neginf=-1e30)
    thr = run.best_f1_threshold(sc, yva)
    m = run.metrics(yva, run.predict(sc, thr))
    train_f1 = run.metrics(ytr, run.predict(np.nan_to_num(run.scores(algo, Xtr, w), nan=0.0, posinf=1e30, neginf=-1e30), thr))["f1"]
    m.update(threshold=float(thr), auc=float(run.auc_score(yva, sc)), train_f1=train_f1)
    return sc, m, diverged


def summary_rows(base, fold_rows):
    """'mean' and 'std' rows over the folds (counts are summed in the mean row)."""
    out = []
    for stat in ("mean", "std"):
        r = dict(base, fold=stat)
        for k in METRIC_KEYS:
            vals = np.array([f[k] for f in fold_rows], dtype=float)
            r[k] = float(vals.mean() if stat == "mean" else vals.std())
        for k in ("tp", "fp", "fn", "tn"):
            r[k] = int(sum(f[k] for f in fold_rows)) if stat == "mean" else ""
        r["diverged"] = int(any(f["diverged"] for f in fold_rows))
        out.append(r)
    return out


def run_job(j, fs, y, folds, test_ids, dirs):
    """Cross-validate one job (all its checkpoints), refit / write the submission if asked,
    save runs.csv rows and the out-of-fold npz. Returns the 'mean' rows."""
    tx, tx_test = fs["tx"], fs["tx_test"]
    n, d = tx.shape
    checkpoints = j["iters"] or [None]
    splits = list(run.kfold_indices(n, folds, j["seed"]))
    acc = {c: {"rows": [], "oof": np.zeros(n, np.float32), "fold": np.zeros(n, np.int8),
               "thr": [], "W": []} for c in checkpoints}
    for f, (tr, va) in enumerate(splits):
        Xtr, ytr = tx[tr], y[tr]
        for c, w, secs in train_path(j, ytr, Xtr):
            sc, m, div = evaluate(j["algo"], w, Xtr, ytr, tx[va], y[va])
            m.update(time_s=secs, diverged=div, fold=f)
            a = acc[c]
            a["rows"].append(m)
            a["oof"][va] = np.clip(sc, -1e30, 1e30)
            a["fold"][va] = f
            a["thr"].append(m["threshold"])
            a["W"].append(w.astype(np.float32))
        del Xtr
        gc.collect()

    # refit on all training data with the last checkpoint, mean fold threshold
    refit = None
    if j["refit"]:
        last = checkpoints[-1]
        jj = dict(j, iters=[last] if last is not None else None)
        w_all = [w for _, w, _ in train_path(jj, y, tx)][-1]
        thr = float(np.mean(acc[last]["thr"]))
        refit = {"w": w_all.astype(np.float32), "thr": thr}
        if j["submit"]:
            y_pred = run.predict(np.nan_to_num(run.scores(j["algo"], tx_test, w_all)), thr)
            os.makedirs(dirs["sub"], exist_ok=True)
            path = os.path.join(dirs["sub"], j["submit"] + ".csv")
            create_csv_submission(test_ids, y_pred, path)
            with open(os.path.join(dirs["sub"], j["submit"] + ".json"), "w") as fh:
                json.dump({"job": j, "threshold": thr, "predicted_positive_rate": float(np.mean(y_pred == 1)),
                           "cv_mean_f1": float(np.mean([r["f1"] for r in acc[last]["rows"]])),
                           "cv_mean_auc": float(np.mean([r["auc"] for r in acc[last]["rows"]]))}, fh, indent=2)
            log(f"  wrote {path} (threshold {thr:.3f}, predicted positive rate {np.mean(y_pred == 1):.4f})")

    means = []
    os.makedirs(os.path.join(dirs["res"], "oof"), exist_ok=True)
    new_file = not os.path.exists(os.path.join(dirs["res"], "runs.csv"))
    with open(os.path.join(dirs["res"], "runs.csv"), "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for c in checkpoints:
            rid = run_id(j, c)
            base = {
                "run_id": rid, "stage": j["stage"], "label": j["label"], "algorithm": j["algo"],
                "featureset": j["featureset"], "n_features": d, "gamma": j["gamma"], "lambda": j["lam"],
                "max_iters": c, "seed": j["seed"], "folds": folds,
            }
            fold_rows = acc[c]["rows"]
            rows = [dict(base, **{k: r[k] for k in METRIC_KEYS + ["tp", "fp", "fn", "tn", "diverged", "fold"]})
                    for r in fold_rows]
            rows += summary_rows(base, fold_rows)
            for r in rows:
                r["diverged"] = int(r["diverged"])
            writer.writerows(rows)
            means.append(rows[-2])
            extra = {}
            if refit is not None and c == checkpoints[-1]:
                extra = {"refit_w": refit["w"], "refit_thr": refit["thr"]}
            np.savez_compressed(
                os.path.join(dirs["res"], "oof", rid + ".npz"), oof_scores=acc[c]["oof"],
                fold_id=acc[c]["fold"], thresholds=np.array(acc[c]["thr"]), weights=np.array(acc[c]["W"]),
                **extra,
            )
    return means


### MAIN ###
def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def done_run_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {r["run_id"] for r in csv.DictReader(fh) if r["fold"] == "mean"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--submissions-dir", default="submissions")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--sub-sample", action="store_true", help="every 50th training row (test of the script)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = p.parse_args()
    dirs = {"res": args.results_dir, "sub": args.submissions_dir}

    jobs = [j for j in build_jobs() if j["stage_no"] in args.stages]
    done = done_run_ids(os.path.join(dirs["res"], "runs.csv"))
    pending = [j for j in jobs if not all(r in done for r in job_run_ids(j))]
    log(f"{len(jobs)} jobs in stages {args.stages}, {len(jobs) - len(pending)} already done, {len(pending)} to run")
    if args.dry_run:
        for j in pending:
            print(f"  {j['stage']:13s} {j['featureset']:20s} {j['algo']:24s} gamma={j['gamma']} lam={j['lam']} "
                  f"iters={j['iters']} seed={j['seed']}{' SUBMIT' if j['submit'] else ''}")
        return
    os.makedirs(os.path.join(dirs["res"], "features"), exist_ok=True)
    with open(os.path.join(dirs["res"], "plan.json"), "w") as fh:
        json.dump(jobs, fh, indent=1)
    with open(os.path.join(dirs["res"], "meta.json"), "w") as fh:
        json.dump({"command": " ".join(sys.argv), "started": datetime.datetime.now().isoformat(),
                   "python": sys.version, "numpy": np.__version__, "folds": args.folds,
                   "sub_sample": args.sub_sample, "feature_sets": FEATURE_SETS}, fh, indent=1)

    x_train, x_test, y, _, test_ids = load_csv_data(args.data_dir, args.sub_sample)
    names = read_feature_names(args.data_dir)
    log(f"data loaded: x_train {x_train.shape}, x_test {x_test.shape}")

    for fs_name in FEATURE_ORDER:
        todo = [j for j in pending if j["featureset"] == fs_name]
        if not todo:
            continue
        flags = SimpleNamespace(**dict(FLAG_DEFAULTS, **FEATURE_SETS[fs_name]))
        t = time.time()
        xa, xb = (x_train.copy(), x_test.copy()) if flags.recode else (x_train, x_test)  # recode edits in place
        tx, tx_test, info = preprocess(xa, xb, names, flags)
        del xa, xb
        with open(os.path.join(dirs["res"], "features", fs_name + ".json"), "w") as fh:
            json.dump({"flags": FEATURE_SETS[fs_name], "n_features": len(info["columns"]),
                       "columns": info["columns"]}, fh)
        log(f"feature set '{fs_name}': {tx.shape[1]} features, built in {time.time() - t:.0f}s, {len(todo)} jobs")
        fs = {"tx": tx, "tx_test": tx_test}
        for j in todo:
            t = time.time()
            log(f"  {j['stage']} {j['algo']} gamma={j['gamma']} lam={j['lam']} iters={j['iters']} seed={j['seed']}")
            try:
                with np.errstate(all="ignore"):
                    means = run_job(j, fs, y, args.folds, test_ids, dirs)
                for r in means:
                    log(f"    {r['run_id']}: F1 {r['f1']:.4f} AUC {r['auc']:.4f} trainF1 {r['train_f1']:.4f}"
                        f"{'  DIVERGED' if r['diverged'] else ''}")
                log(f"  job done in {time.time() - t:.0f}s")
            except Exception:
                log("  JOB FAILED, continuing with the next one:\n" + traceback.format_exc())
                with open(os.path.join(dirs["res"], "errors.log"), "a") as fh:
                    fh.write(f"{j}\n{traceback.format_exc()}\n")
        del fs, tx, tx_test
        gc.collect()
    log("all done")


if __name__ == "__main__":
    main()
