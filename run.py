"""Train and compare all the models of implementations.py, optionally write a submission.

Examples:
    python run.py                         # evaluate every algorithm on a validation split
    python run.py --submit ridge_regression
    python run.py --submit best --output my_submission.csv
"""

import argparse
import datetime
import sys
import time

import numpy as np

from helpers import load_csv_data, create_csv_submission
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
    sigmoid,
)

ALGORITHMS = [
    "mean_squared_error_gd",
    "mean_squared_error_sgd",
    "least_squares",
    "ridge_regression",
    "logistic_regression",
    "reg_logistic_regression",
]
LOGISTIC = {"logistic_regression", "reg_logistic_regression"}


### PREPROCESSING ###
def preprocess(x_train, x_test):
    """Impute NaNs with the train median, standardize, and add a bias column."""
    median = np.nanmedian(x_train, axis=0)
    median = np.where(np.isnan(median), 0.0, median)  # columns that are all NaN
    x_train = np.where(np.isnan(x_train), median, x_train)
    x_test = np.where(np.isnan(x_test), median, x_test)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    add_bias = lambda x: np.c_[np.ones(x.shape[0]), x]
    return add_bias(x_train), add_bias(x_test)


def kfold_indices(n, k, seed):
    """Shuffle once and yield (train_idx, val_idx) for each of the k folds.
    k < 2 falls back to a single hold-out split of size --val-frac (handled by caller)."""
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        yield np.concatenate(folds[:i] + folds[i + 1 :]), folds[i]


### TRAINING ###
def fit(name, y, tx, args):
    """Fit one algorithm. Returns the weights. y is in {-1, 1}."""
    d = tx.shape[1]
    w0 = np.zeros(d)
    y01 = (y + 1) / 2  # logistic regression expects {0, 1}
    if name == "mean_squared_error_gd":
        return mean_squared_error_gd(y, tx, w0, args.max_iters, args.gamma)[0]
    if name == "mean_squared_error_sgd":
        return mean_squared_error_sgd(y, tx, w0, args.max_iters, args.gamma)[0]
    if name == "least_squares":
        try:
            return least_squares(y, tx)[0]
        except np.linalg.LinAlgError:  # collinear features: fall back to min-norm solution
            print("Least squares: singular matrix, using pseudo-inverse instead")
            return np.linalg.pinv(tx) @ y
    if name == "ridge_regression":
        return ridge_regression(y, tx, args.lambda_)[0]
    if name == "logistic_regression":
        return logistic_regression(y01, tx, w0, args.max_iters, args.gamma)[0]
    if name == "reg_logistic_regression":
        return reg_logistic_regression(
            y01, tx, args.lambda_, w0, args.max_iters, args.gamma
        )[0]
    raise ValueError(name)


def scores(name, tx, w):
    """Continuous score; higher means more likely to be class 1."""
    z = tx @ w
    return sigmoid(z) if name in LOGISTIC else z


def default_threshold(name):
    return 0.5 if name in LOGISTIC else 0.0


def predict(s, threshold):
    return np.where(s >= threshold, 1, -1)


def best_f1_threshold(s, y):
    """Threshold on the score maximizing F1 (the classes are heavily imbalanced)."""
    candidates = np.quantile(s, np.linspace(0.5, 0.99, 100))
    f1s = [metrics(y, predict(s, t))["f1"] for t in candidates]
    return candidates[int(np.argmax(f1s))]


### METRICS ###
def auc_score(y, s):
    """ROC AUC from ranks (Mann-Whitney U); threshold independent. Ties get average rank."""
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s))
    # average ranks over tied scores
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    n_pos = np.sum(y == 1)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metrics(y, y_pred):
    tp = int(np.sum((y == 1) & (y_pred == 1)))
    tn = int(np.sum((y == -1) & (y_pred == -1)))
    fp = int(np.sum((y == -1) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == -1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0  # sensitivity / TPR
    specificity = tn / (tn + fp) if tn + fp else 0.0  # TNR
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


METRIC_KEYS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc",
]


def evaluate_all(tx, y, args):
    """K-fold cross-validation of every algorithm.

    For each fold: train on the other folds, pick the F1-optimal threshold on the held-out
    fold (unless --no-tune), compute all metrics. Prints mean (and std for F1/AUC) over folds.
    Returns (best algorithm by mean F1, {algorithm: mean threshold over folds}).
    """
    if args.folds >= 2:
        splits = list(kfold_indices(len(y), args.folds, args.seed))
    else:  # single hold-out split
        idx = np.random.default_rng(args.seed).permutation(len(y))
        n_val = int(len(y) * args.val_frac)
        splits = [(idx[n_val:], idx[:n_val])]
    k = len(splits)
    print(
        f"{k}-fold cross-validation: ~{len(splits[0][0])} train / ~{len(splits[0][1])} val samples "
        f"per fold, {tx.shape[1]} features (incl. bias), positive rate {np.mean(y == 1):.4f}"
    )
    print(
        f"Majority-class baseline: accuracy {np.mean(y == -1):.4f}, F1 0.0000, AUC 0.5000\n"
        "Metrics are means over folds; +- is the std over folds; TP/FP/FN/TN are summed over folds.\n"
    )
    header = (
        f"{'algorithm':<26}{'thr':>8}{'acc':>8}{'bal_acc':>8}{'prec':>8}{'recall':>8}"
        f"{'spec':>8}{'F1':>8}{'F1_std':>8}{'MCC':>8}{'AUC':>8}{'AUC_std':>9}"
        f"{'trainF1':>9}{'time(s)':>9}   TP/FP/FN/TN"
    )
    print(header)
    print("-" * len(header))

    mean_f1, thresholds = {}, {}
    for name in args.algorithms:
        per_fold = {key: [] for key in METRIC_KEYS + ["auc", "train_f1", "thr", "time"]}
        counts = np.zeros(4, dtype=int)
        for tr, va in splits:
            start = time.time()
            np.random.seed(args.seed)  # reproducible SGD
            w = fit(name, y[tr], tx[tr], args)
            per_fold["time"].append(time.time() - start)
            sc = scores(name, tx[va], w)
            thr = default_threshold(name) if args.no_tune else best_f1_threshold(sc, y[va])
            m = metrics(y[va], predict(sc, thr))
            for key in METRIC_KEYS:
                per_fold[key].append(m[key])
            per_fold["auc"].append(auc_score(y[va], sc))
            # F1 on the training folds at the same threshold: big gap to val F1 = overfitting
            per_fold["train_f1"].append(
                metrics(y[tr], predict(scores(name, tx[tr], w), thr))["f1"]
            )
            per_fold["thr"].append(thr)
            counts += [m["tp"], m["fp"], m["fn"], m["tn"]]
        mu = {key: float(np.mean(v)) for key, v in per_fold.items()}
        sd = {key: float(np.std(v)) for key, v in per_fold.items()}
        mean_f1[name], thresholds[name] = mu["f1"], mu["thr"]
        print(
            f"{name:<26}{mu['thr']:>8.3f}{mu['accuracy']:>8.4f}{mu['balanced_accuracy']:>8.4f}"
            f"{mu['precision']:>8.4f}{mu['recall']:>8.4f}{mu['specificity']:>8.4f}"
            f"{mu['f1']:>8.4f}{sd['f1']:>8.4f}{mu['mcc']:>8.4f}{mu['auc']:>8.4f}{sd['auc']:>9.4f}"
            f"{mu['train_f1']:>9.4f}{mu['time']:>9.2f}   "
            f"{counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}"
        )
    best = max(mean_f1, key=mean_f1.get)
    print(f"\nBest by mean CV F1: {best} ({mean_f1[best]:.4f})")
    return best, thresholds


def write_submission(name, threshold, tx, y, tx_test, test_ids, args):
    """Refit on all training data, predict the test set with the CV-derived threshold."""
    np.random.seed(args.seed)
    w = fit(name, y, tx, args)
    y_pred = predict(scores(name, tx_test, w), threshold)
    create_csv_submission(test_ids, y_pred, args.output)
    print(
        f"Wrote {args.output} using {name} (threshold {threshold:.3f}, mean over CV folds); "
        f"predicted positive rate: {np.mean(y_pred == 1):.3f}"
    )


def print_run_config(args, x_train, x_test):
    """Print everything needed to reproduce / catalogue this run."""
    print("=" * 80)
    print(f"Run started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"Command: python {' '.join(sys.argv)}")
    print("Parameters:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:<12} = {v}")
    print(f"  {'init_w':<12} = zeros")
    print(f"  {'sgd_batch':<12} = 1 (as implemented in implementations.py)")
    print("Data:")
    print(f"  x_train {x_train.shape}, x_test {x_test.shape}")
    print(f"  NaN fraction: train {np.isnan(x_train).mean():.4f}, test {np.isnan(x_test).mean():.4f}")
    print("Preprocessing: NaN -> train column median, standardize (train mean/std), add bias column")
    if args.folds >= 2:
        print(f"Validation: {args.folds}-fold cross-validation, shuffled with seed {args.seed}")
    else:
        print(f"Validation: single hold-out split, fraction {args.val_frac}, seed {args.seed}")
    print("=" * 80 + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--sub-sample", action="store_true", help="use every 50th training row (fast)")
    p.add_argument("--folds", type=int, default=5, help="k for k-fold CV (<2: single hold-out split)")
    p.add_argument("--val-frac", type=float, default=0.2, help="hold-out fraction, only used if --folds < 2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iters", type=int, default=300)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--lambda_", "--lambda", dest="lambda_", type=float, default=1e-2)
    p.add_argument("--no-tune", action="store_true", help="do not tune the decision threshold for F1")
    p.add_argument(
        "--submit",
        choices=ALGORITHMS + ["best"],
        help="write a submission csv with this algorithm ('best' = highest mean CV F1)",
    )
    p.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=ALGORITHMS,
        help="only evaluate these algorithms (default: all)",
    )
    p.add_argument("--output", default="submission.csv")
    args = p.parse_args()

    x_train, x_test, y_train, _, test_ids = load_csv_data(args.data_dir, args.sub_sample)
    tx, tx_test = preprocess(x_train, x_test)

    print_run_config(args, x_train, x_test)
    best, thresholds = evaluate_all(tx, y_train, args)
    if args.submit:
        name = best if args.submit == "best" else args.submit
        if name not in thresholds:
            p.error(f"--submit {name} must be one of the evaluated --algorithms")
        write_submission(name, thresholds[name], tx, y_train, tx_test, test_ids, args)


if __name__ == "__main__":
    main()
