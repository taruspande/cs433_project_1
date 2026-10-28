import argparse
import csv
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

### STYLE ###
COL1 = (3.4, 2.5)  # one column of a two-column paper
COL2 = (7.0, 2.7)  # full width
C = {"blue": "#2f6db5", "orange": "#e07b39", "green": "#3a9b5c", "red": "#c8453d", "grey": "#777777",
     "purple": "#7a5aa6"}
ALGO_LABEL = {
    "mean_squared_error_gd": "GD", "mean_squared_error_sgd": "SGD", "least_squares": "LS",
    "ridge_regression": "Ridge", "logistic_regression": "Logistic", "reg_logistic_regression": "Reg. logistic",
}
ALGO_ORDER = list(ALGO_LABEL)
# short x-axis labels of the methods_base figure, in the order of ALGO_ORDER
METHOD_ABBR = {
    "mean_squared_error_gd": "GD", "mean_squared_error_sgd": "SGD", "least_squares": "LS",
    "ridge_regression": "RR", "logistic_regression": "LGD", "reg_logistic_regression": "RLGD",
}
FEATURE_LABEL = {
    "plain": "plain", "inter": "interactions", "oh": "one-hot", "oh_recode": "one-hot + recode",
    "oh_inter": "one-hot + inter.", "oh_inter_recode": "one-hot + inter. + recode",
    "oh_inter_recode_bin": "one-hot + inter. + recode + bins", "oh_dummy": "one-hot + dummy inter.",
    "final": "one-hot + recode + dummy inter.", "final_bin": "final + bins",
    "both": "final + raw inter.", "oh_std": "one-hot (std.)", "final_std": "final (std.)",
    "both_std": "final + raw inter. (std.)",
}
ABLATION_SETS = ["plain", "inter", "oh", "oh_recode", "oh_inter", "oh_inter_recode", "oh_inter_recode_bin",
                 "oh_dummy", "final", "final_bin"]
NUMERIC = ["n_features", "gamma", "lambda", "max_iters", "seed", "folds", "threshold", "accuracy",
           "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "auc", "train_f1", "tp",
           "fp", "fn", "tn", "time_s", "diverged"]


def setup_style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.dpi": 120,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


### LOADING ###
def load_runs(results_dir):
    """{run_id: {"meta": row of the 'mean' line, "folds": [fold rows], "mean": row, "std": row}}"""
    path = os.path.join(results_dir, "runs.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run experiments.py first")
    runs = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for k in NUMERIC:
                r[k] = float(r[k]) if r[k] not in ("", None) else None
            run = runs.setdefault(r["run_id"], {"folds": []})
            if r["fold"] in ("mean", "std"):
                run[r["fold"]] = r
            else:
                run["folds"].append(r)
    for run in runs.values():
        run["meta"] = run["mean"]
    return runs


def find(runs, algo=None, fs=None, stage=None, label=None, lam=None, gamma=None, iters=None, seed=0):
    """All runs matching the given fields (None = any), sorted by (lambda, gamma, iterations)."""
    out = []
    for run in runs.values():
        m = run["meta"]
        for key, want in (("algorithm", algo), ("featureset", fs), ("stage", stage), ("label", label)):
            if want is not None and m[key] != want:
                break
        else:
            if lam is not None and not (m["lambda"] is not None and np.isclose(m["lambda"], lam)):
                continue
            if gamma is not None and not (m["gamma"] is not None and np.isclose(m["gamma"], gamma)):
                continue
            if iters is not None and m["max_iters"] != iters:
                continue
            if seed is not None and m["seed"] != seed:
                continue
            out.append(run)
    out.sort(key=lambda r: (r["meta"]["lambda"] or 0, r["meta"]["gamma"] or 0, r["meta"]["max_iters"] or 0))
    return out


def one(runs, **kw):
    found = find(runs, **kw)
    return found[0] if found else None


def val(run, key, stat="mean"):
    return run[stat][key]


def load_labels(data_dir, sub_sample):
    y = np.genfromtxt(os.path.join(data_dir, "y_train.csv"), delimiter=",", skip_header=1, dtype=int, usecols=1)
    return y[::50] if sub_sample else y


def load_oof(results_dir, run):
    z = np.load(os.path.join(results_dir, "oof", run["meta"]["run_id"] + ".npz"))
    return {k: z[k] for k in z.files}


def low_limit(runs_list, key, margin=0.01):
    """Left/bottom axis limit just below the smallest (value - std) of the shown runs."""
    lows = [val(r, key) - val(r, key, "std") for r in runs_list]
    return max(0.0, float(min(lows)) - margin)


def save(fig, out, name, fmt, dpi):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}.{fmt}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {path}")


### CURVES FROM OUT-OF-FOLD SCORES ###
def roc_pr(y, s):
    """ROC and precision-recall points (one per distinct score) and the ROC AUC."""
    order = np.argsort(-s, kind="mergesort")
    pos = y[order] == 1
    ss = s[order]
    last = np.r_[np.where(np.diff(ss) != 0)[0], len(s) - 1]  # last index of each tied score
    tps, fps = np.cumsum(pos)[last], np.cumsum(~pos)[last]
    P, N = pos.sum(), (~pos).sum()
    tpr, fpr = np.r_[0, tps / P], np.r_[0, fps / N]
    precision, recall = tps / (tps + fps), tps / P
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    keep = np.unique(np.linspace(0, len(fpr) - 1, 1500).astype(int))
    keep_pr = np.unique(np.linspace(0, len(precision) - 1, 1500).astype(int))
    return fpr[keep], tpr[keep], recall[keep_pr], precision[keep_pr], auc


def prf_at(y, s, thr):
    pred = s >= thr
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y != 1))
    fn = np.sum(~pred & (y == 1))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


### FIGURES ###
def fig_methods(runs, a):
    """Stage 1: all six methods on plain features with default parameters."""
    items = [(al, one(runs, algo=al, fs="plain", stage="s1_base")) for al in ALGO_ORDER]
    items = [(al, r) for al, r in items if r]
    if not items:
        return False
    fig, axes = plt.subplots(1, 2, figsize=COL2)
    for ax, key, title in zip(axes, ["f1", "auc"], ["F1", "AUC"]):
        x = np.arange(len(items))
        ax.bar(x, [val(r, key) for _, r in items], yerr=[val(r, key, "std") for _, r in items],
               color=[C["red"] if r["mean"]["diverged"] else C["blue"] for _, r in items], capsize=2)
        ax.set_xticks(x, [METHOD_ABBR[al] for al, _ in items])
        ax.set_title(f"{title}, plain features, default parameters")
        if key == "auc":
            ax.axhline(0.5, color=C["grey"], lw=0.8, ls="--")
            ax.set_ylim(0.4, None)
    save(fig, a.out, "methods_base", a.format, a.dpi)


def fig_sgd(runs, a):
    """Stage 2: SGD step size x iterations."""
    gammas = [1e-2, 1e-3, 1e-4, 1e-5]
    iters = [1000, 10000, 100000, 500000]
    grid = {(g, it): one(runs, algo="mean_squared_error_sgd", stage="s2_sgd", gamma=g, iters=it) for g in gammas for it in iters}
    if not any(grid.values()):
        return False
    fig, axes = plt.subplots(1, 2, figsize=COL2, constrained_layout=True)
    for ax, key, title, (vmin, vmax) in zip(axes, ["auc", "f1"], ["AUC", "F1"], [(0.4, 0.9), (0.0, 0.45)]):
        Z = np.full((len(gammas), len(iters)), np.nan)
        for i, g in enumerate(gammas):
            for j, it in enumerate(iters):
                r = grid[(g, it)]
                if r:
                    Z[i, j] = val(r, key)
        im = ax.imshow(Z, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(iters)), [f"{it:,}" for it in iters])
        ax.set_yticks(range(len(gammas)), [f"{g:g}" for g in gammas])
        ax.set_xlabel("iterations")
        ax.set_ylabel("step size $\\gamma$" if key == "auc" else "")
        ax.set_title(f"SGD: {title}")
        ax.grid(False)
        for i, g in enumerate(gammas):
            for j, it in enumerate(iters):
                r = grid[(g, it)]
                if r:
                    ax.text(j, i, "diverged" if r["mean"]["diverged"] else f"{Z[i, j]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, a.out, "sgd_grid", a.format, a.dpi)


def _band(ax, xs, runs_list, key, **kw):
    y = np.array([val(r, key) for r in runs_list])
    s = np.array([val(r, key, "std") for r in runs_list])
    line, = ax.plot(xs, y, marker="o", ms=3, **kw)
    ax.fill_between(xs, y - s, y + s, color=line.get_color(), alpha=0.15, lw=0)


def fig_convergence(runs, a):
    """Stage 3: the iterative methods against the number of iterations (plain features)."""
    ref = one(runs, algo="ridge_regression", fs="plain", stage="s4_features", label="feature ablation", lam=1e-4)
    series = []
    for al, col in [("mean_squared_error_gd", C["orange"]), ("logistic_regression", C["blue"]),
                    ("reg_logistic_regression", C["green"])]:
        rs = [r for r in find(runs, algo=al, fs="plain", stage="s3_iterative", gamma=0.02)]
        if rs:
            series.append((al, col, rs))
    if not series:
        return False
    fig, axes = plt.subplots(1, 2, figsize=COL2)
    for ax, key, title in zip(axes, ["f1", "auc"], ["F1", "AUC"]):
        for al, col, rs in series:
            _band(ax, [r["meta"]["max_iters"] for r in rs], rs, key, color=col, label=ALGO_LABEL[al])
        for al, col, _ in series:  # the under-trained contrast: gamma 0.001, 400 iterations
            small = one(runs, algo=al, fs="plain", stage="s3_iterative", gamma=0.001)
            if small:
                ax.plot([400], [val(small, key)], marker="x", color=col, ls="none", ms=5)
        if ref:
            ax.axhline(val(ref, key), color=C["grey"], ls="--", lw=0.9, label="ridge (exact)")
        ax.set_xscale("log")
        ax.set_xlabel("iterations ($\\gamma=0.02$; x: $\\gamma=0.001$, 400 it.)")
        ax.set_title(f"{title} vs iterations (plain features)")
    axes[0].legend(loc="lower right")
    save(fig, a.out, "convergence_plain", a.format, a.dpi)


def fig_ablation(runs, a):
    """Stage 4: ridge on the feature sets (lambda 1e-4, seed 0)."""
    items = [(fs, one(runs, algo="ridge_regression", fs=fs, stage="s4_features", label="feature ablation", lam=1e-4))
             for fs in ABLATION_SETS]
    items = [(fs, r) for fs, r in items if r][::-1]
    if not items:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(COL2[0], 3.0), sharey=True)
    y = np.arange(len(items))
    base = {k: val(next(r for fs, r in items if fs == "plain"), k) for k in ("f1", "auc")} if any(fs == "plain" for fs, _ in items) else None
    for ax, key, title in zip(axes, ["f1", "auc"], ["F1", "AUC"]):
        lo = low_limit([r for _, r in items], key)
        ax.barh(y, [val(r, key) for _, r in items], xerr=[val(r, key, "std") for _, r in items],
                color=[C["green"] if fs == "final" else C["blue"] for fs, _ in items], capsize=2, height=0.65)
        if base:
            ax.axvline(base[key], color=C["grey"], ls="--", lw=0.9)
        ax.set_xlim(lo, None)
        ax.set_title(f"{title} (5-fold CV, ridge)")
    axes[0].set_yticks(y, [f"{FEATURE_LABEL[fs]} ({int(r['meta']['n_features'])})" for fs, r in items])
    save(fig, a.out, "feature_ablation", a.format, a.dpi)


def fig_seeds(runs, a):
    """Stage 4: the same ablation on three fold splits (seeds)."""
    sets = ["plain", "oh", "oh_inter", "oh_dummy", "final"]
    data = {}
    for fs in sets:
        pts = [one(runs, algo="ridge_regression", fs=fs, lam=1e-4, seed=s, stage="s4_features") for s in (0, 1, 2)]
        pts = [p for p in pts if p]
        if pts:
            data[fs] = pts
    if not data:
        return False
    fig, ax = plt.subplots(figsize=COL1)
    for i, (fs, pts) in enumerate(data.items()):
        f1s = [val(p, "f1") for p in pts]
        ax.scatter(f1s, [i] * len(f1s), color=C["blue"], s=14, zorder=3)
        ax.plot([np.mean(f1s)] * 2, [i - 0.3, i + 0.3], color=C["red"], lw=1.5, zorder=4)
    ax.set_yticks(range(len(data)), [FEATURE_LABEL[fs] for fs in data])
    ax.invert_yaxis()
    ax.set_xlabel("F1 (dots: seeds 0, 1, 2; red: mean)")
    ax.set_title("Stability over fold splits")
    save(fig, a.out, "seed_stability", a.format, a.dpi)


def fig_lambda(runs, a):
    """Stage 4: ridge lambda sweep on plain and final features."""
    fig, axes = plt.subplots(1, 2, figsize=COL2)
    drawn = False
    for fs, col in (("plain", C["orange"]), ("final", C["blue"])):
        rs = [r for r in find(runs, algo="ridge_regression", fs=fs, stage="s4_features", seed=0)
              if r["meta"]["label"] in ("ridge lambda sweep", "feature ablation")]
        rs = sorted({r["meta"]["lambda"]: r for r in rs}.values(), key=lambda r: r["meta"]["lambda"])
        if len(rs) < 2:
            continue
        drawn = True
        lams = [r["meta"]["lambda"] for r in rs]
        for ax, key in zip(axes, ["f1", "auc"]):
            # one translucent line per fold, then the opaque mean on top
            fold_ids = sorted({f["fold"] for r in rs for f in r["folds"]}, key=int)
            for fid in fold_ids:
                ys = [next(f[key] for f in r["folds"] if f["fold"] == fid) for r in rs]
                ax.plot(lams, ys, color=col, alpha=0.5, lw=0.8)
            ax.plot(lams, [val(r, key) for r in rs], color=col, lw=1.8, marker="o", ms=3,
                    label=f"{'plain' if fs == 'plain' else 'final'} features, mean")
    if not drawn:
        plt.close(fig)
        return False
    for ax, title in zip(axes, ["F1", "AUC"]):
        ax.set_xscale("log")
        ax.set_xlabel("$\\lambda$")
        ax.set_title(f"Ridge: {title} vs $\\lambda$ (thin: single folds)")
    axes[0].plot([], [], color=C["grey"], alpha=0.4, lw=0.8, label="single folds")  # legend entry only
    axes[0].legend(loc="lower left")
    save(fig, a.out, "ridge_lambda", a.format, a.dpi)


def fig_final(runs, a):
    """Stage 5: final models, F1 and AUC (last checkpoint of each)."""
    cands = [
        ("Ridge, plain", one(runs, algo="ridge_regression", fs="plain", stage="s4_features", label="feature ablation", lam=1e-4)),
        ("Logistic, plain", one(runs, algo="logistic_regression", fs="plain", stage="s3_iterative", gamma=0.02, iters=1000)),
        ("Ridge, final", one(runs, algo="ridge_regression", fs="final", stage="s5_final")),
        ("Ridge, + raw inter.", one(runs, algo="ridge_regression", fs="both", stage="s5_final")),
        ("Log., one-hot", one(runs, algo="reg_logistic_regression", fs="oh_std", stage="s5_final", iters=500)),
        ("Log., final", one(runs, algo="reg_logistic_regression", fs="final_std", stage="s5_final", iters=500)),
        ("Log., + raw inter.", one(runs, algo="reg_logistic_regression", fs="both_std", stage="s5_final", iters=500)),
    ]
    cands = [(n, r) for n, r in cands if r]
    if len(cands) < 2:
        return False
    fig, axes = plt.subplots(1, 2, figsize=COL2)
    x = np.arange(len(cands))
    for ax, key, title in zip(axes, ["f1", "auc"], ["F1", "AUC"]):
        lo = low_limit([r for _, r in cands], key)
        ax.bar(x, [val(r, key) for _, r in cands], yerr=[val(r, key, "std") for _, r in cands],
               color=[C["green"] if "final" in n else C["blue"] for n, _ in cands], capsize=2)
        ax.set_xticks(x, [n for n, _ in cands], rotation=30, ha="right")
        ax.set_ylim(lo, None)
        ax.set_title(f"{title} (5-fold CV)")
    save(fig, a.out, "final_models", a.format, a.dpi)


def fig_final_convergence(runs, a):
    """Stage 5: regularized logistic regression on the final feature sets vs iterations."""
    fig, axes = plt.subplots(1, 2, figsize=COL2)
    drawn = False
    for fs, col in (("oh_std", C["orange"]), ("final_std", C["green"]), ("both_std", C["purple"])):
        rs = find(runs, algo="reg_logistic_regression", fs=fs, stage="s5_final")
        if len(rs) < 2:
            continue
        drawn = True
        for ax, key in zip(axes, ["f1", "auc"]):
            _band(ax, [r["meta"]["max_iters"] for r in rs], rs, key, color=col, label=FEATURE_LABEL[fs])
    if not drawn:
        plt.close(fig)
        return False
    for ax, key, title in zip(axes, ["f1", "auc"], ["F1", "AUC"]):
        ref = one(runs, algo="ridge_regression", fs="final", stage="s5_final")
        if ref:
            ax.axhline(val(ref, key), color=C["grey"], ls="--", lw=0.9, label="ridge, final features")
        ax.set_xlabel("iterations ($\\gamma=0.1$)")
        ax.set_title(f"Regularized logistic regression: {title}")
    axes[0].legend(loc="lower right")
    save(fig, a.out, "convergence_final", a.format, a.dpi)


def _curve_runs(runs):
    """(label, run, colour) for the models shown in the curve figures."""
    cands = [
        ("Ridge, plain", one(runs, algo="ridge_regression", fs="plain", stage="s4_features", label="feature ablation", lam=1e-4), C["orange"]),
        ("Ridge, final", one(runs, algo="ridge_regression", fs="final", stage="s5_final"), C["blue"]),
        ("Reg. logistic, final", one(runs, algo="reg_logistic_regression", fs="final_std", stage="s5_final", iters=500), C["green"]),
    ]
    return [c for c in cands if c[1]]


def fig_roc_pr(runs, a):
    cands = _curve_runs(runs)
    if not cands:
        return False
    y = load_labels(a.data_dir, a.sub_sample)
    fig, axes = plt.subplots(1, 2, figsize=COL2)
    for label, run, col in cands:
        s = load_oof(a.results_dir, run)["oof_scores"].astype(float)
        fpr, tpr, rec, prec, auc = roc_pr(y, s)
        axes[0].plot(fpr, tpr, color=col, lw=1.2, label=f"{label} (AUC {auc:.3f})")
        axes[1].plot(rec, prec, color=col, lw=1.2, label=label)
    axes[0].plot([0, 1], [0, 1], color=C["grey"], ls="--", lw=0.8)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC (out-of-fold scores)")
    axes[1].axhline(np.mean(y == 1), color=C["grey"], ls="--", lw=0.8, label="positive rate")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall (out-of-fold scores)", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="upper right")
    save(fig, a.out, "roc_pr", a.format, a.dpi)


def _final_logistic(runs):
    return one(runs, algo="reg_logistic_regression", fs="final_std", stage="s5_final", iters=500)


def fig_threshold(runs, a):
    """Precision / recall / F1 against the decision threshold (final logistic model)."""
    run = _final_logistic(runs)
    if not run:
        return False
    y = load_labels(a.data_dir, a.sub_sample)
    z = load_oof(a.results_dir, run)
    s = z["oof_scores"].astype(float)
    thr = np.quantile(s, np.linspace(0.5, 0.995, 120))
    prf = np.array([prf_at(y, s, t) for t in thr])
    fig, ax = plt.subplots(figsize=COL1)
    for k, name, col in ((0, "precision", C["orange"]), (1, "recall", C["blue"]), (2, "F1", C["green"])):
        ax.plot(thr, prf[:, k], color=col, label=name)
    ax.axvline(float(np.mean(z["thresholds"])), color=C["red"], ls="--", lw=0.9, label="chosen threshold")
    ax.set_xlabel("decision threshold (predicted probability)")
    ax.set_title("Final model: threshold choice")
    ax.legend()
    save(fig, a.out, "threshold_curve", a.format, a.dpi)


def fig_confusion(runs, a):
    run = _final_logistic(runs)
    if not run:
        return False
    y = load_labels(a.data_dir, a.sub_sample)
    z = load_oof(a.results_dir, run)
    pred = z["oof_scores"] >= z["thresholds"][z["fold_id"]]
    M = np.array([[np.sum((y == t) & (pred == p)) for p in (False, True)] for t in (-1, 1)])
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    ax.imshow(M / M.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:,}\n({M[i, j] / M[i].sum():.1%})", ha="center", va="center",
                    color="white" if i == j else "black", fontsize=7)
    ax.set_xticks([0, 1], ["no heart\ndisease", "heart\ndisease"])
    ax.set_yticks([0, 1], ["no heart\ndisease", "heart\ndisease"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Final model, out-of-fold")
    ax.grid(False)
    save(fig, a.out, "confusion_matrix", a.format, a.dpi)


def fig_coefficients(runs, a):
    """Largest weights of the final logistic model (columns are standardized: comparable)."""
    run = _final_logistic(runs)
    path = os.path.join(a.results_dir, "features", "final_std.json")
    if not run or not os.path.exists(path):
        return False
    cols = json.load(open(path))["columns"]
    w = load_oof(a.results_dir, run)["weights"].astype(float).mean(axis=0)
    order = [i for i in np.argsort(-np.abs(w)) if cols[i] != "bias"][:20]
    order = sorted(order, key=lambda i: w[i])
    fig, ax = plt.subplots(figsize=(COL1[0], 3.2))
    ax.barh(range(len(order)), [w[i] for i in order], color=[C["red"] if w[i] > 0 else C["blue"] for i in order])
    ax.set_yticks(range(len(order)), [cols[i] if len(cols[i]) < 28 else cols[i][:27] + "…" for i in order], fontsize=6)
    ax.set_xlabel("weight (standardized columns, mean over folds)")
    ax.set_title("Final model: 20 largest weights")
    save(fig, a.out, "coefficients", a.format, a.dpi)


def fig_calibration(runs, a):
    """Reliability of the predicted probabilities of the final logistic model."""
    run = _final_logistic(runs)
    if not run:
        return False
    y = load_labels(a.data_dir, a.sub_sample)
    s = load_oof(a.results_dir, run)["oof_scores"].astype(float)
    edges = np.quantile(s, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, 9)
    pred = np.array([s[idx == b].mean() for b in range(10)])
    obs = np.array([np.mean(y[idx == b] == 1) for b in range(10)])
    fig, ax = plt.subplots(figsize=COL1)
    ax.plot([0, max(pred.max(), obs.max())], [0, max(pred.max(), obs.max())], color=C["grey"], ls="--", lw=0.8)
    ax.plot(pred, obs, marker="o", ms=3.5, color=C["green"])
    ax.set_xlabel("mean predicted probability (decile)")
    ax.set_ylabel("observed positive rate")
    ax.set_title("Final model: calibration")
    save(fig, a.out, "calibration", a.format, a.dpi)


### TABLES ###
def pm(run, key, digits=4):
    return f"{val(run, key):.{digits}f} $\\pm$ {val(run, key, 'std'):.{digits}f}"


def write_table(out, name, header, rows, caption=None, label=None, wide=False):
    """Write <name>.tex and <name>.md. With a caption the .tex is a complete float (table, or table*
    spanning both columns when wide=True); without one it is the bare tabular. Needs booktabs."""
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name + ".tex"), "w") as fh:
        env = "table*" if wide else "table"
        if caption:
            fh.write(f"% requires \\usepackage{{booktabs}}\n\\begin{{{env}}}[t]\n\\centering\n\\small\n")
        fh.write("\\begin{tabular}{" + "l" + "r" * (len(header) - 1) + "}\n\\toprule\n")
        fh.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            fh.write(" & ".join(str(c).replace("_", "\\_") if i == 0 else str(c) for i, c in enumerate(r)) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
        if caption:
            fh.write(f"\\caption{{{caption}}}\n")
            if label:
                fh.write(f"\\label{{{label}}}\n")
            fh.write(f"\\end{{{env}}}\n")
    with open(os.path.join(out, name + ".md"), "w") as fh:
        fh.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(c).replace("$\\pm$", "±") for c in r) + " |\n")
    print(f"  wrote {os.path.join(out, name)}.tex / .md")


def table_ablation(runs, a):
    rows = []
    base = one(runs, algo="ridge_regression", fs="plain", stage="s4_features", label="feature ablation", lam=1e-4)
    for fs in ABLATION_SETS:
        r = one(runs, algo="ridge_regression", fs=fs, stage="s4_features", label="feature ablation", lam=1e-4)
        if r:
            d = f"{val(r, 'f1') - val(base, 'f1'):+.4f}" if base else ""
            rows.append([FEATURE_LABEL[fs], int(r["meta"]["n_features"]), pm(r, "f1"), pm(r, "auc"), d])
    if not rows:
        return False
    write_table(a.out + "/tables", "ablation", ["feature set", "features", "F1", "AUC", "$\\Delta$F1"], rows)


def table_methods(runs, a):
    rows = []
    for label, r in [
        ("SGD ($\\gamma$=0.01, 300 it.), plain", one(runs, algo="mean_squared_error_sgd", fs="plain", stage="s1_base")),
        ("GD, plain", one(runs, algo="mean_squared_error_gd", fs="plain", stage="s3_iterative", gamma=0.02, iters=1000)),
        ("Least squares, plain", one(runs, algo="least_squares", fs="plain", stage="s1_base")),
        ("Ridge, plain", one(runs, algo="ridge_regression", fs="plain", stage="s4_features", label="feature ablation", lam=1e-4)),
        ("Logistic, plain", one(runs, algo="logistic_regression", fs="plain", stage="s3_iterative", gamma=0.02, iters=1000)),
        ("Reg. logistic, plain", one(runs, algo="reg_logistic_regression", fs="plain", stage="s3_iterative", gamma=0.02, iters=1000)),
        ("Ridge, final features", one(runs, algo="ridge_regression", fs="final", stage="s5_final")),
        ("Logistic, final features", one(runs, algo="logistic_regression", fs="final_std", stage="s5_final", iters=500)),
        ("Reg. logistic, final features", one(runs, algo="reg_logistic_regression", fs="final_std", stage="s5_final", iters=500)),
    ]:
        if r:
            rows.append([label, pm(r, "f1"), pm(r, "auc"), f"{val(r, 'mcc'):.4f}", f"{val(r, 'precision'):.3f}",
                         f"{val(r, 'recall'):.3f}"])
    if not rows:
        return False
    caption = (
        "Performance of the methods, 5-fold cross-validation (mean $\\pm$ std over folds; the decision threshold is "
        "chosen to maximize F1). \\emph{plain}: standardized raw features; \\emph{final features}: one-hot "
        "encoding, recoded special codes and dummy interactions. Parameters: SGD $\\gamma=0.01$, 300 iterations; "
        "GD and logistic regression on plain features $\\gamma=0.02$, 1000 iterations (regularized: $\\lambda=0.01$); "
        "ridge $\\lambda=10^{-4}$; logistic regression on final features $\\gamma=0.1$, 500 iterations "
        "(regularized: $\\lambda=10^{-3}$)."
    )
    write_table(a.out + "/tables", "methods", ["model", "F1", "AUC", "MCC", "precision", "recall"], rows,
                caption=caption, label="tab:methods", wide=True)


FIGURES = {
    "methods": fig_methods, "sgd": fig_sgd, "convergence": fig_convergence, "ablation": fig_ablation,
    "seeds": fig_seeds, "lambda": fig_lambda, "final": fig_final, "final_convergence": fig_final_convergence,
    "roc": fig_roc_pr, "threshold": fig_threshold, "confusion": fig_confusion,
    "coefficients": fig_coefficients, "calibration": fig_calibration,
    "table_ablation": table_ablation, "table_methods": table_methods,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--data-dir", default="dataset", help="for y_train.csv (curves from out-of-fold scores)")
    p.add_argument("--sub-sample", action="store_true", help="the results were made with --sub-sample")
    p.add_argument("--only", nargs="+", choices=list(FIGURES), help="only these figures / tables")
    p.add_argument("--list", action="store_true")
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    if a.list:
        for k, f in FIGURES.items():
            print(f"{k:18s} {(f.__doc__ or '').strip().splitlines()[0] if f.__doc__ else ''}")
        return
    setup_style()
    runs = load_runs(a.results_dir)
    print(f"{len(runs)} runs loaded from {a.results_dir}/runs.csv")
    for name in a.only or FIGURES:
        print(name)
        try:
            if FIGURES[name](runs, a) is False:
                print("  skipped: the runs it needs are not in runs.csv")
        except Exception as e:  # one broken figure must not stop the others
            print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
