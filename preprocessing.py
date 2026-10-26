"""Order:
    1. --recode             special survey codes -> NaN / 0, frequency codes -> numbers
    2. --bin                quantile-bin chosen numeric columns (needs --one-hot)
    3. impute NaN with the train median and standardize with train mean/std
    4. --interactions       products of standardized column pairs
    5. --one-hot            categorical columns -> dummies (NaN is its own level)
    6. --dummy-interactions products of numeric columns and dummies (needs --one-hot)
    7. bias column
    8. --drop-json          drop the columns listed by diagnostics.py --save-json
    9. --standardize-dummies  center/scale all columns (mainly for gradient-descent methods)
"""

import json
import os

import numpy as np

### CONFIGURATION ###
# --recode: BRFSS special codes, checked against the value counts of the training data.
# "nan" codes mean don't know / refused, "zero" codes mean "none" (a real 0).
# Only codes that are special for that column are listed (e.g. 7 and 9 are real day counts
# in PHYSHLTH, and 77 is a real age in _AGE80, so no blanket 7/9/77/99 rule).
RECODE_RULES = {
    "PHYSHLTH": {"zero": [88], "nan": [77, 99]},  # days of poor physical health
    "MENTHLTH": {"zero": [88], "nan": [77, 99]},
    "POORHLTH": {"zero": [88], "nan": [77, 99]},
    "CHILDREN": {"zero": [88], "nan": [99]},
    "AVEDRNK2": {"nan": [77, 99]},
    "DRNK3GE5": {"zero": [88], "nan": [77, 99]},
    "MAXDRNKS": {"nan": [77, 99]},
    "_DRNKWEK": {"nan": [99900]},
}
# 1xx = times per day, 2xx = per week, 3xx = per month, 555 = never, 777/999 = don't know/refused
FREQ_PER_DAY_COLUMNS = ["FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1"]
# 1xx = times per week, 2xx = per month, 888 = never, 777/999 = don't know/refused
FREQ_PER_WEEK_COLUMNS = ["ALCDAY5", "STRENGTH"]

# --bin: numeric columns that get quantile-bin dummies (in addition to the numeric column)
BIN_COLUMNS = [
    "_AGE80", "_BMI5", "PHYSHLTH", "MENTHLTH", "POORHLTH", "ALCDAY5", "AVEDRNK2", "DRNK3GE5",
    "MAXDRNKS", "_DRNKWEK", "HEIGHT3", "WEIGHT2", "CHILDREN", "STRENGTH",
    "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1",
]

# --dummy-interactions: pairs of variables; a categorical variable stands for all its dummies,
# a numeric variable for its standardized column. Every column pair of (a, b) gets a product.
DUMMY_INTERACTION_PAIRS = [
    ("_AGE80", "SEX"),
    ("_AGE80", "_SMOKER3"),
    ("_AGE80", "_RFHYPE5"),
    ("_AGE80", "_RFCHOL"),
    ("_AGE80", "DIABETE3"),
    ("_AGE80", "GENHLTH"),
    ("_AGE80", "_TOTINDA"),
    ("_AGE80", "CVDSTRK3"),
    ("_BMI5", "DIABETE3"),
    ("_BMI5", "_RFHYPE5"),
    ("_BMI5", "SEX"),
    ("SEX", "_SMOKER3"),
    ("SEX", "_RFHYPE5"),
    ("SEX", "_RFCHOL"),
    ("SEX", "DIABETE3"),
    ("_RFHYPE5", "_RFCHOL"),
    ("_RFHYPE5", "DIABETE3"),
    ("_RFCHOL", "DIABETE3"),
    ("GENHLTH", "DIFFWALK"),
]

# --interactions: pairs of raw columns, multiplied after standardization
INTERACTION_PAIRS = [
    # Age Interactions
    ("_AGE80", "SEX"),
    ("_AGE80", "_BMI5"),
    ("_AGE80", "_SMOKER3"),
    ("_AGE80", "_RFHYPE5"),  # high blood pressure
    ("_AGE80", "_RFCHOL"),  # high cholesterol
    ("_AGE80", "DIABETE3"),
    ("_AGE80", "GENHLTH"),
    ("_AGE80", "_TOTINDA"),  # physical activity
    ("_AGE80", "CVDSTRK3"),  # stroke
    # sex differences in risk factors
    ("SEX", "_SMOKER3"),
    ("SEX", "_RFHYPE5"),
    ("SEX", "_RFCHOL"),
    ("SEX", "DIABETE3"),
    # co-occurring conditions
    ("_RFHYPE5", "_RFCHOL"),
    ("_RFHYPE5", "DIABETE3"),
    ("_RFCHOL", "DIABETE3"),
    ("_BMI5", "DIABETE3"),
    ("_BMI5", "_RFHYPE5"),
    ("GENHLTH", "DIFFWALK"),
    ("GENHLTH", "PHYSHLTH"),
]


def read_feature_names(data_dir):
    """Column names of x_train.csv without the Id column (aligned with load_csv_data's x)."""
    with open(os.path.join(data_dir, "x_train.csv")) as f:
        return f.readline().strip().split(",")[1:]


def add_interactions(x_train, x_test, names, pairs=INTERACTION_PAIRS):
    """Append the product of each feature pair as a new column.

    Expects imputed + standardized inputs, so each product is of zero-mean, unit-variance
    values. Everything stays in memory; nothing is written to disk. Pairs naming a column
    that doesn't exist are skipped. Returns (x_train, x_test, added_names, skipped_pairs).
    """
    col = {n: i for i, n in enumerate(names)}
    new_train, new_test, added, skipped = [], [], [], []
    for a, b in pairs:
        if a not in col or b not in col:
            skipped.append((a, b))
            continue
        new_train.append(x_train[:, col[a]] * x_train[:, col[b]])
        new_test.append(x_test[:, col[a]] * x_test[:, col[b]])
        added.append(f"{a}*{b}")
    if added:
        x_train = np.c_[x_train, np.array(new_train).T]
        x_test = np.c_[x_test, np.array(new_test).T]
    return x_train, x_test, added, skipped


def fit_one_hot(x, names, max_levels, min_frac):
    """Decide which columns are categorical and which levels each one gets (train data only).

    A column is categorical if all its non-NaN values are integers and it has at most
    `max_levels` distinct values. Levels covering less than `min_frac` of the rows are merged
    into one "other" category, and NaN becomes its own "missing" category. The most frequent
    category is the dropped baseline (all its dummies 0), so the dummies of one variable are
    never collinear with the bias column.

    Returns a list of encoders: dicts with the column index, the kept level values,
    has_missing / has_other flags, the index of the dropped category and dummy names.
    """
    n = x.shape[0]
    encoders = []
    for j in range(x.shape[1]):
        col = x[:, j]
        nan_mask = np.isnan(col)
        valid = col[~nan_mask]
        if valid.size == 0:
            continue
        vals, cnts = np.unique(valid, return_counts=True)
        if len(vals) > max_levels or not np.all(vals == np.round(vals)):
            continue
        keep = cnts / n >= min_frac
        levels = vals[keep]
        counts = list(cnts[keep])
        labels = [f"{names[j]}=={int(v)}" for v in levels]
        has_missing = bool(nan_mask.any())
        has_other = bool((~keep).any())
        if has_missing:
            counts.append(int(nan_mask.sum()))
            labels.append(f"{names[j]}==missing")
        if has_other:
            counts.append(int(cnts[~keep].sum()))
            labels.append(f"{names[j]}==other")
        if len(counts) < 2:  # constant column: nothing to encode
            continue
        drop = int(np.argmax(counts))
        encoders.append(
            {
                "col": j,
                "levels": levels,
                "has_missing": has_missing,
                "has_other": has_other,
                "drop": drop,
                "labels": [lab for i, lab in enumerate(labels) if i != drop],
            }
        )
    return encoders


def apply_one_hot(x, encoders, out):
    """Write the dummy columns of every encoder into `out` (shape n x total_dummies)."""
    pos = 0
    for enc in encoders:
        col = x[:, enc["col"]]
        nan_mask = np.isnan(col)
        masks = [col == v for v in enc["levels"]]
        if enc["has_missing"]:
            masks.append(nan_mask)
        if enc["has_other"]:  # values outside the kept levels (also unseen test values)
            masks.append(~nan_mask & ~np.isin(col, enc["levels"]))
        for i, m in enumerate(masks):
            if i != enc["drop"]:
                out[:, pos] = m
                pos += 1


### RECODING (--recode) ###
def recode(x, names):
    """Replace special codes in place (x is modified). Returns the names of the changed columns."""
    col = {n: i for i, n in enumerate(names)}
    changed = []

    for name, rule in RECODE_RULES.items():
        if name not in col:
            continue
        v = x[:, col[name]]  # view: edits write into x
        v[np.isin(v, rule.get("nan", []))] = np.nan
        v[np.isin(v, rule.get("zero", []))] = 0.0
        changed.append(name)

    for name in FREQ_PER_DAY_COLUMNS:
        if name in col:
            v = x[:, col[name]].copy()
            out = np.full(v.shape, np.nan)
            out[(v >= 101) & (v <= 199)] = v[(v >= 101) & (v <= 199)] - 100
            out[(v >= 201) & (v <= 299)] = (v[(v >= 201) & (v <= 299)] - 200) / 7
            out[(v >= 301) & (v <= 399)] = (v[(v >= 301) & (v <= 399)] - 300) / 30
            out[v == 555] = 0.0
            x[:, col[name]] = out
            changed.append(name)

    for name in FREQ_PER_WEEK_COLUMNS:
        if name in col:
            v = x[:, col[name]].copy()
            out = np.full(v.shape, np.nan)
            out[(v >= 101) & (v <= 199)] = v[(v >= 101) & (v <= 199)] - 100
            out[(v >= 201) & (v <= 299)] = (v[(v >= 201) & (v <= 299)] - 200) * 7 / 30
            out[v == 888] = 0.0
            x[:, col[name]] = out
            changed.append(name)

    if "HEIGHT3" in col:  # feet*100 + inches, or 9000 + cm  ->  cm
        v = x[:, col["HEIGHT3"]].copy()
        out = np.full(v.shape, np.nan)
        imperial = (v >= 200) & (v <= 711)
        out[imperial] = (v[imperial] // 100) * 30.48 + (v[imperial] % 100) * 2.54
        metric = (v >= 9000) & (v <= 9998)
        out[metric] = v[metric] - 9000
        x[:, col["HEIGHT3"]] = out
        changed.append("HEIGHT3")

    if "WEIGHT2" in col:  # pounds, or 9000 + kg  ->  kg
        v = x[:, col["WEIGHT2"]].copy()
        out = np.full(v.shape, np.nan)
        imperial = (v >= 50) & (v <= 999)
        out[imperial] = v[imperial] * 0.45359237
        metric = (v >= 9000) & (v <= 9998)
        out[metric] = v[metric] - 9000
        x[:, col["WEIGHT2"]] = out
        changed.append("WEIGHT2")

    return changed


### BINNING (--bin) ###
def fit_bins(x, names, columns, n_bins):
    """Quantile bin edges per column, from the training data. Returns {name: edges}."""
    col = {n: i for i, n in enumerate(names)}
    edges = {}
    for name in columns:
        if name not in col:
            continue
        v = x[:, col[name]]
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        # repeated quantiles (e.g. many zeros) collapse into fewer bins
        e = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
        if len(e):
            edges[name] = e
    return edges


def apply_bins(x, names, edges):
    """Bin ids (as floats, NaN stays NaN) for the columns in `edges`, shape n x len(edges)."""
    col = {n: i for i, n in enumerate(names)}
    out = np.empty((x.shape[0], len(edges)))
    for k, (name, e) in enumerate(edges.items()):
        v = x[:, col[name]]
        out[:, k] = np.where(np.isnan(v), np.nan, np.searchsorted(e, v, side="right"))
    return out


### FULL PIPELINE ###
def _flag(cfg, key, default=False):
    return getattr(cfg, key, default)


def preprocess(x_train, x_test, names, cfg):
    """Build the model matrices. Returns (tx_train, tx_test, info).

    `cfg` is any object with the attributes of add_feature_args() (e.g. argparse Namespace).
    Missing attributes count as "off". x_train / x_test may be modified in place by --recode.
    info["columns"] holds the name of every output column, info has the details of each step.
    """
    names = list(names)
    one_hot = _flag(cfg, "one_hot")
    max_levels = _flag(cfg, "max_levels", 10)
    min_level_frac = _flag(cfg, "min_level_frac", 0.005)
    if (_flag(cfg, "bin") or _flag(cfg, "dummy_interactions")) and not one_hot:
        raise ValueError("--bin and --dummy-interactions need --one-hot")
    info = {
        "added": [], "skipped": [], "n_categorical": 0, "n_dummies": 0, "recoded": [],
        "binned": [], "dummy_interactions": [], "dummy_interactions_skipped": [],
        "dropped": 0, "drop_unmatched": [], "columns": [],
    }

    if _flag(cfg, "recode"):
        info["recoded"] = recode(x_train, names)
        recode(x_test, names)

    if _flag(cfg, "bin"):
        edges = fit_bins(x_train, names, BIN_COLUMNS, min(_flag(cfg, "bins", 8), max_levels))
        if edges:
            x_train = np.hstack([x_train, apply_bins(x_train, names, edges)])
            x_test = np.hstack([x_test, apply_bins(x_test, names, edges)])
            names += [f"{n}_bin" for n in edges]
            info["binned"] = list(edges)

    raw_train, raw_test = x_train, x_test  # keep NaNs for the one-hot encoder
    median = np.nanmedian(x_train, axis=0)
    median = np.where(np.isnan(median), 0.0, median)  # columns that are all NaN
    x_train = np.where(np.isnan(x_train), median, x_train)
    x_test = np.where(np.isnan(x_test), median, x_test)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    d0 = x_train.shape[1]
    if _flag(cfg, "interactions"):
        x_train, x_test, info["added"], info["skipped"] = add_interactions(x_train, x_test, names)
    names_std = names + info["added"]  # names of the columns of x_train / x_test now

    if not one_hot:
        columns = ["bias"] + names_std
        tx = np.c_[np.ones(x_train.shape[0]), x_train]
        tx_test = np.c_[np.ones(x_test.shape[0]), x_test]
    else:
        encoders = fit_one_hot(raw_train, names, max_levels, min_level_frac)
        cat_cols = {enc["col"] for enc in encoders}
        # numeric block: original non-categorical columns, then the interaction columns
        keep_cols = [j for j in range(d0) if j not in cat_cols] + list(range(d0, x_train.shape[1]))
        numeric_names = [names_std[j] for j in keep_cols]
        n_dummies = sum(len(enc["labels"]) for enc in encoders)
        info["n_categorical"], info["n_dummies"] = len(encoders), n_dummies
        columns = ["bias"] + numeric_names + [lab for enc in encoders for lab in enc["labels"]]

        # variable -> its column indices in the output (a categorical variable = all its dummies)
        var_cols = {n: [1 + k] for k, n in enumerate(numeric_names)}
        pos = 1 + len(numeric_names)
        for enc in encoders:
            var_cols[names[enc["col"]]] = list(range(pos, pos + len(enc["labels"])))
            pos += len(enc["labels"])
        pairs = []  # (column i, column j, name) of --dummy-interactions
        if _flag(cfg, "dummy_interactions"):
            for a, b in DUMMY_INTERACTION_PAIRS:
                if a not in var_cols or b not in var_cols:
                    info["dummy_interactions_skipped"].append((a, b))
                    continue
                pairs += [
                    (i, j, f"{columns[i]}*{columns[j]}")
                    for i in var_cols[a]
                    for j in var_cols[b]
                    if i != j
                ]
            info["dummy_interactions"] = [p[2] for p in pairs]
            columns += info["dummy_interactions"]

        def build(x_std, x_raw):
            out = np.zeros((x_std.shape[0], len(columns)))
            out[:, 0] = 1.0  # bias
            out[:, 1 : 1 + len(keep_cols)] = x_std[:, keep_cols]
            start = 1 + len(keep_cols)
            apply_one_hot(x_raw, encoders, out[:, start : start + n_dummies])
            for k, (i, j, _) in enumerate(pairs):
                out[:, start + n_dummies + k] = out[:, i] * out[:, j]
            return out

        tx, tx_test = build(x_train, raw_train), build(x_test, raw_test)

    drop_json = _flag(cfg, "drop_json", None)
    if drop_json:
        with open(drop_json) as f:
            drop = set(json.load(f)["columns_to_drop"]) - {"bias"}
        keep = [i for i, c in enumerate(columns) if c not in drop]
        info["dropped"] = len(columns) - len(keep)
        info["drop_unmatched"] = sorted(drop - set(columns))  # not in this feature set
        tx, tx_test = tx[:, keep], tx_test[:, keep]
        columns = [columns[i] for i in keep]

    if _flag(cfg, "standardize_dummies"):
        # center and scale every column (except the bias) with train mean/std: one-hot and
        # product columns are mostly 0 and badly scaled, which makes gradient descent crawl
        mean = tx[:, 1:].mean(axis=0)
        std = tx[:, 1:].std(axis=0)
        std[std == 0] = 1.0
        for m in (tx, tx_test):
            m[:, 1:] -= mean
            m[:, 1:] /= std

    info["columns"] = columns
    return tx, tx_test, info


### COMMAND LINE FLAGS ###
def add_feature_args(p):
    """Add the feature-pipeline flags to an argparse parser (shared by run.py, diagnostics.py)."""
    p.add_argument("--recode", action="store_true", help="recode special survey codes (see RECODE_RULES)")
    p.add_argument("--bin", action="store_true", help="quantile-bin numeric columns (needs --one-hot)")
    p.add_argument("--bins", type=int, default=8, help="number of quantile bins for --bin")
    p.add_argument(
        "--interactions",
        action="store_true",
        help="add pairwise interaction features (see INTERACTION_PAIRS), kept in memory only",
    )
    p.add_argument(
        "--one-hot",
        action="store_true",
        help="one-hot encode categorical columns (few integer levels), in memory only",
    )
    p.add_argument("--max-levels", type=int, default=10, help="max distinct values of a categorical column")
    p.add_argument(
        "--min-level-frac",
        type=float,
        default=0.005,
        help="levels rarer than this fraction of rows are merged into 'other'",
    )
    p.add_argument(
        "--dummy-interactions",
        action="store_true",
        help="products of numeric columns and one-hot dummies (see DUMMY_INTERACTION_PAIRS)",
    )
    p.add_argument(
        "--standardize-dummies",
        action="store_true",
        help="standardize all final columns (helps gradient descent / logistic regression converge)",
    )
    p.add_argument(
        "--drop-json",
        metavar="PATH",
        help="drop the 'columns_to_drop' of a diagnostics.py --save-json report (same flags!)",
    )


def validate_feature_args(p, args):
    """Reject flag combinations that need --one-hot."""
    if (args.bin or args.dummy_interactions) and not args.one_hot:
        p.error("--bin and --dummy-interactions need --one-hot")
