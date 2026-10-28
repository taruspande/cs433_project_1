# CS-433 Project 1: predicting heart disease (MICHD) from BRFSS survey data

Binary classification (label -1 / +1) of about 330k survey respondents, scored with F1.
Only Python and NumPy are used for the models, features and metrics (matplotlib is used by `plots.py` for visualization only).

This file documents the files that are tracked in the repository and where everything is.

## Repository layout

```
README.md                 this file
.gitignore                what is not pushed (data, venv, logs, shell scripts, large result files)

implementations.py        the six required methods (+ their loss / gradient helpers)
helpers.py                course helpers: load_csv_data, create_csv_submission
preprocessing.py          feature pipeline (recoding, bins, interactions, one-hot, ...)
run.py                    train / cross-validate / write a submission
experiments.py            the whole experiment study of the report (writes results/ and submissions/)
diagnostics.py            linear-dependency analysis of the feature matrix (writes column_report.json)
plots.py                  figures and tables for the report (reads results/, writes figures/)
column_report.json        output of diagnostics.py, used by run.py --drop-json

results/                  outputs of experiments.py
submissions/              the final submission files
figures/                  the figures and tables of the report
grading_tests/            the course's public tests (signature checks of implementations.py)
```

The data is not in the repository. Put the competition files in a folder `dataset/` next to `run.py`:
`x_train.csv`, `y_train.csv`, `x_test.csv` (and `sample_submission.csv`).

## The code

### `implementations.py`
The six functions required by the project, with the requested signatures, returning `(w, loss)`:
`mean_squared_error_gd`, `mean_squared_error_sgd` (batch size 1), `least_squares` (normal equations),
`ridge_regression` (adds `2 * N * lambda_` to the diagonal of the normal equations), `logistic_regression` and
`reg_logistic_regression` (gradient descent, labels in {0, 1}). The helpers `sigmoid`, `compute_loss`,
`compute_gradient`, `compute_logistic_loss` and `compute_logistic_gradient` are in the same file.

### `helpers.py`
`load_csv_data(data_path, sub_sample=False)` reads the three csv files (`sub_sample=True` keeps every 50th training row)
and `create_csv_submission(ids, y_pred, name)` writes a file in the format of `sample_submission.csv`.

### `preprocessing.py`
Builds the model matrices from the raw columns. Every step except imputation, standardization and the bias column is an
option, selected with command-line flags (`add_feature_args()` defines them; `run.py` and `diagnostics.py` share them):

| order | flag | what it does |
|---|---|---|
| 1 | `--recode` | special survey codes -> NaN or 0 (`RECODE_RULES`), frequency codes -> numbers, height/weight -> cm/kg |
| 2 | `--bin` | adds quantile-bin dummies for 20 numeric columns (`BIN_COLUMNS`, `--bins`); needs `--one-hot` |
| 3 | (always) | NaN -> training median, standardize with training mean/std |
| 4 | `--interactions` | products of 20 raw column pairs (`INTERACTION_PAIRS`) |
| 5 | `--one-hot` | columns with at most `--max-levels` integer levels -> 0/1 dummies; NaN is its own level, levels under `--min-level-frac` merged into "other", the most frequent level of each variable dropped |
| 6 | `--dummy-interactions` | products of numeric columns and dummies, and of pairs of dummies (`DUMMY_INTERACTION_PAIRS`); needs `--one-hot` |
| 7 | (always) | bias column |
| 8 | `--drop-json PATH` | drops the columns listed in a `diagnostics.py --save-json` report (must be made with the same flags) |
| 9 | `--standardize-dummies` | centers and scales every column except the bias (helps gradient descent / logistic regression converge) |

All statistics (medians, means, bin edges, one-hot levels) come from the training rows only; nothing is written to disk.
`preprocess(x_train, x_test, names, cfg)` returns the two matrices and an `info` dict with the name of every column.

### `run.py`
Trains the methods of `implementations.py` with k-fold cross-validation and reports, per algorithm, the mean over folds of
accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, AUC, training F1, time and the confusion counts.
The decision threshold is chosen per fold to maximize F1 (`--no-tune` turns this off). With `--submit <algorithm|best>`
it refits on all training data and writes the predictions for the test set using the mean of the per-fold thresholds.
Main options: `--algorithms`, `--folds` (default 5), `--seed`, `--gamma`, `--lambda_`, `--max-iters`, `--sub-sample`, `--output`,
plus the feature flags above.

```
python run.py --algorithms ridge_regression --lambda_ 0.0001 --one-hot --recode --dummy-interactions
```

### `experiments.py`
Runs the whole study of the report in one process (data loaded once, each feature set built once), in five stages:
1. base comparison of the six methods on plain features,
2. SGD step size x iterations,
3. gradient descent / logistic regression with convergence checkpoints,
4. feature ablation with ridge (10 feature sets, 3 seeds, lambda sweep, least squares),
5. the final models (logistic regression on the final features, ridge, and the variants).

`python experiments.py --dry-run` prints the plan, `--stages 1 2` runs some stages, `--sub-sample` is a fast test.
Runs already present in `results/runs.csv` are skipped, so an interrupted study can be restarted.
Outputs go to `results/` and, for the two final models, to `submissions/`.

### `diagnostics.py`
Analyses the linear dependencies of the feature matrix built by `preprocessing.py` (same flags as `run.py`):
constant columns, groups of identical columns, the null space of X^T X (eigenvalues), columns with too few non-zero rows,
and a Gram-Schmidt elimination of redundant columns. `--save-json PATH` writes every category and the list `columns_to_drop`.

```
python diagnostics.py --one-hot --interactions --recode --dummy-interactions --save-json column_report.json
```

### `plots.py`
Reads `results/runs.csv` and the saved out-of-fold scores in `results/oof/` and writes the figures and tables to `figures/`
(`--list` shows them, `--only` selects some, `--format png|pdf`, `--dpi`). Needs matplotlib.

## Data files

### `column_report.json`
Output of `diagnostics.py --save-json` for the flags `--one-hot --interactions --recode --dummy-interactions` (878 features):
64 groups of identical columns, 179 columns to drop (51 with too little support, 128 redundant), 699 columns kept.
It is only valid for that feature set.

### `results/`
- `runs.csv`: one row per run and cross-validation fold, plus `mean` and `std` rows, with all metrics (87 runs).
- `plan.json` and `meta.json`: the list of experiments and the command / versions of the study.
- `features/<set>.json`: the column names of every feature set used.
- `oof/<run_id>.npz` (not tracked, about 100 MB): out-of-fold scores, fold ids, thresholds and weights of every run.
  `plots.py` needs it for the ROC / PR / threshold / confusion / calibration / coefficient figures; regenerate it with `python experiments.py`.

### `submissions/`
`reg_logistic_final.csv` and `ridge_final.csv` are test-set predictions in the competition format.
The `.json` next to each holds the parameters, the decision threshold, the predicted positive rate and the cross-validation scores.

### `figures/`
`.pdf` and `.png` of every figure made by `plots.py`, and `tables/` with the ablation and method tables (`.tex` and `.md`).

| file | shows |
|---|---|
| `methods_base` | the six methods, plain features, default parameters |
| `sgd_grid` | SGD step size x iterations |
| `convergence_plain` | gradient methods vs iterations (plain features) |
| `feature_ablation` | ridge on the feature sets |
| `seed_stability` | the ablation over three fold splits |
| `ridge_lambda` | ridge vs lambda, per fold and mean |
| `final_models` | the final models compared |
| `convergence_final` | regularized logistic regression vs iterations on the final features |
| `roc_pr`, `threshold_curve`, `confusion_matrix`, `calibration`, `coefficients` | the final logistic model (out-of-fold) |

### `grading_tests/`
The course's public tests, which check the signatures of `implementations.py` (see `grading_tests/INSTRUCTIONS.md`).
Run `pytest --github_link <repo-url-or-local-path> .` from that folder in the environment of `grading_tests/environment.yml`.

## Final model

Regularized logistic regression (`reg_logistic_regression`) with `lambda_ = 0.001`, `gamma = 0.1`, 500 iterations from zero weights,
on the features `--one-hot --recode --dummy-interactions --standardize-dummies` (858 columns).
Five-fold cross-validation (seed 0): F1 0.4393, AUC 0.8656; decision threshold 0.2228. It was written by `experiments.py` to
`submissions/reg_logistic_final.csv`; the same settings with `run.py`:

```
python run.py --algorithms reg_logistic_regression --lambda_ 0.001 --gamma 0.1 --max-iters 500 \
    --one-hot --recode --dummy-interactions --standardize-dummies \
    --submit reg_logistic_regression --output reg_logistic_final.csv
```

## Requirements
Python 3 with NumPy (models, features, metrics); matplotlib only for `plots.py`.
