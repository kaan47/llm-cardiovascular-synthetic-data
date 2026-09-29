"""Evaluation of LLM-generated synthetic cardiovascular cohorts.

Reproduces every table and number reported in
"Large Language Models as Clinical Data Synthesizers" (Kara and Gunel).

Usage
-----
    python run_analysis.py                  # full analysis (about 35 min on one CPU core)
    python run_analysis.py --quick          # reduced settings for a functional check
    python run_analysis.py --jobs 4         # parallel model fitting

Inputs are read from ``data/`` and all outputs are written to ``results/``.
Time-consuming steps are cached in ``results/cache/`` so that an interrupted
run resumes where it stopped.
"""

import argparse
import hashlib
import json
import os
import pickle
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, linregress, norm, rankdata, spearmanr, truncnorm
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
MODELS = ["ChatGPT", "Claude", "DeepSeek", "Gemini"]
RUNS = [1, 2, 3]
TARGET = "target"
CONTINUOUS = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
LABELS = {
    "age": "Age",
    "sex": "Sex",
    "chest pain type": "Chest pain type",
    "resting bp s": "Resting BP",
    "cholesterol": "Cholesterol",
    "fasting blood sugar": "Fasting blood sugar",
    "resting ecg": "Resting ECG",
    "max heart rate": "Max heart rate",
    "exercise angina": "Exercise angina",
    "oldpeak": "Oldpeak",
    "ST slope": "ST slope",
    "target": "Heart disease",
}
CATEGORIES = {
    "sex": {0: "Female", 1: "Male"},
    "chest pain type": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fasting blood sugar": {0: "<= 120 mg/dl", 1: "> 120 mg/dl"},
    "resting ecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exercise angina": {0: "No", 1: "Yes"},
    "ST slope": {0: "Code 0 (undocumented)", 1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "target": {0: "No", 1: "Yes"},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="folder with real_data.csv and synthetic/")
    parser.add_argument("--out", default="results", help="output folder")
    parser.add_argument("--jobs", type=int, default=1, help="parallel jobs for model fitting")
    parser.add_argument("--quick", action="store_true", help="reduced settings for a functional check")
    return parser.parse_args()


ARGS = parse_args()
N_JOBS = ARGS.jobs
if ARGS.quick:
    N_BOOT, N_REP = 200, 2
    N_ITER_EXT, CV_EXT = 2, 3
    N_OUTER, N_ITER_IN, CV_IN = 2, 2, 2
else:
    N_BOOT, N_REP = 1000, 10  # bootstrap resamples, benchmark repetitions
    N_ITER_EXT, CV_EXT = 10, 5  # external validation: candidates, CV folds
    N_OUTER, N_ITER_IN, CV_IN = 5, 5, 3  # nested CV: outer folds, inner candidates, inner folds

TABLES = os.path.join(ARGS.out, "tables")
CACHE = os.path.join(ARGS.out, "cache")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
NUMBERS = {}


def save(df, name):
    df.to_csv(os.path.join(TABLES, name), index=False)
    return df


def cached(key, compute):
    path = os.path.join(CACHE, hashlib.md5(key.encode()).hexdigest()[:16] + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    result = compute()
    with open(path, "wb") as fh:
        pickle.dump(result, fh)
    return result


def mean_sd(values, digits=3):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return f"{values.mean():.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def label(col):
    return LABELS.get(col, col)


def run_name(key):
    return f"{key[0]}-{key[1]}"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
real = pd.read_csv(os.path.join(ARGS.data, "real_data.csv"))
FEATURES = [c for c in real.columns if c != TARGET]
COLUMNS = list(real.columns)
real_unique = real.drop_duplicates().reset_index(drop=True)

synthetic = {}
for model in MODELS:
    for run in RUNS:
        df = pd.read_csv(os.path.join(ARGS.data, "synthetic", f"{model.lower()}_run{run}.csv"))
        if list(df.columns) != COLUMNS:
            raise ValueError(f"unexpected columns in {model} run {run}")
        synthetic[(model, run)] = df.apply(pd.to_numeric)

print(f"real data: {len(real)} records, {len(real_unique)} unique")
print("synthetic cohorts:", {run_name(k): len(v) for k, v in synthetic.items()})

# ----------------------------------------------------------------------------
# 1  Data audit (Table S3)
# ----------------------------------------------------------------------------
real_values = real.values.astype(float)
scaler = StandardScaler().fit(real_unique[FEATURES])


def max_equal_fields(df):
    """Largest number of the 12 fields that a synthetic record shares with any single real record."""
    return np.array([(real_values == row).sum(1).max() for row in df.values.astype(float)])


def real_records_with_copy(df):
    """Real records that have a copy (at least 11 of 12 identical fields) in a synthetic cohort."""
    rows = df.values.astype(float)
    return np.array([((rows == row).sum(1) >= 11).any() for row in real_values])


def nearest_within(df):
    dist = cdist(scaler.transform(df[FEATURES]), scaler.transform(df[FEATURES]))
    np.fill_diagonal(dist, np.inf)
    return dist.min(1)


def template_score(df):
    """Row lag with the most similar records, and that similarity relative to random ordering."""
    z = scaler.transform(df[FEATURES])
    by_lag = [np.abs(z[:-lag] - z[lag:]).mean() for lag in range(1, 41)]
    shuffled = np.mean([np.abs(z - z[np.random.default_rng(s).permutation(len(z))]).mean() for s in range(20)])
    return int(np.argmin(by_lag) + 1), min(by_lag) / shuffled


equal_fields = {k: max_equal_fields(df) for k, df in synthetic.items()}
reference_diversity = np.median([np.median(nearest_within(real_unique.sample(200, random_state=s))) for s in range(20)])

audit = []
for (model, run), df in synthetic.items():
    lag, lag_ratio = template_score(df)
    near = equal_fields[(model, run)]
    audit.append(
        {
            "Model": model,
            "Run": run,
            "Records": len(df),
            "Within-dataset duplicates": int(df.duplicated().sum()),
            "Exact copies (12/12 fields)": int((near == 12).sum()),
            "Near-exact copies (>=11/12 fields)": int((near >= 11).sum()),
            "% records near-exact": round(100 * (near >= 11).mean(), 1),
            "Relative within-dataset diversity": round(np.median(nearest_within(df)) / reference_diversity, 2),
            "Template lag (rows)": lag,
            "Lag similarity ratio": round(lag_ratio, 3),
            "Non-integer ages": int((df.age % 1 != 0).sum()),
            "Cholesterol = 0": int((df.cholesterol == 0).sum()),
            "Target prevalence": round(df[TARGET].mean(), 3),
        }
    )
audit = save(pd.DataFrame(audit), "TableS3_data_audit.csv")
NUMBERS["reference_within_nn"] = round(reference_diversity, 3)

# ----------------------------------------------------------------------------
# 2  Univariate fidelity (Tables S4-S7)
# ----------------------------------------------------------------------------
ks_threshold = {
    c: np.percentile([ks_2samp(real[c], real[c].sample(200, random_state=s)).statistic for s in range(200)], 95)
    for c in FEATURES
}
ks_rows = []
for (model, run), df in synthetic.items():
    for c in FEATURES:
        test = ks_2samp(real[c], df[c])
        ks_rows.append(
            {
                "Model": model,
                "Run": run,
                "Feature": label(c),
                "KS": test.statistic,
                "p": test.pvalue,
                "Threshold": ks_threshold[c],
            }
        )
ks = pd.DataFrame(ks_rows)
ks["p>0.05"] = ks.p > 0.05
ks["Above threshold"] = ks.KS > ks.Threshold
save(ks.round(4), "TableS5_ks_tests.csv")
NUMBERS["ks_nonsignificant_features"] = ks.groupby(["Model", "Run"])["p>0.05"].sum().unstack().to_dict("index")

describe = []
for c in COLUMNS:
    row = {"Variable": label(c), "Real": f"{real[c].mean():.2f} ({real[c].min():g}–{real[c].max():g})"}
    for model in MODELS:
        means = [synthetic[(model, run)][c].mean() for run in RUNS]
        pooled = pd.concat([synthetic[(model, run)][c] for run in RUNS])
        row[model] = f"{np.mean(means):.2f} [{min(means):.2f}–{max(means):.2f}] ({pooled.min():g}–{pooled.max():g})"
    describe.append(row)
save(pd.DataFrame(describe), "TableS4_descriptive_statistics.csv")

sd_ratio = [
    {"Model": k[0], "Run": k[1], **{label(c): round(df[c].std() / real[c].std(), 2) for c in CONTINUOUS}}
    for k, df in synthetic.items()
]
save(pd.DataFrame(sd_ratio), "TableS6_sd_ratios.csv")

frequencies = []
for c, levels in CATEGORIES.items():
    for code, name in levels.items():
        row = {"Variable": label(c), "Category": name, "Real": round(100 * (real[c] == code).mean(), 1)}
        for model in MODELS:
            shares = [100 * (synthetic[(model, run)][c] == code).mean() for run in RUNS]
            row[model] = f"{np.mean(shares):.1f} [{min(shares):.1f}–{max(shares):.1f}]"
        frequencies.append(row)
save(pd.DataFrame(frequencies), "TableS7_category_frequencies.csv")

# ----------------------------------------------------------------------------
# 3  Relational fidelity (Tables S8-S9)
# ----------------------------------------------------------------------------
UPPER = np.triu_indices(len(COLUMNS), 1)


def correlation_mad(df, ref=real):
    a, b = df[COLUMNS].corr().fillna(0).values, ref[COLUMNS].corr().values
    return np.mean(np.abs(a[UPPER] - b[UPPER]))


def sign_agreement(df, ref=real):
    a, b = df[COLUMNS].corr().fillna(0).values, ref[COLUMNS].corr().values
    keep = np.abs(b[UPPER]) >= 0.1
    return np.mean(np.sign(a[UPPER][keep]) == np.sign(b[UPPER][keep]))


def cramers_v(x, y):
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))


def outcome_association(df):
    values = []
    for c in FEATURES:
        if c in CONTINUOUS:
            values.append(np.corrcoef(df[c], df[TARGET])[0, 1] if df[c].std() > 0 else 0)
        else:
            values.append(cramers_v(df[c], df[TARGET]))
    return np.array(values)


def independent_marginals(n, seed):
    """Baseline that uses exactly the information of Table 1: independent truncated normals."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in COLUMNS:
        mu, sd, lo, hi = round(real[c].mean(), 2), round(real[c].std(), 2), real[c].min(), real[c].max()
        x = truncnorm.rvs((lo - mu) / sd, (hi - mu) / sd, loc=mu, scale=sd, size=n, random_state=rng)
        out[c] = np.clip(np.round(x, 1) if c == "oldpeak" else np.round(x), lo, hi)
    return pd.DataFrame(out)[COLUMNS]


association_real = outcome_association(real)
continuous_idx = [FEATURES.index(c) for c in CONTINUOUS]
importance_real = (
    RandomForestClassifier(n_estimators=300, random_state=SEED)
    .fit(real_unique[FEATURES], real_unique[TARGET])
    .feature_importances_
)

relational = []
for (model, run), df in synthetic.items():
    importance = (
        RandomForestClassifier(n_estimators=300, random_state=SEED).fit(df[FEATURES], df[TARGET]).feature_importances_
    )
    association = outcome_association(df)
    relational.append(
        {
            "Model": model,
            "Run": run,
            "Correlation MAD": correlation_mad(df),
            "Sign agreement": sign_agreement(df),
            "Outcome-association MAE": np.mean(np.abs(association - association_real)),
            "Outcome-association sign agreement": np.mean(
                np.sign(association[continuous_idx]) == np.sign(association_real[continuous_idx])
            ),
            "Importance rank correlation": spearmanr(importance, importance_real).correlation,
        }
    )
relational = pd.DataFrame(relational)

null_mad = [correlation_mad(real.sample(200, random_state=s)) for s in range(200)]
NUMBERS["correlation_mad_real_subsample"] = [round(np.mean(null_mad), 3), round(np.percentile(null_mad, 95), 3)]
NUMBERS["correlation_mad_independent_marginals"] = round(
    np.mean([correlation_mad(independent_marginals(200, s)) for s in range(20)]), 3
)

PAIRS = [
    ("age", "max heart rate"),
    ("age", "resting bp s"),
    ("resting bp s", "cholesterol"),
    ("max heart rate", "oldpeak"),
]
regression = []
for x, y in PAIRS:
    for (model, run), df in [(("Real", 0), real)] + list(synthetic.items()):
        fit = linregress(df[x], df[y])
        residual = df[y] - (fit.intercept + fit.slope * df[x])
        regression.append(
            {
                "Pair": f"{label(x)} -> {label(y)}",
                "Model": model,
                "Run": run,
                "Slope": fit.slope,
                "CI low": fit.slope - 1.96 * fit.stderr,
                "CI high": fit.slope + 1.96 * fit.stderr,
                "r": fit.rvalue,
                "Residual SD": residual.std(),
            }
        )
save(pd.DataFrame(regression).round(3), "TableS8_bivariate_regression.csv")

# ----------------------------------------------------------------------------
# 4  Risk maps and SHAP interactions (Table S9)
# ----------------------------------------------------------------------------
mutual_info = mutual_info_classif(real[FEATURES], real[TARGET], random_state=SEED)
TOP4 = [FEATURES[i] for i in np.argsort(mutual_info)[-4:][::-1]]
NUMBERS["top4_mutual_information"] = [label(c) for c in TOP4]
LEVELS = {"ST slope": [1, 2, 3], "chest pain type": [1, 2, 3, 4], "exercise angina": [0, 1]}
HR_EDGES = np.percentile(real["max heart rate"], [0, 20, 40, 60, 80, 100])


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def axis_values(feature):
    if feature in LEVELS:
        return [[v] for v in LEVELS[feature]]
    return [np.linspace(HR_EDGES[i], HR_EDGES[i + 1], 8) for i in range(5)]


def risk_map(df, f1, f2, kind):
    """Predicted risk for every category combination that exists in the data."""
    if kind == "RF":
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=SEED)
    else:
        model = logistic()
    model.fit(df[[f1, f2]], df[TARGET])
    grid = np.zeros((len(axis_values(f2)), len(axis_values(f1))))
    for i, v2 in enumerate(axis_values(f2)):
        for j, v1 in enumerate(axis_values(f1)):
            points = pd.DataFrame([(a, b) for a in v1 for b in v2], columns=[f1, f2])
            grid[i, j] = model.predict_proba(points)[:, 1].mean()
    return grid


def safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def shap_interactions(df):
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED).fit(df[FEATURES], df[TARGET])
    values = shap.TreeExplainer(model).shap_interaction_values(df[FEATURES].sample(min(300, len(df)), random_state=0))
    if isinstance(values, list):
        values = values[1]
    elif np.ndim(values) == 4:
        values = values[..., 1]
    matrix = np.abs(values).mean(0)
    np.fill_diagonal(matrix, 0)
    return matrix


FEATURE_PAIRS = list(combinations(TOP4, 2))
maps_real = {kind: {p: risk_map(real, *p, kind) for p in FEATURE_PAIRS} for kind in ["RF", "LR"]}
shap_real = cached("shap_real", lambda: shap_interactions(real))
UPPER_F = np.triu_indices(len(FEATURES), 1)

higher_order = []
for (model, run), df in synthetic.items():
    row = {"Model": model, "Run": run}
    matrix = cached(f"shap_{model}-{run}", lambda df=df: shap_interactions(df))
    row["SHAP interaction similarity"] = spearmanr(shap_real[UPPER_F], matrix[UPPER_F]).correlation
    for kind in ["RF", "LR"]:
        agreement = [safe_corr(maps_real[kind][p].ravel(), risk_map(df, *p, kind).ravel()) for p in FEATURE_PAIRS]
        row[f"Risk-map agreement ({kind})"] = np.nanmean(agreement)
    higher_order.append(row)
fidelity = save(
    relational.merge(pd.DataFrame(higher_order), on=["Model", "Run"]).round(3), "TableS9_fidelity_per_run.csv"
)

# ----------------------------------------------------------------------------
# 5  Disclosure risk (Table S10)
# ----------------------------------------------------------------------------
real_rest, real_holdout = train_test_split(real_unique, test_size=200, random_state=1, stratify=real_unique[TARGET])


def distance_to_closest(a, b):
    dist = cdist(scaler.transform(a[FEATURES]), scaler.transform(b[FEATURES]))
    dist.sort(1)
    return dist[:, 0], dist[:, 1]


reference_dcr, _ = distance_to_closest(real_holdout, real_rest)
dcr_p5 = np.percentile(reference_dcr, 5)
QUASI_IDENTIFIERS = ["age", "sex", "resting bp s", "cholesterol"]
qi_scaler = StandardScaler().fit(real_unique[QUASI_IDENTIFIERS])


def attribute_inference(released):
    """Accuracy of inferring the disease status of held-out patients from their closest released record."""
    idx = cdist(
        qi_scaler.transform(real_holdout[QUASI_IDENTIFIERS]), qi_scaler.transform(released[QUASI_IDENTIFIERS])
    ).argmin(1)
    return np.mean(released[TARGET].values[idx] == real_holdout[TARGET].values)


release_reference = np.array([attribute_inference(real_rest.sample(200, random_state=s)) for s in range(100)])
privacy = []
for (model, run), df in synthetic.items():
    d1, d2 = distance_to_closest(df, real_unique)
    privacy.append(
        {
            "Model": model,
            "Run": run,
            "Median DCR": np.median(d1),
            "% DCR below reference 5th pct": 100 * np.mean(d1 < dcr_p5),
            "% identical in 11 features": 100 * np.mean(d1 == 0),
            "Median NNDR": np.nanmedian(d1 / np.where(d2 == 0, np.nan, d2)),
            "Attribute-inference accuracy": attribute_inference(df),
        }
    )
privacy = save(pd.DataFrame(privacy).round(3), "TableS10_privacy.csv")
NUMBERS["privacy_reference"] = {
    "median_dcr_new_patients": round(np.median(reference_dcr), 3),
    "dcr_5th_percentile": round(dcr_p5, 3),
    "attribute_inference_release_mean": round(release_reference.mean(), 3),
    "attribute_inference_release_p95": round(np.percentile(release_reference, 95), 3),
    "majority_class": round(max(real_holdout[TARGET].mean(), 1 - real_holdout[TARGET].mean()), 3),
}

# ----------------------------------------------------------------------------
# 6  Machine learning frameworks
# ----------------------------------------------------------------------------
SEARCH_SPACES = {
    "AdaBoost": (
        lambda: AdaBoostClassifier(random_state=SEED),
        {"n_estimators": [50, 100, 150], "learning_rate": [0.01, 0.1, 1.0]},
    ),
    "GBM": (
        lambda: GradientBoostingClassifier(random_state=SEED),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 4, 5], "subsample": [1.0]},
    ),
    "Stochastic GBM": (
        lambda: GradientBoostingClassifier(random_state=SEED),
        {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.6, 0.8],
        },
    ),
    "XGBoost": (
        lambda: XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=1, verbosity=0),
        {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
        },
    ),
    "LightGBM": (
        lambda: LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1, subsample_freq=1),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
        },
    ),
    "Random Forest": (
        lambda: RandomForestClassifier(random_state=SEED, n_jobs=1),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    ),
    "Random Forest Boosting": (
        lambda: AdaBoostClassifier(estimator=RandomForestClassifier(random_state=SEED, n_jobs=1), random_state=SEED),
        {
            "estimator__n_estimators": [25, 50],
            "estimator__max_depth": [3, 5, None],
            "n_estimators": [10, 25],
            "learning_rate": [0.1, 1.0],
        },
    ),
}
BASE_LEARNERS = list(SEARCH_SPACES)
FRAMEWORKS = sorted(
    BASE_LEARNERS + ["Super Learner (Hist-Meta)", "Super Learner (LR-Meta)", "Super Learner (XGB-Meta)", "Blending"]
)


def standardize(train, test):
    idx = [FEATURES.index(c) for c in CONTINUOUS]
    sc = StandardScaler().fit(train[:, idx])
    train, test = train.copy(), test.copy()
    train[:, idx] = sc.transform(train[:, idx])
    test[:, idx] = sc.transform(test[:, idx])
    return train, test


def tune(x, y, n_iter, folds):
    tuned = {}
    for name, (make, space) in SEARCH_SPACES.items():
        n_candidates = int(np.prod([len(v) for v in space.values()]))
        cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
        if n_candidates <= n_iter:
            search = GridSearchCV(make(), space, cv=cv, scoring="roc_auc", n_jobs=N_JOBS)
        else:
            search = RandomizedSearchCV(
                make(), space, n_iter=n_iter, cv=cv, scoring="roc_auc", n_jobs=N_JOBS, random_state=SEED
            )
        search.fit(x, y)
        tuned[name] = (search.best_estimator_, search.best_params_)
    return tuned


def predict_frameworks(x, y, x_test, n_iter, folds):
    """Tune on (x, y) only and return test-set probabilities of all 11 frameworks."""
    x, x_test = standardize(x, x_test)
    tuned = tune(x, y, n_iter, folds)
    kfold = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out_of_fold = np.zeros((len(x), len(BASE_LEARNERS)))
    fold_test = np.zeros((5, len(x_test), len(BASE_LEARNERS)))
    full_test = np.zeros((len(x_test), len(BASE_LEARNERS)))
    for j, name in enumerate(BASE_LEARNERS):
        estimator = tuned[name][0]
        full_test[:, j] = clone(estimator).fit(x, y).predict_proba(x_test)[:, 1]
        for i, (train_idx, val_idx) in enumerate(kfold.split(x, y)):
            fold_model = clone(estimator).fit(x[train_idx], y[train_idx])
            out_of_fold[val_idx, j] = fold_model.predict_proba(x[val_idx])[:, 1]
            fold_test[i, :, j] = fold_model.predict_proba(x_test)[:, 1]
    predictions = {name: full_test[:, j] for j, name in enumerate(BASE_LEARNERS)}
    predictions["Super Learner (Hist-Meta)"] = (
        HistGradientBoostingClassifier(random_state=SEED, max_iter=100)
        .fit(out_of_fold, y)
        .predict_proba(full_test)[:, 1]
    )
    predictions["Super Learner (LR-Meta)"] = logistic().fit(out_of_fold, y).predict_proba(full_test)[:, 1]
    predictions["Super Learner (XGB-Meta)"] = (
        XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )
        .fit(out_of_fold, y)
        .predict_proba(full_test)[:, 1]
    )
    predictions["Blending"] = (
        LogisticRegression(max_iter=2000).fit(out_of_fold, y).predict_proba(fold_test.mean(0))[:, 1]
    )
    return predictions, {name: tuned[name][1] for name in BASE_LEARNERS}


def calibration_slope(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return LogisticRegression(C=1e6, max_iter=5000).fit(logit, y).coef_[0, 0]


def evaluate(y, p):
    predicted = (p >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y, p),
        "Accuracy": accuracy_score(y, predicted),
        "Precision": precision_score(y, predicted, zero_division=0),
        "Recall": recall_score(y, predicted),
        "F1": f1_score(y, predicted, zero_division=0),
        "Brier": brier_score_loss(y, p),
        "Calibration slope": calibration_slope(y, p),
    }


def rank_auc(y, p):
    ranks = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_ci(y, p, n_boot=N_BOOT):
    rng = np.random.default_rng(0)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            values.append(rank_auc(y[idx], p[idx]))
    return np.percentile(values, [2.5, 97.5])


# ----------------------------------------------------------------------------
# 7  External validation (Tables S12, S13, S17)
# ----------------------------------------------------------------------------
x_real, y_real = real[FEATURES].values.astype(float), real[TARGET].values
is_unique = ~real.duplicated().values
has_cholesterol = (real.cholesterol > 0).values
external, tuned_params, prediction_frames = [], [], []
for (model, run), df in synthetic.items():
    start = time.time()
    key = f"external_{model}-{run}_{N_ITER_EXT}_{CV_EXT}"
    predictions, params = cached(
        key,
        lambda df=df: predict_frameworks(
            df[FEATURES].values.astype(float), df[TARGET].values, x_real, N_ITER_EXT, CV_EXT
        ),
    )
    copied = real_records_with_copy(df)
    for name, prm in params.items():
        tuned_params.append(
            {
                "Model": model,
                "Run": run,
                "Base learner": name,
                "Selected hyperparameters": ", ".join(f"{k}={v}" for k, v in prm.items()),
            }
        )
    for framework in FRAMEWORKS:
        p = predictions[framework]
        y, pu = y_real[is_unique], p[is_unique]
        low, high = bootstrap_ci(y, pu)
        external.append(
            {
                "Model": model,
                "Run": run,
                "Framework": framework,
                **evaluate(y, pu),
                "AUC CI low": low,
                "AUC CI high": high,
                "AUC (all 1,190)": roc_auc_score(y_real, p),
                "AUC (cholesterol > 0)": roc_auc_score(
                    y_real[is_unique & has_cholesterol], p[is_unique & has_cholesterol]
                ),
                "AUC (copied patients excluded)": roc_auc_score(y_real[is_unique & ~copied], p[is_unique & ~copied]),
                "Patients excluded (copies)": int((is_unique & copied).sum()),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "Model": model,
                    "Run": run,
                    "Framework": framework,
                    "record": np.arange(len(real)),
                    "unique": is_unique,
                    "y": y_real,
                    "p": np.round(p, 5),
                }
            )
        )
    print(f"external validation {model}-{run}: {time.time() - start:.0f} s", flush=True)
external = save(pd.DataFrame(external).round(4), "TableS12_external_validation.csv")
save(pd.DataFrame(tuned_params), "TableS17_selected_hyperparameters.csv")
pd.concat(prediction_frames).to_csv(os.path.join(TABLES, "external_predictions.csv"), index=False)
sensitivity = (
    external.groupby(["Model", "Run"])[
        [
            "AUC",
            "AUC (all 1,190)",
            "AUC (cholesterol > 0)",
            "AUC (copied patients excluded)",
            "Patients excluded (copies)",
        ]
    ]
    .mean()
    .reset_index()
)
save(sensitivity.round(3), "TableS13_external_sensitivity.csv")


# ----------------------------------------------------------------------------
# 8  Benchmark and learning curve (Tables S15, S16)
# ----------------------------------------------------------------------------
def gaussian_copula(train, n, seed):
    """Gaussian copula with empirical marginals fitted to the real training records."""
    rng = np.random.default_rng(seed)
    x = train.values.astype(float)
    u = np.column_stack([rankdata(x[:, j], method="average") / (len(x) + 1) for j in range(x.shape[1])])
    corr = np.corrcoef(norm.ppf(u), rowvar=False)
    samples = norm.cdf(rng.multivariate_normal(np.zeros(x.shape[1]), corr, size=n))
    out = np.column_stack([np.quantile(x[:, j], samples[:, j], method="inverted_cdf") for j in range(x.shape[1])])
    return pd.DataFrame(out, columns=train.columns)


def sequential_cart(train, n, seed, min_leaf=5):
    """Sequential CART synthesis: each variable is drawn from the leaf of a tree fitted on the preceding variables."""
    rng = np.random.default_rng(seed)
    cols = list(train.columns)
    out = pd.DataFrame(index=range(n))
    out[cols[0]] = rng.choice(train[cols[0]].values, size=n)
    for j in range(1, len(cols)):
        previous, y = cols[:j], train[cols[j]].values
        tree = DecisionTreeRegressor(min_samples_leaf=min_leaf, random_state=seed).fit(train[previous].values, y)
        train_leaf, new_leaf = tree.apply(train[previous].values), tree.apply(out[previous].values)
        donors = {leaf: y[train_leaf == leaf] for leaf in np.unique(train_leaf)}
        out[cols[j]] = [rng.choice(donors[leaf]) for leaf in new_leaf]
    return out[cols]


CLASSIFIERS = {
    "Random Forest": (
        lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
        {"min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.5]},
    ),
    "Logistic regression": (
        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        {"logisticregression__C": [0.01, 0.1, 1, 10]},
    ),
}


def tuned_test_auc(train, test_sets, classifier):
    make, grid = CLASSIFIERS[classifier]
    search = GridSearchCV(
        make(), grid, cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="roc_auc", n_jobs=N_JOBS
    )
    search.fit(train[FEATURES], train[TARGET])
    return [roc_auc_score(t[TARGET], search.predict_proba(t[FEATURES])[:, 1]) for t in test_sets]


def without_copies(test, df):
    rows = df.values.astype(float)
    return test[[not ((rows == r).sum(1) >= 11).any() for r in test.values.astype(float)]]


LC_SIZES = [25, 50, 100, 200, 400]


def benchmark_split(rep):
    train, test = train_test_split(real_unique, test_size=0.25, random_state=100 + rep, stratify=real_unique[TARGET])
    sources = {
        ("Real (full training partition)", 0): train,
        ("Real (n = 200)", 0): train.sample(200, random_state=rep),
        ("Gaussian copula", 0): gaussian_copula(train, 200, rep),
        ("Sequential CART", 0): sequential_cart(train, 200, rep),
        ("Independent marginals", 0): independent_marginals(200, rep),
    }
    sources.update(synthetic)
    rows = []
    for (source, run), df in sources.items():
        tests = [test, without_copies(test, df)] if source in MODELS else [test, test]
        for classifier in CLASSIFIERS:
            auc, auc_clean = tuned_test_auc(df, tests, classifier)
            rows.append(
                {
                    "rep": rep,
                    "Source": source,
                    "Run": run,
                    "Classifier": classifier,
                    "AUC": auc,
                    "AUC (copied test patients removed)": auc_clean,
                    "n test": len(tests[1]),
                }
            )
    for size in LC_SIZES + [len(train)]:
        subset = (
            train
            if size == len(train)
            else train.groupby(TARGET, group_keys=False).apply(
                lambda g: g.sample(int(round(size * len(g) / len(train))), random_state=rep)
            )
        )
        for classifier in CLASSIFIERS:
            rows.append(
                {
                    "rep": rep,
                    "Source": "Learning curve",
                    "Run": size,
                    "Classifier": classifier,
                    "AUC": tuned_test_auc(subset, [test], classifier)[0],
                }
            )
    for source in ["Gaussian copula", "Sequential CART"]:
        rows_train = train.values.astype(float)
        share = np.mean([((rows_train == r).sum(1) >= 11).any() for r in sources[(source, 0)].values.astype(float)])
        rows.append({"rep": rep, "Source": source, "Run": 0, "Classifier": "copy rate", "AUC": share})
    return rows


bench = pd.DataFrame(
    sum([cached(f"benchmark_{rep}", lambda rep=rep: benchmark_split(rep)) for rep in range(N_REP)], [])
)
save(bench.round(4), "TableS15_benchmark_raw.csv")
scores = bench[(bench.Source != "Learning curve") & (bench.Classifier != "copy rate")]
summary = (
    scores.groupby(["Source", "Classifier"])
    .agg(
        AUC_mean=("AUC", "mean"),
        AUC_sd=("AUC", "std"),
        AUC_copies_removed=("AUC (copied test patients removed)", "mean"),
    )
    .reset_index()
)
run_sd = (
    scores[scores.Source.isin(MODELS)]
    .groupby(["Source", "Classifier", "Run"])
    .AUC.mean()
    .groupby(["Source", "Classifier"])
    .std()
    .rename("SD across runs")
    .reset_index()
)
save(summary.merge(run_sd, on=["Source", "Classifier"], how="left").round(3), "TableS15_benchmark_summary.csv")
NUMBERS["copy_rate_record_access_generators"] = (
    bench[bench.Classifier == "copy rate"].groupby("Source").AUC.mean().round(3).to_dict()
)

curve = bench[bench.Source == "Learning curve"].groupby(["Classifier", "Run"]).AUC.agg(["mean", "std"]).reset_index()


def effective_size(auc, classifier):
    c = curve[curve.Classifier == classifier].sort_values("Run")
    sizes, values = np.log(c.Run.values), np.maximum.accumulate(c["mean"].values)
    if auc <= values[0]:
        return f"< {int(c.Run.values[0])}"
    if auc >= values[-1]:
        return f">= {int(c.Run.values[-1])}"
    return int(round(np.exp(np.interp(auc, values, sizes))))


ess = []
for model in MODELS:
    for classifier in CLASSIFIERS:
        for run in RUNS:
            auc = scores[(scores.Source == model) & (scores.Run == run) & (scores.Classifier == classifier)].AUC.mean()
            ess.append(
                {
                    "Model": model,
                    "Run": run,
                    "Classifier": classifier,
                    "Benchmark AUC": round(auc, 3),
                    "Effective real sample size": effective_size(auc, classifier),
                }
            )
save(pd.DataFrame(ess), "TableS16_effective_sample_size.csv")
save(curve.round(3), "TableS16_learning_curve.csv")

# ----------------------------------------------------------------------------
# 9  Internal validation by nested cross-validation (Table S11)
# ----------------------------------------------------------------------------
internal = []
for (model, run), df in synthetic.items():
    start = time.time()
    x, y = df[FEATURES].values.astype(float), df[TARGET].values
    groups = df[COLUMNS].astype(str).agg("|".join, axis=1).factorize()[0]
    out_of_fold = {framework: np.zeros(len(df)) for framework in FRAMEWORKS}
    outer = StratifiedGroupKFold(N_OUTER, shuffle=True, random_state=SEED)
    for i, (train_idx, test_idx) in enumerate(outer.split(x, y, groups)):
        key = f"internal_{model}-{run}_fold{i}_{N_OUTER}_{N_ITER_IN}_{CV_IN}"
        predictions, _ = cached(
            key, lambda a=train_idx, b=test_idx: predict_frameworks(x[a], y[a], x[b], N_ITER_IN, CV_IN)
        )
        for framework in FRAMEWORKS:
            out_of_fold[framework][test_idx] = predictions[framework]
    for framework in FRAMEWORKS:
        internal.append({"Model": model, "Run": run, "Framework": framework, **evaluate(y, out_of_fold[framework])})
    print(f"nested cross-validation {model}-{run}: {time.time() - start:.0f} s", flush=True)
internal = save(pd.DataFrame(internal).round(4), "TableS11_internal_nested_cv.csv")

# ----------------------------------------------------------------------------
# 10  Summary tables (Tables 4, 5 and S14)
# ----------------------------------------------------------------------------
merged = internal[["Model", "Run", "Framework", "AUC"]].merge(
    external, on=["Model", "Run", "Framework"], suffixes=(" internal", "")
)
table5 = []
for framework in FRAMEWORKS:
    row = {"Framework": framework}
    for model in MODELS:
        row[model] = mean_sd(merged[(merged.Framework == framework) & (merged.Model == model)]["AUC"])
    table5.append(row)
save(pd.DataFrame(table5), "Table5_external_auc_by_framework.csv")

run_means = (
    merged.groupby(["Model", "Run"])[
        ["AUC internal", "AUC", "AUC (copied patients excluded)", "Brier", "Calibration slope"]
    ]
    .mean()
    .reset_index()
)
run_means = run_means.rename(columns={"AUC": "AUC external"})
save(run_means.round(4), "TableS14_run_level_means.csv")
groups_ext = [run_means[run_means.Model == m]["AUC external"] for m in MODELS]
NUMBERS["model_comparison"] = {
    "anova_p": float(stats.f_oneway(*groups_ext).pvalue),
    "kruskal_wallis_p": float(stats.kruskal(*groups_ext).pvalue),
}
pairwise = []
for a, b in combinations(MODELS, 2):
    xa, xb = run_means[run_means.Model == a]["AUC external"], run_means[run_means.Model == b]["AUC external"]
    pairwise.append(
        {
            "Comparison": f"{a} vs {b}",
            "Mean difference": xa.mean() - xb.mean(),
            "Welch t p": stats.ttest_ind(xa, xb, equal_var=False).pvalue,
        }
    )
pairwise = pd.DataFrame(pairwise)
order = np.argsort(pairwise["Welch t p"].values)
holm = np.minimum(1, np.maximum.accumulate(pairwise["Welch t p"].values[order] * np.arange(len(pairwise), 0, -1)))
pairwise["Holm-adjusted p"] = np.nan
pairwise.loc[order, "Holm-adjusted p"] = holm
save(pairwise.round(4), "TableS14_model_comparison.csv")

table4 = []
for model in MODELS:
    a, f, p, r = (
        audit[audit.Model == model],
        fidelity[fidelity.Model == model],
        privacy[privacy.Model == model],
        run_means[run_means.Model == model],
    )
    b = scores[scores.Source == model]
    ks_counts = [NUMBERS["ks_nonsignificant_features"][model][run] for run in RUNS]
    table4.append(
        {
            "Model": model,
            "Near-exact copies of real records (%)": mean_sd(a["% records near-exact"], 1),
            "Within-dataset duplicates": mean_sd(a["Within-dataset duplicates"], 1),
            "Within-dataset diversity (real = 1)": mean_sd(a["Relative within-dataset diversity"], 2),
            "Features without K–S difference (of 11)": mean_sd(ks_counts, 1),
            "Correlation MAD": mean_sd(f["Correlation MAD"]),
            "Correlation sign agreement": mean_sd(f["Sign agreement"], 2),
            "SHAP interaction similarity": mean_sd(f["SHAP interaction similarity"], 2),
            "Risk-map agreement (LR)": mean_sd(f["Risk-map agreement (LR)"], 2),
            "Attribute-inference accuracy": mean_sd(p["Attribute-inference accuracy"]),
            "Internal AUC (nested CV)": mean_sd(r["AUC internal"]),
            "External AUC (11 frameworks)": mean_sd(r["AUC external"]),
            "External AUC, copied patients excluded": mean_sd(r["AUC (copied patients excluded)"]),
            "External Brier score": mean_sd(r["Brier"]),
            "Benchmark AUC, Random Forest": mean_sd(b[b.Classifier == "Random Forest"].groupby("Run").AUC.mean()),
            "Benchmark AUC, logistic regression": mean_sd(
                b[b.Classifier == "Logistic regression"].groupby("Run").AUC.mean()
            ),
        }
    )
save(
    pd.DataFrame(table4)
    .set_index("Model")
    .T.reset_index()
    .rename(columns={"index": "Metric (mean ± SD across 3 runs)"}),
    "Table4_model_summary.csv",
)


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    return obj


with open(os.path.join(ARGS.out, "key_numbers.json"), "w") as fh:
    json.dump(to_builtin(NUMBERS), fh, indent=2)
print("finished; tables written to", TABLES)
