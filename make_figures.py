"""Figures of "Large Language Models as Clinical Data Synthesizers".

Creates the main figures (Fig1-Fig9) and the supplementary figures (FigS1-FigS7)
at print size (174 mm width) as PNG and PDF files in ``figures/``.
Run ``run_analysis.py`` first; this script reads its tables from ``results/tables``.

Usage
-----
    python make_figures.py            # 300 dpi PNG and vector PDF
    python make_figures.py --dpi 600
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch
from itertools import combinations
from scipy.stats import ks_2samp, spearmanr
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import shap

parser = argparse.ArgumentParser(description="Create the figures of the article.")
parser.add_argument("--data", default="data")
parser.add_argument("--results", default="results")
parser.add_argument("--out", default="figures")
parser.add_argument("--dpi", type=int, default=300)
ARGS = parser.parse_args()
DPI = ARGS.dpi
DATA, RES, OUT = ARGS.data, os.path.join(ARGS.results, "tables"), ARGS.out
os.makedirs(OUT, exist_ok=True)
SEED = 42
MODELS = ["ChatGPT", "Claude", "DeepSeek", "Gemini"]
RUNS = [1, 2, 3]
W2 = 174 / 25.4  # double-column width in inches
COL = {
    "Real": "#4D4D4D",
    "ChatGPT": "#009E73",
    "Claude": "#E69F00",
    "DeepSeek": "#CC79A7",
    "Gemini": "#0072B2",
}  # Okabe-Ito
RUN_MK = {1: "o", 2: "s", 3: "^"}
RUN_LS = {1: "-", 2: "--", 3: ":"}
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
NICE = {
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
SHORT = {
    "age": "Age",
    "sex": "Sex",
    "chest pain type": "Chest pain",
    "resting bp s": "Rest. BP",
    "cholesterol": "Cholest.",
    "fasting blood sugar": "Fast. BS",
    "resting ecg": "Rest. ECG",
    "max heart rate": "Max HR",
    "exercise angina": "Ex. angina",
    "oldpeak": "Oldpeak",
    "ST slope": "ST slope",
    "target": "Disease",
}
lab = lambda c: NICE.get(c, c)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def letter(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


# ------------------------------------------------------------------ data
r = pd.read_csv(os.path.join(DATA, "real_data.csv"))
T = "target"
FEAT = [c for c in r.columns if c != T]
COLS = list(r.columns)
CONT = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
ru = r.drop_duplicates().reset_index(drop=True)
SYN = {(m, k): pd.read_csv(os.path.join(DATA, "synthetic", f"{m.lower()}_run{k}.csv")) for m in MODELS for k in RUNS}
for d in SYN.values():
    assert list(d.columns) == COLS
EXT = pd.read_csv(os.path.join(RES, "TableS12_external_validation.csv"))
INT = pd.read_csv(os.path.join(RES, "TableS11_internal_nested_cv.csv"))
BENCH = pd.read_csv(os.path.join(RES, "TableS15_benchmark_raw.csv"))
PRED = pd.read_csv(os.path.join(RES, "external_predictions.csv"))
sc_all = StandardScaler().fit(ru[FEAT])
R_vals = r.values.astype(float)
NEAR = {k: np.array([(R_vals == x).sum(1).max() for x in d.values.astype(float)]) for k, d in SYN.items()}
ks = pd.DataFrame(
    [{"Model": k[0], "Run": k[1], "Feature": c, "p": ks_2samp(r[c], d[c]).pvalue} for k, d in SYN.items() for c in FEAT]
)


# ================================================================== Fig 1: study design
def box(ax, x, y, w, h, text, fc, ec="#333", fs=6.8, bold=False, tc="k"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=0.7))
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        color=tc,
        linespacing=1.25,
    )


def arrow(ax, p, q, **kw):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=7,
            lw=0.8,
            color=kw.get("color", "#333"),
            ls=kw.get("ls", "-"),
            shrinkA=0,
            shrinkB=0,
        )
    )


fig, ax = plt.subplots(figsize=(W2, 3.9))
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(0, 1.1)
ax.axis("off")
box(ax, 0, 0.70, 0.2, 0.2, "Reference dataset\n1,190 patients, 5 cohorts\n11 features + outcome", "#eeeeee", fs=6)
box(ax, 0, 0.42, 0.2, 0.2, "Univariate summary only\n(Table 1: mean, SD,\nmin, max)", "#fff4d6", fs=6)
box(ax, 0, 0.14, 0.2, 0.2, "Identical minimal prompt\nin a new conversation", "#fff4d6", fs=6)
arrow(ax, (0.1, 0.70), (0.1, 0.62))
arrow(ax, (0.1, 0.42), (0.1, 0.34))
ys = [0.77, 0.57, 0.37, 0.17]
labels = {
    "ChatGPT": "ChatGPT\nGPT-5.6 Luna, Think",
    "Claude": "Claude\nSonnet 5.5, medium effort",
    "DeepSeek": "DeepSeek\nV4.1-Flash, DeepThink",
    "Gemini": "Gemini\n3.1 Pro, extended thinking",
}
for m, yv in zip(MODELS, ys):
    box(ax, 0.255, yv, 0.19, 0.13, labels[m], COL[m] + "33", ec=COL[m], fs=5.9)
    arrow(ax, (0.2, 0.24), (0.255, yv + 0.065), color="#888")
    for i, k in enumerate(RUNS):
        ax.add_patch(
            FancyBboxPatch(
                (0.465 + i * 0.027, yv + 0.035), 0.02, 0.06, boxstyle="round,pad=0.002", fc=COL[m], ec="k", lw=0.4
            )
        )
        if yv == ys[-1]:
            ax.text(0.475 + i * 0.027, yv - 0.02, str(k), ha="center", va="top", fontsize=5.5)
    arrow(ax, (0.445, yv + 0.065), (0.465, yv + 0.065), color="#888")
ax.text(0.502, 0.085, "generation runs 1–3\n(independent)", ha="center", va="top", fontsize=5.6)
box(ax, 0.575, 0.38, 0.11, 0.26, "12 synthetic\ncohorts\n≈ 200 records\neach", "#f2f2f2", fs=6)
for yv in ys:
    arrow(ax, (0.55, yv + 0.065), (0.575, 0.51), color="#aaa")
pillars = [
    ("Fidelity", "univariate distributions\ncorrelation structure\nrisk maps and interactions", "#dfeaf5", 0.70, 0.24),
    (
        "Memorization and privacy",
        "verbatim copies of real records\ndistance to closest record\nattribute-inference attack",
        "#f6e0e8",
        0.40,
        0.24,
    ),
    (
        "Predictive utility",
        "11 ML frameworks\ninternal: nested cross-validation\nexternal: 918 unique real patients\nbenchmark against real data and\nrecord-access generators",
        "#e2f2e7",
        0.03,
        0.31,
    ),
]
for t, sub, fc, y0, h in pillars:
    ax.add_patch(
        FancyBboxPatch((0.725, y0), 0.275, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec="#333", lw=0.7)
    )
    ax.text(0.8625, y0 + h - 0.04, t, ha="center", va="center", fontsize=6.6, fontweight="bold")
    ax.text(0.8625, y0 + (h - 0.06) / 2, sub, ha="center", va="center", fontsize=5.8, linespacing=1.25)
    arrow(ax, (0.685, 0.51), (0.725, y0 + h / 2), color="#888")
ax.plot([0.1, 0.1, 0.8625], [0.905, 1.0, 1.0], ls="--", lw=0.8, color="#b03a2e")
arrow(ax, (0.8625, 1.0), (0.8625, 0.945), color="#b03a2e")
ax.text(
    0.48,
    1.02,
    "real records are used only as test data, never for training, tuning or model selection",
    ha="center",
    va="bottom",
    fontsize=6,
    color="#b03a2e",
    style="italic",
)
save(fig, "Fig1")

# ================================================================== Fig 2: continuous distributions
UNITS = {
    "age": "Age (years)",
    "resting bp s": "Resting BP (mm Hg)",
    "cholesterol": "Cholesterol (mg/dl)",
    "max heart rate": "Max heart rate (bpm)",
    "oldpeak": "Oldpeak (mm)",
}
fig, axes = plt.subplots(5, 4, figsize=(W2, 6.6))
for i, c in enumerate(CONT):
    lo, hi = np.percentile(np.concatenate([r[c].values] + [d[c].values for d in SYN.values()]), [0.5, 99.5])
    for j, m in enumerate(MODELS):
        a = axes[i, j]
        sns.kdeplot(r[c], ax=a, color="#9e9e9e", fill=True, alpha=0.45, lw=0, clip=(lo, hi), warn_singular=False)
        for k in RUNS:
            sns.kdeplot(SYN[(m, k)][c], ax=a, color=COL[m], lw=1.1, ls=RUN_LS[k], clip=(lo, hi), warn_singular=False)
        a.set_xlim(lo, hi)
        a.set_yticks([])
        a.set_ylabel("")
        a.set_xlabel(UNITS[c])
        a.spines["left"].set_visible(False)
        n = int((ks[(ks.Model == m) & (ks.Feature == c)].p > 0.05).sum())
        a.text(
            0.98,
            0.96,
            f"K–S p > 0.05\nin {n}/3 runs",
            transform=a.transAxes,
            ha="right",
            va="top",
            fontsize=5.8,
            color="#333",
        )
        if i == 0:
            a.set_title(m, color=COL[m], fontweight="bold", fontsize=8)
fig.legend(
    handles=[Patch(color="#9e9e9e", alpha=0.45, label="Real data (n = 1,190)")]
    + [Line2D([0], [0], color="k", ls=RUN_LS[k], lw=1.1, label=f"Generation run {k}") for k in RUNS],
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.02),
    frameon=False,
)
fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=0.9, w_pad=0.6)
save(fig, "Fig2")


# ================================================================== Fig 3: correlation deviation
def corr_dev(d):
    return d[COLS].corr().fillna(0).values - r[COLS].corr().values


fig = plt.figure(figsize=(W2, 6.9))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045], wspace=0.08, hspace=0.35)
mask = np.triu(np.ones((12, 12)), 0).astype(bool)
L = [SHORT[c] for c in COLS]
cax = fig.add_subplot(gs[:, 2])
for i, m in enumerate(MODELS):
    a = fig.add_subplot(gs[i // 2, i % 2])
    E = np.mean([corr_dev(SYN[(m, k)]) for k in RUNS], 0)
    annot = np.where(
        np.abs(E) >= 0.2, np.vectorize(lambda v: f"{v:+.2f}".replace("+0.", "+.").replace("-0.", "−."))(E), ""
    )
    sns.heatmap(
        E,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=a,
        cbar=i == 0,
        cbar_ax=cax if i == 0 else None,
        cbar_kws={"label": "Correlation in synthetic data − correlation in real data (mean of 3 runs)"},
        xticklabels=L,
        yticklabels=L if i % 2 == 0 else False,
        annot_kws={"size": 5.2},
        linewidths=0.4,
        linecolor="white",
        square=True,
    )
    a.set_title(m, color=COL[m], fontweight="bold", fontsize=8)
    plt.setp(a.get_xticklabels(), rotation=55, ha="right")
    letter(a, "abcd"[i], -0.02 if i % 2 else -0.2)
save(fig, "Fig3")

# ================================================================== Fig 4 / S3: risk maps on real category combinations; Fig S2: continuous LR surfaces
mi = mutual_info_classif(r[FEAT], r[T], random_state=SEED)
TOP4 = [FEAT[i] for i in np.argsort(mi)[-4:][::-1]]
PAIRS = list(combinations(TOP4, 2))
LEVELS = {"ST slope": [1, 2, 3], "chest pain type": [1, 2, 3, 4], "exercise angina": [0, 1]}
LVL = {
    "ST slope": ["Up", "Flat", "Down"],
    "chest pain type": ["TA", "AA", "NA", "AS"],
    "exercise angina": ["No", "Yes"],
}
HR_E = np.percentile(r["max heart rate"], [0, 20, 40, 60, 80, 100])


def axis_values(f):
    if f in LEVELS:
        return [[v] for v in LEVELS[f]], LVL[f]
    return [np.linspace(HR_E[i], HR_E[i + 1], 8) for i in range(5)], [f"≤{HR_E[i + 1]:.0f}" for i in range(4)] + [
        f">{HR_E[4]:.0f}"
    ]


def lr_pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def risk_table(d, f1, f2, kind):
    mdl = (
        RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=SEED) if kind == "RF" else lr_pipe()
    ).fit(d[[f1, f2]], d[T])
    v1, l1 = axis_values(f1)
    v2, l2 = axis_values(f2)
    Z = np.zeros((len(v2), len(v1)))
    for i, a2 in enumerate(v2):
        for j, a1 in enumerate(v1):
            Z[i, j] = mdl.predict_proba(pd.DataFrame([(x, y) for x in a1 for y in a2], columns=[f1, f2]))[:, 1].mean()
    return Z, l1, l2


def safe_r(a, b):
    return np.nan if (np.std(a) == 0 or np.std(b) == 0) else np.corrcoef(a, b)[0, 1]


AX = {
    "ST slope": "ST slope",
    "chest pain type": "Chest pain type",
    "max heart rate": "Max heart rate (bpm)",
    "exercise angina": "Exercise angina",
}


def risk_figure(kind, name):
    RTr = {p: risk_table(r, *p, kind) for p in PAIRS}
    RTs = {(m, k): {p: risk_table(SYN[(m, k)], *p, kind) for p in PAIRS} for m in MODELS for k in RUNS}
    fig = plt.figure(figsize=(W2, 8.4))
    gs = fig.add_gridspec(len(PAIRS), 6, width_ratios=[1, 1, 1, 1, 1, 0.06], wspace=0.1, hspace=0.75)
    cax = fig.add_subplot(gs[:, 5])
    for ri, p in enumerate(PAIRS):
        Zr, l1, l2 = RTr[p]
        for ci, m in enumerate(["Real"] + MODELS):
            a = fig.add_subplot(gs[ri, ci])
            Z = Zr if m == "Real" else np.mean([RTs[(m, k)][p][0] for k in RUNS], 0)
            sns.heatmap(
                Z,
                ax=a,
                cmap="RdBu_r",
                vmin=0,
                vmax=1,
                annot=True,
                fmt=".2f",
                annot_kws={"size": 5.2},
                cbar=(ri == 0 and ci == 0),
                cbar_ax=cax if (ri == 0 and ci == 0) else None,
                cbar_kws={"label": "Predicted probability of heart disease"},
                linewidths=0.6,
                linecolor="white",
                xticklabels=l1,
                yticklabels=l2 if ci == 0 else False,
            )
            a.invert_yaxis()
            a.set_xlabel(AX[p[0]], fontsize=6.3, labelpad=1)
            a.set_ylabel(AX[p[1]] if ci == 0 else "", fontsize=6.3)
            a.tick_params(axis="x", labelsize=5.4, rotation=35 if p[0] == "max heart rate" else 0, pad=1)
            a.tick_params(axis="y", labelsize=5.4, rotation=0, pad=1)
            head = ("Real data" if m == "Real" else m) if ri == 0 else ""
            if m != "Real":
                rv = np.nanmean([safe_r(Zr.ravel(), RTs[(m, k)][p][0].ravel()) for k in RUNS])
                head = (head + "\n" if head else "") + f"r = {rv:.2f}"
            a.set_title(head, fontsize=7 if ri == 0 else 6.3, color=COL.get(m, "k"), fontweight="bold", pad=2)
    save(fig, name)


risk_figure("RF", "Fig4")
risk_figure("LR", "FigS3")
GR = {}
for p in PAIRS:
    xx, yy = np.meshgrid(
        np.linspace(r[p[0]].min() if p[0] != "ST slope" else 1, r[p[0]].max(), 60),
        np.linspace(r[p[1]].min() if p[1] != "ST slope" else 1, r[p[1]].max(), 60),
    )
    GR[p] = (xx, yy, pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=list(p)))
fig = plt.figure(figsize=(W2, 8.4))
gs = fig.add_gridspec(len(PAIRS), 6, width_ratios=[1, 1, 1, 1, 1, 0.06], wspace=0.12, hspace=0.6)
cax = fig.add_subplot(gs[:, 5])
TICK = {
    "ST slope": ([1, 2, 3], ["Up", "Flat", "Down"]),
    "chest pain type": ([1, 2, 3, 4], ["TA", "AA", "NA", "AS"]),
    "exercise angina": ([0, 1], ["No", "Yes"]),
}
for ri, p in enumerate(PAIRS):
    xx, yy, g = GR[p]
    Sr = lr_pipe().fit(r[list(p)], r[T]).predict_proba(g)[:, 1]
    for ci, m in enumerate(["Real"] + MODELS):
        a = fig.add_subplot(gs[ri, ci])
        S = (
            Sr
            if m == "Real"
            else np.mean([lr_pipe().fit(SYN[(m, k)][list(p)], SYN[(m, k)][T]).predict_proba(g)[:, 1] for k in RUNS], 0)
        )
        cs = a.contourf(xx, yy, S.reshape(xx.shape), levels=np.linspace(0, 1, 21), cmap="RdBu_r")
        for f, setter, getter in [(p[0], a.set_xticks, a.set_xticklabels), (p[1], a.set_yticks, a.set_yticklabels)]:
            if f in TICK:
                setter(TICK[f][0])
                getter(TICK[f][1])
        a.tick_params(labelsize=5.4, pad=1)
        a.set_xlabel(AX[p[0]], fontsize=6.3, labelpad=1)
        a.set_ylabel(AX[p[1]] if ci == 0 else "", fontsize=6.3)
        if ci > 0:
            a.set_yticklabels([])
        head = ("Real data" if m == "Real" else m) if ri == 0 else ""
        if m != "Real":
            head = (head + "\n" if head else "") + f"r = {np.corrcoef(Sr, S)[0, 1]:.2f}"
        a.set_title(head, fontsize=7 if ri == 0 else 6.3, color=COL.get(m, "k"), fontweight="bold", pad=2)
fig.colorbar(cs, cax=cax, label="Predicted probability of heart disease", ticks=np.linspace(0, 1, 6))
save(fig, "FigS2")


# ================================================================== Fig 5: SHAP interaction matrices
def shap_matrix(d):
    mdl = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED).fit(d[FEAT], d[T])
    iv = shap.TreeExplainer(mdl).shap_interaction_values(d[FEAT].sample(min(300, len(d)), random_state=0))
    iv = iv[..., 1] if np.ndim(iv) == 4 else (iv[1] if isinstance(iv, list) else iv)
    M = np.abs(iv).mean(0)
    np.fill_diagonal(M, 0)
    return M


SHr = shap_matrix(r)
SHs = {k: shap_matrix(d) for k, d in SYN.items()}
IU = np.triu_indices(len(FEAT), 1)
fig = plt.figure(figsize=(W2, 5.3))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12, hspace=0.42)
cax = fig.add_subplot(gs[:, 3])
FL = [SHORT[c] for c in FEAT]
panels = [("Real data", SHr / SHr.max(), "Real")] + [
    (m, np.mean([SHs[(m, k)] / SHs[(m, k)].max() for k in RUNS], 0), m) for m in MODELS
]
for i, (t, M, m) in enumerate(panels):
    a = fig.add_subplot(gs[i // 3, i % 3])
    msk = np.triu(np.ones_like(M), 0).astype(bool)
    sns.heatmap(
        M,
        mask=msk,
        ax=a,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        square=True,
        cbar=i == 0,
        cbar_ax=cax if i == 0 else None,
        cbar_kws={"label": "Mean |SHAP interaction value| (scaled to the strongest pair)"},
        xticklabels=FL,
        yticklabels=FL if i % 3 == 0 else False,
        linewidths=0.3,
        linecolor="white",
    )
    plt.setp(a.get_xticklabels(), rotation=60, ha="right", fontsize=5.6)
    plt.setp(a.get_yticklabels(), fontsize=5.6)
    sub = (
        ""
        if m == "Real"
        else "\nsimilarity to real: "
        + f"{np.mean([spearmanr(SHr[IU], SHs[(m, k)][IU]).correlation for k in RUNS]):.2f}"
    )
    a.set_title(t + sub, color=COL[m], fontweight="bold", fontsize=7)
    letter(a, "abcde"[i], -0.25 if i % 3 == 0 else -0.05)
a = fig.add_subplot(gs[1, 2])
for j, m in enumerate(MODELS):
    for k in RUNS:
        a.scatter(
            j + (k - 2) * 0.18,
            spearmanr(SHr[IU], SHs[(m, k)][IU]).correlation,
            color=COL[m],
            marker=RUN_MK[k],
            s=26,
            edgecolor="k",
            lw=0.4,
            zorder=3,
        )
a.set_xticks(range(4))
a.set_xticklabels(MODELS, rotation=30, ha="right")
a.set_ylabel("Similarity to real\n(Spearman ρ)")
a.axhline(0, color="grey", lw=0.5)
a.set_ylim(-0.1, 1)
a.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker=RUN_MK[k],
            color="w",
            markerfacecolor="grey",
            markeredgecolor="k",
            markersize=4.5,
            label=f"Run {k}",
        )
        for k in RUNS
    ],
    loc="upper left",
    frameon=False,
    fontsize=6,
)
letter(a, "f", -0.3)
save(fig, "Fig5")

# ================================================================== Fig 6: memorization and privacy
r_rest, r_ho = train_test_split(ru, test_size=200, random_state=1, stratify=ru[T])


def dcr(A, B):
    d_ = cdist(sc_all.transform(A[FEAT]), sc_all.transform(B[FEAT]))
    return d_.min(1)


ref = dcr(r_ho, r_rest)
QI = ["age", "sex", "resting bp s", "cholesterol"]
scq = StandardScaler().fit(ru[QI])


def aia(rel):
    idx = cdist(scq.transform(r_ho[QI]), scq.transform(rel[QI])).argmin(1)
    return np.mean(rel[T].values[idx] == r_ho[T].values)


AIA = np.array([aia(r_rest.sample(200, random_state=s)) for s in range(100)])
majority = max(r_ho[T].mean(), 1 - r_ho[T].mean())
fig = plt.figure(figsize=(W2, 3.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 0.8], wspace=0.5)
a = fig.add_subplot(gs[0])
cats = [
    ("12 of 12 fields (verbatim)", lambda v: v == 12, "#7f0000"),
    ("11 of 12", lambda v: v == 11, "#d7301f"),
    ("10 of 12", lambda v: v == 10, "#fc8d59"),
    ("≤ 9 of 12", lambda v: v <= 9, "#e6e6e6"),
]
keys = [(m, k) for m in MODELS for k in RUNS]
ypos = np.arange(len(keys))[::-1]
left = np.zeros(len(keys))
for cname, fn, c_ in cats:
    v = np.array([100 * fn(NEAR[k]).mean() for k in keys])
    a.barh(ypos, v, left=left, color=c_, height=0.75, edgecolor="white", lw=0.4, label=cname)
    left += v
a.set_yticks(ypos)
a.set_yticklabels([f"{m} {k}" for m, k in keys], fontsize=6)
[t.set_color(COL[k[0]]) for t, k in zip(a.get_yticklabels(), keys)]
g1 = 100 * (NEAR[("Gemini", 1)] == 12).mean()
a.text(
    g1 / 2,
    ypos[keys.index(("Gemini", 1))],
    f"{g1:.0f}%",
    color="white",
    fontsize=6,
    fontweight="bold",
    ha="center",
    va="center",
)
a.set_xlim(0, 100)
a.set_xlabel("Synthetic records (%)")
a.legend(
    title="Fields identical to the closest\nreal record",
    loc="upper center",
    bbox_to_anchor=(0.45, -0.17),
    ncol=2,
    frameon=False,
    fontsize=5.8,
    title_fontsize=6,
)
letter(a, "a", -0.3)
a = fig.add_subplot(gs[1])
xs = np.linspace(0, 4, 400)
a.plot(xs, 100 * np.searchsorted(np.sort(ref), xs, side="right") / len(ref), color="k", lw=2.2, alpha=0.55)
for m in MODELS:
    for k in RUNS:
        dd = np.sort(dcr(SYN[(m, k)], ru))
        a.plot(xs, 100 * np.searchsorted(dd, xs, side="right") / len(dd), color=COL[m], ls=RUN_LS[k], lw=1)
z = 100 * np.mean(dcr(SYN[("Gemini", 1)], ru) == 0)
a.annotate(
    f"Gemini run 1: {z:.0f}% of\nrecords identical to a\nreal patient in all\n11 features",
    xy=(0.03, z),
    xytext=(2.35, 6),
    fontsize=5.5,
    color=COL["Gemini"],
    arrowprops=dict(arrowstyle="->", color=COL["Gemini"], lw=0.7),
    bbox=dict(boxstyle="round,pad=.25", fc="white", ec=COL["Gemini"], lw=0.6),
)
a.set_xlim(0, 4)
a.set_ylim(0, 101)
a.set_xlabel("Distance to closest real record\n(standardized features)")
a.set_ylabel("Synthetic records within distance (%)")
a.legend(
    handles=[Line2D([0], [0], color="k", lw=2.2, alpha=0.55, label="New real patient (reference)")]
    + [Line2D([0], [0], color=COL[m], lw=1.2, label=m) for m in MODELS]
    + [Line2D([0], [0], color="grey", ls=RUN_LS[k], lw=1, label=f"Run {k}") for k in RUNS],
    loc="upper center",
    bbox_to_anchor=(0.55, -0.3),
    fontsize=5.5,
    frameon=False,
    ncol=2,
    handlelength=1.8,
)
letter(a, "b", -0.25)
a = fig.add_subplot(gs[2])
a.axhspan(np.percentile(AIA, 5), np.percentile(AIA, 95), color="#bdbdbd", alpha=0.5, lw=0)
a.axhline(AIA.mean(), color="#757575", lw=0.8)
a.axhline(majority, color="k", ls=":", lw=0.8)
for j, m in enumerate(MODELS):
    for k in RUNS:
        a.scatter(
            j + (k - 2) * 0.2, aia(SYN[(m, k)]), color=COL[m], marker=RUN_MK[k], s=24, edgecolor="k", lw=0.4, zorder=3
        )
a.set_xticks(range(4))
a.set_xticklabels(MODELS, rotation=30, ha="right")
[t.set_color(COL[m]) for t, m in zip(a.get_xticklabels(), MODELS)]
a.set_ylabel("Accuracy of inferring disease status")
a.set_ylim(0.45, 0.78)
a.legend(
    handles=[
        Patch(color="#bdbdbd", alpha=0.5, label="Release of 200 real\npatients (5th–95th pct)"),
        Line2D([0], [0], color="k", ls=":", lw=0.8, label="Majority class"),
    ]
    + [
        Line2D(
            [0],
            [0],
            marker=RUN_MK[k],
            color="w",
            markerfacecolor="grey",
            markeredgecolor="k",
            markersize=4,
            label=f"Run {k}",
        )
        for k in RUNS
    ],
    loc="upper left",
    fontsize=5.5,
    frameon=False,
)
letter(a, "c", -0.35)
save(fig, "Fig6")

# ================================================================== Fig 7: internal vs external
m4 = INT[["Model", "Run", "Framework", "AUC"]].merge(
    EXT[["Model", "Run", "Framework", "AUC"]], on=["Model", "Run", "Framework"], suffixes=(" internal", " external")
)
bm = BENCH[~BENCH.Source.isin(["Learning curve"]) & (BENCH.Classifier != "copy rate")]
real_ref = bm[bm.Source == "Real (full training partition)"].groupby("Classifier").AUC.mean()
fig, ax = plt.subplots(1, 2, figsize=(W2, 3.2), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.3})
a = ax[0]
a.fill_between([0, 1.05], [0, 1.05], 0, color="#f4cccc", alpha=0.4, lw=0)
a.text(
    0.985,
    0.36,
    "internal validation\noverestimates\nreal-world AUC",
    ha="right",
    fontsize=6,
    color="#a33",
    style="italic",
)
for m in MODELS:
    for k in RUNS:
        s_ = m4[(m4.Model == m) & (m4.Run == k)]
        a.scatter(
            s_["AUC internal"],
            s_["AUC external"],
            color=COL[m],
            marker=RUN_MK[k],
            s=13,
            edgecolor="k",
            lw=0.3,
            alpha=0.9,
        )
a.plot([0, 1.05], [0, 1.05], "k--", lw=0.7)
a.axhline(0.5, color="grey", ls=":", lw=0.6)
a.axvline(0.5, color="grey", ls=":", lw=0.6)
a.set_xlim(0.35, 1.01)
a.set_ylim(0.15, 1.0)
a.set_xlabel("Internal AUC (nested CV on synthetic data)")
a.set_ylabel("External AUC (918 unique real patients)")
a.legend(
    handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL[m], markeredgecolor="k", markersize=5, label=m)
        for m in MODELS
    ]
    + [
        Line2D(
            [0],
            [0],
            marker=RUN_MK[k],
            color="w",
            markerfacecolor="grey",
            markeredgecolor="k",
            markersize=4.5,
            label=f"Run {k}",
        )
        for k in RUNS
    ],
    loc="upper left",
    fontsize=6,
    frameon=False,
    ncol=2,
)
letter(a, "a", -0.17)
a = ax[1]
for j, m in enumerate(MODELS):
    for k in RUNS:
        v = m4[(m4.Model == m) & (m4.Run == k)]["AUC external"]
        x = j + (k - 2) * 0.22
        a.vlines(x, v.min(), v.max(), color=COL[m], lw=2.2, alpha=0.55)
        a.scatter(x, v.mean(), marker=RUN_MK[k], s=34, color=COL[m], edgecolor="k", lw=0.5, zorder=3)
a.axhspan(real_ref.min(), real_ref.max(), color="#4D4D4D", alpha=0.18, lw=0)
a.text(3.42, real_ref.mean(), "trained on\nreal data", va="center", fontsize=6)
a.axhline(0.5, color="grey", ls=":", lw=0.7)
a.text(3.42, 0.5, "chance", va="center", fontsize=6, color="grey")
a.set_xticks(range(4))
a.set_xticklabels(MODELS)
[t.set_color(COL[m]) for t, m in zip(a.get_xticklabels(), MODELS)]
a.set_xlim(-0.5, 3.95)
a.set_ylim(0.15, 1.0)
a.set_ylabel("External AUC (918 unique real patients)")
a.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker=RUN_MK[k],
            color="w",
            markerfacecolor="grey",
            markeredgecolor="k",
            markersize=4.5,
            label=f"Run {k}: mean of 11 frameworks",
        )
        for k in RUNS
    ]
    + [Line2D([0], [0], color="grey", lw=2.2, alpha=0.55, label="range over 11 frameworks")],
    loc="lower left",
    fontsize=6,
    frameon=False,
)
letter(a, "b", -0.15)
save(fig, "Fig7")

# ================================================================== Fig 8: benchmark and learning curve
GROUPS = [
    ("Real data", ["Real (full training partition)", "Real (n = 200)"], "#4D4D4D"),
    ("Record-access generators", ["Sequential CART", "Gaussian copula"], "#6a3d9a"),
    ("LLMs, summary statistics only", ["Claude", "ChatGPT", "Gemini", "DeepSeek"], None),
    ("Baseline, summary statistics only", ["Independent marginals"], "#9e9e9e"),
]
LBL = {
    "Real (full training partition)": "Real data, full training set (≈ 688)",
    "Real (n = 200)": "Real data, 200 records",
    "Sequential CART": "Sequential CART, 200",
    "Gaussian copula": "Gaussian copula, 200",
    "Independent marginals": "Independent marginals, 200",
}
LC = BENCH[BENCH.Source == "Learning curve"].groupby(["Classifier", "Run"]).AUC.agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(W2, 3.6), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.42})
a = ax[0]
y = 0
yt, yl, ycol = [], [], []
for g, srcs, gc in GROUPS:
    a.text(0.16, y + 0.25, g, fontsize=6.3, fontweight="bold", color="#444")
    y -= 0.7
    for s_ in srcs:
        c_ = COL.get(s_, gc)
        for off, clf, mk in [(0.16, "Random Forest", "o"), (-0.16, "Logistic regression", "s")]:
            v = bm[(bm.Source == s_) & (bm.Classifier == clf)]
            lo, hi = np.percentile(v.AUC, [2.5, 97.5])
            a.plot([lo, hi], [y + off] * 2, color=c_, lw=1.3, alpha=0.8)
            a.scatter(v.AUC.mean(), y + off, marker=mk, s=18, color=c_, edgecolor="k", lw=0.4, zorder=3)
        yt.append(y)
        yl.append(LBL.get(s_, f"{s_}, 3 runs × 200"))
        ycol.append(c_)
        y -= 1
    y -= 0.25
    a.axhline(y + 0.62, color="#e0e0e0", lw=0.6)
a.set_yticks(yt)
a.set_yticklabels(yl, fontsize=6.2)
[t.set_color(c) for t, c in zip(a.get_yticklabels(), ycol)]
a.axvline(0.5, color="k", ls=":", lw=0.7)
a.set_xlim(0.15, 1.0)
a.set_ylim(y + 0.3, 0.9)
a.spines["left"].set_visible(False)
a.tick_params(axis="y", length=0)
a.set_xlabel("AUC on held-out real patients\n(marker = mean, line = 95% range over 10 splits)")
a.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="grey",
            markeredgecolor="k",
            markersize=4.5,
            label="Random Forest",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="grey",
            markeredgecolor="k",
            markersize=4.5,
            label="Logistic regression",
        ),
    ],
    loc="upper left",
    bbox_to_anchor=(0.0, 0.95),
    fontsize=6,
    frameon=False,
)
letter(a, "a", -0.62, 1.0)
a = ax[1]
xs_ = [25, 50, 100, 200, 400, int(LC.Run.max())]
for clf, ls, mk in [("Random Forest", "-", "o"), ("Logistic regression", "--", "s")]:
    c_ = LC[LC.Classifier == clf].sort_values("Run")
    a.errorbar(
        c_.Run,
        c_["mean"],
        yerr=c_["std"],
        fmt=mk + ls,
        color="k",
        ms=3,
        capsize=2,
        lw=1,
        elinewidth=0.7,
        label=f"Real data, {clf}",
    )
for ii, m in enumerate(["Claude", "ChatGPT", "Gemini", "DeepSeek"]):
    rr = bm[(bm.Source == m) & (bm.Classifier == "Random Forest")].groupby("Run").AUC.mean()
    a.axhline(rr.mean(), color=COL[m], lw=1.3)
    a.vlines(xs_[-1] * (1.04 + 0.035 * ii), rr.min(), rr.max(), color=COL[m], lw=3, alpha=0.7)
    a.text(
        xs_[-1] * 1.22,
        rr.mean(),
        f"{m} {rr.mean():.2f}\n(runs {rr.min():.2f}–{rr.max():.2f})",
        color=COL[m],
        va="center",
        fontsize=5.8,
        fontweight="bold",
    )
a.set_xscale("log")
a.set_xticks(xs_)
a.set_xticklabels([str(v) for v in xs_])
a.minorticks_off()
a.set_xlim(20, xs_[-1] * 1.18)
a.set_ylim(0.15, 1.0)
a.set_xlabel("Real training records (log scale)")
a.set_ylabel("AUC on held-out real patients")
a.legend(loc="lower left", fontsize=5.8, frameon=False)
letter(a, "b", -0.2, 1.0)
save(fig, "Fig8")

# ================================================================== Fig 9: dashboard
fid = []
for k, d in SYN.items():
    ca, cb = d[COLS].corr().fillna(0).values, r[COLS].corr().values
    iu = np.triu_indices(12, 1)
    msk = np.abs(cb[iu]) >= 0.1
    fid.append(
        {
            "Model": k[0],
            "Run": k[1],
            "KS": int((ks[(ks.Model == k[0]) & (ks.Run == k[1])].p > 0.05).sum()),
            "MAD": np.mean(np.abs(ca[iu] - cb[iu])),
            "Sign": np.mean(np.sign(ca[iu][msk]) == np.sign(cb[iu][msk])),
            "SHAP": spearmanr(SHr[IU], SHs[k][IU]).correlation,
            "Copies": 100 * (NEAR[k] >= 11).mean(),
            "AIA": aia(d),
            "Int": INT[(INT.Model == k[0]) & (INT.Run == k[1])].AUC.mean(),
            "Ext": EXT[(EXT.Model == k[0]) & (EXT.Run == k[1])].AUC.mean(),
        }
    )
fid = pd.DataFrame(fid)
null = [
    np.mean(
        np.abs(
            r.sample(200, random_state=s)[COLS].corr().values[np.triu_indices(12, 1)]
            - r[COLS].corr().values[np.triu_indices(12, 1)]
        )
    )
    for s in range(200)
]
P9 = [
    ("KS", "Features without K–S difference\n(of 11)", None, (0, 11.5)),
    ("MAD", "Mean absolute correlation deviation", np.percentile(null, 95), None),
    ("Sign", "Correlation sign agreement", 1, (0, 1.05)),
    ("SHAP", "Interaction similarity (Spearman)", None, (-0.1, 1)),
    ("Copies", "Near-verbatim copies of real records (%)", None, None),
    ("AIA", "Attribute-inference accuracy", AIA.mean(), None),
    ("Int", "Internal AUC (nested CV)", None, (0.3, 1.02)),
    ("Ext", "External AUC (real patients)", real_ref.mean(), (0.2, 1.0)),
    (None, None, None, None),
]
fig, axes = plt.subplots(3, 3, figsize=(W2, 5.8))
axes = axes.ravel()
for i, (c, t, refl, yl_) in enumerate(P9):
    a = axes[i]
    if c is None:
        a.axis("off")
        a.legend(
            handles=[
                Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor=COL[m], markeredgecolor="k", markersize=6, label=m
                )
                for m in MODELS
            ]
            + [
                Line2D(
                    [0],
                    [0],
                    marker=RUN_MK[k],
                    color="w",
                    markerfacecolor="grey",
                    markeredgecolor="k",
                    markersize=5,
                    label=f"Run {k}",
                )
                for k in RUNS
            ]
            + [
                Line2D([0], [0], color="k", lw=2, label="Mean of 3 runs"),
                Line2D([0], [0], color="k", ls="--", lw=0.8, label="Real-data reference"),
            ],
            loc="center",
            frameon=False,
            fontsize=7,
        )
        continue
    for j, m in enumerate(MODELS):
        for k in RUNS:
            v = fid[(fid.Model == m) & (fid.Run == k)][c].values[0]
            a.scatter(j + (k - 2) * 0.17, v, color=COL[m], marker=RUN_MK[k], s=22, edgecolor="k", lw=0.4, zorder=3)
        a.hlines(fid[fid.Model == m][c].mean(), j - 0.32, j + 0.32, color=COL[m], lw=2)
    if refl is not None:
        a.axhline(refl, color="k", ls="--", lw=0.8)
    if yl_:
        a.set_ylim(*yl_)
    a.set_xticks(range(4))
    a.set_xticklabels(MODELS, fontsize=6)
    [tt.set_color(COL[m]) for tt, m in zip(a.get_xticklabels(), MODELS)]
    a.set_title(t, fontsize=6.8, pad=3)
    letter(a, "abcdefgh"[i], -0.2, 1.02)
fig.tight_layout(h_pad=1.2, w_pad=1.0)
save(fig, "Fig9")

# ================================================================== Fig S1: categorical distributions
CAT = {
    "sex": {0: "Female", 1: "Male"},
    "chest pain type": {1: "Typ.", 2: "Atyp.", 3: "Non-ang.", 4: "Asym."},
    "fasting blood sugar": {0: "≤ 120", 1: "> 120"},
    "resting ecg": {0: "Normal", 1: "ST-T", 2: "LVH"},
    "exercise angina": {0: "No", 1: "Yes"},
    "ST slope": {1: "Up", 2: "Flat", 3: "Down"},
    "target": {0: "No", 1: "Yes"},
}
fig, axes = plt.subplots(2, 4, figsize=(W2, 3.9))
axes = axes.ravel()
w = 0.16
for i, (c, lv) in enumerate(CAT.items()):
    a = axes[i]
    x = np.arange(len(lv))
    a.bar(x - 2 * w, [100 * (r[c] == v).mean() for v in lv], w, color="#9e9e9e", edgecolor="k", lw=0.3)
    for j, m in enumerate(MODELS):
        vals = np.array([[100 * (SYN[(m, k)][c] == v).mean() for v in lv] for k in RUNS])
        a.bar(x + (j - 1) * w, vals.mean(0), w, color=COL[m], edgecolor="k", lw=0.3)
        for kk, k in enumerate(RUNS):
            a.scatter(x + (j - 1) * w, vals[kk], color="white", edgecolor="k", s=5, marker=RUN_MK[k], zorder=3, lw=0.4)
    a.set_xticks(x)
    a.set_xticklabels(list(lv.values()), fontsize=6)
    a.set_xlabel(lab(c))
    a.set_ylabel("Records (%)" if i % 4 == 0 else "")
    letter(a, "abcdefg"[i], -0.25, 1.0)
axes[-1].axis("off")
axes[-1].legend(
    handles=[Patch(color="#9e9e9e", label="Real data")]
    + [Patch(color=COL[m], label=f"{m} (mean of 3 runs)") for m in MODELS]
    + [
        Line2D(
            [0],
            [0],
            marker=RUN_MK[k],
            color="w",
            markerfacecolor="white",
            markeredgecolor="k",
            markersize=4.5,
            label=f"Run {k}",
        )
        for k in RUNS
    ],
    loc="center",
    frameon=False,
    fontsize=6.5,
)
fig.tight_layout()
save(fig, "FigS1")

# ================================================================== Fig S4: bivariate correlations
PAIRS2 = [
    ("age", "max heart rate"),
    ("age", "resting bp s"),
    ("resting bp s", "cholesterol"),
    ("max heart rate", "oldpeak"),
]
fig, axes = plt.subplots(1, 4, figsize=(W2, 2.2), sharey=True)
for i, (x_, y_) in enumerate(PAIRS2):
    a = axes[i]
    a.axhline(np.corrcoef(r[x_], r[y_])[0, 1], color=COL["Real"], lw=1.2, ls="--")
    a.axhline(0, color="grey", lw=0.4)
    for j, m in enumerate(MODELS):
        for k in RUNS:
            a.scatter(
                j + (k - 2) * 0.17,
                np.corrcoef(SYN[(m, k)][x_], SYN[(m, k)][y_])[0, 1],
                color=COL[m],
                marker=RUN_MK[k],
                s=20,
                edgecolor="k",
                lw=0.4,
                zorder=3,
            )
    a.set_xticks(range(4))
    a.set_xticklabels(MODELS, rotation=35, ha="right", fontsize=6)
    a.set_ylim(-1.05, 1.05)
    a.set_xlabel(f"{lab(x_)} vs {lab(y_)}", fontsize=6.5)
    letter(a, "abcd"[i], -0.15, 1.0)
axes[0].set_ylabel("Pearson correlation")
axes[3].legend(
    handles=[Line2D([0], [0], color=COL["Real"], ls="--", lw=1.2, label="Real data")]
    + [
        Line2D(
            [0],
            [0],
            marker=RUN_MK[k],
            color="w",
            markerfacecolor="grey",
            markeredgecolor="k",
            markersize=4,
            label=f"Run {k}",
        )
        for k in RUNS
    ],
    loc="lower right",
    fontsize=5.8,
    frameon=False,
)
fig.tight_layout()
save(fig, "FigS4")

# ================================================================== Fig S5: calibration
fig, axes = plt.subplots(1, 4, figsize=(W2, 2.75), sharey=True)
for i, m in enumerate(MODELS):
    a = axes[i]
    for k in RUNS:
        s_ = PRED[(PRED.Model == m) & (PRED.Run == k) & (PRED.Framework == "Super Learner (LR-Meta)") & PRED.unique]
        g_ = (
            pd.DataFrame({"p": s_.p.values, "y": s_.y.values, "b": pd.qcut(s_.p.values, 10, duplicates="drop")})
            .groupby("b", observed=True)
            .agg(p=("p", "mean"), y=("y", "mean"))
        )
        e_ = EXT[(EXT.Model == m) & (EXT.Run == k) & (EXT.Framework == "Super Learner (LR-Meta)")].iloc[0]
        a.plot(
            g_.p,
            g_.y,
            marker=RUN_MK[k],
            ls=RUN_LS[k],
            color=COL[m],
            lw=0.9,
            ms=2.8,
            label=f"Run {k}: slope {e_['Calibration slope']:.2f}, Brier {e_.Brier:.3f}",
        )
    a.plot([0, 1], [0, 1], "k--", lw=0.6)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_aspect("equal")
    a.set_xlabel("Predicted risk (decile mean)")
    a.set_title(m, color=COL[m], fontweight="bold", fontsize=7.5)
    a.legend(fontsize=5.4, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, handlelength=1.8)
axes[0].set_ylabel("Observed proportion with disease")
fig.tight_layout(w_pad=0.8)
save(fig, "FigS5")

# ================================================================== Fig S6: external AUC per framework and run
FW_ORDER = [
    "Random Forest",
    "Random Forest Boosting",
    "AdaBoost",
    "GBM",
    "Stochastic GBM",
    "XGBoost",
    "LightGBM",
    "Blending",
    "Super Learner (LR-Meta)",
    "Super Learner (Hist-Meta)",
    "Super Learner (XGB-Meta)",
]
H = EXT.pivot_table(index="Framework", columns=["Model", "Run"], values="AUC").reindex(FW_ORDER)
H = H[[(m, k) for m in MODELS for k in RUNS]]
fig, a = plt.subplots(figsize=(W2, 3.4))
sns.heatmap(
    H,
    ax=a,
    cmap="RdYlBu",
    vmin=0.2,
    vmax=1,
    center=0.6,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 5.5},
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "External AUC (918 unique real patients)", "shrink": 0.8},
    xticklabels=[f"{m} {k}" for m, k in H.columns],
)
[t.set_color(COL[mk[0]]) for t, mk in zip(a.get_xticklabels(), H.columns)]
plt.setp(a.get_xticklabels(), rotation=45, ha="right")
a.set_xlabel("")
a.set_ylabel("")
for xv in [3, 6, 9]:
    a.axvline(xv, color="k", lw=1)
save(fig, "FigS6")

# ================================================================== Fig S7: robustness of external AUC to the test-set definition
SENS = [
    ("AUC", "Primary: 918 unique records"),
    ("AUC (all 1,190)", "All 1,190 records"),
    ("AUC (cholesterol > 0)", "Non-zero cholesterol"),
    ("AUC (copied patients excluded)", "Copied real records excluded"),
]
fig, a = plt.subplots(figsize=(W2, 2.5))
mk = ["o", "D", "v", "P"]
keys = [(m, k) for m in MODELS for k in RUNS]
for si, (c, l) in enumerate(SENS):
    v = [EXT[(EXT.Model == m) & (EXT.Run == k)][c].mean() for m, k in keys]
    a.scatter(
        np.arange(12) + (si - 1.5) * 0.15,
        v,
        marker=mk[si],
        s=16,
        color=[COL[m] for m, _ in keys],
        edgecolor="k",
        lw=0.4,
        label=l,
        zorder=3,
    )
a.set_xticks(range(12))
a.set_xticklabels([f"{m} {k}" for m, k in keys], rotation=45, ha="right", fontsize=6)
[t.set_color(COL[m]) for t, (m, _) in zip(a.get_xticklabels(), keys)]
for xv in [2.5, 5.5, 8.5]:
    a.axvline(xv, color="#ddd", lw=0.6)
a.axhline(0.5, color="grey", ls=":", lw=0.6)
a.set_ylabel("Mean external AUC (11 frameworks)")
a.set_ylim(0.2, 1)
a.legend(
    handles=[
        Line2D([0], [0], marker=mk[i], color="w", markerfacecolor="grey", markeredgecolor="k", markersize=4.5, label=l)
        for i, (_, l) in enumerate(SENS)
    ],
    loc="lower left",
    ncol=2,
    fontsize=6,
    frameon=False,
)
fig.tight_layout()
save(fig, "FigS7")
print("done")
