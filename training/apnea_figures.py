"""
Report figures for the sleep apnea mini-project.
Needs apnea_starter2.py and apnea_starter3.py in the same folder, plus the cached
CSVs they created (features_v2_base.csv, features_v3_edr.csv, results_*.csv).

Run:  python apnea_figures.py      -> saves PNGs into ./figures
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from xgboost import DMatrix

from apnea_starter2 import (
    TIME, POINCARE, FREQ, CONTEXT_OFFSETS,
    zscore_per_record, add_context, smooth,
)
from apnea_starter3 import (
    EDR_FEATS, ROLL_WINDOW, add_rolling, make_xgb, make_rf, cv_probs,
)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------- data
def load_features():
    base = pd.read_csv("features_v2_base.csv")
    edr = pd.read_csv("features_v3_edr.csv")
    df = (base.merge(edr, on=["record", "minute"], how="left")
              .sort_values(["record", "minute"]).reset_index(drop=True))
    all_feats = TIME + POINCARE + FREQ + EDR_FEATS
    df = zscore_per_record(df, all_feats)
    z_all = [c + "_z" for c in all_feats]
    df = add_context(df, z_all)
    ctx = [f"{c}_t{o:+d}" for o in CONTEXT_OFFSETS for c in z_all]
    df = add_rolling(df, z_all)
    roll = [f"{c}_roll{ROLL_WINDOW}" for c in z_all]
    return df, z_all + ctx + roll


def base_name(col):
    """Map e.g. 'sdnn_z_t-1' or 'edr_vlf_z_roll7' back to 'sdnn' / 'edr_vlf'."""
    col = col.replace(f"_roll{ROLL_WINDOW}", "")
    for o in CONTEXT_OFFSETS:
        col = col.replace(f"_t{o:+d}", "")
    return col[:-2] if col.endswith("_z") else col


# ---------------------------------------------------------------- figures
def fig_ablation():
    rows = [("0. v1 baseline (XGB, HRV)", 0.697, 0.769)]
    if os.path.exists("results_v2_ablation.csv"):
        v2 = pd.read_csv("results_v2_ablation.csv")
        v2 = v2[~v2["setup"].str.startswith("5b")]
        rows += [(f"v2 | {r.setup}", r.accuracy, r.auc) for r in v2.itertuples()]
    if os.path.exists("results_v3.csv"):
        v3 = pd.read_csv("results_v3.csv")
        keep = v3[v3["setup"].str.startswith(("B.", "D."))]
        rows += [(f"v3 | {r.setup}", r.accuracy, r.auc) for r in keep.itertuples()]

    labels = [r[0] for r in rows]
    acc = [r[1] for r in rows]
    auc = [r[2] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(y - 0.2, acc, height=0.4, label="Accuracy")
    ax.barh(y + 0.2, auc, height=0.4, label="AUC")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.6, 0.9)
    ax.set_xlabel("Score (record-wise 5-fold cross-validation)")
    ax.set_title("Effect of each improvement on apnea detection")
    ax.legend(loc="upper right")
    for i, (a, u) in enumerate(zip(acc, auc)):
        ax.text(a + 0.003, i - 0.2, f"{a:.3f}", va="center", fontsize=8)
        ax.text(u + 0.003, i + 0.2, f"{u:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/1_ablation.png", dpi=200)
    plt.close(fig)


def fig_roc(y, curves):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, prob in curves.items():
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve (held-out records)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/2_roc.png", dpi=200)
    plt.close(fig)


def fig_confusion(y, pred):
    cm = confusion_matrix(y, pred)
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1%})", ha="center", va="center",
                    color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Apnea"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Apnea"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (final ensemble)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/3_confusion_matrix.png", dpi=200)
    plt.close(fig)


def fig_importance(df, cols):
    """Mean |SHAP value| per feature, using XGBoost's built-in SHAP (no extra install).
    The model is fit on all records here ONLY to explain it, not to report performance."""
    X = df[cols].fillna(0).values
    y = df["label"].values
    model = make_xgb(y)
    model.fit(X, y)
    contrib = model.get_booster().predict(DMatrix(X), pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=cols)
    imp = imp.groupby(base_name).sum().sort_values(ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index[::-1], imp.values[::-1])
    ax.set_xlabel("Mean |SHAP value| (summed over time offsets)")
    ax.set_title("Most influential features")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/4_feature_importance.png", dpi=200)
    plt.close(fig)
    imp.round(4).to_csv("feature_importance.csv", header=["mean_abs_shap"])
    print("\nTop features:\n", imp.round(3).to_string())


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    df, cols = load_features()
    y = df["label"].values
    print(f"Rows: {len(df)} | features: {len(cols)}")

    print("Running cross-validation for the final models...")
    p_xgb, fold = cv_probs(df, cols, make_xgb)
    p_rf, _ = cv_probs(df, cols, make_rf)
    p_ens = smooth(df, (p_xgb + p_rf) / 2)
    p_xgb_s = smooth(df, p_xgb)
    pred = (p_ens >= 0.5).astype(int)

    fig_ablation()
    fig_roc(y, {"XGBoost": p_xgb_s, "Ensemble XGB+RF": p_ens})
    fig_confusion(y, pred)
    fig_importance(df, cols)

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"\nFinal ensemble: acc={(tp + tn) / len(y):.3f} sens={tp / (tp + fn):.3f} "
          f"spec={tn / (tn + fp):.3f} AUC={roc_auc_score(y, p_ens):.3f}")
    print(f"Figures saved in ./{FIG_DIR}")