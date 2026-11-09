"""
Apnea-ECG starter: download -> per-minute HRV features -> RF / XGBoost
with record-wise cross-validation (no patient leakage).

Setup:  pip install wfdb numpy scipy pandas scikit-learn xgboost
Run:    python apnea_starter.py
"""
import os
import numpy as np
import pandas as pd
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

DATA_DIR = "apnea-ecg"
FS = 100  # sampling rate of Apnea-ECG (Hz)

# 35 labelled records: a = apnea, b = borderline, c = control
RECORDS = (
    [f"a{i:02d}" for i in range(1, 21)]
    + [f"b{i:02d}" for i in range(1, 6)]
    + [f"c{i:02d}" for i in range(1, 11)]
)


# ---------------------------------------------------------------- download
def download():
    """Download only the files we need (signal, header, apnea labels, QRS)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    files = []
    for r in RECORDS:
        files += [f"{r}.dat", f"{r}.hea", f"{r}.apn", f"{r}.qrs"]
    wfdb.dl_files("apnea-ecg", DATA_DIR, files)


# ---------------------------------------------------------------- loading
def load_record(rec):
    """Return R-peak times (seconds) and per-minute labels (1 = apnea, 0 = normal)."""
    path = os.path.join(DATA_DIR, rec)
    r_times = wfdb.rdann(path, "qrs").sample / FS
    apn = wfdb.rdann(path, "apn")
    labels = np.array([1 if s == "A" else 0 for s in apn.symbol])
    return r_times, labels


# ---------------------------------------------------------------- features
def hrv_features(r_times):
    """HRV features from a window of R-peak times. Returns dict or None."""
    rr = np.diff(r_times)
    keep = (rr > 0.3) & (rr < 2.0)  # drop implausible beats
    rr_c, t = rr[keep], r_times[1:][keep]
    if len(rr_c) < 20:
        return None

    d = np.diff(rr_c)
    mean_rr, sdnn = rr_c.mean(), rr_c.std()
    feats = {
        "mean_rr": mean_rr,
        "mean_hr": 60 / mean_rr,
        "sdnn": sdnn,
        "cv_rr": sdnn / mean_rr,
        "rmssd": np.sqrt(np.mean(d ** 2)),
        "pnn50": np.mean(np.abs(d) > 0.05),
        "lf": np.nan, "hf": np.nan, "lf_hf": np.nan,
    }

    # frequency domain (works best for windows >= 3 min)
    try:
        fs_i = 4
        tt = np.arange(t[0], t[-1], 1 / fs_i)
        rr_i = interp1d(t, rr_c, kind="linear")(tt)
        rr_i = rr_i - rr_i.mean()
        f, p = welch(rr_i, fs=fs_i, nperseg=min(len(rr_i), 256))
        df = f[1] - f[0]
        lf = p[(f >= 0.04) & (f < 0.15)].sum() * df
        hf = p[(f >= 0.15) & (f < 0.40)].sum() * df
        feats.update(lf=lf, hf=hf, lf_hf=lf / hf if hf > 0 else np.nan)
    except Exception:
        pass
    return feats


def build_dataset(context=1):
    """
    One row per minute. Features use a window of (2*context + 1) minutes
    centred on that minute: context=0 -> 1 min, 1 -> 3 min, 2 -> 5 min.
    Changing `context` is your segment-length experiment.
    """
    rows = []
    for rec in RECORDS:
        r_times, labels = load_record(rec)
        for m in range(len(labels)):
            start = max(0, m - context) * 60
            end = (m + context + 1) * 60
            lo, hi = np.searchsorted(r_times, [start, end])
            feats = hrv_features(r_times[lo:hi])
            if feats is None:
                continue
            feats.update(record=rec, minute=m, label=labels[m])
            rows.append(feats)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- evaluation
def evaluate(df, name, model_fn, n_splits=5):
    feature_cols = [c for c in df.columns if c not in ("record", "minute", "label")]
    X = df[feature_cols].fillna(0).values
    y = df["label"].values
    groups = df["record"].values  # split by RECORD, never by minute

    oof_pred = np.zeros(len(y))
    oof_prob = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = model_fn(y[tr])
        model.fit(X[tr], y[tr])
        oof_prob[te] = model.predict_proba(X[te])[:, 1]
        oof_pred[te] = (oof_prob[te] >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, oof_pred).ravel()
    print(f"\n===== {name} =====")
    print(classification_report(y, oof_pred, target_names=["normal", "apnea"], digits=3))
    print(f"Sensitivity: {tp / (tp + fn):.3f} | Specificity: {tn / (tn + fp):.3f} "
          f"| AUC: {roc_auc_score(y, oof_prob):.3f}")


def make_rf(y_train):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  n_jobs=-1, random_state=42)


def make_xgb(y_train):
    spw = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         scale_pos_weight=spw, eval_metric="logloss", random_state=42)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    download()

    for context in (0, 1, 2):  # 1-, 3-, 5-minute windows
        df = build_dataset(context)
        df.to_csv(f"features_{2 * context + 1}min.csv", index=False)
        print(f"\n#### Window = {2 * context + 1} min | rows = {len(df)} "
              f"| apnea share = {df['label'].mean():.2%}")
        evaluate(df, f"Random Forest ({2 * context + 1} min)", make_rf)
        evaluate(df, f"XGBoost ({2 * context + 1} min)", make_xgb)

    # Next steps:
    #  - feature ablation: drop feature groups and re-run evaluate()
    #  - SHAP on the XGBoost model for feature importance
    #  - 1D-CNN on raw ECG minutes (read signal with wfdb.rdrecord)