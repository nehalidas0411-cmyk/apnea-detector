"""
Apnea-ECG v2: richer HRV features + per-record normalisation + temporal context
+ prediction smoothing, evaluated as a step-by-step ablation.

Keep this next to apnea_starter.py (it reuses the downloaded 'apnea-ecg' folder).
Run:  python apnea_v2.py
"""
import os
import numpy as np
import pandas as pd
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

DATA_DIR = "apnea-ecg"
FS = 100
WINDOW_CONTEXT = 2          # 2 -> 5-minute window centred on each minute
CONTEXT_OFFSETS = (-2, -1, 1, 2)   # neighbouring minutes stacked as extra features
SMOOTH_WINDOW = 5           # minutes, for smoothing predicted probabilities

RECORDS = (
    [f"a{i:02d}" for i in range(1, 21)]
    + [f"b{i:02d}" for i in range(1, 6)]
    + [f"c{i:02d}" for i in range(1, 11)]
)

TIME = ["mean_rr", "sdnn", "cv_rr", "rmssd", "pnn50", "pnn20", "rr_range", "rr_iqr"]
POINCARE = ["sd1", "sd2", "sd_ratio"]
FREQ = ["vlf", "lf", "hf", "lf_hf", "vlf_ratio"]


# ---------------------------------------------------------------- data
def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    files = []
    for r in RECORDS:
        for ext in ("dat", "hea", "apn", "qrs"):
            if not os.path.exists(os.path.join(DATA_DIR, f"{r}.{ext}")):
                files.append(f"{r}.{ext}")
    if files:
        wfdb.dl_files("apnea-ecg", DATA_DIR, files)


def load_record(rec):
    path = os.path.join(DATA_DIR, rec)
    r_times = wfdb.rdann(path, "qrs").sample / FS
    apn = wfdb.rdann(path, "apn")
    labels = np.array([1 if s == "A" else 0 for s in apn.symbol])
    return r_times, labels


# ---------------------------------------------------------------- features
def window_features(r_times):
    rr = np.diff(r_times)
    keep = (rr > 0.3) & (rr < 2.0)
    rr_c, t = rr[keep], r_times[1:][keep]
    if len(rr_c) < 60:
        return None

    d = np.diff(rr_c)
    m, sd = rr_c.mean(), rr_c.std()
    sd1 = np.std(d) / np.sqrt(2)
    sd2 = np.sqrt(max(2 * sd ** 2 - 0.5 * np.std(d) ** 2, 1e-12))
    f = {
        "mean_rr": m, "sdnn": sd, "cv_rr": sd / m,
        "rmssd": np.sqrt(np.mean(d ** 2)),
        "pnn50": np.mean(np.abs(d) > 0.05),
        "pnn20": np.mean(np.abs(d) > 0.02),
        "rr_range": np.ptp(rr_c),
        "rr_iqr": np.subtract(*np.percentile(rr_c, [75, 25])),
        "sd1": sd1, "sd2": sd2, "sd_ratio": sd1 / sd2,
        "vlf": np.nan, "lf": np.nan, "hf": np.nan, "lf_hf": np.nan, "vlf_ratio": np.nan,
    }

    # Frequency domain, including the VLF band where apnea cycles show up
    try:
        fs_i = 4
        tt = np.arange(t[0], t[-1], 1 / fs_i)
        rr_i = interp1d(t, rr_c, kind="linear")(tt)
        rr_i = rr_i - rr_i.mean()
        fr, p = welch(rr_i, fs=fs_i, nperseg=min(len(rr_i), 1024))
        df_ = fr[1] - fr[0]
        band = lambda lo, hi: p[(fr >= lo) & (fr < hi)].sum() * df_
        vlf, lf, hf = band(0.01, 0.04), band(0.04, 0.15), band(0.15, 0.40)
        total = vlf + lf + hf + 1e-12
        f.update(
            vlf=np.log(vlf + 1e-9), lf=np.log(lf + 1e-9), hf=np.log(hf + 1e-9),
            lf_hf=np.log((lf + 1e-9) / (hf + 1e-9)), vlf_ratio=vlf / total,
        )
    except Exception:
        pass
    return f


def build_base():
    rows = []
    for rec in RECORDS:
        r_times, labels = load_record(rec)
        for m in range(len(labels)):
            lo, hi = np.searchsorted(
                r_times, [max(0, m - WINDOW_CONTEXT) * 60, (m + WINDOW_CONTEXT + 1) * 60]
            )
            f = window_features(r_times[lo:hi])
            if f is None:
                continue
            f.update(record=rec, minute=m, label=labels[m])
            rows.append(f)
    return pd.DataFrame(rows).sort_values(["record", "minute"]).reset_index(drop=True)


def zscore_per_record(df, cols):
    """Remove each person's baseline: z-score every feature within its own record."""
    g = df.groupby("record")[cols]
    z = (df[cols] - g.transform("mean")) / (g.transform("std") + 1e-9)
    z.columns = [c + "_z" for c in cols]
    return pd.concat([df, z], axis=1)


def add_context(df, cols, offsets=CONTEXT_OFFSETS):
    """Add features from neighbouring minutes (apnea occurs in multi-minute episodes)."""
    g = df.groupby("record")
    new = {f"{c}_t{o:+d}": g[c].shift(-o) for o in offsets for c in cols}
    return pd.concat([df, pd.DataFrame(new)], axis=1)


# ---------------------------------------------------------------- evaluation
def smooth(df, prob, w=SMOOTH_WINDOW):
    s = pd.Series(prob, index=df.index)
    return (s.groupby(df["record"])
             .transform(lambda x: x.rolling(w, center=True, min_periods=1).mean())
             .values)


def make_xgb(y):
    spw = (y == 0).sum() / max(1, (y == 1).sum())
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         scale_pos_weight=spw, eval_metric="logloss", random_state=42)


def make_rf(y):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  n_jobs=-1, random_state=42)


def evaluate(df, cols, name, model_fn, smooth_preds=False, n_splits=5):
    X = df[cols].fillna(0).values
    y = df["label"].values
    groups = df["record"].values       # split by record, never by minute
    prob = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = model_fn(y[tr])
        model.fit(X[tr], y[tr])
        prob[te] = model.predict_proba(X[te])[:, 1]
    if smooth_preds:
        prob = smooth(df, prob)
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    row = {
        "setup": name, "n_features": len(cols),
        "accuracy": (tp + tn) / len(y), "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp), "f1_apnea": f1_score(y, pred),
        "auc": roc_auc_score(y, prob),
    }
    print(f"{name:<45} acc={row['accuracy']:.3f} sens={row['sensitivity']:.3f} "
          f"spec={row['specificity']:.3f} F1={row['f1_apnea']:.3f} AUC={row['auc']:.3f}")
    return row


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    download()

    if os.path.exists("features_v2_base.csv"):
        df = pd.read_csv("features_v2_base.csv")
    else:
        df = build_base()
        df.to_csv("features_v2_base.csv", index=False)

    all_feats = TIME + POINCARE + FREQ
    df = zscore_per_record(df, all_feats)
    z_feats = [c + "_z" for c in all_feats]
    df = add_context(df, z_feats)
    ctx_feats = [f"{c}_t{o:+d}" for o in CONTEXT_OFFSETS for c in z_feats]

    print(f"\nRows: {len(df)} | apnea share: {df['label'].mean():.2%}\n")
    results = [
        evaluate(df, TIME, "1. Time-domain HRV only (raw)", make_xgb),
        evaluate(df, all_feats, "2. + Poincare + VLF/LF/HF (raw)", make_xgb),
        evaluate(df, z_feats, "3. Same, per-record z-scored", make_xgb),
        evaluate(df, z_feats + ctx_feats, "4. + neighbouring-minute context", make_xgb),
        evaluate(df, z_feats + ctx_feats, "5. + prediction smoothing", make_xgb, smooth_preds=True),
        evaluate(df, z_feats + ctx_feats, "5b. Random Forest, same as 5", make_rf, smooth_preds=True),
    ]
    pd.DataFrame(results).round(3).to_csv("results_v2_ablation.csv", index=False)
    print("\nSaved: results_v2_ablation.csv (use this table in your report)")