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

from apnea_starter2 import (
    RECORDS, DATA_DIR, FS, WINDOW_CONTEXT, TIME, POINCARE, FREQ,
    download, load_record, build_base, zscore_per_record, add_context,
    smooth, CONTEXT_OFFSETS,
)

EDR_FEATS = ["edr_cv", "edr_vlf", "edr_resp", "edr_ratio", "edr_peak_freq"]
ROLL_WINDOW = 7


# ---------------------------------------------------------------- EDR features
def beat_amplitudes(rec, r_times):
    """Peak deviation of the ECG around each R-peak (tracks breathing)."""
    sig = wfdb.rdrecord(os.path.join(DATA_DIR, rec)).p_signal[:, 0]
    idx = np.round(r_times * FS).astype(int)
    offs = np.arange(-5, 6)
    win = sig[np.clip(idx[:, None] + offs, 0, len(sig) - 1)]
    return np.abs(win - np.median(sig)).max(axis=1)


def edr_window_features(t, amp):
    nan = {k: np.nan for k in EDR_FEATS}
    if len(t) < 60:
        return nan
    lo, hi = np.percentile(amp, [1, 99])
    amp = np.clip(amp, lo, hi)
    try:
        fs_i = 4
        tt = np.arange(t[0], t[-1], 1 / fs_i)
        a = interp1d(t, amp, kind="linear")(tt)
        cv = a.std() / (a.mean() + 1e-9)
        a = a - a.mean()
        fr, p = welch(a, fs=fs_i, nperseg=min(len(a), 1024))
        df_ = fr[1] - fr[0]
        band = lambda l, h: p[(fr >= l) & (fr < h)].sum() * df_
        vlf, resp = band(0.01, 0.04), band(0.15, 0.40)
        m = (fr >= 0.1) & (fr <= 0.5)
        return {
            "edr_cv": cv,
            "edr_vlf": np.log(vlf + 1e-9),
            "edr_resp": np.log(resp + 1e-9),
            "edr_ratio": vlf / (vlf + resp + 1e-12),
            "edr_peak_freq": fr[m][np.argmax(p[m])],
        }
    except Exception:
        return nan


def build_edr():
    rows = []
    for rec in RECORDS:
        r_times, labels = load_record(rec)
        amps = beat_amplitudes(rec, r_times)
        for m in range(len(labels)):
            lo, hi = np.searchsorted(
                r_times, [max(0, m - WINDOW_CONTEXT) * 60, (m + WINDOW_CONTEXT + 1) * 60]
            )
            f = edr_window_features(r_times[lo:hi], amps[lo:hi])
            f.update(record=rec, minute=m)
            rows.append(f)
        print(f"  EDR done: {rec}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- helpers
def add_rolling(df, cols, w=ROLL_WINDOW):
    g = df.groupby("record")
    new = {
        f"{c}_roll{w}": g[c].transform(lambda x: x.rolling(w, center=True, min_periods=1).mean())
        for c in cols
    }
    return pd.concat([df, pd.DataFrame(new)], axis=1)


def make_xgb(y):
    spw = (y == 0).sum() / max(1, (y == 1).sum())
    return XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=5, reg_lambda=5,
        scale_pos_weight=spw, eval_metric="logloss", random_state=42,
    )


def make_rf(y):
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=10, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )


def cv_probs(df, cols, model_fn, n_splits=5):
    X = df[cols].fillna(0).values
    y = df["label"].values
    prob = np.zeros(len(y))
    fold = np.zeros(len(y), dtype=int)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, df["record"].values)):
        model = model_fn(y[tr])
        model.fit(X[tr], y[tr])
        prob[te] = model.predict_proba(X[te])[:, 1]
        fold[te] = k
    return prob, fold


def report(df, prob, fold, name, smooth_preds=True):
    if smooth_preds:
        prob = smooth(df, prob)
    y = df["label"].values
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    fold_acc = [(pred[fold == k] == y[fold == k]).mean() for k in np.unique(fold)]
    row = {
        "setup": name, "accuracy": (tp + tn) / len(y), "acc_fold_std": np.std(fold_acc),
        "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "f1_apnea": f1_score(y, pred), "auc": roc_auc_score(y, prob),
    }
    print(f"{name:<42} acc={row['accuracy']:.3f} (+/-{row['acc_fold_std']:.3f} across folds) "
          f"sens={row['sensitivity']:.3f} spec={row['specificity']:.3f} "
          f"F1={row['f1_apnea']:.3f} AUC={row['auc']:.3f}")
    return row


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    download()

    base_csv, edr_csv = "features_v2_base.csv", "features_v3_edr.csv"
    base = pd.read_csv(base_csv) if os.path.exists(base_csv) else build_base()
    base.to_csv(base_csv, index=False)

    if os.path.exists(edr_csv):
        edr = pd.read_csv(edr_csv)
    else:
        print("Building EDR features (one-off, a few minutes)...")
        edr = build_edr()
        edr.to_csv(edr_csv, index=False)

    df = base.merge(edr, on=["record", "minute"], how="left")
    df = df.sort_values(["record", "minute"]).reset_index(drop=True)

    hrv_feats = TIME + POINCARE + FREQ
    all_feats = hrv_feats + EDR_FEATS
    df = zscore_per_record(df, all_feats)
    z_hrv = [c + "_z" for c in hrv_feats]
    z_all = [c + "_z" for c in all_feats]

    df = add_context(df, z_all)
    ctx_hrv = [f"{c}_t{o:+d}" for o in CONTEXT_OFFSETS for c in z_hrv]
    ctx_all = [f"{c}_t{o:+d}" for o in CONTEXT_OFFSETS for c in z_all]

    df = add_rolling(df, z_all)
    roll_all = [f"{c}_roll{ROLL_WINDOW}" for c in z_all]

    print(f"\nRows: {len(df)} | apnea share: {df['label'].mean():.2%}\n")
    results = []

    set_v2 = z_hrv + ctx_hrv
    p, f = cv_probs(df, set_v2, make_xgb)
    results.append(report(df, p, f, "A. v2 features, regularised XGB"))

    set_edr = z_all + ctx_all
    p, f = cv_probs(df, set_edr, make_xgb)
    results.append(report(df, p, f, "B. + ECG-derived respiration"))

    set_full = z_all + ctx_all + roll_all
    p_xgb, f = cv_probs(df, set_full, make_xgb)
    results.append(report(df, p_xgb, f, "C. + rolling-mean features (XGB)"))

    p_rf, _ = cv_probs(df, set_full, make_rf)
    results.append(report(df, p_rf, f, "C2. Same, Random Forest"))

    results.append(report(df, (p_xgb + p_rf) / 2, f, "D. Ensemble XGB + RF (final)"))

    pd.DataFrame(results).round(3).to_csv("results_v3.csv", index=False)
    print("\nSaved: results_v3.csv")