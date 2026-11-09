"""Inference pipeline: beat times (+ optional beat amplitudes) -> per-minute apnea probability.
Mirrors training: 5-min window features, per-record z-score, neighbouring-minute context,
rolling mean, XGBoost + Random Forest average, 5-minute smoothing, threshold 0.5."""
import json, os
import numpy as np
from .features import (HRV, EDR, CONTEXT_OFFSETS, ROLL_WINDOW, SMOOTH_WINDOW, WINDOW_CONTEXT,
                       window_features, edr_window_features)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_MINUTES = 30


def columns(base):
    z = [c + "_z" for c in base]
    ctx = [f"{c}_t{o:+d}" for o in CONTEXT_OFFSETS for c in z]
    roll = [f"{c}_roll{ROLL_WINDOW}" for c in z]
    return z + ctx + roll


def _shift(a, o):
    """pandas groupby shift(-o): value from row i+o, NaN outside the record."""
    out = np.full_like(a, np.nan)
    n = len(a)
    if o >= 0:
        out[:n - o] = a[o:]
    else:
        out[-o:] = a[:n + o]
    return out


def _roll(a, w):
    """rolling(w, center=True, min_periods=1).mean(), ignoring NaN."""
    n, h = len(a), w // 2
    out = np.full(n, np.nan)
    for i in range(n):
        seg = a[max(0, i - h): i + h + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def build_matrix(r_times, r_amps=None):
    """Returns (minutes, X, use_edr). Minutes whose window has too few beats are dropped, as in
    training, so context shifts act on neighbouring kept rows."""
    r_times = np.asarray(r_times, float)
    use_edr = r_amps is not None
    if use_edr:
        r_amps = np.asarray(r_amps, float)
    rows, mins = [], []
    for m in range(int(np.ceil(r_times[-1] / 60))):
        lo, hi = np.searchsorted(r_times, [max(0, m - WINDOW_CONTEXT) * 60, (m + WINDOW_CONTEXT + 1) * 60])
        f = window_features(r_times[lo:hi])
        if f is None:
            continue
        if use_edr:
            f.update(edr_window_features(r_times[lo:hi], r_amps[lo:hi]))
        rows.append(f)
        mins.append(m)
    if len(rows) < MIN_MINUTES:
        raise ValueError(f"Need at least {MIN_MINUTES} usable minutes of beats, got {len(rows)}.")
    base = HRV + EDR if use_edr else HRV
    raw = np.array([[r[c] for c in base] for r in rows], float)
    mu, sd = np.nanmean(raw, 0), np.nanstd(raw, 0, ddof=1)
    z = (raw - mu) / (sd + 1e-9)
    parts = [z]
    for o in CONTEXT_OFFSETS:
        parts.append(np.column_stack([_shift(z[:, j], o) for j in range(z.shape[1])]))
    parts.append(np.column_stack([_roll(z[:, j], ROLL_WINDOW) for j in range(z.shape[1])]))
    return np.array(mins), np.nan_to_num(np.hstack(parts), nan=0.0), use_edr


class RandomForestNP:
    """Random-forest inference in numpy from arrays exported by training/train_export.py."""
    def __init__(self, path):
        d = np.load(path)
        self.left, self.right, self.feat = d["left"], d["right"], d["feat"]
        self.thr, self.p1, self.offs = d["thr"], d["p1"], d["offs"]

    def predict_proba1(self, X):
        X = X.astype(np.float32)
        n, ar = len(X), np.arange(len(X))
        total = np.zeros(n)
        for t in range(len(self.offs) - 1):
            a, b = self.offs[t], self.offs[t + 1]
            L, R, F, T = self.left[a:b], self.right[a:b], self.feat[a:b], self.thr[a:b]
            node = np.zeros(n, int)
            while True:
                internal = L[node] != -1
                if not internal.any():
                    break
                go_left = X[ar, np.where(internal, F[node], 0)] <= T[node]
                node = np.where(internal, np.where(go_left, L[node], R[node]), node)
            total += self.p1[a:b][node]
        return total / (len(self.offs) - 1)


class Models:
    def __init__(self, d=os.path.join(ROOT, "models")):
        import xgboost as xgb
        self.xgb = xgb
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        assert self.meta["columns"]["full"] == columns(HRV + EDR), "column mismatch (full)"
        assert self.meta["columns"]["hrv"] == columns(HRV), "column mismatch (hrv)"
        self.sets = {}
        for key in ("full", "hrv"):
            b = xgb.Booster()
            b.load_model(os.path.join(d, f"xgb_{key}.json"))
            self.sets[key] = (b, RandomForestNP(os.path.join(d, f"rf_{key}.npz")))

    def predict(self, r_times, r_amps=None):
        mins, X, use_edr = build_matrix(r_times, r_amps)
        key = "full" if use_edr else "hrv"
        b, rf = self.sets[key]
        p = (b.predict(self.xgb.DMatrix(X)) + rf.predict_proba1(X)) / 2
        return mins, _roll(p, SMOOTH_WINDOW), key
