"""Per-window feature extraction. Numpy/scipy only; copied verbatim from
training/apnea_starter2.py (window_features) and apnea_starter3.py (edr_window_features)
so serving and training produce identical numbers (checked by training/check_parity.py)."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch

TIME = ["mean_rr", "sdnn", "cv_rr", "rmssd", "pnn50", "pnn20", "rr_range", "rr_iqr"]
POINCARE = ["sd1", "sd2", "sd_ratio"]
FREQ = ["vlf", "lf", "hf", "lf_hf", "vlf_ratio"]
HRV = TIME + POINCARE + FREQ
EDR_FEATS = ["edr_cv", "edr_vlf", "edr_resp", "edr_ratio", "edr_peak_freq"]
EDR = EDR_FEATS
CONTEXT_OFFSETS = (-2, -1, 1, 2)
ROLL_WINDOW = 7
SMOOTH_WINDOW = 5
WINDOW_CONTEXT = 2


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
