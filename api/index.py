import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from apnea.pipeline import Models, ROOT

app = FastAPI(title="Apnea detection API")
_models = None


def models():
    global _models
    if _models is None:
        _models = Models()
    return _models


def samples():
    d = os.path.join(ROOT, "samples")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npz")) if os.path.isdir(d) else []


def summarise(mins, p, key, truth=None):
    pred = (p >= 0.5).astype(int)
    out = {
        "model": "XGBoost + Random Forest (HRV + ECG-derived respiration)" if key == "full"
                 else "XGBoost + Random Forest (HRV only, no amplitude data supplied)",
        "minutes": mins.tolist(), "prob": np.round(p, 4).tolist(), "pred": pred.tolist(),
        "n_minutes": int(len(mins)), "flagged_minutes": int(pred.sum()),
        "flagged_pct": round(float(pred.mean()) * 100, 1),
    }
    if truth is not None:
        y = truth[mins].astype(int)
        tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
        out["truth"] = y.tolist()
        out["accuracy"] = round((tp + tn) / len(y), 3)
        out["sensitivity"] = round(tp / max(1, (y == 1).sum()), 3)
        out["specificity"] = round(tn / max(1, (y == 0).sum()), 3)
    return out


@app.get("/api/health")
def health():
    try:
        models()
        return {"ok": True, "samples": samples()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/samples")
def list_samples():
    out = []
    for s in samples():
        d = np.load(os.path.join(ROOT, "samples", s + ".npz"))
        out.append({"id": s, "minutes": int(len(d["labels"])),
                    "apnea_minutes": int(d["labels"].sum())})
    return out


@app.get("/api/analyze/sample/{sid}")
def analyze_sample(sid: str):
    if sid not in samples():
        raise HTTPException(404, "Unknown sample")
    d = np.load(os.path.join(ROOT, "samples", sid + ".npz"))
    mins, p, key = models().predict(d["r_times"], d["r_amps"])
    res = summarise(mins, p, key, d["labels"])
    res["note"] = "This recording was part of the training data, so the scores here are not a held-out test."
    return res


class Series(BaseModel):
    r_times: Optional[List[float]] = None   # beat times in seconds
    rr: Optional[List[float]] = None        # or RR intervals in seconds
    r_amps: Optional[List[float]] = None    # optional beat amplitudes (same length as r_times)


@app.post("/api/analyze")
def analyze(s: Series):
    if s.r_times:
        t = np.asarray(s.r_times, float)
    elif s.rr:
        t = np.cumsum(np.asarray(s.rr, float))
    else:
        raise HTTPException(422, "Provide r_times or rr.")
    if len(t) > 200_000 or not np.all(np.diff(t) > 0):
        raise HTTPException(422, "Beat times must be increasing and under 200,000 beats.")
    amps = None
    if s.r_amps:
        if len(s.r_amps) != len(t):
            raise HTTPException(422, "r_amps must match r_times in length.")
        amps = s.r_amps
    try:
        mins, p, key = models().predict(t, amps)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return summarise(mins, p, key)
