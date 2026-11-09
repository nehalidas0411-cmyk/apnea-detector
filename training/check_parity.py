"""Confirms the serving pipeline (apnea/pipeline.py) reproduces the training feature matrix.
  python check_parity.py"""
import numpy as np
from common import load_df, columns, HRV, EDR
from apnea_starter2 import load_record
from apnea_starter3 import beat_amplitudes
from apnea.pipeline import build_matrix

df = load_df()
for rec in ("c01", "a02"):
    t, _ = load_record(rec)
    mins, X, _ = build_matrix(t, beat_amplitudes(rec, t))
    ref = df[df.record == rec].sort_values("minute")
    assert (ref.minute.values == mins).all(), "minute rows differ"
    ref_X = ref[columns(HRV + EDR)].fillna(0).values
    print(rec, "max abs diff:", np.abs(X - ref_X).max())
