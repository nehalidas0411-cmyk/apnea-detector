"""Export a few records (beat times, beat amplitudes, minute labels) for the website demo.
  python make_samples.py     (needs the apnea-ecg folder from apnea_starter.py)"""
import os
import numpy as np
from apnea_starter2 import load_record
from apnea_starter3 import beat_amplitudes

OUT = os.path.join("..", "samples")
os.makedirs(OUT, exist_ok=True)
for rec in ("a01", "a05", "b02", "c03"):
    t, labels = load_record(rec)
    amps = beat_amplitudes(rec, t)
    np.savez_compressed(os.path.join(OUT, rec + ".npz"), r_times=t.astype(np.float64),
                        r_amps=amps.astype(np.float32), labels=labels.astype(np.int8))
    print(rec, len(t), "beats,", len(labels), "minutes,", int(labels.sum()), "apnea")
