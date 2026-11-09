import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from apnea_starter2 import zscore_per_record, add_context
from apnea_starter3 import add_rolling
from apnea.features import HRV, EDR
from apnea.pipeline import columns


def load_df():
    """Same feature table as apnea_figures.load_features, from the cached CSVs."""
    base = pd.read_csv("features_v2_base.csv")
    edr = pd.read_csv("features_v3_edr.csv")
    df = base.merge(edr, on=["record", "minute"], how="left").sort_values(["record", "minute"]).reset_index(drop=True)
    allf = HRV + EDR
    df = zscore_per_record(df, allf)
    z = [c + "_z" for c in allf]
    df = add_context(df, z)
    df = add_rolling(df, z)
    return df
