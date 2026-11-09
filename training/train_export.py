"""Train the final models on all 35 records and export them for the API.
Run from this folder after apnea_starter3.py has produced features_v2_base.csv and features_v3_edr.csv.
  python train_export.py
Writes ../models/{xgb_full,xgb_hrv}.json, {rf_full,rf_hrv}.npz, meta.json"""
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score
from common import load_df, columns, HRV, EDR
from apnea_starter2 import smooth
from apnea_starter3 import make_xgb, make_rf, cv_probs
from xgboost import DMatrix

OUT = os.path.join("..", "models")
os.makedirs(OUT, exist_ok=True)


def export_rf(rf, path):
    L, R, F, T, P, offs = [], [], [], [], [], [0]
    for e in rf.estimators_:
        t = e.tree_
        v = t.value[:, 0, :]
        L.append(t.children_left); R.append(t.children_right); F.append(t.feature)
        T.append(t.threshold); P.append(v[:, 1] / v.sum(1)); offs.append(offs[-1] + t.node_count)
    np.savez_compressed(path, left=np.concatenate(L).astype(np.int32), right=np.concatenate(R).astype(np.int32),
                        feat=np.concatenate(F).astype(np.int32), thr=np.concatenate(T).astype(np.float64),
                        p1=np.concatenate(P).astype(np.float32), offs=np.array(offs, np.int64))


def main():
    df = load_df()
    y = df["label"].values
    meta = {"columns": {}, "cv": {}}
    from apnea.pipeline import RandomForestNP
    for key, base in (("full", HRV + EDR), ("hrv", HRV)):
        cols = columns(base)
        meta["columns"][key] = cols
        X = df[cols].fillna(0).values
        xgb, rf = make_xgb(y), make_rf(y)
        xgb.fit(X, y); rf.fit(X, y)
        xgb.get_booster().save_model(os.path.join(OUT, f"xgb_{key}.json"))
        export_rf(rf, os.path.join(OUT, f"rf_{key}.npz"))
        # parity checks: exported models must reproduce the trained ones
        import xgboost as xgb_lib
        b = xgb_lib.Booster(); b.load_model(os.path.join(OUT, f"xgb_{key}.json"))
        d1 = np.abs(b.predict(DMatrix(X)) - xgb.predict_proba(X)[:, 1]).max()
        d2 = np.abs(RandomForestNP(os.path.join(OUT, f"rf_{key}.npz")).predict_proba1(X) - rf.predict_proba(X)[:, 1]).max()
        print(f"[{key}] max prob diff  xgb={d1:.2e}  rf={d2:.2e}")
        assert d1 < 1e-5 and d2 < 1e-5
        px, _ = cv_probs(df, cols, make_xgb)
        pr, _ = cv_probs(df, cols, make_rf)
        pe = smooth(df, (px + pr) / 2)
        acc = float(((pe >= 0.5) == y).mean())
        meta["cv"][key] = {"accuracy": round(acc, 3), "auc": round(float(roc_auc_score(y, pe)), 3)}
        print(f"[{key}] record-wise 5-fold CV: {meta['cv'][key]}")
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
    print("Saved models to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
