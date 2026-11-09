# Sleep apnea detection from single-lead ECG

Static site (`public/`) plus a FastAPI backend (`api/`, `apnea/`) that runs the trained XGBoost + Random Forest ensemble.

## 1. Train and export (once, locally)
Copy your `apnea-ecg/` folder and the cached `features_v2_base.csv`, `features_v3_edr.csv` into `training/`
(or let `apnea_starter3.py` rebuild them), then:

    pip install wfdb numpy scipy pandas scikit-learn xgboost
    cd training
    python train_export.py     # writes ../models (checks exported models match the trained ones)
    python make_samples.py     # writes ../samples for the website demo
    python check_parity.py     # serving features == training features (expect ~1e-12)

## 2. Run locally
    pip install -r requirements.txt uvicorn
    uvicorn api.index:app --reload     # API on http://127.0.0.1:8000/api/health
    (open public/index.html through any static server, or deploy)

## 3. Deploy on Vercel
Commit everything including `models/` and `samples/`, import the repo in Vercel, deploy. No build settings needed.

## API
- GET  /api/health
- GET  /api/samples
- GET  /api/analyze/sample/{id}
- POST /api/analyze  {"r_times":[...seconds...], "r_amps":[...optional...]}  or  {"rr":[...seconds...]}
  Needs at least 30 usable minutes. Without r_amps the HRV-only ensemble is used.

Research prototype, not a medical device.
