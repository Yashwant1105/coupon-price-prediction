# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, json, pandas as pd, numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = "src/best_model.pkl"
META_PATH = "src/model_meta.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    raise RuntimeError("Model or meta file missing. Make sure src/best_model.pkl and src/model_meta.json exist.")

model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)

FEATURES = meta["features"]          
THRESHOLD = meta.get("threshold", 0.3)
BEST_IT = meta.get("best_iteration", None)

#Load lookup tables (saved from notebook) 

CUST_DEMO_CSV = "src/cust_demo_feat.csv"
CUST_TRANS_CSV = "src/cust_trans_features.csv"
CAMPAIGN_CSV = "src/campaign_feat.csv"

#create an empty DataFrame so API still runs
def _load_csv_or_empty(path, index_col):
    if os.path.exists(path):
        df = pd.read_csv(path)
        # ensure id column exists
        if index_col not in df.columns:
            raise RuntimeError(f"{path} is missing required column '{index_col}'")
        return df.set_index(index_col)
    else:
        # return empty df with no columns (safe fallback)
        return pd.DataFrame(columns=[]).astype(float)

cust_demo_df  = _load_csv_or_empty(CUST_DEMO_CSV, "customer_id")
cust_trans_df = _load_csv_or_empty(CUST_TRANS_CSV, "customer_id")
campaign_df   = _load_csv_or_empty(CAMPAIGN_CSV, "campaign_id")

# Convert lookup tables to dict-of-dicts (fast lookups)
cust_demo_dict  = cust_demo_df.to_dict(orient="index") if not cust_demo_df.empty else {}
cust_trans_dict = cust_trans_df.to_dict(orient="index") if not cust_trans_df.empty else {}
campaign_dict   = campaign_df.to_dict(orient="index") if not campaign_df.empty else {}

# ---- API app and request schema ----
app = FastAPI(title="Coupon Purchase Prediction API",
              description="Predict probability a coupon will be redeemed. Supply only customer_id, campaign_id, coupon_id.",
              version="1.0")

# Enable CORS so frontend (Lovable/ngrok site) can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev: allow all.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    customer_id: int
    campaign_id: int
    coupon_id: int

# ---- Helper: build full feature row in the same order as FEATURES ----
def build_feature_row(customer_id: int, campaign_id: int, coupon_id: int):
    # Start with zero values for all features
    row = {f: 0 for f in FEATURES}

    # If the original pipeline included ids as features, put them in
    if "customer_id" in row:
        row["customer_id"] = int(customer_id)
    if "campaign_id" in row:
        row["campaign_id"] = int(campaign_id)
    if "coupon_id" in row:
        row["coupon_id"] = int(coupon_id)

    # Merge demographic features if present
    demo = cust_demo_dict.get(int(customer_id), {})
    for k, v in demo.items():
        if k in row and k != "customer_id":
            # cast to numeric if possible
            try:
                row[k] = float(v) if pd.notna(v) else 0.0
            except Exception:
                row[k] = 0.0

    # Merge transaction aggregates if present
    tx = cust_trans_dict.get(int(customer_id), {})
    for k, v in tx.items():
        if k in row and k != "customer_id":
            try:
                row[k] = float(v) if pd.notna(v) else 0.0
            except Exception:
                row[k] = 0.0

    # Merge campaign features if present
    camp = campaign_dict.get(int(campaign_id), {})
    for k, v in camp.items():
        if k in row and k != "campaign_id":
            try:
                row[k] = float(v) if pd.notna(v) else 0.0
            except Exception:
                row[k] = 0.0

    # Build DataFrame in FEATURES order and coerce to numeric (fill missing with 0)
    df = pd.DataFrame([row], columns=FEATURES).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

# ---- Endpoints ----
@app.get("/")
def root():
    return {"message": "Coupon Purchase Prediction API is running!"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        X = build_feature_row(req.customer_id, req.campaign_id, req.coupon_id)
        # LightGBM booster uses .predict; sklearn wrapper might have predict_proba
        if hasattr(model, "predict_proba") and not hasattr(model, "predict"):  # unlikely, but safe
            proba = model.predict_proba(X)[:, 1][0]
        else:
            # For Booster: model.predict(X) returns probs
            proba = model.predict(X, num_iteration=BEST_IT)[0] if BEST_IT else model.predict(X)[0]

        recommendation = "Likely to purchase ✅" if proba >= THRESHOLD else "Unlikely to purchase ❌"
        return {
            "purchase_probability": float(proba),
            "recommendation": recommendation
        }
    except Exception as e:
        # return HTTP 400 with a helpful message
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
