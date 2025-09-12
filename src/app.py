from fastapi import FastAPI
import joblib
import pandas as pd

# Load trained model
model = joblib.load("src/best_model.pkl")

# Create FastAPI app
app = FastAPI(
    title="Coupon Purchase Prediction API",
    description="Predicts probability of a coupon being purchased (redeemed) by a customer.",
    version="1.0"
)

# Root endpoint
@app.get("/")
def root():
    return {"message": "Coupon Purchase Prediction API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(campaign_id: int, coupon_id: int, customer_id: int):
    """
    Predicts purchase probability given campaign_id, coupon_id, and customer_id.
    """
    # Build dataframe from inputs
    input_data = pd.DataFrame([{
        "campaign_id": campaign_id,
        "coupon_id": coupon_id,
        "customer_id": customer_id
    }])

    # LightGBM requires predict_proba
    if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(input_data)[:, 1][0]
    else:  # in case LightGBM native booster
        pred_proba = model.predict(input_data)[0]

    return {
        "purchase_probability": float(pred_proba),
        "recommendation": "Likely to purchase ✅" if pred_proba > 0.5 else "Unlikely to purchase ❌"
    }
