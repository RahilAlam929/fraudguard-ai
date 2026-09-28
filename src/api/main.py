import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="FraudGuard AI",
    description="Credit card fraud detection API",
    version="1.0.0",
)

MODEL_PATH = "models/random_forest.joblib"
SCALER_PATH = "models/feature_scaler.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "FraudGuard AI API",
        "model": "random_forest",
    }


@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.model_dump()

    input_data = pd.DataFrame(
        [[data[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    input_data[["Time", "Amount"]] = scaler.transform(
        input_data[["Time", "Amount"]]
    )

    probability = float(model.predict_proba(input_data)[0][1])

    threshold = 0.50
    prediction = int(probability >= threshold)

    if probability >= 0.70:
        risk_level = "HIGH"
    elif probability >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fraud_probability": round(probability, 6),
        "prediction": prediction,
        "is_fraud": bool(prediction),
        "risk_level": risk_level,
        "threshold": threshold,
    }
