import json
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model" if (BASE_DIR / "model").exists() else BASE_DIR.parent / "model"

model_reg = joblib.load(MODEL_DIR / "regression_model.pkl")
model_clf = joblib.load(MODEL_DIR / "classification_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

with open(MODEL_DIR / "feature_columns.json") as f:
    feature_columns = json.load(f)

@app.get("/health")
def health():
    return {"status" : "ok"}

class RegressionInput(BaseModel):
    features: dict

class ClassificationInput(BaseModel):
    features: dict

@app.post("/predict/regression")
def predict_regression(input: RegressionInput):
    row = pd.DataFrame([input.features])
    row = row.reindex(columns=feature_columns['regression'], fill_value=0)
    prediction = model_reg.predict(row)
    return {"predicted_sales": float(prediction[0])}

@app.post("/predict/classification")
def predict_classification(input: ClassificationInput):
    row = pd.DataFrame([input.features])
    row = row.reindex(columns=feature_columns['classification'], fill_value=0)
    row_scaled = scaler.transform(row)
    prediction = model_clf.predict(row_scaled)
    probability = model_clf.predict_proba(row_scaled)[0][1]
    return {
        "is_profitable": int(prediction[0]),
        "probability_profitable": float(probability)
    }

