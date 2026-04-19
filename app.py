from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model, scaler, and columns
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Define input schema (simple user-friendly input)
class LoanRequest(BaseModel):
    duration: int
    credit_amount: int
    age: int

@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict(request: LoanRequest):

    # Step 1: Convert input to dataframe
    data = pd.DataFrame([{
        "duration": request.duration,
        "credit_amount": request.credit_amount,
        "age": request.age
    }])

    # Step 2: Encode
    data_encoded = pd.get_dummies(data)

    # Step 3: Align with training columns
    data_encoded = data_encoded.reindex(columns=columns, fill_value=0)

    # Step 4: Scale
    data_scaled = scaler.transform(data_encoded)

    # Step 5: Predict
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    # Step 6: Return professional output
    if prediction == 1:
        return {
            "decision": "Approved",
            "risk_level": "Low",
            "confidence": float(round(probability, 2))
        }
    else:
        return {
            "decision": "Rejected",
            "risk_level": "High",
            "confidence": float(round(probability, 2))
        }
