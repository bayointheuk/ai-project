from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

# IMPORTANT: Load training columns
# (we will create this file next)
columns = joblib.load("columns.pkl")

# New input format (simple fields)
class LoanRequest(BaseModel):
    duration: int
    credit_amount: int
    age: int

@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict(request: LoanRequest):

    # Step 1: Convert to dataframe
    data = pd.DataFrame([{
        "duration": request.duration,
        "credit_amount": request.credit_amount,
        "age": request.age
    }])

    # Step 2: One-hot encode
    data_encoded = pd.get_dummies(data)

    # Step 3: Align columns with training
    data_encoded = data_encoded.reindex(columns=columns, fill_value=0)

    # Step 4: Scale
    data_scaled = scaler.transform(data_encoded)

    # Step 5: Predict
    prediction = model.predict(data_scaled)[0]

    if prediction == 1:
        return {"prediction": "Approved", "risk": "Low"}
    else:
        return {"prediction": "Rejected", "risk": "High"}
