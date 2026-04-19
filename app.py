from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

class LoanRequest(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict(request: LoanRequest):
    data_array = np.array(request.features).reshape(1, -1)
    data_scaled = scaler.transform(data_array)
    prediction = model.predict(data_scaled)[0]

    if prediction == 1:
        return {"prediction": "Approved", "risk": "Low"}
    else:
        return {"prediction": "Rejected", "risk": "High"}
