# AI Loan Risk Prediction API BY AJAYI ADEBAYO

## 🚀 Overview
This project is a production-style machine learning API that predicts whether a loan application should be approved or rejected.

The system is deployed in the cloud, I used render.com and provides real-time predictions via a REST API.

---

## ⚙️ Tech Stack
- Python
- Scikit-learn
- FastAPI
- Pandas / NumPy
- Render (Cloud Deployment)

---

## 🧠 Features
- Machine learning model for credit risk prediction
- Data preprocessing and feature engineering pipeline
- REST API with FastAPI
- Live cloud deployment

---

## 📡 API Endpoint
POST /predict

---

## 📥 Example Input
```json
{
  "duration": 5,
  "credit_amount": 40000,
  "age": 41
}

EXAMPLE OUTPUT
{
  "decision": "Approved",
  "risk_level": "Low",
  "confidence": 0.87

LIVE PROJECT URL https://ai-project-s22c.onrender.com/docs

💡 Use Case

This project simulates how financial institutions evaluate loan applications using machine learning models.


}
