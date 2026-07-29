from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="CLV Prediction API")

model = joblib.load('models/clv_model.pkl')

class CustomerFeatures(BaseModel):
    frequency: int
    recency: float
    customer_age_days: float
    avg_order_value: float

@app.get("/")
def health_check():
    return {"status": "CLV Prediction API is running"}

@app.post("/predict")
def predict_clv(customer: CustomerFeatures):
    input_df = pd.DataFrame([customer.dict()])
    prediction = model.predict(input_df)[0]
    
    return {"predicted_clv": round(float(prediction), 2)}