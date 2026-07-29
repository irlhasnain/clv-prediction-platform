from fastapi.testclient import TestClient
import sys
sys.path.append('.')
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "CLV Prediction API is running"

def test_predict_endpoint():
    payload = {
        "frequency": 5,
        "recency": 30.0,
        "customer_age_days": 365.0,
        "avg_order_value": 150.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_clv" in response.json()

def test_predict_invalid_input():
    payload = {"frequency": "not_a_number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  