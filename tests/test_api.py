import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

sample_customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.integration
def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Churn Prediction API" in response.json()["message"]


@pytest.mark.integration
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.integration
def test_predict_endpoint_returns_200(client):
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200


@pytest.mark.integration
def test_predict_output_fields(client):
    response = client.post("/predict", json=sample_customer)
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert "message" in data


@pytest.mark.integration
def test_predict_probability_range(client):
    response = client.post("/predict", json=sample_customer)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


@pytest.mark.unit
def test_predict_invalid_input(client):
    response = client.post("/predict", json={"gender": "Unknown"})
    assert response.status_code == 422