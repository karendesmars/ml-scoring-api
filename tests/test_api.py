from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PATIENT = {
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_response():
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_out_of_range_sex():
    invalid_patient = {**VALID_PATIENT, "sex": 5}
    response = client.post("/predict", json=invalid_patient)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    incomplete_patient = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    response = client.post("/predict", json=incomplete_patient)
    assert response.status_code == 422
