from fastapi import FastAPI, HTTPException

from app.model import predict
from app.schemas import PatientFeatures, PredictionResponse

app = FastAPI(
    title="Heart Disease Scoring API",
    description="Serves a LogisticRegression model trained on the Cleveland heart disease dataset.",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Basic health check, used by the deployment platform to confirm the service is up."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_heart_disease(features: PatientFeatures):
    """Predict whether a patient has heart disease from their clinical parameters."""
    try:
        prediction, probability = predict(features.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictionResponse(prediction=prediction, probability=probability)
