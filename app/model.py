from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "heart_disease_model.joblib"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


@lru_cache
def get_model():
    """Load the trained model once and keep it in memory for the life of the process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run `python model/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict(features: dict) -> tuple[int, float]:
    """Run a prediction for one patient. Returns (prediction, probability_of_class_1)."""
    model = get_model()
    row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return prediction, probability
