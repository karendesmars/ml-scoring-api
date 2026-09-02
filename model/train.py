"""
Trains the heart disease classifier served by this API and saves it to model/heart_disease_model.joblib.

This is the same LogisticRegression setup (C=0.20433597178569418, solver="liblinear") used and
cross-validated in the pandas-heart-disease portfolio project: github.com/karendesmars/pandas-heart-disease

Run from the project root:
    python model/train.py
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart-disease.csv"
MODEL_PATH = ROOT / "model" / "heart_disease_model.joblib"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def train():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(C=0.20433597178569418, solver="liblinear")
    model.fit(X_train, y_train)

    test_accuracy = model.score(X_test, y_test)
    print(f"Test accuracy on the held-out split: {test_accuracy:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
