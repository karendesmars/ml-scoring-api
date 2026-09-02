# ML Scoring API

**Model served:** LogisticRegression trained on the [Cleveland heart disease dataset](https://github.com/karendesmars/pandas-heart-disease), 88.5% test accuracy
**Tools:** FastAPI, Pydantic, scikit-learn, Docker
**Goal:** Serve a trained classification model as a production-style HTTP API, not just a notebook. This project focuses on the ML engineering layer (API design, input validation, testing, containerization, deployment) rather than on training a new model.

---

## Project Structure

```
ml-scoring-api/
├── app/
│   ├── main.py                        # FastAPI app: /health and /predict endpoints
│   ├── model.py                       # Model loading and prediction logic
│   └── schemas.py                     # Pydantic request/response models, with field validation
├── model/
│   ├── train.py                       # Trains and saves the model (reproducible from scratch)
│   └── heart_disease_model.joblib     # Trained model, tracked in git (small file)
├── data/
│   └── heart-disease.csv              # Training data, reused from pandas-heart-disease
├── tests/
│   └── test_api.py                    # pytest tests for the API (health check, valid/invalid input)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Why reuse an existing model instead of training a new one

The model itself (a LogisticRegression on 13 clinical features) was already built and validated in [pandas-heart-disease](https://github.com/karendesmars/pandas-heart-disease). The point of this project is the engineering layer around a model: a clean API contract, input validation, automated tests, a container image, and a real deployment. `model/train.py` retrains and re-saves the exact same model from the same data, so the project is self-contained and reproducible without depending on the other repo.

---

## Running locally

```
conda env create -f environment.yml   # or: pip install -r requirements.txt
python model/train.py                 # trains and saves the model (only needed once)
uvicorn app.main:app --reload
```

The interactive API docs are then available at `http://127.0.0.1:8000/docs`.

---

## API

### `GET /health`
Basic health check, used by the deployment platform to confirm the service is running.

### `POST /predict`
Takes the 13 clinical features as JSON and returns a prediction.

Example request body:
```json
{
  "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
  "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
  "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}
```

Example response:
```json
{ "prediction": 0, "probability": 0.275 }
```

Invalid input (missing fields, or a value outside the ranges observed in the training data) returns a `422` with details on which field failed validation, instead of a silent wrong prediction.

---

## Example: model behavior on two contrasting profiles

```bash
# Profile A: younger, healthier-looking indicators
curl -X POST https://ml-scoring-api-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"age":29,"sex":0,"cp":0,"trestbps":110,"chol":180,"fbs":0,"restecg":0,"thalach":190,"exang":0,"oldpeak":0.0,"slope":2,"ca":0,"thal":2}'
# -> {"prediction":1,"probability":0.942}

# Profile B: older, several risk factors
curl -X POST https://ml-scoring-api-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"age":68,"sex":1,"cp":3,"trestbps":160,"chol":286,"fbs":1,"restecg":2,"thalach":108,"exang":1,"oldpeak":3.2,"slope":1,"ca":3,"thal":3}'
# -> {"prediction":0,"probability":0.029}
```

This looks backwards at first: the younger profile is predicted as higher risk, the older one with more risk factors as lower risk. This is not a bug in the API. It comes from the underlying model itself: as documented in [pandas-heart-disease](https://github.com/karendesmars/pandas-heart-disease), `sex` has a strong negative coefficient in this LogisticRegression (being male pushes the prediction toward "no disease"). Profile A is female (`sex=0`), Profile B is male (`sex=1`), and that one field carries enough weight to flip the result even against several other risk factors going the other way. This is a known limitation of the model on this specific 303-patient dataset, not something to fix at the API level, the API is correctly serving whatever the model predicts.

---

## Running with Docker

```
docker build -t ml-scoring-api .
docker run -p 8000:8000 ml-scoring-api
```

---

## Tests

```
pytest tests/ -v
```

Covers: health check, a valid prediction request, an out-of-range field, and a missing required field.

---

## Deployment

Deployed on [Railway](https://railway.app) directly from this repo's `Dockerfile` (no local Docker installation needed on the deploying machine, Railway builds the image from source).

Live URL: https://ml-scoring-api-production.up.railway.app

```
curl https://ml-scoring-api-production.up.railway.app/health
curl -X POST https://ml-scoring-api-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```
