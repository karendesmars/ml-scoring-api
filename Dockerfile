FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY model/heart_disease_model.joblib model/heart_disease_model.joblib

EXPOSE 8000

# Shell form (not exec form) so $PORT expands. Railway/Render inject their own PORT
# at runtime; 8000 is only the fallback for local `docker run` without -e PORT=...
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
