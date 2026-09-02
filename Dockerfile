FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY model/heart_disease_model.joblib model/heart_disease_model.joblib
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

# entrypoint.sh resolves $PORT (Railway/Render inject their own; 8000 is the local
# fallback) and uses `exec` so uvicorn becomes PID 1 and receives SIGTERM directly.
CMD ["./entrypoint.sh"]
