# Production FastAPI review score prediction service
# Build: docker build -t review-predict .
# Run:   docker run -p 8000:8000 review-predict
#
# Before building, copy a review experiment into experiments/final_model/ (see README).

FROM python:3.10-slim AS builder

# Build deps for ML libs (e.g. numpy/scikit-learn/shap)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# No extra system deps in runtime image

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY experiments/final_model/ ./experiments/final_model/

# Serve loads from this dir (model.joblib, preprocessor.joblib, feature_config.json)
ENV EXPERIMENT_DIR=/app/experiments/final_model

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
