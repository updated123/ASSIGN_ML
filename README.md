# Review Score Prediction (Multi-class 1–5)

Multi-class review score prediction with time-based train/val/test split, delivery-time-safe features, and optional SHAP explainability.

## Structure

```
src/
├── train.py                        # Training (LR + XGBoost/RF), time-based split
├── evaluate.py                     # Metrics, error analysis, limitations
├── serve.py                        # FastAPI prediction service (review score)
├── serve_delivery.py               # FastAPI prediction service (delivery duration)
├── feature_preparation_review.py   # Feature pipeline and config
├── train_delivery_duration.py      # Delivery duration training
├── evaluate_delivery_duration.py   # Delivery duration evaluation
├── feature_preparation_delivery_duration.py
├── data.py                         # Raw data loading
experiments/                        # One folder per run (model, preprocessor, config, reports)
notebooks/                          # EDA, feature config, reports
```

## Setup

```bash
cd ASSIGN_ML
pip install pandas numpy scikit-learn xgboost fastapi uvicorn pydantic joblib shap
```

## Training and evaluation

1. **Feature config** (if not already present):

   ```bash
   python -m src.feature_preparation_review
   ```

2. **Train** (uses `notebooks/review_feature_config.json`):

   ```bash
   python -m src.train
   ```

3. **Evaluate** (latest run, or specify `--run <run_id>`):

   ```bash
   python -m src.evaluate
   python -m src.evaluate --run 20260226_175329
   ```

## Prediction API (serve)

Loads **model** and **preprocessor** from `experiments/` (looks for `model.joblib` / `model.pkl` and `preprocessor.joblib` / `preprocessor.pkl` in the latest run, or in `EXPERIMENT_DIR` if set).

### Run the server

```bash
uvicorn src.serve:app --reload
```

Default: `http://localhost:8000`. Docs: `http://localhost:8000/docs`.

### Example request

**POST /predict** with a JSON body (bucket values must match training, e.g. `val_q0`, `fr_low`, `dd_q1`):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_order_value": 150,
    "freight_ratio": 0.21,
    "item_count": 2,
    "approval_delay_hours": 5,
    "delivery_delay_days": -2,
    "late_delivery_flag": 0,
    "primary_category": "beleza_saude",
    "seller_state": "SP",
    "customer_state": "RJ",
    "payment_type_primary": "credit_card",
    "order_value_bucket": "val_q2",
    "freight_ratio_bucket": "fr_low",
    "delivery_delay_bucket": "dd_q1"
  }'
```

Example response:

```json
{
  "predicted_review_score": 5,
  "class_probabilities": {
    "1": 0.02,
    "2": 0.03,
    "3": 0.08,
    "4": 0.22,
    "5": 0.65
  },
  "top_shap_features": [
    {"feature": "delivery_delay_days", "impact": -0.12},
    {"feature": "late_delivery_flag", "impact": -0.08},
    {"feature": "total_order_value", "impact": 0.05}
  ]
}
```

### Optional environment

- **EXPERIMENT_DIR** – Path to experiment folder (default: latest under `experiments/`).

Artifacts written by training are `model.joblib` and `preprocessor.joblib`; the server also accepts `model.pkl` and `preprocessor.pkl` if you provide them (e.g. by copying or symlinking).

---

## Delivery Duration Prediction API (`serve_delivery`)

Regression API for **delivery_duration_days**. Loads `model.joblib` / `model.pkl`, `preprocessor.joblib` / `preprocessor.pkl`, and `feature_config.json` from the latest delivery experiment (or `DELIVERY_EXPERIMENT_DIR`). Uses log-transform internally when configured; returns predicted days and top-3 SHAP feature impacts.

### Run the server

```bash
uvicorn src.serve_delivery:app --reload
```

Default: `http://localhost:8000`. Docs: `http://localhost:8000/docs`.

### Example request

**POST /predict** with a JSON body:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "intra_state": 1,
    "total_freight_value": 22.5,
    "freight_ratio": 0.18,
    "average_weight_per_item": 450,
    "avg_item_price": 80,
    "total_weight_g": 900,
    "total_order_value": 160,
    "total_volume_cm3": 12000,
    "bulky_flag": 0,
    "item_count": 2,
    "primary_category": "beleza_saude",
    "payment_type_primary": "credit_card",
    "seller_state": "SP",
    "customer_state": "RJ",
    "customer_region": "SE",
    "seller_region": "SE"
  }'
```

Example response:

```json
{
  "predicted_delivery_days": 12.47,
  "top_shap_features": [
    {"feature": "intra_state", "impact": -1.84},
    {"feature": "total_freight_value", "impact": 0.92},
    {"feature": "bulky_flag", "impact": 0.55}
  ]
}
```

Optional: **distance_km** (float) in the body to set distance bucket; if omitted, `distance_km_bucket` is `d_unknown`.

### Optional environment

- **DELIVERY_EXPERIMENT_DIR** – Path to delivery experiment folder (default: latest run in `experiments/` whose `feature_config.json` has `target: "delivery_duration_days"`).

Artifacts: `model.joblib`, `preprocessor.joblib`, `feature_config.json` (or `.pkl` equivalents).

---

## Docker

The API can run in a container. The image expects **experiments/final_model/** to contain:

- `model.joblib` (or `model.pkl`)
- `preprocessor.joblib` (or `preprocessor.pkl`)
- `feature_config.json`

Copy (or symlink) your chosen run into `final_model` before building:

```bash
mkdir -p experiments/final_model
cp experiments/<run_id>/model.joblib experiments/final_model/
cp experiments/<run_id>/preprocessor.joblib experiments/final_model/
cp experiments/<run_id>/feature_config.json experiments/final_model/
```

Build and run:

```bash
docker build -t review-predict .
docker run -p 8000:8000 review-predict
```

Then call `http://localhost:8000/predict` as in the curl example above.
