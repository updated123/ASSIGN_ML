# AI Chat Session Summary – Review Score Prediction Project

This document summarizes the work done in the AI-assisted session for the ASSIGN_ML review prediction project.

---

## 1. Exploratory Data Analysis (EDA)

- **notebooks/eda.py**: Full EDA script that loads all CSVs from `./data`, prints shape/schema/missing values per dataset, builds an order-level analytical dataset with documented joins, and engineers features (e.g. `delivery_delay_days`, `late_delivery_flag`, `total_order_value`, `freight_ratio`, `item_count`, time-based features).
- **Review score handling**: Raw `review_score` is integers 1–5 only. For multiple reviews per order, the **latest review** (by `review_creation_date`) is kept; no averaging. Reports generate only Score 1–5 with counts and percentages.
- **Outputs**: `notebooks/eda_results.txt` (full quantitative report), `notebooks/eda_insights.txt` (structured ML insights, class imbalance, candidate problems).

---

## 2. Feature Preparation for Review Score

- **src/feature_preparation_review.py**: Leakage-safe feature pipeline (delivery-time only; no review text or timestamps). Defines initial feature set, quantile buckets, **historical aggregates** (seller/category review stats from past data only), interaction features, correlation diagnostics, and feature removal. Exports `prepare_splits_from_config()` for train/evaluate.
- **Outputs**: `notebooks/review_feature_config.json`, `notebooks/review_feature_preparation_report.txt`.

---

## 3. Training Pipeline

- **src/train.py**: Time-based split (70% train / 15% val / 15% test) by `order_purchase_timestamp` to avoid future leakage. Trains (A) multinomial logistic regression (baseline) and (B) XGBoost (or RandomForest) with light tuning; selects best by validation macro-F1. Saves model, preprocessor, config, and experiment metadata to `experiments/<run_id>/`.

---

## 4. Evaluation Pipeline

- **src/evaluate.py**: Loads a run, computes accuracy, macro-F1, weighted-F1, per-class P/R/F1, confusion matrix, macro ROC-AUC (OvR). Performs error analysis (confusion patterns, segments, failure modes), writes limitations and learnings. Saves reports under the same `experiments/<run_id>/`.

---

## 5. FastAPI Prediction Service

- **src/serve.py**: Loads model and preprocessor from `experiments/` (or `EXPERIMENT_DIR`). **POST /predict** with Pydantic-validated request; returns `predicted_review_score`, `class_probabilities`, and optional **SHAP** top-3 features for tree models. Handles 422/500/503 appropriately.

---

## 6. Docker

- **Dockerfile**: Multi-stage build with `python:3.10-slim`; copies `src/` and `experiments/final_model/`; sets `EXPERIMENT_DIR=/app/experiments/final_model`; runs `uvicorn src.serve:app --host 0.0.0.0 --port 8000`.
- **.dockerignore**: Excludes other experiment runs and dev artifacts; only `experiments/final_model/` is included.

---

## 7. Final Model Directory

- **experiments/final_model/**: Created and populated with the best run’s artifacts (`model.joblib`, `preprocessor.joblib`, `feature_config.json`) from run `20260226_175329` (XGBoost) for Docker and production use.

---

## 8. Project Structure (Result)

```
ASSIGN_ML/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── serve.py
│   ├── feature_preparation_review.py
│   ├── data.py
│   └── ...
├── experiments/
│   ├── final_model/          # model.joblib, preprocessor.joblib, feature_config.json
│   └── <run_id>/             # per-run artifacts and reports
├── notebooks/
│   ├── eda.py
│   ├── eda_results.txt
│   ├── eda_insights.txt
│   └── review_feature_config.json
├── ai_chat_logs/             # This summary and chat-related logs
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

*Session summary generated for reference. Raw chat transcripts may be stored in the IDE’s agent-transcripts folder.*
