"""
Production FastAPI service for multi-class review score prediction.

Loads model and preprocessor from experiments/ at startup (model.joblib / model.pkl,
preprocessor.joblib / preprocessor.pkl). Exposes POST /predict with Pydantic validation,
optional SHAP explainability for tree-based models, and structured error handling.

Assumptions:
- Experiment dir is chosen via env EXPERIMENT_DIR or latest run in experiments/.
- Request payload may omit historical aggregates and primary_seller_id; defaults applied.
- late_freight_ratio is derived as late_delivery_flag * freight_ratio (no leakage).
"""

from pathlib import Path
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Load .joblib by default (train.py saves these); fallback to .pkl
try:
    import joblib
    def _load_artifact(path: Path):
        return joblib.load(path)
except ImportError:
    import pickle
    def _load_artifact(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# So joblib can unpickle _LabelEncodedModel saved by train.py
try:
    from src.train import _LabelEncodedModel  # noqa: F401
except ImportError:
    from train import _LabelEncodedModel  # noqa: F401

# ---------------------------------------------------------------------------
# App and global model state (loaded once at startup)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Review Score Prediction API",
    description="Multi-class review score (1-5) prediction with optional SHAP explanations.",
    version="1.0.0",
)

_model = None
_preprocessor = None
_config = None
_feature_names_out = None  # same order as transform_to_matrix output
_shap_explainer = None
_shap_model_raw = None  # underlying tree model for SHAP (unwrap _LabelEncodedModel if needed)


def _get_experiment_dir() -> Path:
    """Resolve experiment directory: env EXPERIMENT_DIR or latest run with model."""
    if os.environ.get("EXPERIMENT_DIR"):
        p = Path(os.environ["EXPERIMENT_DIR"])
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p
    if not EXPERIMENTS_DIR.exists():
        raise FileNotFoundError("experiments/ not found; run training first.")
    dirs = sorted(EXPERIMENTS_DIR.glob("*"), key=lambda x: x.name, reverse=True)
    for d in dirs:
        if not d.is_dir():
            continue
        if (d / "model.joblib").exists() or (d / "model.pkl").exists():
            return d
    raise FileNotFoundError("No experiment with model.joblib or model.pkl found.")


def _load_model_and_preprocessor():
    """Load model, preprocessor, config; build feature name list for SHAP."""
    global _model, _preprocessor, _config, _feature_names_out, _shap_explainer, _shap_model_raw
    exp_dir = _get_experiment_dir()
    model_path = exp_dir / "model.joblib" if (exp_dir / "model.joblib").exists() else exp_dir / "model.pkl"
    prep_path = exp_dir / "preprocessor.joblib" if (exp_dir / "preprocessor.joblib").exists() else exp_dir / "preprocessor.pkl"
    config_path = exp_dir / "feature_config.json"
    if not model_path.exists() or not prep_path.exists():
        raise FileNotFoundError(f"Missing {model_path.name} or {prep_path.name} in {exp_dir}")
    _model = _load_artifact(model_path)
    _preprocessor = _load_artifact(prep_path)
    if config_path.exists():
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            _config = json.load(f)
    else:
        _config = {
            "numeric": getattr(_preprocessor, "numeric", []),
            "categorical_onehot": getattr(_preprocessor, "categorical_onehot", []),
            "categorical_ordinal": getattr(_preprocessor, "categorical_ordinal", []),
        }
    # Feature names in same order as transform_to_matrix: numeric + one-hot + ordinal
    num = _preprocessor.get("numeric", _config.get("numeric", []))
    cat_onehot = _preprocessor.get("categorical_onehot", _config.get("categorical_onehot", []))
    enc = _preprocessor.get("encoder")
    if enc is not None and hasattr(enc, "get_feature_names_out"):
        cat_names = list(enc.get_feature_names_out(cat_onehot))
    else:
        cat_names = [f"cat_{i}" for i in range(len(cat_onehot))]
    ordinal = _preprocessor.get("categorical_ordinal", [])
    _feature_names_out = list(num) + list(cat_names) + list(ordinal)
    # SHAP: only for tree-based (unwrap wrapper if present)
    _shap_model_raw = getattr(_model, "model", _model)
    if _is_tree_model(_shap_model_raw):
        try:
            import shap
            # One-row background (zeros); same feature count as transform output. No target leakage.
            n_features = len(_feature_names_out)
            _shap_explainer = shap.TreeExplainer(
                _shap_model_raw,
                np.zeros((1, n_features), dtype=np.float32),
            )
        except Exception:
            _shap_explainer = None
    else:
        _shap_explainer = None


def _is_tree_model(m) -> bool:
    """True if model is tree-based (XGBoost, RF, etc.) for SHAP TreeExplainer."""
    if m is None:
        return False
    cls = type(m).__name__
    return "XGB" in cls or "RandomForest" in cls or "GradientBoost" in cls or "LGBM" in cls


@app.on_event("startup")
def startup():
    """Load model and preprocessor once at startup."""
    try:
        _load_model_and_preprocessor()
    except Exception as e:
        # Log and re-raise so app fails fast
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Startup failed: {e}") from e


# ---------------------------------------------------------------------------
# Request schema (Pydantic)
# ---------------------------------------------------------------------------
# Bucket values from our pipeline: order_value_bucket e.g. "val_q0","val_q1",...
# freight_ratio_bucket "fr_low","fr_mid","fr_high"; delivery_delay_bucket "dd_q0",...
# User example used "mid","low","early" - we accept both style and validate loosely.
class PredictRequest(BaseModel):
    total_order_value: float = Field(..., ge=0, le=1e7, description="Order total value")
    freight_ratio: float = Field(..., ge=0, le=2, description="Freight / order value ratio")
    item_count: int = Field(..., ge=0, le=100)
    approval_delay_hours: float = Field(..., ge=0, le=720)
    delivery_delay_days: float = Field(..., description="Negative = early")
    late_delivery_flag: int = Field(..., ge=0, le=1)
    primary_category: str = Field(..., min_length=1, max_length=128)
    seller_state: str = Field(..., min_length=1, max_length=8)
    customer_state: str = Field(..., min_length=1, max_length=8)
    payment_type_primary: str = Field(..., min_length=1, max_length=32)
    order_value_bucket: str = Field(..., min_length=1, max_length=32)
    freight_ratio_bucket: str = Field(..., min_length=1, max_length=32)
    delivery_delay_bucket: str = Field(..., min_length=1, max_length=32)
    # Optional: historical and seller id (defaults avoid missing columns)
    primary_seller_id: Optional[str] = Field(None, max_length=64)
    seller_hist_avg_review: Optional[float] = Field(None, ge=0, le=5)
    seller_hist_1star_rate: Optional[float] = Field(None, ge=0, le=1)
    seller_hist_5star_rate: Optional[float] = Field(None, ge=0, le=1)
    category_hist_avg_review: Optional[float] = Field(None, ge=0, le=5)
    category_hist_dissatisfaction_rate: Optional[float] = Field(None, ge=0, le=1)

    @field_validator("primary_category", "seller_state", "customer_state", "payment_type_primary", "order_value_bucket", "freight_ratio_bucket", "delivery_delay_bucket", mode="before")
    @classmethod
    def strip_str(cls, v):
        if isinstance(v, str):
            return v.strip() or "__missing__"
        return v


# ---------------------------------------------------------------------------
# Transform single request to matrix (same order as training)
# ---------------------------------------------------------------------------
def _request_to_dataframe(req: PredictRequest) -> pd.DataFrame:
    """Build one-row DataFrame with all features expected by preprocessor."""
    late_freight = req.late_delivery_flag * req.freight_ratio
    # Defaults for optional historical (no leakage; neutral if not provided)
    seller_avg = req.seller_hist_avg_review if req.seller_hist_avg_review is not None else 0.0
    seller_1 = req.seller_hist_1star_rate if req.seller_hist_1star_rate is not None else 0.0
    seller_5 = req.seller_hist_5star_rate if req.seller_hist_5star_rate is not None else 0.0
    cat_avg = req.category_hist_avg_review if req.category_hist_avg_review is not None else 0.0
    cat_dis = req.category_hist_dissatisfaction_rate if req.category_hist_dissatisfaction_rate is not None else 0.0
    row = {
        "total_order_value": req.total_order_value,
        "freight_ratio": req.freight_ratio,
        "item_count": req.item_count,
        "approval_delay_hours": req.approval_delay_hours,
        "delivery_delay_days": req.delivery_delay_days,
        "late_delivery_flag": req.late_delivery_flag,
        "late_freight_ratio": late_freight,
        "seller_hist_avg_review": seller_avg,
        "seller_hist_1star_rate": seller_1,
        "seller_hist_5star_rate": seller_5,
        "category_hist_avg_review": cat_avg,
        "category_hist_dissatisfaction_rate": cat_dis,
        "primary_category": req.primary_category,
        "seller_state": req.seller_state,
        "customer_state": req.customer_state,
        "payment_type_primary": req.payment_type_primary,
        "order_value_bucket": req.order_value_bucket,
        "freight_ratio_bucket": req.freight_ratio_bucket,
        "delivery_delay_bucket": req.delivery_delay_bucket,
        "primary_seller_id": req.primary_seller_id if req.primary_seller_id is not None else "__unknown__",
    }
    return pd.DataFrame([row])


def _transform_to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Apply preprocessor to DataFrame; same logic as train/evaluate."""
    preprocessor = _preprocessor
    config = _config
    num = preprocessor.get("numeric", config.get("numeric", []))
    num = [c for c in num if c in df.columns]
    cat_onehot = preprocessor.get("categorical_onehot", config.get("categorical_onehot", []))
    cat_onehot = [c for c in cat_onehot if c in df.columns]
    ordinal_map = preprocessor.get("ordinal_map", {})
    n_ord = preprocessor.get("n_ordinal", len(ordinal_map))
    Xn = df[num].fillna(0)
    Xn_scaled = preprocessor["scaler"].transform(Xn)
    Xc = df[cat_onehot].fillna("__missing__").astype(str)
    Xc_enc = preprocessor["encoder"].transform(Xc)
    seller_ord = np.full((len(df), 1), n_ord)
    if "primary_seller_id" in df.columns and ordinal_map:
        seller_ord = df["primary_seller_id"].map(ordinal_map).fillna(n_ord).astype(int).values.reshape(-1, 1)
    return np.hstack([Xn_scaled, Xc_enc, seller_ord])


def _top_shap_features(X: np.ndarray, predicted_class_index: int) -> list[dict]:
    """Return top 3 features by |SHAP| for the predicted class. Lightweight, no plotting."""
    if _shap_explainer is None or _feature_names_out is None:
        return []
    try:
        import shap
        # For multi-class, shap_values is list of (n_samples, n_features) per class
        shap_values = _shap_explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[predicted_class_index]
        else:
            sv = shap_values
        if sv.ndim == 3:
            sv = sv[predicted_class_index]
        row = sv[0] if sv.ndim == 2 else sv
        impact_by_idx = [(i, float(row[i])) for i in range(len(row))]
        impact_by_idx.sort(key=lambda x: abs(x[1]), reverse=True)
        names = _feature_names_out
        return [
            {"feature": names[i] if i < len(names) else f"feature_{i}", "impact": imp}
            for i, imp in impact_by_idx[:3]
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class PredictResponse(BaseModel):
    predicted_review_score: int
    class_probabilities: dict[str, float]
    top_shap_features: list[dict[str, float]] = []


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict review score (1-5) from order features.
    Returns predicted class, class probabilities, and top 3 SHAP features (if tree model).
    """
    global _model, _preprocessor
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        df = _request_to_dataframe(request)
        X = _transform_to_matrix(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")
    try:
        pred = _model.predict(X)
        proba = _model.predict_proba(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    # Decode to 1-5 (model may be wrapped _LabelEncodedModel)
    predicted_score = int(pred[0])
    classes = getattr(_model, "classes_", [1, 2, 3, 4, 5])
    if hasattr(classes, "tolist"):
        classes = classes.tolist()
    proba_row = proba[0]
    class_probabilities = {str(c): float(proba_row[i]) for i, c in enumerate(classes)}
    # SHAP: predicted_class_index for tree model (0-4 if encoded, else 1-5)
    predicted_class_index = list(classes).index(predicted_score) if predicted_score in classes else 0
    top_shap_features = _top_shap_features(X, predicted_class_index)
    return PredictResponse(
        predicted_review_score=predicted_score,
        class_probabilities=class_probabilities,
        top_shap_features=top_shap_features,
    )


@app.get("/health")
def health():
    """Health check; confirms model is loaded."""
    return {"status": "ok", "model_loaded": _model is not None}


# ---------------------------------------------------------------------------
# Sample curl (see also README.md)
# ---------------------------------------------------------------------------
# curl -X POST http://localhost:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{
#     "total_order_value": 150,
#     "freight_ratio": 0.21,
#     "item_count": 2,
#     "approval_delay_hours": 5,
#     "delivery_delay_days": -2,
#     "late_delivery_flag": 0,
#     "primary_category": "beleza_saude",
#     "seller_state": "SP",
#     "customer_state": "RJ",
#     "payment_type_primary": "credit_card",
#     "order_value_bucket": "val_q2",
#     "freight_ratio_bucket": "fr_low",
#     "delivery_delay_bucket": "dd_q1"
#   }'
#
# Alternative bucket names (if your training used different labels):
#   "order_value_bucket": "mid", "freight_ratio_bucket": "low", "delivery_delay_bucket": "early"
#
# Run: uvicorn src.serve:app --reload
