"""
Production FastAPI service for delivery duration regression.

Loads model, preprocessor, and feature config at startup (model.joblib / model.pkl,
preprocessor.joblib / preprocessor.pkl). Exposes POST /predict with Pydantic validation,
SHAP TreeExplainer for top-3 feature impact, and structured error handling.

Artifacts: set DELIVERY_EXPERIMENT_DIR to an experiment folder, or uses latest run
in experiments/ whose feature_config.json has target "delivery_duration_days".
"""

from pathlib import Path
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

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

# Reuse transform logic from training (single source of truth)
try:
    from src.train_delivery_duration import transform_to_matrix as _transform_to_matrix_train
except ImportError:
    from train_delivery_duration import transform_to_matrix as _transform_to_matrix_train

# ---------------------------------------------------------------------------
# App and global state (loaded once at startup)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Delivery Duration Prediction API",
    description="Regression API for delivery_duration_days with optional SHAP explainability.",
    version="1.0.0",
)

_model = None
_preprocessor = None
_config = None
_use_log_target = True
_feature_names_out = None
_shap_explainer = None
_shap_model_raw = None


def _is_tree_model(m) -> bool:
    if m is None:
        return False
    cls = type(m).__name__
    return "XGB" in cls or "RandomForest" in cls or "GradientBoost" in cls or "LGBM" in cls


def _get_experiment_dir() -> Path:
    """Resolve delivery experiment dir: env DELIVERY_EXPERIMENT_DIR or latest run with delivery config."""
    if os.environ.get("DELIVERY_EXPERIMENT_DIR"):
        p = Path(os.environ["DELIVERY_EXPERIMENT_DIR"])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
    if not EXPERIMENTS_DIR.exists():
        raise FileNotFoundError("experiments/ not found; run training first.")
    dirs = sorted(EXPERIMENTS_DIR.glob("*"), key=lambda x: x.name, reverse=True)
    for d in dirs:
        if not d.is_dir():
            continue
        cfg_path = d / "feature_config.json"
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("target") == "delivery_duration_days":
                if (d / "model.joblib").exists() or (d / "model.pkl").exists():
                    if (d / "preprocessor.joblib").exists() or (d / "preprocessor.pkl").exists():
                        return d
        except Exception:
            continue
    raise FileNotFoundError("No delivery duration experiment (model + preprocessor + feature_config) found.")


def _load_model_and_preprocessor():
    global _model, _preprocessor, _config, _use_log_target, _feature_names_out, _shap_explainer, _shap_model_raw
    exp_dir = _get_experiment_dir()
    model_path = exp_dir / "model.joblib" if (exp_dir / "model.joblib").exists() else exp_dir / "model.pkl"
    prep_path = exp_dir / "preprocessor.joblib" if (exp_dir / "preprocessor.joblib").exists() else exp_dir / "preprocessor.pkl"
    config_path = exp_dir / "feature_config.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact in {exp_dir}. Expected model.joblib or model.pkl.")
    if not prep_path.exists():
        raise FileNotFoundError(f"Missing preprocessor in {exp_dir}. Expected preprocessor.joblib or preprocessor.pkl.")
    _model = _load_artifact(model_path)
    _preprocessor = _load_artifact(prep_path)
    with open(config_path, "r", encoding="utf-8") as f:
        _config = json.load(f)
    _use_log_target = _config.get("target_log_transform", True)
    num = _preprocessor.get("numeric", _config.get("numeric", []))
    cat = _preprocessor.get("categorical", _config.get("categorical", []))
    enc = _preprocessor.get("encoder")
    if enc is not None and hasattr(enc, "get_feature_names_out"):
        cat_names = list(enc.get_feature_names_out(cat))
    else:
        cat_names = [f"cat_{i}" for i in range(len(cat))]
    _feature_names_out = list(num) + list(cat_names)
    _shap_model_raw = _model
    if _is_tree_model(_shap_model_raw):
        try:
            import shap
            n_features = len(_feature_names_out)
            _shap_explainer = shap.TreeExplainer(
                _shap_model_raw,
                np.zeros((1, n_features), dtype=np.float32),
            )
        except Exception:
            _shap_explainer = None
    else:
        _shap_explainer = None


@app.on_event("startup")
def startup():
    try:
        _load_model_and_preprocessor()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Startup failed: {e}") from e


# ---------------------------------------------------------------------------
# Request schema (Pydantic)
# ---------------------------------------------------------------------------
class PredictDeliveryRequest(BaseModel):
    intra_state: int = Field(..., ge=0, le=1, description="1 = same state, 0 = different state")
    total_freight_value: float = Field(..., ge=0, description="Total freight value")
    freight_ratio: float = Field(..., ge=0, le=2, description="Freight / order value ratio")
    average_weight_per_item: float = Field(..., ge=0, description="Average weight per item (g)")
    avg_item_price: float = Field(..., ge=0, description="Average item price")
    total_weight_g: float = Field(..., ge=0, description="Total weight (g)")
    total_order_value: float = Field(..., ge=0, description="Total order value")
    total_volume_cm3: float = Field(..., ge=0, description="Total volume (cm³)")
    bulky_flag: int = Field(..., ge=0, le=1, description="Bulky order flag")
    item_count: int = Field(..., ge=0, le=100, description="Number of items")
    primary_category: str = Field(..., min_length=1, max_length=128)
    payment_type_primary: str = Field(..., min_length=1, max_length=32)
    seller_state: str = Field(..., min_length=1, max_length=8)
    customer_state: str = Field(..., min_length=1, max_length=8)
    customer_region: str = Field(..., min_length=1, max_length=8)
    seller_region: str = Field(..., min_length=1, max_length=8)
    # Optional: for distance_km_bucket when geo is available
    distance_km: Optional[float] = Field(None, ge=0, description="Distance in km (optional)")

    @field_validator("primary_category", "payment_type_primary", "seller_state", "customer_state", "customer_region", "seller_region", mode="before")
    @classmethod
    def strip_str(cls, v):
        if isinstance(v, str):
            return v.strip() or "__missing__"
        return v


# ---------------------------------------------------------------------------
# Build one-row DataFrame with derived features (same as training pipeline)
# ---------------------------------------------------------------------------
def _request_to_dataframe(req: PredictDeliveryRequest) -> pd.DataFrame:
    region_to_region = f"{req.seller_region}_{req.customer_region}"
    if req.distance_km is not None:
        d = req.distance_km
        if d <= 500:
            distance_km_bucket = "d_0_500"
        elif d <= 1500:
            distance_km_bucket = "d_500_1500"
        else:
            distance_km_bucket = "d_1500_plus"
    else:
        distance_km_bucket = "d_unknown"
    inter_state_x_bulky_flag = (1 - req.intra_state) * req.bulky_flag
    row = {
        "intra_state": req.intra_state,
        "total_freight_value": req.total_freight_value,
        "freight_ratio": req.freight_ratio,
        "average_weight_per_item": req.average_weight_per_item,
        "avg_item_price": req.avg_item_price,
        "total_weight_g": req.total_weight_g,
        "total_order_value": req.total_order_value,
        "total_volume_cm3": req.total_volume_cm3,
        "bulky_flag": req.bulky_flag,
        "item_count": req.item_count,
        "inter_state_x_bulky_flag": inter_state_x_bulky_flag,
        "primary_category": req.primary_category,
        "payment_type_primary": req.payment_type_primary,
        "seller_state": req.seller_state,
        "customer_state": req.customer_state,
        "customer_region": req.customer_region,
        "seller_region": req.seller_region,
        "region_to_region": region_to_region,
        "distance_km_bucket": distance_km_bucket,
    }
    return pd.DataFrame([row])


def _transform_to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Apply preprocessor; same columns and order as training."""
    return _transform_to_matrix_train(df, _preprocessor)


def _top_shap_features(X: np.ndarray, top_k: int = 3) -> list[dict]:
    """Top K features by |SHAP| for regression. Single sample, no leakage."""
    if _shap_explainer is None or _feature_names_out is None:
        return []
    try:
        import shap
        shap_values = _shap_explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values
        row = sv[0] if sv.ndim == 2 else sv
        impact_by_idx = [(i, float(row[i])) for i in range(len(row))]
        impact_by_idx.sort(key=lambda x: abs(x[1]), reverse=True)
        names = _feature_names_out
        out = []
        for i, imp in impact_by_idx[:top_k]:
            name = names[i] if i < len(names) else f"feature_{i}"
            out.append({"feature": name, "impact": round(imp, 2)})
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class PredictDeliveryResponse(BaseModel):
    predicted_delivery_days: float
    top_shap_features: list[dict] = []


@app.post("/predict", response_model=PredictDeliveryResponse)
def predict(request: PredictDeliveryRequest):
    """
    Predict delivery_duration_days from order features.
    Returns predicted days (original scale) and top 3 SHAP feature impacts (if tree model).
    """
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        df = _request_to_dataframe(request)
        # Ensure all columns expected by preprocessor exist
        num = _preprocessor.get("numeric", [])
        cat = _preprocessor.get("categorical", [])
        for c in num + cat:
            if c not in df.columns:
                raise HTTPException(status_code=422, detail=f"Missing field: {c}")
        X = _transform_to_matrix(df)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")
    try:
        pred = _model.predict(X)
        raw_pred = float(pred[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    if _use_log_target:
        predicted_days = np.expm1(raw_pred)
    else:
        predicted_days = raw_pred
    predicted_days = round(max(0.0, predicted_days), 2)
    top_shap_features = _top_shap_features(X, top_k=3)
    return PredictDeliveryResponse(
        predicted_delivery_days=predicted_days,
        top_shap_features=top_shap_features,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}
