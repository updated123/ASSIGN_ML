"""
Training pipeline for delivery duration regression.

Target: delivery_duration_days (or log1p for training when target_log_transform=True).
Time-based split 70/15/15; preprocessing fit on train only; no future leakage.
Models: (A) Ridge regression, (B) XGBoost/RandomForest; best by validation MAE.
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from src.feature_preparation_delivery_duration import (
        prepare_splits_delivery_duration,
        TARGET,
        PROJECT_ROOT,
    )
except ImportError:
    from feature_preparation_delivery_duration import (
        prepare_splits_delivery_duration,
        TARGET,
        PROJECT_ROOT,
    )

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIG_PATH = PROJECT_ROOT / "notebooks" / "delivery_duration_feature_config.json"
RANDOM_STATE = 42

# Time-based split: no future data in train/val
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
TARGET_LOG = "target_log"  # log1p(delivery_duration_days)


def _mape(y_true, y_pred, epsilon=1e-8):
    """Mean Absolute Percentage Error; clip pred to avoid div by zero."""
    y_pred = np.clip(y_pred, epsilon, None)
    return np.mean(np.abs((y_true - y_pred) / y_pred)) * 100


def build_preprocessor(config, train_df, target_cap_p99=None):
    """Fit scaler on numeric and one-hot encoder on categorical (train only). Optionally store target_cap_p99 for winsorization."""
    numeric = [c for c in config["numeric"] if c in train_df.columns]
    categorical = [c for c in config["categorical"] if c in train_df.columns]
    Xn = train_df[numeric].fillna(0)
    scaler = StandardScaler()
    scaler.fit(Xn)
    Xc = train_df[categorical].fillna("__missing__").astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(Xc)
    out = {
        "scaler": scaler,
        "encoder": encoder,
        "numeric": numeric,
        "categorical": categorical,
    }
    if target_cap_p99 is not None:
        out["target_cap_p99"] = float(target_cap_p99)
    return out


def transform_to_matrix(df, preprocessor):
    """Transform DataFrame to numeric matrix (same order as training)."""
    num = preprocessor["numeric"]
    cat = preprocessor["categorical"]
    Xn = df[num].fillna(0)
    Xn_scaled = preprocessor["scaler"].transform(Xn)
    Xc = df[cat].fillna("__missing__").astype(str)
    Xc_enc = preprocessor["encoder"].transform(Xc)
    return np.hstack([Xn_scaled, Xc_enc])


def compute_metrics(y_true, y_pred):
    """MAE, RMSE, MAPE, R² (all in original scale)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = _mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "r2": float(r2)}


def train_baseline_ridge(X_train, y_train, X_val, y_val):
    """(A) Ridge regression on scaled features; capture coefficients."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        m.fit(X_train, y_train)
    return m


def train_tree_regressor(X_train, y_train, X_val, y_val):
    """(B) Tree regressor with stronger regularization to reduce overfitting. Returns (model, name, hyperparams)."""
    best_model = None
    best_val_mae = np.inf
    best_name = ""
    best_hp = {}
    try:
        import xgboost as xgb
        # Reduced max_depth (4, 5), add reg_alpha/reg_lambda to compare val vs test gap
        for n_est, depth in [(150, 4), (200, 5), (150, 5)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = xgb.XGBRegressor(
                    n_estimators=n_est, max_depth=depth, learning_rate=0.05,
                    reg_alpha=0.1, reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                )
                m.fit(X_train, y_train)
            val_pred = m.predict(X_val)
            mae = mean_absolute_error(y_val, val_pred)
            if mae < best_val_mae:
                best_val_mae = mae
                best_model = m
                best_name = "xgboost"
                best_hp = {"n_estimators": n_est, "max_depth": depth, "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 1.0}
    except ImportError:
        pass
    if best_model is None:
        from sklearn.ensemble import RandomForestRegressor
        for n_est in [100, 150]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = RandomForestRegressor(n_estimators=n_est, max_depth=6, random_state=RANDOM_STATE)
                m.fit(X_train, y_train)
            val_pred = m.predict(X_val)
            mae = mean_absolute_error(y_val, val_pred)
            if mae < best_val_mae:
                best_val_mae = mae
                best_model = m
                best_name = "random_forest"
                best_hp = {"n_estimators": n_est, "max_depth": 6}
    return best_model, best_name, best_hp


def run_training(config_path=None):
    """
    Load config and splits; fit preprocessor on train; apply log1p to target after split if config says so.
    Train Ridge and tree model; pick best by validation MAE. Save model, preprocessor, experiment.json.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_path or CONFIG_PATH
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}. Run: python -m src.feature_preparation_delivery_duration")

    train_df, val_df, test_df, config = prepare_splits_delivery_duration(
        config_path=config_path,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
    )

    use_log_target = config.get("target_log_transform", True)
    winsorize_p99 = config.get("target_winsorize_p99", False)
    y_train_raw = train_df[TARGET].values.copy()
    y_val_raw = val_df[TARGET].values.copy()
    y_test_raw = test_df[TARGET].values.copy()
    target_cap_p99 = None
    if winsorize_p99:
        target_cap_p99 = float(np.percentile(y_train_raw, 99))
        y_train_raw = np.minimum(y_train_raw, target_cap_p99)
        y_val_raw = np.minimum(y_val_raw, target_cap_p99)
        y_test_raw = np.minimum(y_test_raw, target_cap_p99)
    if use_log_target:
        y_train = np.log1p(y_train_raw)
        y_val = np.log1p(y_val_raw)
        y_test = np.log1p(y_test_raw)
    else:
        y_train = y_train_raw
        y_val = y_val_raw
        y_test = y_test_raw

    preprocessor = build_preprocessor(config, train_df, target_cap_p99=target_cap_p99)
    X_train = transform_to_matrix(train_df, preprocessor)
    X_val = transform_to_matrix(val_df, preprocessor)
    X_test = transform_to_matrix(test_df, preprocessor)

    # (A) Ridge
    ridge = train_baseline_ridge(X_train, y_train, X_val, y_val)
    ridge_val_pred_log = ridge.predict(X_val)
    ridge_test_pred_log = ridge.predict(X_test)
    if use_log_target:
        ridge_val_pred = np.expm1(ridge_val_pred_log)
        ridge_test_pred = np.expm1(ridge_test_pred_log)
    else:
        ridge_val_pred = ridge_val_pred_log
        ridge_test_pred = ridge_test_pred_log
    ridge_val_metrics = compute_metrics(y_val_raw, ridge_val_pred)
    ridge_test_metrics = compute_metrics(y_test_raw, ridge_test_pred)

    # Ridge on raw target (log vs raw comparison)
    ridge_raw = train_baseline_ridge(X_train, y_train_raw, X_val, y_val_raw)
    ridge_raw_val_pred = ridge_raw.predict(X_val)
    ridge_raw_test_pred = ridge_raw.predict(X_test)
    ridge_raw_val_metrics = compute_metrics(y_val_raw, ridge_raw_val_pred)
    ridge_raw_test_metrics = compute_metrics(y_test_raw, ridge_raw_test_pred)

    # (B) Tree
    tree_model, tree_name, tree_hyperparams = train_tree_regressor(X_train, y_train, X_val, y_val)
    tree_val_pred_log = tree_model.predict(X_val)
    tree_test_pred_log = tree_model.predict(X_test)
    if use_log_target:
        tree_val_pred = np.expm1(tree_val_pred_log)
        tree_test_pred = np.expm1(tree_test_pred_log)
    else:
        tree_val_pred = tree_val_pred_log
        tree_test_pred = tree_test_pred_log
    tree_val_metrics = compute_metrics(y_val_raw, tree_val_pred)
    tree_test_metrics = compute_metrics(y_test_raw, tree_test_pred)

    # Best by validation MAE
    if tree_val_metrics["mae"] <= ridge_val_metrics["mae"]:
        best_model = tree_model
        best_name = tree_name
        best_val_metrics = tree_val_metrics
        best_test_metrics = tree_test_metrics
    else:
        best_model = ridge
        best_name = "ridge"
        best_val_metrics = ridge_val_metrics
        best_test_metrics = ridge_test_metrics

    # Coefficients / feature importance for interpretability
    ridge_coef = None
    feature_importance = None
    if best_name == "ridge" and hasattr(best_model, "coef_"):
        ridge_coef = best_model.coef_.tolist()
    elif hasattr(best_model, "feature_importances_"):
        # Tree: build feature names (numeric + one-hot categories)
        num_names = preprocessor["numeric"]
        cat_names = list(preprocessor["encoder"].get_feature_names_out(preprocessor["categorical"]))
        feat_names = num_names + cat_names
        feature_importance = dict(zip(feat_names, best_model.feature_importances_.tolist()))

    # Validation vs test gap (overfitting indicator)
    val_test_gap_mae = best_test_metrics["mae"] - best_val_metrics["mae"]
    val_test_gap_r2 = best_test_metrics["r2"] - best_val_metrics["r2"]

    experiment = {
        "run_id": run_id,
        "model_type": best_name,
        "hyperparameters": {"alpha": 1.0} if best_name == "ridge" else tree_hyperparams,
        "target": TARGET,
        "target_log_transform": use_log_target,
        "target_winsorize_p99": winsorize_p99,
        "target_cap_p99": target_cap_p99,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "feature_set": {"numeric": config["numeric"], "categorical": config["categorical"]},
        "validation_metrics": best_val_metrics,
        "test_metrics": best_test_metrics,
        "val_test_gap_mae": float(val_test_gap_mae),
        "val_test_gap_r2": float(val_test_gap_r2),
        "ridge_val_metrics": ridge_val_metrics,
        "ridge_test_metrics": ridge_test_metrics,
        "ridge_raw_val_metrics": ridge_raw_val_metrics,
        "ridge_raw_test_metrics": ridge_raw_test_metrics,
        "log_vs_raw_comparison": {
            "ridge_log_val_mae": ridge_val_metrics["mae"], "ridge_log_test_mae": ridge_test_metrics["mae"],
            "ridge_raw_val_mae": ridge_raw_val_metrics["mae"], "ridge_raw_test_mae": ridge_raw_test_metrics["mae"],
        },
        "tree_val_metrics": tree_val_metrics,
        "tree_test_metrics": tree_test_metrics,
        "notes": "Best model by validation MAE. Tree: stronger reg (max_depth 4-5, reg_alpha/lambda). Log vs raw: Ridge comparison.",
        "ridge_coefficients": ridge_coef,
        "feature_importance": feature_importance,
    }

    joblib.dump(best_model, run_dir / "model.joblib")
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
    with open(run_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(run_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2)

    with open(run_dir / "metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Run: {run_id}\nModel: {best_name}\nTarget transform: log1p={use_log_target}\n")
        f.write(f"Val  MAE={best_val_metrics['mae']:.4f} RMSE={best_val_metrics['rmse']:.4f} R2={best_val_metrics['r2']:.4f}\n")
        f.write(f"Test MAE={best_test_metrics['mae']:.4f} RMSE={best_test_metrics['rmse']:.4f} R2={best_test_metrics['r2']:.4f}\n")

    print(f"Run {run_id} | Best: {best_name} | log_target={use_log_target}")
    print(f"  Val  MAE={best_val_metrics['mae']:.4f} RMSE={best_val_metrics['rmse']:.4f} MAPE={best_val_metrics['mape']:.2f}% R2={best_val_metrics['r2']:.4f}")
    print(f"  Test MAE={best_test_metrics['mae']:.4f} RMSE={best_test_metrics['rmse']:.4f} MAPE={best_test_metrics['mape']:.2f}% R2={best_test_metrics['r2']:.4f}")
    print(f"  Artifacts: {run_dir}")
    return run_id, experiment, best_model, preprocessor


if __name__ == "__main__":
    run_training()
