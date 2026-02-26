"""
Training pipeline for multi-class review score prediction.
Target: review_score ∈ {1, 2, 3, 4, 5}.

Uses finalized feature configuration (numeric, categorical, quantile buckets,
historical aggregates). Strict time-based train/val/test split.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

try:
    from src.feature_preparation_review import (
        prepare_splits_from_config,
        TARGET,
        PROJECT_ROOT,
    )
except ImportError:
    from feature_preparation_review import (
        prepare_splits_from_config,
        TARGET,
        PROJECT_ROOT,
    )

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIG_PATH = PROJECT_ROOT / "notebooks" / "review_feature_config.json"

# ---------------------------------------------------------------------------
# Train / Validation / Test methodology (time-based)
# ---------------------------------------------------------------------------
# Why temporal splitting prevents future leakage:
#   At inference we only have data up to "now". If we trained with random split,
#   the model could see future orders (by order_purchase_timestamp) in training,
#   learning period-specific patterns that won't exist when we predict later.
#   Time-based split ensures we train only on the past and validate/test on the
#   future, so no information from the future is ever used for training.
#
# Why random splitting would inflate performance:
#   With random split, validation/test orders can be from the same time window
#   as training. Seasonal and temporal patterns (e.g. holiday behavior, seller
#   performance in the same period) would be shared, inflating metrics. In
#   production we predict truly future orders, so temporal split gives a
#   realistic estimate.
#
# How this simulates production rollout:
#   We deploy at time T. Train = orders before T (e.g. 70%), val = next 15%,
#   test = final 15%. Validation tunes hyperparameters; test is held out once
#   for final reporting. In production we'd retrain on all data up to T and
#   predict orders after T, which matches the test setup.
# ---------------------------------------------------------------------------

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
CLASSES = [1, 2, 3, 4, 5]


class _LabelEncodedModel:
    """Wraps XGBoost model trained on 0-4 labels; decode predictions to 1-5."""
    def __init__(self, model, label_encoder):
        self.model = model
        self.le = label_encoder
        self.classes_ = np.array(self.le.classes_)  # 1,2,3,4,5

    def predict(self, X):
        return self.le.inverse_transform(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)  # columns in order 0,1,2,3,4 -> 1,2,3,4,5


def time_based_split(df, time_col="order_purchase_timestamp"):
    """Split by order_purchase_timestamp: Train 70%, Val 15%, Test 15%."""
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    t1 = int(n * TRAIN_FRAC)
    t2 = int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


def build_preprocessor(config, train_df):
    """
    Fit preprocessor only on training data (no val/test).
    - Numeric: StandardScaler
    - Categorical one-hot: OneHotEncoder(handle_unknown='ignore')
    - primary_seller_id: ordinal map from train frequency order
    """
    numeric = [c for c in config["numeric"] if c in train_df.columns]
    cat_onehot = [c for c in config["categorical_onehot"] if c in train_df.columns]
    cat_ordinal = list(config.get("categorical_ordinal", []))

    Xn = train_df[numeric].fillna(0)
    scaler = StandardScaler()
    scaler.fit(Xn)

    Xc = train_df[cat_onehot].fillna("__missing__").astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(Xc)

    ordinal_map = {}
    if "primary_seller_id" in cat_ordinal and "primary_seller_id" in train_df.columns:
        seller_order = train_df["primary_seller_id"].value_counts().index.tolist()
        ordinal_map = {s: i for i, s in enumerate(seller_order)}

    return {
        "scaler": scaler,
        "encoder": encoder,
        "numeric": numeric,
        "categorical_onehot": cat_onehot,
        "categorical_ordinal": cat_ordinal,
        "ordinal_map": ordinal_map,
        "n_ordinal": len(ordinal_map),
    }


def transform_to_matrix(df, preprocessor):
    """Transform DataFrame to numeric matrix using fitted preprocessor."""
    num = preprocessor["numeric"]
    cat_onehot = preprocessor["categorical_onehot"]
    cat_ordinal = preprocessor["categorical_ordinal"]
    ordinal_map = preprocessor["ordinal_map"]
    n_ord = preprocessor["n_ordinal"]

    Xn = df[num].fillna(0)
    Xn_scaled = preprocessor["scaler"].transform(Xn)

    Xc = df[cat_onehot].fillna("__missing__").astype(str)
    Xc_enc = preprocessor["encoder"].transform(Xc)

    seller_ord = np.full((len(df), 1), n_ord)
    if "primary_seller_id" in df.columns and ordinal_map:
        seller_ord = df["primary_seller_id"].map(ordinal_map).fillna(n_ord).astype(int).values.reshape(-1, 1)

    return np.hstack([Xn_scaled, Xc_enc, seller_ord])


def train_baseline_lr(X_train, y_train, X_val, y_val):
    """
    (A) Baseline: Multinomial Logistic Regression.
    - Proper scaling (caller provides already-scaled X).
    - class_weight="balanced" for imbalance.
    - Returns model and per-class coefficients for interpretation.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(
            max_iter=2000,
            random_state=42,
            multi_class="multinomial",
            class_weight="balanced",
        )
        m.fit(X_train, y_train)
    return m


def train_stronger_model(X_train, y_train, X_val, y_val):
    """
    (B) Stronger: Gradient boosting (XGBoost or LightGBM) with multi-class objective.
    Handles imbalance via sample_weight (balanced). XGBoost expects labels 0..n_class-1,
    so we encode 1-5 -> 0-4 and store encoder to map predictions back.
    """
    from sklearn.utils.class_weight import compute_class_weight
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)  # 1,2,3,4,5 -> 0,1,2,3,4
    y_val_enc = le.transform(y_val)
    cw = compute_class_weight(
        "balanced",
        classes=np.unique(y_train_enc),
        y=y_train_enc,
    )
    sample_weight = np.array([cw[list(np.unique(y_train_enc)).index(y)] for y in y_train_enc])

    best_model = None
    best_val_macro_f1 = -1
    best_name = ""
    best_le = None

    best_hyperparams = {}
    try:
        import xgboost as xgb
        for n_est, depth in [(150, 6), (200, 8), (250, 6)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = xgb.XGBClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    learning_rate=0.05,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    objective="multi:softprob",
                    num_class=5,
                )
                m.fit(X_train, y_train_enc, sample_weight=sample_weight, verbose=False)
            y_val_pred_enc = m.predict(X_val)
            y_val_pred = le.inverse_transform(y_val_pred_enc)
            macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
            if macro > best_val_macro_f1:
                best_val_macro_f1 = macro
                best_model = m
                best_name = "xgboost"
                best_le = le
                best_hyperparams = {"n_estimators": n_est, "max_depth": depth, "learning_rate": 0.05}
    except ImportError:
        pass

    if best_model is None:
        from sklearn.ensemble import RandomForestClassifier
        for n_est in [150, 200]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=42,
                )
                m.fit(X_train, y_train)
            y_val_pred = m.predict(X_val)
            macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
            if macro > best_val_macro_f1:
                best_val_macro_f1 = macro
                best_model = m
                best_name = "random_forest"
                best_le = None
                best_hyperparams = {"n_estimators": n_est, "max_depth": 12}

    if best_le is not None:
        best_model = _LabelEncodedModel(best_model, best_le)
    return best_model, best_name, best_hyperparams


def run_training(config_path=None):
    """
    Full training pipeline:
    - Load config and prepare train/val/test (time-based, leakage-safe features).
    - Fit preprocessor on train only.
    - Train baseline LR and stronger model; compare on validation macro-F1.
    - Save best model and artifacts to experiments/<run_id>.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_path or CONFIG_PATH
    if not Path(config_path).exists():
        raise FileNotFoundError(
            f"Feature config not found: {config_path}. Run: python -m src.feature_preparation_review"
        )

    train_df, val_df, test_df, config = prepare_splits_from_config(
        config_path=config_path,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
    )

    preprocessor = build_preprocessor(config, train_df)
    X_train = transform_to_matrix(train_df, preprocessor)
    X_val = transform_to_matrix(val_df, preprocessor)
    X_test = transform_to_matrix(test_df, preprocessor)
    y_train = train_df[TARGET].values
    y_val = val_df[TARGET].values
    y_test = test_df[TARGET].values

    # (A) Baseline LR
    lr = train_baseline_lr(X_train, y_train, X_val, y_val)
    lr_val_pred = lr.predict(X_val)
    lr_val_macro = f1_score(y_val, lr_val_pred, average="macro", zero_division=0)
    lr_val_weighted = f1_score(y_val, lr_val_pred, average="weighted", zero_division=0)
    lr_val_acc = accuracy_score(y_val, lr_val_pred)

    # (B) Stronger model
    stronger, stronger_name, stronger_hp = train_stronger_model(X_train, y_train, X_val, y_val)
    strong_val_pred = stronger.predict(X_val)
    strong_val_macro = f1_score(y_val, strong_val_pred, average="macro", zero_division=0)
    strong_val_weighted = f1_score(y_val, strong_val_pred, average="weighted", zero_division=0)
    strong_val_acc = accuracy_score(y_val, strong_val_pred)

    # Pick best by validation macro-F1 (primary metric for imbalanced multi-class)
    if strong_val_macro >= lr_val_macro:
        best_model = stronger
        best_name = stronger_name
        best_val_macro = strong_val_macro
        best_val_weighted = strong_val_weighted
        best_val_acc = strong_val_acc
        best_hyperparams = stronger_hp
    else:
        best_model = lr
        best_name = "logistic_regression"
        best_val_macro = lr_val_macro
        best_val_weighted = lr_val_weighted
        best_val_acc = lr_val_acc
        best_hyperparams = {"class_weight": "balanced", "multi_class": "multinomial"}

    # Test set metrics (for experiment log; full evaluation in evaluate.py)
    y_test_pred = best_model.predict(X_test)
    test_macro = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    test_weighted = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Per-class coefficients for LR (for interpretation)
    lr_coef_per_class = None
    if hasattr(best_model, "coef_") and best_name == "logistic_regression":
        lr_coef_per_class = {
            int(c): [float(x) for x in best_model.coef_[i]]
            for i, c in enumerate(best_model.classes_)
        }

    experiment = {
        "run_id": run_id,
        "model_type": best_name,
        "target": TARGET,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "hyperparameters": best_hyperparams,
        "feature_set": {
            "numeric": config["numeric"],
            "categorical_onehot": config["categorical_onehot"],
            "categorical_ordinal": config.get("categorical_ordinal", []),
        },
        "metrics": {
            "val_accuracy": float(best_val_acc),
            "val_macro_f1": float(best_val_macro),
            "val_weighted_f1": float(best_val_weighted),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_macro),
            "test_weighted_f1": float(test_weighted),
        },
        "notes": (
            "Baseline LR vs stronger model; best selected by validation macro-F1. "
            "Class distribution: 5 dominant (~58%); macro-F1 prioritizes rare classes."
        ),
        "lr_coef_per_class": lr_coef_per_class,
    }

    joblib.dump(best_model, run_dir / "model.joblib")
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
    with open(run_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(run_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2)

    # Brief metrics summary for quick view
    with open(run_dir / "metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Run: {run_id}\nModel: {best_name}\n")
        f.write(f"Val  - Accuracy: {best_val_acc:.4f}  Macro-F1: {best_val_macro:.4f}  Weighted-F1: {best_val_weighted:.4f}\n")
        f.write(f"Test - Accuracy: {test_acc:.4f}  Macro-F1: {test_macro:.4f}  Weighted-F1: {test_weighted:.4f}\n")

    print(f"Run {run_id} | Best: {best_name}")
    print(f"  Val  Macro-F1: {best_val_macro:.4f}  Weighted-F1: {best_val_weighted:.4f}  Acc: {best_val_acc:.4f}")
    print(f"  Test Macro-F1: {test_macro:.4f}  Weighted-F1: {test_weighted:.4f}  Acc: {test_acc:.4f}")
    print(f"  Artifacts: {run_dir}")
    return run_id, experiment, best_model, preprocessor


if __name__ == "__main__":
    run_id, exp, _, _ = run_training()
    print("Training complete. Run: python -m src.evaluate")
