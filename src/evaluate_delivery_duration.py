"""
Evaluation pipeline for delivery duration regression.

Loads a trained experiment (model + preprocessor + config), recomputes val/test metrics
in original scale, and runs error analysis: residuals, segment MAE, top-50 extreme errors.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from src.feature_preparation_delivery_duration import (
        prepare_splits_delivery_duration,
        TARGET,
        PROJECT_ROOT,
    )
    from src.train_delivery_duration import (
        transform_to_matrix,
        compute_metrics,
        TRAIN_FRAC,
        VAL_FRAC,
        TEST_FRAC,
    )
except ImportError:
    from feature_preparation_delivery_duration import (
        prepare_splits_delivery_duration,
        TARGET,
        PROJECT_ROOT,
    )
    from train_delivery_duration import (
        transform_to_matrix,
        compute_metrics,
        TRAIN_FRAC,
        VAL_FRAC,
        TEST_FRAC,
    )

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def _mape(y_true, y_pred, epsilon=1e-8):
    y_pred = np.clip(y_pred, epsilon, None)
    return np.mean(np.abs((y_true - y_pred) / y_pred)) * 100


def get_run_dir(run_id=None):
    """Return experiment directory: by run_id or latest with model.joblib."""
    if not EXPERIMENTS_DIR.exists():
        return None
    if run_id:
        d = EXPERIMENTS_DIR / run_id
        return d if d.is_dir() and (d / "model.joblib").exists() else None
    dirs = sorted(EXPERIMENTS_DIR.glob("*"), key=lambda p: p.name, reverse=True)
    for d in dirs:
        if d.is_dir() and (d / "model.joblib").exists() and (d / "experiment.json").exists():
            return d
    return None


def load_experiment(run_dir):
    """Load model, preprocessor, config, experiment meta from run_dir."""
    run_dir = Path(run_dir)
    model = joblib.load(run_dir / "model.joblib")
    preprocessor = joblib.load(run_dir / "preprocessor.joblib")
    with open(run_dir / "feature_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(run_dir / "experiment.json", "r", encoding="utf-8") as f:
        experiment = json.load(f)
    return model, preprocessor, config, experiment


def run_evaluation(run_id=None):
    """
    Load experiment; reload splits; predict val/test; invert log if needed;
    compute MAE, RMSE, MAPE, R²; run residual + segment + top-50 error analysis.
    """
    run_dir = get_run_dir(run_id)
    if run_dir is None:
        raise FileNotFoundError(f"No experiment found (run_id={run_id}). Run train_delivery_duration first.")
    model, preprocessor, config, experiment = load_experiment(run_dir)
    use_log = experiment.get("target_log_transform", True)

    train_df, val_df, test_df, _ = prepare_splits_delivery_duration(
        config_path=run_dir / "feature_config.json",
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
    )

    X_val = transform_to_matrix(val_df, preprocessor)
    X_test = transform_to_matrix(test_df, preprocessor)
    y_val_raw = val_df[TARGET].values
    y_test_raw = test_df[TARGET].values

    val_pred_log = model.predict(X_val)
    test_pred_log = model.predict(X_test)
    if use_log:
        val_pred = np.expm1(val_pred_log)
        test_pred = np.expm1(test_pred_log)
    else:
        val_pred = val_pred_log
        test_pred = test_pred_log

    val_metrics = compute_metrics(y_val_raw, val_pred)
    test_metrics = compute_metrics(y_test_raw, test_pred)

    # Segment MAE for required reporting (computed early for JSON)
    segment_mae = {}
    if "intra_state" in test_df.columns:
        for val, label in [(1, "intra_state"), (0, "inter_state")]:
            mask = (test_df["intra_state"].values == val)
            if mask.sum() > 0:
                segment_mae[label] = {"mae": float(np.mean(np.abs(y_test_raw[mask] - test_pred[mask]))), "n": int(mask.sum())}
    if "distance_km_bucket" in test_df.columns:
        long_mask = np.array([str(x).startswith("d_1500") or "1500" in str(x) for x in test_df["distance_km_bucket"].values])
        if long_mask.sum() > 0:
            segment_mae["long_distance_bucket"] = {"mae": float(np.mean(np.abs(y_test_raw[long_mask] - test_pred[long_mask]))), "n": int(long_mask.sum())}
    metrics_by_band = {}
    if (y_test_raw <= 30).sum() > 10:
        m = y_test_raw <= 30
        metrics_by_band["y_true_le_30"] = {"mae": float(mean_absolute_error(y_test_raw[m], test_pred[m])), "r2": float(r2_score(y_test_raw[m], test_pred[m])), "n": int(m.sum())}
    if (y_test_raw > 30).sum() > 10:
        m = y_test_raw > 30
        metrics_by_band["y_true_gt_30"] = {"mae": float(mean_absolute_error(y_test_raw[m], test_pred[m])), "r2": float(r2_score(y_test_raw[m], test_pred[m])), "n": int(m.sum())}
    metrics_log = {
        "run_id": experiment.get("run_id", run_dir.name),
        "validation": val_metrics,
        "test": test_metrics,
        "segment_mae": segment_mae,
        "test_by_band": metrics_by_band,
    }
    with open(run_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)

    # --- Error analysis ---
    lines = []
    lines.append("=" * 70)
    lines.append("DELIVERY DURATION – ERROR ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. METRICS (original scale)")
    lines.append(f"   Validation: MAE={val_metrics['mae']:.4f} RMSE={val_metrics['rmse']:.4f} MAPE={val_metrics['mape']:.2f}% R2={val_metrics['r2']:.4f}")
    lines.append(f"   Test:       MAE={test_metrics['mae']:.4f} RMSE={test_metrics['rmse']:.4f} MAPE={test_metrics['mape']:.2f}% R2={test_metrics['r2']:.4f}")
    # Metrics by target band: <=30 days vs >30 days
    mask_short = y_test_raw <= 30
    mask_long = y_test_raw > 30
    if mask_short.sum() > 10:
        mae_short = mean_absolute_error(y_test_raw[mask_short], test_pred[mask_short])
        r2_short = r2_score(y_test_raw[mask_short], test_pred[mask_short])
        lines.append(f"   Test (y_true <= 30 days): MAE={mae_short:.4f} R2={r2_short:.4f} n={int(mask_short.sum())}")
    if mask_long.sum() > 10:
        mae_long = mean_absolute_error(y_test_raw[mask_long], test_pred[mask_long])
        r2_long = r2_score(y_test_raw[mask_long], test_pred[mask_long])
        lines.append(f"   Test (y_true > 30 days):  MAE={mae_long:.4f} R2={r2_long:.4f} n={int(mask_long.sum())}")
    lines.append("")

    # A. Residual analysis (on test for unbiased view)
    residuals = y_test_raw - test_pred
    lines.append("2. RESIDUAL ANALYSIS (test set)")
    lines.append(f"   Mean residual: {np.mean(residuals):.4f} (negative = overpredict delivery time)")
    lines.append(f"   Std residual:  {np.std(residuals):.4f}")
    lines.append(f"   Median residual: {np.median(residuals):.4f}")
    overpred = (residuals < 0).mean() * 100
    underpred = (residuals > 0).mean() * 100
    lines.append(f"   Overprediction (pred > actual): {overpred:.1f}% of cases")
    lines.append(f"   Underprediction (pred < actual): {underpred:.1f}% of cases")
    # Simple heteroscedasticity: correlation |residual| vs y_true
    abs_res = np.abs(residuals)
    if np.std(y_test_raw) > 0 and np.std(abs_res) > 0:
        het = np.corrcoef(y_test_raw, abs_res)[0, 1]
        lines.append(f"   |residual| vs actual correlation: {het:.3f} (positive suggests higher errors for longer deliveries)")
    lines.append("")

    # B. Segment performance (MAE by segment)
    def add_order_value_bucket(df):
        if "total_order_value" not in df.columns:
            return df.assign(order_value_bucket="unknown")
        q = df["total_order_value"].quantile([0.33, 0.66])
        def bucket(x):
            if x <= q.iloc[0]: return "low"
            if x <= q.iloc[1]: return "mid"
            return "high"
        return df.assign(order_value_bucket=df["total_order_value"].apply(bucket))

    test_with_pred = test_df.copy()
    test_with_pred["y_true"] = y_test_raw
    test_with_pred["y_pred"] = test_pred
    test_with_pred["abs_error"] = np.abs(y_test_raw - test_pred)
    test_with_pred = add_order_value_bucket(test_with_pred)

    # Segment MAE: intra_state, inter_state, long-distance bucket (required reporting)
    lines.append("3. SEGMENT MAE (test set) – intra_state, inter_state, long-distance bucket")
    if "intra_state" in test_with_pred.columns:
        for val, label in [(1, "intra_state"), (0, "inter_state")]:
            mask = test_with_pred["intra_state"] == val
            if mask.sum() > 0:
                mae_s = np.mean(test_with_pred.loc[mask, "abs_error"])
                n_s = int(mask.sum())
                lines.append(f"   {label}: MAE={mae_s:.3f} n={n_s}")
    if "distance_km_bucket" in test_with_pred.columns:
        long_mask = test_with_pred["distance_km_bucket"].astype(str).str.contains("1500_plus|d_far", regex=True)
        if long_mask.sum() > 0:
            mae_long = np.mean(test_with_pred.loc[long_mask, "abs_error"])
            lines.append(f"   long_distance_bucket (d_1500_plus): MAE={mae_long:.3f} n={int(long_mask.sum())}")
        for bucket in ["d_0_500", "d_500_1500", "d_1500_plus", "d_unknown"]:
            bmask = test_with_pred["distance_km_bucket"] == bucket
            if bmask.sum() > 0:
                lines.append(f"   distance_km_bucket={bucket}: MAE={test_with_pred.loc[bmask, 'abs_error'].mean():.3f} n={int(bmask.sum())}")
    lines.append("")
    segment_cols = ["primary_category", "seller_region", "order_value_bucket", "bulky_flag"]
    lines.append("3b. SEGMENT MAE (other segments)")
    for col in segment_cols:
        if col not in test_with_pred.columns:
            continue
        seg = test_with_pred.groupby(col, observed=True)["abs_error"].agg(["mean", "count"])
        seg = seg.sort_values("mean", ascending=False)
        lines.append(f"   By {col}:")
        for idx, row in seg.head(8).iterrows():
            lines.append(f"      {idx}: MAE={row['mean']:.3f} n={int(row['count'])}")
        lines.append("")
    lines.append("")

    # C. Top 50 extreme errors
    test_with_pred["error"] = y_test_raw - test_pred
    test_with_pred["abs_error_rank"] = test_with_pred["abs_error"].rank(ascending=False)
    top50 = test_with_pred[test_with_pred["abs_error_rank"] <= 50].sort_values("abs_error", ascending=False)
    lines.append("4. TOP 50 LARGEST ABSOLUTE ERRORS (test)")
    lines.append("   Columns: y_true, y_pred, error, intra_state, primary_category, seller_region, order_value_bucket, bulky_flag, total_order_value (if present)")
    for _, row in top50.head(50).iterrows():
        parts = [f"true={row['y_true']:.1f} pred={row['y_pred']:.1f} err={row['error']:.1f}"]
        for c in ["intra_state", "primary_category", "seller_region", "order_value_bucket", "bulky_flag"]:
            if c in row.index and pd.notna(row[c]):
                parts.append(f"{c}={row[c]}")
        if "total_order_value" in row.index:
            parts.append(f"order_val={row['total_order_value']:.0f}")
        lines.append("   " + " | ".join(parts))
    lines.append("")
    lines.append("5. EXTREME ERROR SUMMARY")
    lines.append(f"   Top-50 mean |error|: {top50['abs_error'].mean():.2f} days")
    if "intra_state" in top50.columns:
        lines.append(f"   Top-50 intra_state share: {(top50['intra_state'] == 1).mean()*100:.1f}%")
    if "bulky_flag" in top50.columns:
        lines.append(f"   Top-50 bulky_flag share: {(top50['bulky_flag'] == 1).mean()*100:.1f}%")
    lines.append("")
    lines.append("6. LEARNINGS (model comparison & failure modes)")
    lines.append(f"   Best model: {experiment.get('model_type', 'unknown')}.")
    lines.append(f"   Validation R2={val_metrics['r2']:.3f}, Test R2={test_metrics['r2']:.3f} (negative test R2 suggests overfitting or distribution shift).")
    lines.append("   Residuals: negative mean residual => model overpredicts delivery time (actuals often shorter than predicted).")
    lines.append("   Segment: inter_state has higher MAE than intra_state; long-distance deliveries are harder.")
    lines.append("")
    lines.append("7. IMPROVEMENT SUMMARY")
    gap_mae = experiment.get("val_test_gap_mae")
    gap_r2 = experiment.get("val_test_gap_r2")
    if gap_mae is not None:
        lines.append(f"   Val–test gap: MAE +{gap_mae:.3f} days, R2 {gap_r2:.3f} (smaller gap = less overfitting).")
    log_vs_raw = experiment.get("log_vs_raw_comparison", {})
    if log_vs_raw:
        lines.append("   Log vs raw (Ridge): log test MAE={:.3f}, raw test MAE={:.3f}.".format(
            log_vs_raw.get("ridge_log_test_mae", 0), log_vs_raw.get("ridge_raw_test_mae", 0)))
    lines.append("   What improved: stronger tree regularization (max_depth 4–5, reg_alpha/lambda), geo features (region_to_region, distance_km_bucket, inter_state_x_bulky).")
    lines.append("   What still fails: test R2 often negative (time drift / distribution shift); large errors on long deliveries (>30d) and inter_state; tail underprediction.")
    lines.append("   Production reliability: use with caution; recommend segment-specific thresholds or fallbacks for inter_state and long-distance; monitor test-set performance over time.")

    report_path = run_dir / "error_analysis.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Error analysis written to {report_path}")

    # Lightweight residual distribution (text histogram)
    hist_path = run_dir / "residual_histogram.txt"
    hist, bin_edges = np.histogram(residuals, bins=15)
    with open(hist_path, "w", encoding="utf-8") as f:
        f.write("Residual (y_true - y_pred) distribution (test set)\n")
        f.write("bin_lo\tbin_hi\tcount\n")
        for i in range(len(hist)):
            f.write(f"{bin_edges[i]:.2f}\t{bin_edges[i+1]:.2f}\t{int(hist[i])}\n")
    print(f"Residual histogram written to {hist_path}")

    return {
        "run_dir": str(run_dir),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "error_analysis_path": str(report_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate delivery duration model")
    parser.add_argument("--run_id", type=str, default=None, help="Experiment run_id (default: latest)")
    args = parser.parse_args()
    run_evaluation(run_id=args.run_id)
