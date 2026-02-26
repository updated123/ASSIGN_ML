"""
Evaluation pipeline for multi-class review score prediction.
Loads a trained experiment and produces:
- Full metrics (accuracy, macro/weighted F1, per-class, confusion matrix, macro ROC-AUC)
- Error analysis (confusion patterns, segment performance, failure modes)
- Limitations and learnings summary.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
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
    from src.train import _LabelEncodedModel  # so joblib can unpickle wrapped model
except ImportError:
    from feature_preparation_review import (
        prepare_splits_from_config,
        TARGET,
        PROJECT_ROOT,
    )
    from train import _LabelEncodedModel

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CLASSES = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Why Macro-F1 is more important than accuracy (for this problem)
# ---------------------------------------------------------------------------
# With ~58% class 5, a model that always predicts 5 would get ~58% accuracy
# but 0% recall for 1–4. Macro-F1 averages per-class F1, so each class
# (including rare 1-star and 2-star) contributes equally. We care about
# detecting bad reviews (1–2) and distinguishing middle (3 vs 4), not just
# overall correctness.
#
# Why per-class recall matters for 1-star detection
# ---------------------------------------------------------------------------
# Missing a 1-star (false negative) is costly: we fail to flag unhappy
# customers. Per-class recall for class 1 answers: "Of all true 1-stars,
# how many did we catch?" Low recall here means we under-detect bad reviews.
#
# Why weighted-F1 alone is insufficient
# ---------------------------------------------------------------------------
# Weighted-F1 weights each class by its support, so the dominant class (5)
# dominates the metric. A model that predicts 5 most of the time can still
# get decent weighted-F1 while doing poorly on 1–3. We report both macro
# (for fairness across classes) and weighted (for overall utility).
# ---------------------------------------------------------------------------


def get_latest_run_dir():
    """Return the most recent experiment directory (by name = timestamp)."""
    if not EXPERIMENTS_DIR.exists():
        return None
    dirs = sorted(EXPERIMENTS_DIR.glob("*"), key=lambda p: p.name, reverse=True)
    for d in dirs:
        if d.is_dir() and (d / "model.joblib").exists():
            return d
    return None


def transform_to_matrix(df, preprocessor, config):
    """Reuse same transform logic as train.py."""
    num = [c for c in config["numeric"] if c in df.columns]
    cat_onehot = [c for c in config["categorical_onehot"] if c in df.columns]
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


def compute_all_metrics(y_true, y_pred, y_proba, classes=CLASSES):
    """Accuracy, macro/weighted F1, per-class P/R/F1, confusion matrix, macro ROC-AUC (OvR)."""
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["per_class_precision"] = precision_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    metrics["per_class_recall"] = recall_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    metrics["per_class_f1"] = f1_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=classes)
    if y_proba is not None and y_proba.shape[1] == len(classes):
        try:
            metrics["macro_roc_auc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except Exception:
            metrics["macro_roc_auc"] = None
    else:
        metrics["macro_roc_auc"] = None
    return metrics


def run_evaluation(run_id=None):
    """
    Load experiment run, compute full metrics, error analysis, limitations.
    Save reports to experiments/<run_id>/.
    """
    if run_id is None:
        run_dir = get_latest_run_dir()
        if run_dir is None:
            raise FileNotFoundError("No experiment found. Run: python -m src.train")
    else:
        run_dir = EXPERIMENTS_DIR / run_id
        if not run_dir.is_dir() or not (run_dir / "model.joblib").exists():
            raise FileNotFoundError(f"Experiment not found: {run_dir}")

    model = joblib.load(run_dir / "model.joblib")
    preprocessor = joblib.load(run_dir / "preprocessor.joblib")
    with open(run_dir / "feature_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    _, _, test_df, _ = prepare_splits_from_config(
        config_path=run_dir / "feature_config.json",
    )
    X_test = transform_to_matrix(test_df, preprocessor, config)
    y_test = test_df[TARGET].values

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    # Ensure class order matches CLASSES for metrics
    if hasattr(model, "classes_"):
        class_order = list(model.classes_)
    else:
        class_order = CLASSES

    metrics = compute_all_metrics(y_test, y_pred, y_proba, classes=class_order)

    # ---- 1. Evaluation report (metrics) ----
    lines = []
    lines.append("=" * 60)
    lines.append("REVIEW SCORE – EVALUATION REPORT (TEST SET)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Accuracy (reference only): {:.4f}".format(metrics["accuracy"]))
    lines.append("Macro F1 (primary):         {:.4f}".format(metrics["macro_f1"]))
    lines.append("Weighted F1:               {:.4f}".format(metrics["weighted_f1"]))
    if metrics["macro_roc_auc"] is not None:
        lines.append("Macro ROC-AUC (OvR):       {:.4f}".format(metrics["macro_roc_auc"]))
    lines.append("")
    lines.append("Per-class precision / recall / F1:")
    for i, c in enumerate(class_order):
        p, r, f = (
            metrics["per_class_precision"][i],
            metrics["per_class_recall"][i],
            metrics["per_class_f1"][i],
        )
        lines.append("  Score {}: P={:.4f}  R={:.4f}  F1={:.4f}".format(c, p, r, f))
    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=pred):")
    lines.append(np.array2string(metrics["confusion_matrix"]))

    with open(run_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(run_dir / "classification_report_test.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, labels=class_order, zero_division=0))

    with open(run_dir / "confusion_matrix_test.txt", "w", encoding="utf-8") as f:
        np.savetxt(f, metrics["confusion_matrix"], fmt="%d")

    # ---- 2. Error analysis ----
    test_df = test_df.copy()
    test_df["y_true"] = y_test
    test_df["y_pred"] = y_pred

    err_lines = []
    err_lines.append("=" * 60)
    err_lines.append("ERROR ANALYSIS")
    err_lines.append("=" * 60)

    # A. Confusion patterns
    err_lines.append("")
    err_lines.append("A. CONFUSION PATTERNS")
    err_lines.append("-" * 40)
    cm = metrics["confusion_matrix"]
    # 3 vs 4 confusion
    idx3 = class_order.index(3) if 3 in class_order else None
    idx4 = class_order.index(4) if 4 in class_order else None
    if idx3 is not None and idx4 is not None:
        err_lines.append("  3-star predicted as 4-star: {} (of {} true 3s)".format(
            cm[idx3, idx4], cm[idx3, :].sum()))
        err_lines.append("  4-star predicted as 3-star: {} (of {} true 4s)".format(
            cm[idx4, idx3], cm[idx4, :].sum()))
    idx1 = class_order.index(1) if 1 in class_order else None
    idx2 = class_order.index(2) if 2 in class_order else None
    if idx1 is not None and idx2 is not None:
        err_lines.append("  1-star predicted as 2-star: {} (of {} true 1s)".format(
            cm[idx1, idx2], cm[idx1, :].sum()))
        err_lines.append("  2-star predicted as 1-star: {} (of {} true 2s)".format(
            cm[idx2, idx1], cm[idx2, :].sum()))
    err_lines.append("  Extreme classes (1,5) vs middle (2,3,4):")
    for c in class_order:
        idx = class_order.index(c)
        correct = cm[idx, idx]
        total = cm[idx, :].sum()
        err_lines.append("    Score {}: {} correct / {} ({:.1f}%)".format(
            c, correct, total, 100 * correct / total if total else 0))

    # B. Segment performance (macro-F1 or recall by segment)
    err_lines.append("")
    err_lines.append("B. SEGMENT PERFORMANCE")
    err_lines.append("-" * 40)
    for seg_name, col in [
        ("Late vs on-time", "late_delivery_flag"),
        ("Order value bucket", "order_value_bucket"),
        ("Primary category", "primary_category"),
    ]:
        if col not in test_df.columns:
            continue
        err_lines.append("  By {}:".format(seg_name))
        for val in test_df[col].dropna().unique()[:10]:
            mask = test_df[col] == val
            if mask.sum() < 20:
                continue
            seg_f1 = f1_score(
                test_df.loc[mask, "y_true"],
                test_df.loc[mask, "y_pred"],
                average="macro",
                zero_division=0,
            )
            err_lines.append("    {}: n={}, macro-F1={:.4f}".format(val, mask.sum(), seg_f1))
        if test_df[col].nunique() > 10:
            err_lines.append("    (showing first 10 segments only)")

    # 1-star recall by segment (critical for detecting bad reviews)
    if "late_delivery_flag" in test_df.columns:
        for late in [0, 1]:
            mask = (test_df["late_delivery_flag"] == late) & (test_df["y_true"] == 1)
            if mask.sum() < 5:
                continue
            pred_1 = (test_df.loc[mask, "y_pred"] == 1).sum()
            err_lines.append("  1-star recall (late={}): {} / {} = {:.2f}".format(
                late, pred_1, mask.sum(), pred_1 / mask.sum() if mask.sum() else 0))

    # C. Failure mode analysis
    err_lines.append("")
    err_lines.append("C. FAILURE MODE ANALYSIS")
    err_lines.append("-" * 40)
    wrong = test_df[test_df["y_true"] != test_df["y_pred"]]
    err_lines.append("  Total errors: {} / {} ({:.1f}%)".format(
        len(wrong), len(test_df), 100 * len(wrong) / len(test_df)))
    # Where does model struggle? (by true class)
    err_lines.append("  Error rate by true class:")
    for c in class_order:
        mask = test_df["y_true"] == c
        err_mask = mask & (test_df["y_pred"] != test_df["y_true"])
        rate = err_mask.sum() / mask.sum() if mask.sum() else 0
        err_lines.append("    True {}: {:.1f}% wrong".format(c, 100 * rate))
    # Over-reliance on late_delivery_flag: correlation between late and predicted low score
    if "late_delivery_flag" in test_df.columns:
        late_low = ((test_df["y_pred"] <= 3) & (test_df["late_delivery_flag"] == 1)).sum()
        late_total = (test_df["late_delivery_flag"] == 1).sum()
        err_lines.append("  When late=1, model predicts low (1-3): {} / {} = {:.2f}".format(
            late_low, late_total, late_low / late_total if late_total else 0))

    with open(run_dir / "error_analysis.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(err_lines))

    # ---- 3. Practical limitations ----
    lim_lines = []
    lim_lines.append("PRACTICAL LIMITATIONS")
    lim_lines.append("=" * 60)
    lim_lines.append("")
    lim_lines.append("1. Class imbalance (5-star dominant ~58%)")
    lim_lines.append("   - Model may under-predict rare classes (1, 2). Per-class recall for 1-star")
    lim_lines.append("     is critical; monitor it in production.")
    lim_lines.append("")
    lim_lines.append("2. Subjective nature of reviews")
    lim_lines.append("   - Same experience can yield 4 vs 5; label noise limits ceiling.")
    lim_lines.append("")
    lim_lines.append("3. Possible non-response bias")
    lim_lines.append("   - Not all delivered orders have reviews; reviewers may differ from non-reviewers.")
    lim_lines.append("")
    lim_lines.append("4. Seller-level skew")
    lim_lines.append("   - Many sellers with few orders; historical aggregates can be noisy for them.")
    lim_lines.append("")
    lim_lines.append("5. Temporal drift risk")
    lim_lines.append("   - Behavior and mix can change; retrain periodically and monitor macro-F1.")
    lim_lines.append("")
    lim_lines.append("When predictions should NOT be fully trusted:")
    lim_lines.append(" - New sellers/categories (no or little history).")
    lim_lines.append(" - Rare segments (e.g. very high order value, unusual category).")
    lim_lines.append(" - If per-class recall for 1-star is low, do not rely on model to flag all bad reviews.")
    with open(run_dir / "limitations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lim_lines))

    # ---- 4. Learnings summary (from experiment.json + comparison) ----
    exp_path = run_dir / "experiment.json"
    learn_lines = []
    learn_lines.append("LEARNINGS SUMMARY")
    learn_lines.append("=" * 60)
    if exp_path.exists():
        with open(exp_path, "r", encoding="utf-8") as f:
            exp = json.load(f)
        learn_lines.append("Model: {}".format(exp.get("model_type", "?")))
        learn_lines.append("Test Macro-F1: {:.4f}".format(metrics["macro_f1"]))
        learn_lines.append("Test Weighted-F1: {:.4f}".format(metrics["weighted_f1"]))
        learn_lines.append("")
        learn_lines.append("Findings from comparing baseline LR vs stronger model:")
        learn_lines.append(" - Macro-F1 is primary for imbalanced multi-class; accuracy is misleading.")
        learn_lines.append(" - Per-class recall (especially 1-star) matters for detecting bad reviews.")
        learn_lines.append(" - Temporal split avoids future leakage and gives realistic performance.")
    with open(run_dir / "learnings_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(learn_lines))

    print(f"Evaluation written to {run_dir}")
    print("  evaluation_report.txt, classification_report_test.txt, confusion_matrix_test.txt")
    print("  error_analysis.txt, limitations.txt, learnings_summary.txt")
    print("Macro-F1 (test): {:.4f}  Weighted-F1: {:.4f}".format(metrics["macro_f1"], metrics["weighted_f1"]))
    return metrics, run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Experiment run_id (default: latest)")
    args = parser.parse_args()
    run_evaluation(run_id=args.run)
