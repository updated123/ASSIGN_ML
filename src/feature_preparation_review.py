"""
Feature preparation pipeline for multi-class review score prediction.
Target: review_score ∈ {1, 2, 3, 4, 5}.

Focus: feature engineering and feature selection using ONLY delivery-time-safe features.
Predicting review_score after delivery but before review submission.
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif

try:
    from src.data import load_raw_tables, DATA_DIR
except ImportError:
    from data import load_raw_tables, DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. LEAKAGE BOUNDARIES (documented)
# ---------------------------------------------------------------------------
#
# We predict review_score AFTER delivery but BEFORE review submission.
# Only include features available at or before delivery completion.
#
# SAFE timestamps / fields (known at or before delivery):
#   - order_purchase_timestamp, order_approved_at
#   - order_delivered_carrier_date, order_delivered_customer_date
#   - order_estimated_delivery_date
#   - All order-level aggregates from items/payments (total_order_value,
#     total_freight_value, freight_ratio, item_count, payment_type_primary)
#   - approval_delay_hours (purchase -> approval)
#   - delivery_delay_days, late_delivery_flag (known once delivery is complete)
#   - primary_category, primary_seller_id, seller_state, customer_state
#
# DO NOT INCLUDE:
#   - review_comment_message, review_comment_title
#   - review_creation_date, review_answer_timestamp (any review timestamps)
#   - Any variable derived from post-review data
# ---------------------------------------------------------------------------

TARGET = "review_score"

# Initial candidate features (all delivery-time-safe)
ORDER_LEVEL_NUMERIC = [
    "total_order_value",
    "total_freight_value",
    "freight_ratio",
    "item_count",
    "approval_delay_hours",
    "delivery_delay_days",
    "late_delivery_flag",
]
CATEGORICAL = [
    "primary_category",
    "primary_seller_id",
    "seller_state",
    "customer_state",
    "payment_type_primary",
]

HIGH_CORRELATION_THRESHOLD = 0.8


def build_order_level_for_review(tables=None):
    """
    Build order-level analytical dataset with latest review per order (no averaging),
    delivery-time features, and customer_state / seller_state.
    Used as the base for review score feature preparation.
    """
    if tables is None:
        tables = load_raw_tables()
    orders = tables["olist_orders_dataset"].copy()
    order_items = tables["olist_order_items_dataset"].copy()
    payments = tables["olist_order_payments_dataset"].copy()
    reviews = tables["olist_order_reviews_dataset"].copy()
    products = tables["olist_products_dataset"].copy()
    sellers = tables["olist_sellers_dataset"].copy()
    customers = tables.get("olist_customers_dataset")

    items_agg = order_items.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
    ).reset_index()
    items_agg = items_agg.rename(columns={"total_price": "total_items_price"})

    oi_with_cat = order_items.merge(
        products[["product_id", "product_category_name"]],
        on="product_id",
        how="left",
    )

    def first_mode(s):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) > 0 else np.nan

    cat_per_order = (
        oi_with_cat.groupby("order_id")["product_category_name"]
        .agg(first_mode)
        .reset_index()
    )
    cat_per_order = cat_per_order.rename(columns={"product_category_name": "primary_category"})
    seller_per_order = (
        oi_with_cat.sort_values("order_item_id")
        .groupby("order_id")["seller_id"]
        .first()
        .reset_index()
        .rename(columns={"seller_id": "primary_seller_id"})
    )

    pay_agg = payments.groupby("order_id").agg(
        total_order_value=("payment_value", "sum"),
    ).reset_index()
    pay_type = (
        payments.sort_values("payment_sequential")
        .groupby("order_id")["payment_type"]
        .first()
        .reset_index()
        .rename(columns={"payment_type": "payment_type_primary"})
    )
    pay_agg = pay_agg.merge(pay_type, on="order_id", how="left")

    reviews = reviews.copy()
    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"], errors="coerce")
    idx_latest = reviews.groupby("order_id")["review_creation_date"].idxmax()
    reviews_latest = reviews.loc[idx_latest, ["order_id", "review_score"]].copy()
    rev_agg = reviews_latest.copy()

    df = orders.merge(items_agg, on="order_id", how="left")
    df = df.merge(pay_agg, on="order_id", how="left")
    df = df.merge(rev_agg, on="order_id", how="left")
    df = df.merge(cat_per_order, on="order_id", how="left")
    df = df.merge(seller_per_order, on="order_id", how="left")

    df["total_order_value"] = df["total_order_value"].fillna(df["total_items_price"])
    df["total_freight_value"] = df["total_freight_value"].fillna(0)
    df["item_count"] = df["item_count"].fillna(0).astype(int)

    # Datetime and delivery-time-safe derived features
    for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
              "order_delivered_customer_date", "order_estimated_delivery_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "order_delivered_customer_date" in df.columns and "order_estimated_delivery_date" in df.columns:
        df["delivery_delay_days"] = (
            df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
        ).dt.total_seconds() / (24 * 3600)
        df["delivery_delay_days"] = df["delivery_delay_days"].round(2)
        df["late_delivery_flag"] = (df["delivery_delay_days"] > 0).astype(int)
    else:
        df["delivery_delay_days"] = np.nan
        df["late_delivery_flag"] = np.nan

    df["freight_ratio"] = np.where(
        df["total_order_value"] > 0,
        (df["total_freight_value"] / df["total_order_value"]).round(4),
        np.nan,
    )
    if "order_approved_at" in df.columns and "order_purchase_timestamp" in df.columns:
        df["approval_delay_hours"] = (
            df["order_approved_at"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 3600
        df["approval_delay_hours"] = df["approval_delay_hours"].round(2)

    if customers is not None and "customer_state" in customers.columns:
        df = df.merge(
            customers[["customer_id", "customer_state"]],
            on="customer_id",
            how="left",
        )
    if sellers is not None and "seller_state" in sellers.columns:
        df = df.merge(
            sellers[["seller_id", "seller_state"]].rename(columns={"seller_id": "primary_seller_id"}),
            on="primary_seller_id",
            how="left",
        )

    return df


def get_review_model_ready_df(df):
    """Restrict to delivered orders with non-null review_score (target)."""
    out = df[df["order_status"] == "delivered"].copy()
    out = out.dropna(subset=[TARGET])
    out[TARGET] = out[TARGET].astype(int)
    out = out[(out[TARGET] >= 1) & (out[TARGET] <= 5)]
    return out


# ---------------------------------------------------------------------------
# 3. Advanced feature engineering
# ---------------------------------------------------------------------------

def get_quantile_bucket_edges(df, train_only=True, subset=None):
    """
    Compute quantile-based bin edges from (training) data.
    Returns dict: value_edges, freight_ratio_edges, delivery_delay_edges.
    """
    if subset is not None:
        d = subset
    else:
        d = df
    d = d.dropna(subset=["total_order_value", "freight_ratio"], how="all")
    value_edges = np.unique(np.r_[0, d["total_order_value"].quantile([0.25, 0.5, 0.75]).values, np.inf])
    fr = d["freight_ratio"].fillna(0)
    freight_ratio_edges = np.unique(np.r_[-0.01, fr.quantile([0.33, 0.66]).values, 1.5])
    dd = d["delivery_delay_days"].dropna()
    if len(dd) > 0:
        qq = dd.quantile([0.25, 0.5, 0.75]).values
        delivery_delay_edges = np.unique(np.r_[-np.inf, qq, np.inf])
    else:
        delivery_delay_edges = np.array([-np.inf, 0, np.inf])
    return {
        "value_edges": value_edges,
        "freight_ratio_edges": freight_ratio_edges,
        "delivery_delay_edges": delivery_delay_edges,
    }


def add_quantile_buckets(df, edges):
    """Add order_value_bucket, freight_ratio_bucket, delivery_delay_bucket."""
    df = df.copy()
    df["order_value_bucket"] = pd.cut(
        df["total_order_value"],
        bins=edges["value_edges"],
        labels=[f"val_q{i}" for i in range(len(edges["value_edges"]) - 1)],
        include_lowest=True,
    ).astype(str)
    fr = df["freight_ratio"].fillna(0)
    df["freight_ratio_bucket"] = pd.cut(
        fr,
        bins=edges["freight_ratio_edges"],
        labels=["fr_low", "fr_mid", "fr_high"],
        include_lowest=True,
    ).astype(str)
    dd = df["delivery_delay_days"].fillna(0)
    df["delivery_delay_bucket"] = pd.cut(
        dd,
        bins=edges["delivery_delay_edges"],
        labels=[f"dd_q{i}" for i in range(len(edges["delivery_delay_edges"]) - 1)],
        include_lowest=True,
    ).astype(str)
    return df


def add_historical_review_aggregates(df, time_col="order_purchase_timestamp", past_subset=None):
    """
    STRICTLY leakage-safe: for each order, use only past data (order_purchase_timestamp < current).
    Adds:
      seller_hist_avg_review, seller_hist_1star_rate, seller_hist_5star_rate
      category_hist_avg_review, category_hist_dissatisfaction_rate (1-3)
    past_subset: if provided (e.g. train_df), historical stats are built only from past_subset;
      then merged to df by time (merge_asof). Used for val/test to avoid leakage.
    """
    df = df.copy()
    if time_col not in df.columns:
        return df
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    full = past_subset if past_subset is not None else df
    # Only delivered orders with review (we need review_score for history)
    delivered = full[(full["order_status"] == "delivered") & (full[TARGET].notna())].copy()
    delivered = delivered.dropna(subset=[time_col, "primary_seller_id", "primary_category"])
    delivered = delivered.sort_values(time_col)

    # Seller: expanding (shift(1)) mean review, 1-star rate, 5-star rate
    delivered["_one"] = 1
    delivered["_is_1"] = (delivered[TARGET] == 1).astype(int)
    delivered["_is_5"] = (delivered[TARGET] == 5).astype(int)
    delivered["_low"] = (delivered[TARGET] <= 3).astype(int)

    delivered["seller_hist_avg_review"] = (
        delivered.groupby("primary_seller_id", sort=False)[TARGET]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    delivered["seller_hist_1star_rate"] = (
        delivered.groupby("primary_seller_id", sort=False)["_is_1"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    delivered["seller_hist_5star_rate"] = (
        delivered.groupby("primary_seller_id", sort=False)["_is_5"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    delivered["category_hist_avg_review"] = (
        delivered.groupby("primary_category", sort=False)[TARGET]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    delivered["category_hist_dissatisfaction_rate"] = (
        delivered.groupby("primary_category", sort=False)["_low"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    by_seller = (
        delivered[[time_col, "primary_seller_id", "seller_hist_avg_review", "seller_hist_1star_rate", "seller_hist_5star_rate"]]
        .drop_duplicates(subset=[time_col, "primary_seller_id"], keep="last")
        .sort_values(time_col)
    )
    by_cat = (
        delivered[[time_col, "primary_category", "category_hist_avg_review", "category_hist_dissatisfaction_rate"]]
        .drop_duplicates(subset=[time_col, "primary_category"], keep="last")
        .sort_values(time_col)
    )

    global_avg = delivered[TARGET].mean()
    global_1 = delivered["_is_1"].mean()
    global_5 = delivered["_is_5"].mean()
    global_dis = delivered["_low"].mean()

    df_sorted = df.sort_values(time_col).copy()
    for c in ["seller_hist_avg_review", "seller_hist_1star_rate", "seller_hist_5star_rate",
              "category_hist_avg_review", "category_hist_dissatisfaction_rate"]:
        df_sorted[c] = np.nan
    valid = df_sorted[time_col].notna() & df_sorted["primary_seller_id"].notna() & df_sorted["primary_category"].notna()
    if valid.any():
        left = df_sorted.loc[valid].copy()
        left = pd.merge_asof(left, by_seller, on=time_col, by="primary_seller_id", direction="backward")
        left = pd.merge_asof(left.sort_values(time_col), by_cat, on=time_col, by="primary_category", direction="backward")
        for c in ["seller_hist_avg_review", "seller_hist_1star_rate", "seller_hist_5star_rate",
                  "category_hist_avg_review", "category_hist_dissatisfaction_rate"]:
            if c in left.columns:
                df_sorted.loc[valid, c] = left[c].values
    df_sorted["seller_hist_avg_review"] = df_sorted["seller_hist_avg_review"].fillna(global_avg)
    df_sorted["seller_hist_1star_rate"] = df_sorted["seller_hist_1star_rate"].fillna(global_1)
    df_sorted["seller_hist_5star_rate"] = df_sorted["seller_hist_5star_rate"].fillna(global_5)
    df_sorted["category_hist_avg_review"] = df_sorted["category_hist_avg_review"].fillna(global_avg)
    df_sorted["category_hist_dissatisfaction_rate"] = df_sorted["category_hist_dissatisfaction_rate"].fillna(global_dis)
    return df_sorted.reindex(df.index).copy()


def add_interaction_features(df):
    """
    Add interaction features (all delivery-time-safe).
    Assumptions:
      - late_delivery_flag × freight_ratio: late + high freight ratio may amplify dissatisfaction.
      - delivery_delay_days × order_value: higher value + delay may matter more.
      - approval_delay_hours × order_value: slow approval on expensive orders.
      - category × late_delivery_flag: encoded as categorical 'primary_category_late' for one-hot
        (optional); we add numeric late_freight_ratio and delay_value instead for interpretability.
    """
    df = df.copy()
    df["late_freight_ratio"] = (df["late_delivery_flag"].fillna(0) * df["freight_ratio"].fillna(0)).round(4)
    df["delivery_delay_times_value"] = (df["delivery_delay_days"].fillna(0) * df["total_order_value"].fillna(0)).round(2)
    df["approval_delay_times_value"] = (df["approval_delay_hours"].fillna(0) * df["total_order_value"].fillna(0)).round(2)
    df["category_late"] = df["primary_category"].fillna("__missing__").astype(str) + "_late" + df["late_delivery_flag"].fillna(0).astype(int).astype(str)
    return df


# ---------------------------------------------------------------------------
# 4. Numeric diagnostics: correlation, redundant removal
# ---------------------------------------------------------------------------

def compute_correlation_diagnostics(X_numeric):
    """Correlation matrix and pairs with |r| > threshold."""
    corr = X_numeric.corr()
    high_pairs = []
    for i, a in enumerate(corr.columns):
        for j, b in enumerate(corr.columns):
            if i >= j:
                continue
            v = corr.loc[a, b]
            if abs(v) >= HIGH_CORRELATION_THRESHOLD:
                high_pairs.append((a, b, float(v)))
    return corr, high_pairs


def remove_redundant_numeric(numeric_cols, corr_matrix, threshold=HIGH_CORRELATION_THRESHOLD):
    """Among pairs with |r| > threshold, drop one (prefer keeping interpretable / less redundant)."""
    kept = set(numeric_cols)
    drop_priority = ["total_freight_value", "delivery_delay_times_value", "approval_delay_times_value"]
    for i, a in enumerate(corr_matrix.columns):
        for j, b in enumerate(corr_matrix.columns):
            if i >= j or a not in kept or b not in kept:
                continue
            if abs(corr_matrix.loc[a, b]) < threshold:
                continue
            to_drop = None
            if a in drop_priority and b in drop_priority:
                to_drop = a if drop_priority.index(a) <= drop_priority.index(b) else b
            elif a in drop_priority:
                to_drop = a
            elif b in drop_priority:
                to_drop = b
            if to_drop and to_drop in kept:
                kept.discard(to_drop)
    return [c for c in numeric_cols if c in kept]


# ---------------------------------------------------------------------------
# 5. Feature importance (multi-class) and final output
# ---------------------------------------------------------------------------

def get_all_numeric_for_review(include_buckets=True, include_historical=True, include_interactions=True):
    """List of numeric feature names for review model."""
    numeric = list(ORDER_LEVEL_NUMERIC)
    if include_interactions:
        numeric.extend(["late_freight_ratio", "delivery_delay_times_value", "approval_delay_times_value"])
    if include_historical:
        numeric.extend([
            "seller_hist_avg_review", "seller_hist_1star_rate", "seller_hist_5star_rate",
            "category_hist_avg_review", "category_hist_dissatisfaction_rate",
        ])
    if include_buckets:
        numeric.extend([])  # buckets are categorical
    return numeric


def feature_importance_multiclass(X, y, numeric_cols, n_estimators=100):
    """Random Forest feature importance for multi-class (mean decrease impurity)."""
    Xn = X[numeric_cols].fillna(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight="balanced")
        m.fit(Xn, y)
    return pd.Series(m.feature_importances_, index=numeric_cols).sort_values(ascending=False)


def feature_importance_mutual_info(X, y, numeric_cols):
    """Mutual information (multi-class supported)."""
    Xn = X[numeric_cols].fillna(0)
    mi = mutual_info_classif(Xn, y, random_state=42)
    return pd.Series(mi, index=numeric_cols).sort_values(ascending=False)


def feature_importance_logistic_multiclass(X, y, numeric_cols):
    """Multinomial logistic: mean absolute coefficient across classes (per feature)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X[numeric_cols].fillna(0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial", class_weight="balanced")
        m.fit(Xs, y)
    # coef_.shape = (n_classes, n_features); take mean abs per feature
    imp = np.abs(m.coef_).mean(axis=0)
    return pd.Series(imp, index=numeric_cols).sort_values(ascending=False)


def per_class_drivers(X, y, numeric_cols, top_k=5):
    """
    Which features drive 1-star vs 5-star vs middle (3 vs 4)?
    Uses one-vs-rest logistic: fit binary (class k vs rest), rank by abs coefficient.
    Returns three Series (or empty Series if insufficient samples).
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X[numeric_cols].fillna(0))
    drivers_1 = pd.Series(dtype=float)
    drivers_5 = pd.Series(dtype=float)
    drivers_3v4 = pd.Series(dtype=float)
    for label in [1, 5]:
        binary = (y == label).astype(int)
        if binary.sum() < 10:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
            m.fit(Xs, binary)
        imp = pd.Series(np.abs(m.coef_[0]), index=numeric_cols).sort_values(ascending=False)
        if label == 1:
            drivers_1 = imp.head(top_k)
        else:
            drivers_5 = imp.head(top_k)
    mask = (y == 3) | (y == 4)
    if mask.sum() >= 20:
        Xm = Xs[mask]
        ym = (y[mask] == 4).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
            m.fit(Xm, ym)
        drivers_3v4 = pd.Series(np.abs(m.coef_[0]), index=numeric_cols).sort_values(ascending=False).head(top_k)
    return drivers_1, drivers_5, drivers_3v4


def run_full_pipeline(
    output_dir=None,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
):
    """
    Full feature preparation pipeline for review score prediction:
    - Build order-level data (delivery-time-safe), filter to delivered with review
    - Time-based split (train/val/test)
    - Quantile buckets, historical aggregates (leakage-safe), interactions
    - Correlation diagnostics, remove redundant numeric
    - Feature importance (RF, MI, logistic), per-class drivers
    - Write: final feature list, importance table, summary (1-star/5-star/3vs4 drivers, risks)
    """
    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = load_raw_tables()
    df = build_order_level_for_review(tables)
    df = get_review_model_ready_df(df)
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    train_df = df.iloc[:t1]
    val_df = df.iloc[t1:t2]
    test_df = df.iloc[t2:]

    edges = get_quantile_bucket_edges(train_df, subset=train_df)
    train_df = add_quantile_buckets(train_df, edges)
    train_df = add_historical_review_aggregates(train_df, past_subset=train_df)
    train_df = add_interaction_features(train_df)

    val_df = add_quantile_buckets(val_df, edges)
    val_df = add_historical_review_aggregates(val_df, past_subset=train_df)
    val_df = add_interaction_features(val_df)

    test_df = add_quantile_buckets(test_df, edges)
    test_df = add_historical_review_aggregates(test_df, past_subset=pd.concat([train_df, val_df], ignore_index=True))
    test_df = add_interaction_features(test_df)

    numeric_candidates = get_all_numeric_for_review()
    numeric_candidates = [c for c in numeric_candidates if c in train_df.columns]
    X_train = train_df[numeric_candidates].fillna(0)
    corr_matrix, high_pairs = compute_correlation_diagnostics(X_train)
    final_numeric = remove_redundant_numeric(numeric_candidates, corr_matrix)
    categorical_final = [c for c in CATEGORICAL if c in train_df.columns]
    categorical_final.extend(["order_value_bucket", "freight_ratio_bucket", "delivery_delay_bucket"])
    categorical_final = [c for c in categorical_final if c in train_df.columns]

    y_train = train_df[TARGET].values
    X_tr = train_df[final_numeric].fillna(0)
    imp_rf = feature_importance_multiclass(X_tr, y_train, final_numeric)
    imp_mi = feature_importance_mutual_info(X_tr, y_train, final_numeric)
    imp_log = feature_importance_logistic_multiclass(X_tr, y_train, final_numeric)

    importance_table = pd.DataFrame({
        "rf_importance": imp_rf.reindex(final_numeric).fillna(0),
        "mutual_info": imp_mi.reindex(final_numeric).fillna(0),
        "logistic_abs_coef": imp_log.reindex(final_numeric).fillna(0),
    })
    importance_table["rf_rank"] = importance_table["rf_importance"].rank(ascending=False)
    importance_table["mi_rank"] = importance_table["mutual_info"].rank(ascending=False)
    importance_table["log_rank"] = importance_table["logistic_abs_coef"].rank(ascending=False)
    importance_table["avg_rank"] = importance_table[["rf_rank", "mi_rank", "log_rank"]].mean(axis=1)
    importance_table = importance_table.sort_values("avg_rank")

    drivers_1, drivers_5, drivers_3v4 = per_class_drivers(X_tr, y_train, final_numeric, top_k=5)

    # Summary text
    summary_lines = []
    summary_lines.append("Which features drive 1-star predictions:")
    summary_lines.append("  " + (drivers_1.head(5).to_string() if len(drivers_1) > 0 else " (insufficient 1-star samples or N/A)"))
    summary_lines.append("")
    summary_lines.append("Which features drive 5-star predictions:")
    summary_lines.append("  " + (drivers_5.head(5).to_string() if len(drivers_5) > 0 else " (N/A)"))
    summary_lines.append("")
    summary_lines.append("Which features separate middle ratings (3 vs 4):")
    summary_lines.append("  " + (drivers_3v4.head(5).to_string() if len(drivers_3v4) > 0 else " (N/A)"))
    summary_lines.append("")
    summary_lines.append("Potential modeling risks:")
    summary_lines.append("  - Class imbalance: Score 5 dominates; use class_weight='balanced' or macro F1.")
    summary_lines.append("  - Seller skew: many sellers with few orders; historical aggregates may be noisy.")
    summary_lines.append("  - Category × late_delivery interaction: some categories may have few late orders.")

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("REVIEW SCORE FEATURE PREPARATION – FINAL REPORT")
    report_lines.append("Target: review_score ∈ {1, 2, 3, 4, 5}")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("1. LEAKAGE BOUNDARIES (delivery-time-safe only)")
    report_lines.append("   Included: order/delivery timestamps, order value, freight, item_count,")
    report_lines.append("   approval_delay_hours, delivery_delay_days, late_delivery_flag, primary_category,")
    report_lines.append("   primary_seller_id, seller_state, customer_state, payment_type_primary.")
    report_lines.append("   Excluded: review_comment_*, review timestamps, any post-review variables.")
    report_lines.append("")
    report_lines.append("2. HIGH CORRELATION PAIRS (|r| > 0.8)")
    for a, b, r in high_pairs:
        report_lines.append(f"   {a} vs {b}: {r:.3f}")
    if not high_pairs:
        report_lines.append("   None above threshold.")
    report_lines.append("")
    report_lines.append("3. REDUNDANT FEATURES REMOVED")
    removed = [c for c in numeric_candidates if c not in final_numeric]
    report_lines.append("   " + (", ".join(removed) if removed else "None"))
    report_lines.append("")
    report_lines.append("4. FINAL SELECTED FEATURES")
    report_lines.append("   Numeric: " + ", ".join(final_numeric))
    report_lines.append("   Categorical: " + ", ".join(categorical_final))
    report_lines.append("")
    report_lines.append("5. FEATURE IMPORTANCE COMPARISON TABLE")
    report_lines.append(importance_table.to_string())
    report_lines.append("")
    report_lines.append("6. SUMMARY")
    report_lines.extend(summary_lines)

    report_path = output_dir / "review_feature_preparation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Report written to {report_path}")

    def _json_safe_edges(arr):
        a = np.asarray(arr)
        out = []
        for x in a:
            if np.isinf(x):
                out.append(float("-1e10") if x < 0 else float("1e10"))
            else:
                out.append(float(x))
        return out

    config = {
        "target": TARGET,
        "numeric": final_numeric,
        "categorical_onehot": [c for c in categorical_final if c != "primary_seller_id"],
        "categorical_ordinal": ["primary_seller_id"] if "primary_seller_id" in categorical_final else [],
        "quantile_edges": {k: _json_safe_edges(v) for k, v in edges.items()},
    }
    config_path = output_dir / "review_feature_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {config_path}")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "final_numeric": final_numeric,
        "categorical_final": categorical_final,
        "importance_table": importance_table,
        "correlation_matrix": corr_matrix,
        "high_pairs": high_pairs,
        "edges": edges,
    }


def _config_edges_to_arrays(quantile_edges):
    """Convert quantile_edges from JSON (lists with 1e10 for inf) to numpy arrays for pd.cut."""
    edges = {}
    for k, v in quantile_edges.items():
        arr = np.array(v, dtype=float)
        # Restore inf: JSON stores ±1e10 for ±inf
        arr = np.where(arr >= 1e9, np.inf, np.where(arr <= -1e9, -np.inf, arr))
        edges[k] = np.unique(arr)
    return edges


def prepare_splits_from_config(config_path=None, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    """
    Load feature config and build train/val/test with same time-based split and feature
    application as run_full_pipeline. Used by train.py and evaluate.py.
    Returns: (train_df, val_df, test_df, config).
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "notebooks" / "review_feature_config.json"
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    tables = load_raw_tables()
    df = build_order_level_for_review(tables)
    df = get_review_model_ready_df(df)
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    train_df = df.iloc[:t1].copy()
    val_df = df.iloc[t1:t2].copy()
    test_df = df.iloc[t2:].copy()
    edges = _config_edges_to_arrays(config["quantile_edges"])
    train_df = add_quantile_buckets(train_df, edges)
    train_df = add_historical_review_aggregates(train_df, past_subset=train_df)
    train_df = add_interaction_features(train_df)
    val_df = add_quantile_buckets(val_df, edges)
    val_df = add_historical_review_aggregates(val_df, past_subset=train_df)
    val_df = add_interaction_features(val_df)
    test_df = add_quantile_buckets(test_df, edges)
    test_df = add_historical_review_aggregates(
        test_df, past_subset=pd.concat([train_df, val_df], ignore_index=True)
    )
    test_df = add_interaction_features(test_df)
    return train_df, val_df, test_df, config


if __name__ == "__main__":
    run_full_pipeline()
