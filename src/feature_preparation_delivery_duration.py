"""
Feature preparation and diagnostics for delivery duration regression.

Target: delivery_duration_days = (order_delivered_customer_date - order_purchase_timestamp) in days.
Prediction time: order approval time (only approval-time-safe features).

LEAKAGE BOUNDARIES (approval-time-safe only)
-------------------------------------------
We predict delivery_duration_days at order approval time for logistics planning.
Include only features known at or before order approval.

SAFE (available at or before approval):
- order_purchase_timestamp: known at purchase.
- seller/customer location (zip, state): known at order placement.
- product category, order value, freight value, item_count: from order/items, known at placement.
- product dimensions/weight: from product catalog, known at placement.
- payment type: known at approval.
- seller_state, customer_state: known at placement.
- Historical seller metrics: computed from past orders only (time-aware), no current order info.

DO NOT INCLUDE (post-approval or post-delivery):
- order_delivered_customer_date: defines target; unknown until delivery.
- order_estimated_delivery_date: set at placement but reflects promise, not actual; can leak.
- delivery_delay_days, late_delivery_flag: require delivered_customer_date.
- review_score, any review data: post-delivery.
- order_delivered_carrier_date: post-approval (optional to exclude for consistency).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    from src.data import load_raw_tables, get_geo_zip_centroids
except ImportError:
    from data import load_raw_tables, get_geo_zip_centroids

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "notebooks"
TARGET = "delivery_duration_days"

# Brazilian state -> region (simplified)
STATE_TO_REGION = {
    "AC": "N", "AL": "NE", "AP": "N", "AM": "N", "BA": "NE", "CE": "NE",
    "DF": "CO", "ES": "SE", "GO": "CO", "MA": "NE", "MT": "CO", "MS": "CO",
    "MG": "SE", "PA": "N", "PB": "NE", "PR": "S", "PE": "NE", "PI": "NE",
    "RJ": "SE", "RN": "NE", "RS": "S", "RO": "N", "RR": "N", "SC": "S",
    "SP": "SE", "SE": "NE", "TO": "N",
}


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))
    return R * c


def build_delivery_duration_dataset(tables=None):
    """
    Build order-level dataset for delivery duration regression.
    Target computed only for delivered orders. All features are approval-time-safe.
    """
    if tables is None:
        tables = load_raw_tables()
    orders = tables["olist_orders_dataset"].copy()
    order_items = tables["olist_order_items_dataset"].copy()
    payments = tables["olist_order_payments_dataset"].copy()
    products = tables["olist_products_dataset"].copy()
    customers = tables.get("olist_customers_dataset")
    sellers = tables.get("olist_sellers_dataset")

    for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_customer_date"]:
        if c in orders.columns:
            orders[c] = pd.to_datetime(orders[c], errors="coerce")

    # Items aggregate: item_count, total_price, total_freight
    items_agg = order_items.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
    ).reset_index()
    items_agg = items_agg.rename(columns={"total_price": "total_items_price"})

    # Product dimensions/weight per order (from order_items + products)
    oi_prod = order_items.merge(
        products[["product_id", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]],
        on="product_id",
        how="left",
    )
    oi_prod["volume_cm3"] = (
        oi_prod["product_length_cm"].fillna(0) * oi_prod["product_height_cm"].fillna(0) * oi_prod["product_width_cm"].fillna(0)
    )
    prod_agg = oi_prod.groupby("order_id").agg(
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
    ).reset_index()

    # Primary category and seller
    oi_cat = order_items.merge(
        products[["product_id", "product_category_name"]],
        on="product_id",
        how="left",
    )
    def first_mode(s):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) > 0 else np.nan
    cat_per_order = oi_cat.groupby("order_id")["product_category_name"].agg(first_mode).reset_index()
    cat_per_order = cat_per_order.rename(columns={"product_category_name": "primary_category"})
    seller_per_order = (
        oi_cat.sort_values("order_item_id").groupby("order_id")["seller_id"].first().reset_index()
        .rename(columns={"seller_id": "primary_seller_id"})
    )

    # Payments
    pay_agg = payments.groupby("order_id").agg(
        total_order_value=("payment_value", "sum"),
    ).reset_index()
    pay_type = (
        payments.sort_values("payment_sequential").groupby("order_id")["payment_type"].first().reset_index()
        .rename(columns={"payment_type": "payment_type_primary"})
    )
    pay_agg = pay_agg.merge(pay_type, on="order_id", how="left")

    df = orders.merge(items_agg, on="order_id", how="left")
    df = df.merge(prod_agg, on="order_id", how="left")
    df = df.merge(pay_agg, on="order_id", how="left")
    df = df.merge(cat_per_order, on="order_id", how="left")
    df = df.merge(seller_per_order, on="order_id", how="left")

    df["total_order_value"] = df["total_order_value"].fillna(df["total_items_price"])
    df["total_freight_value"] = df["total_freight_value"].fillna(0)
    df["item_count"] = df["item_count"].fillna(0).astype(int)

    # Target: only for delivered
    df["delivery_duration_days"] = np.nan
    mask = (df["order_status"] == "delivered") & df["order_delivered_customer_date"].notna() & df["order_purchase_timestamp"].notna()
    df.loc[mask, "delivery_duration_days"] = (
        (df.loc[mask, "order_delivered_customer_date"] - df.loc[mask, "order_purchase_timestamp"]).dt.total_seconds() / (24 * 3600)
    )
    df["delivery_duration_days"] = df["delivery_duration_days"].round(2)

    # A. Order complexity
    df["freight_ratio"] = np.where(
        df["total_order_value"] > 0,
        (df["total_freight_value"] / df["total_order_value"]).round(4),
        np.nan,
    )
    df["avg_item_price"] = np.where(
        df["item_count"] > 0,
        (df["total_order_value"] / df["item_count"]).round(2),
        np.nan,
    )

    # B. Product size & weight
    df["total_weight_g"] = df["total_weight_g"].fillna(0)
    df["total_volume_cm3"] = df["total_volume_cm3"].fillna(0)
    df["average_weight_per_item"] = np.where(
        df["item_count"] > 0,
        (df["total_weight_g"] / df["item_count"]).round(2),
        np.nan,
    )
    # bulky_flag: top quartile of volume (computed later on train) or simple threshold
    df["bulky_flag"] = 0  # set in add_bulky_flag()

    # Customers & sellers
    if customers is not None:
        df = df.merge(
            customers[["customer_id", "customer_zip_code_prefix", "customer_state"]],
            on="customer_id",
            how="left",
        )
    if sellers is not None:
        df = df.merge(
            sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]].rename(columns={"seller_id": "primary_seller_id"}),
            on="primary_seller_id",
            how="left",
        )

    # C. Geographic: distance, intra_state, region
    geo = get_geo_zip_centroids(tables)
    if geo is not None and len(geo) > 0:
        geo = geo.set_index("zip_code_prefix")
        df["_cz"] = df["customer_zip_code_prefix"].astype(str)
        df["_sz"] = df["seller_zip_code_prefix"].astype(str)
        df = df.merge(geo.rename(columns={"lat": "lat_c", "lng": "lng_c"}), left_on="_cz", right_index=True, how="left")
        df = df.merge(geo.rename(columns={"lat": "lat_s", "lng": "lng_s"}), left_on="_sz", right_index=True, how="left")
        valid = df["lat_c"].notna() & df["lng_c"].notna() & df["lat_s"].notna() & df["lng_s"].notna()
        df["distance_km"] = np.nan
        df.loc[valid, "distance_km"] = _haversine_km(
            df.loc[valid, "lat_c"].values, df.loc[valid, "lng_c"].values,
            df.loc[valid, "lat_s"].values, df.loc[valid, "lng_s"].values,
        )
        df = df.drop(columns=[c for c in ["_cz", "_sz", "lat_c", "lng_c", "lat_s", "lng_s"] if c in df.columns])
    else:
        df["distance_km"] = np.nan

    df["intra_state"] = (
        (df["customer_state"].astype(str).str.upper() == df["seller_state"].astype(str).str.upper()).astype(int)
    )
    df["customer_region"] = df["customer_state"].astype(str).str.upper().map(STATE_TO_REGION).fillna("other")
    df["seller_region"] = df["seller_state"].astype(str).str.upper().map(STATE_TO_REGION).fillna("other")

    return df


def add_bulky_flag(df, volume_p90=None):
    """Set bulky_flag=1 for orders in top 10% by total_volume_cm3. If volume_p90 from train, use it."""
    df = df.copy()
    if volume_p90 is None:
        volume_p90 = df["total_volume_cm3"].quantile(0.9)
    df["bulky_flag"] = (df["total_volume_cm3"] >= volume_p90).astype(int)
    return df


def add_geo_and_interaction_features(df):
    """
    Add geographic and interaction features for delivery duration model.
    - distance_km_bucket: categorical (d_0_500, d_500_1500, d_1500_plus, d_unknown).
    - region_to_region: seller_region + '_' + customer_region.
    - inter_state_x_bulky_flag: (1 - intra_state) * bulky_flag (inter-state and bulky).
    """
    df = df.copy()
    # distance_km buckets (approval-time-safe)
    if "distance_km" in df.columns:
        d = df["distance_km"].fillna(-1)
        df["distance_km_bucket"] = pd.cut(
            d,
            bins=[-2, 0, 500, 1500, 1e6],
            labels=["d_unknown", "d_0_500", "d_500_1500", "d_1500_plus"],
        ).astype(str)
    else:
        df["distance_km_bucket"] = "d_unknown"
    # region_to_region
    sr = df.get("seller_region", pd.Series("other", index=df.index)).fillna("other").astype(str)
    cr = df.get("customer_region", pd.Series("other", index=df.index)).fillna("other").astype(str)
    df["region_to_region"] = sr + "_" + cr
    # inter_state × bulky_flag: 1 when different state and bulky
    intra = df.get("intra_state", 0)
    bulky = df.get("bulky_flag", 0)
    df["inter_state_x_bulky_flag"] = ((1 - intra) * bulky).astype(int)
    return df


def add_historical_seller_features(df, time_col="order_purchase_timestamp", past_subset=None):
    """
    Leakage-safe: for each order use only past delivered orders (by time_col).
    Adds: seller_hist_avg_delivery_days, seller_hist_late_rate, seller_hist_volume.
    late = delivered after (purchase + 30 days) as proxy when estimated not used.
    """
    df = df.copy()
    if time_col not in df.columns:
        return df
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    full = past_subset if past_subset is not None else df
    delivered = full[(full["order_status"] == "delivered") & (full[TARGET].notna())].copy()
    delivered = delivered.dropna(subset=[time_col, "primary_seller_id"])
    delivered = delivered.sort_values(time_col)
    # Late proxy: duration > 30 days (simple threshold)
    delivered["_late"] = (delivered[TARGET] > 30).astype(int)
    delivered["seller_hist_avg_delivery_days"] = (
        delivered.groupby("primary_seller_id", sort=False)[TARGET]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    delivered["seller_hist_late_rate"] = (
        delivered.groupby("primary_seller_id", sort=False)["_late"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    delivered["seller_hist_volume"] = (
        delivered.groupby("primary_seller_id", sort=False)[time_col]
        .transform(lambda x: x.shift(1).expanding().count())
    )
    by_seller = (
        delivered[[time_col, "primary_seller_id", "seller_hist_avg_delivery_days", "seller_hist_late_rate", "seller_hist_volume"]]
        .drop_duplicates(subset=[time_col, "primary_seller_id"], keep="last")
        .sort_values(time_col)
    )
    glob_avg = delivered[TARGET].mean()
    glob_late = delivered["_late"].mean()
    df_sorted = df.sort_values(time_col).copy()
    for c in ["seller_hist_avg_delivery_days", "seller_hist_late_rate", "seller_hist_volume"]:
        df_sorted[c] = np.nan
    valid = df_sorted[time_col].notna() & df_sorted["primary_seller_id"].notna()
    if valid.any():
        left = df_sorted.loc[valid].copy()
        left = pd.merge_asof(left, by_seller, on=time_col, by="primary_seller_id", direction="backward")
        for c in ["seller_hist_avg_delivery_days", "seller_hist_late_rate", "seller_hist_volume"]:
            if c in left.columns:
                df_sorted.loc[valid, c] = left[c].values
    df_sorted["seller_hist_avg_delivery_days"] = df_sorted["seller_hist_avg_delivery_days"].fillna(glob_avg)
    df_sorted["seller_hist_late_rate"] = df_sorted["seller_hist_late_rate"].fillna(glob_late)
    df_sorted["seller_hist_volume"] = df_sorted["seller_hist_volume"].fillna(0)
    return df_sorted.reindex(df.index).copy()


def get_model_ready_delivery(df):
    """Restrict to delivered orders with non-null target."""
    out = df[(df["order_status"] == "delivered") & (df[TARGET].notna())].copy()
    out[TARGET] = out[TARGET].astype(float)
    return out


# ---------- Numeric feature list for diagnostics ----------
def get_numeric_candidates():
    return [
        "item_count", "total_order_value", "total_freight_value", "freight_ratio", "avg_item_price",
        "total_weight_g", "total_volume_cm3", "average_weight_per_item", "bulky_flag",
        "distance_km", "intra_state",
        "seller_hist_avg_delivery_days", "seller_hist_late_rate", "seller_hist_volume",
    ]


def get_categorical_candidates():
    return ["primary_category", "payment_type_primary", "seller_state", "customer_state", "customer_region", "seller_region"]


# ---------------------------------------------------------------------------
# 3. Data diagnostics
# ---------------------------------------------------------------------------
def diagnose_target(df, target_col=TARGET):
    """Target distribution, skewness, outliers (>60 days), log recommendation."""
    s = df[target_col].dropna()
    out = {
        "n": len(s),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "q01": float(s.quantile(0.01)),
        "q99": float(s.quantile(0.99)),
        "skew": float(s.skew()),
        "outliers_gt_60": int((s > 60).sum()),
        "pct_gt_60": float((s > 60).mean() * 100),
    }
    out["recommend_log"] = out["skew"] > 1.5 or out["outliers_gt_60"] > 0.01 * out["n"]
    return out


def correlation_analysis(df, numeric_cols, threshold=0.8):
    """Pearson and Spearman; high pairs > threshold."""
    df_n = df[numeric_cols].dropna(how="all")
    pearson = df_n.corr(method="pearson")
    spearman = df_n.corr(method="spearman")
    high_pairs = []
    for i, a in enumerate(pearson.columns):
        for j, b in enumerate(pearson.columns):
            if i >= j:
                continue
            p, s = pearson.loc[a, b], spearman.loc[a, b]
            if abs(p) >= threshold or abs(s) >= threshold:
                high_pairs.append((a, b, float(p), float(s)))
    return {"pearson": pearson, "spearman": spearman, "high_pairs": high_pairs}


def vif_analysis(df, numeric_cols, vif_threshold=10):
    """VIF for numeric features; flag high VIF."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df[numeric_cols].fillna(0)
    X = X.loc[:, (X.std() > 0)]
    if X.shape[1] < 2:
        return {"vif": pd.Series(dtype=float), "high_vif": []}
    vif = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns,
    )
    high_vif = vif[vif >= vif_threshold].index.tolist()
    return {"vif": vif, "high_vif": high_vif}


def feature_target_correlation(df, numeric_cols, target_col=TARGET):
    """Correlation of each feature with target (Pearson)."""
    s = df[target_col].dropna()
    valid = df.loc[s.index][numeric_cols].fillna(0)
    corrs = valid.corrwith(df.loc[valid.index, target_col])
    corrs = corrs.reindex(numeric_cols).fillna(0)
    return corrs.loc[corrs.abs().sort_values(ascending=False).index]


def segment_means(df, target_col=TARGET, group_cols=None):
    """Mean duration by category, seller_region, distance bucket, order value bucket."""
    if group_cols is None:
        group_cols = ["primary_category", "seller_region", "payment_type_primary"]
    out = {}
    for col in group_cols:
        if col not in df.columns:
            continue
        out[col] = df.groupby(col)[target_col].agg(["mean", "count"]).round(4)
    work = df.copy()
    if "distance_km" in work.columns:
        work["distance_bucket"] = pd.cut(work["distance_km"].fillna(0), bins=[-1, 500, 1000, 2000, 1e6], labels=["d_500", "d_1k", "d_2k", "d_far"])
        out["distance_bucket"] = work.groupby("distance_bucket", observed=True)[target_col].agg(["mean", "count"]).round(4)
    if "total_order_value" in work.columns:
        try:
            work["value_bucket"] = pd.qcut(work["total_order_value"].fillna(0), q=4, labels=["v_low", "v_mid1", "v_mid2", "v_high"], duplicates="drop")
        except Exception:
            work["value_bucket"] = "v_mid1"
        out["value_bucket"] = work.groupby("value_bucket", observed=True)[target_col].agg(["mean", "count"]).round(4)
    return out


# ---------------------------------------------------------------------------
# 4. Feature selection (correlation-based, VIF drop)
# ---------------------------------------------------------------------------
def select_features_correlation_based(df, numeric_cols, target_col=TARGET, min_corr=0.01):
    """Keep features with |corr(target)| >= min_corr; rank by abs correlation."""
    corrs = feature_target_correlation(df, numeric_cols, target_col)
    kept = corrs[corrs.abs() >= min_corr].index.tolist()
    return kept, corrs


def remove_redundant_vif(df, numeric_cols, vif_threshold=10):
    """Iteratively drop feature with highest VIF until all < threshold."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        return numeric_cols
    remaining = [c for c in numeric_cols if c in df.columns]
    X = df[remaining].fillna(0)
    X = X.loc[:, (X.std() > 0)]
    while X.shape[1] > 1:
        vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
        if vif.max() < vif_threshold:
            break
        drop = vif.idxmax()
        remaining = [c for c in remaining if c != drop]
        X = X.drop(columns=[drop])
    return remaining


# ---------------------------------------------------------------------------
# Prepare splits for training (used by train_delivery_duration.py)
# ---------------------------------------------------------------------------
def prepare_splits_delivery_duration(config_path=None, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    """
    Load config, build delivery duration dataset with approval-time-safe features,
    apply time-based split. Historical seller features use only past data (no leakage).
    Returns: (train_df, val_df, test_df, config).
    """
    if config_path is None:
        config_path = OUTPUT_DIR / "delivery_duration_feature_config.json"
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    tables = load_raw_tables()
    df = build_delivery_duration_dataset(tables)
    df = add_bulky_flag(df)
    df = add_geo_and_interaction_features(df)
    df = get_model_ready_delivery(df)
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    train_df = df.iloc[:t1].copy()
    val_df = df.iloc[t1:t2].copy()
    test_df = df.iloc[t2:].copy()
    train_df = add_historical_seller_features(train_df, past_subset=train_df)
    val_df = add_historical_seller_features(val_df, past_subset=train_df)
    test_df = add_historical_seller_features(test_df, past_subset=pd.concat([train_df, val_df], ignore_index=True))
    return train_df, val_df, test_df, config


# ---------------------------------------------------------------------------
# 5. Full pipeline and report
# ---------------------------------------------------------------------------
def run_full_pipeline(output_dir=None):
    """
    Build dataset, add features, run diagnostics, feature selection, write report.
    Returns: (df_ready, final_numeric, report_dict).
    """
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = load_raw_tables()
    df = build_delivery_duration_dataset(tables)
    df = add_bulky_flag(df)
    df = add_geo_and_interaction_features(df)
    df = get_model_ready_delivery(df)
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    # Time-based split for historical (use first 70% as "train" for historical)
    n = len(df)
    train_end = int(n * 0.7)
    train_df = df.iloc[:train_end]
    df = add_historical_seller_features(df, past_subset=train_df)

    # Re-get model ready after historical
    df = get_model_ready_delivery(df)
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    train_df = df.iloc[:int(len(df) * 0.7)]

    numeric_candidates = [c for c in get_numeric_candidates() if c in df.columns]
    cat_candidates = [c for c in get_categorical_candidates() if c in df.columns]

    # Diagnostics
    target_diag = diagnose_target(df)
    corr_diag = correlation_analysis(df, numeric_candidates, threshold=0.8)
    try:
        vif_diag = vif_analysis(df, numeric_candidates, vif_threshold=10)
    except Exception:
        vif_diag = {"vif": pd.Series(), "high_vif": []}
    feat_target = feature_target_correlation(df, numeric_candidates)
    segment_means_dict = segment_means(df, group_cols=["primary_category", "seller_region"])

    # Feature selection: drop high VIF, then correlation-based
    after_vif = remove_redundant_vif(df, numeric_candidates, vif_threshold=10)
    final_numeric, corr_rank = select_features_correlation_based(df, after_vif, min_corr=0.005)
    if not final_numeric:
        final_numeric = after_vif[:]

    # Importance comparison (placeholder: use abs correlation as proxy)
    importance_table = feat_target.reindex(final_numeric).fillna(0).abs().sort_values(ascending=False)
    importance_table = pd.DataFrame({"abs_corr_with_target": importance_table})

    # Report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("DELIVERY DURATION – FEATURE PREPARATION & DIAGNOSTICS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("1. LEAKAGE BOUNDARIES (approval-time-safe only)")
    report_lines.append("   Safe: order_purchase_timestamp, seller/customer location, product category,")
    report_lines.append("   order value, freight, item_count, product dimensions, payment type, historical seller.")
    report_lines.append("   Excluded: order_delivered_customer_date, order_estimated_delivery_date,")
    report_lines.append("   delivery_delay_days, late_delivery_flag, review_score.")
    report_lines.append("")
    report_lines.append("2. TARGET DISTRIBUTION (delivery_duration_days)")
    report_lines.append(f"   n={target_diag['n']}, mean={target_diag['mean']:.2f}, median={target_diag['median']:.2f}, std={target_diag['std']:.2f}")
    report_lines.append(f"   skewness={target_diag['skew']:.2f}, outliers (>60 days)={target_diag['outliers_gt_60']} ({target_diag['pct_gt_60']:.2f}%)")
    report_lines.append(f"   Recommend log-transform: {target_diag['recommend_log']}")
    report_lines.append("")
    report_lines.append("3. HIGH CORRELATION PAIRS (|r| > 0.8)")
    for t in corr_diag["high_pairs"][:20]:
        report_lines.append(f"   {t[0]} vs {t[1]}: Pearson={t[2]:.3f}, Spearman={t[3]:.3f}")
    if not corr_diag["high_pairs"]:
        report_lines.append("   None.")
    report_lines.append("")
    report_lines.append("4. HIGH VIF (multicollinearity)")
    report_lines.append(f"   {vif_diag.get('high_vif', [])}")
    report_lines.append("")
    report_lines.append("5. FEATURE–TARGET RANKING (|correlation|)")
    report_lines.append(feat_target.abs().sort_values(ascending=False).head(20).to_string())
    report_lines.append("")
    report_lines.append("6. FEATURE IMPORTANCE COMPARISON (|correlation| with target)")
    report_lines.append(importance_table.to_string())
    report_lines.append("")
    report_lines.append("7. FINAL SELECTED FEATURES")
    report_lines.append("   Numeric: " + ", ".join(final_numeric))
    report_lines.append("   Categorical: " + ", ".join(cat_candidates))
    report_lines.append("")
    report_lines.append("8. REASONING & RISKS")
    report_lines.append("   - Strongest predictors: see section 5 (e.g. intra_state, freight, weight/volume).")
    report_lines.append("   - Seller/location dominance: historical seller and distance can overfit to few sellers/regions.")
    report_lines.append("   - Overfitting risk: avoid using too many seller-specific dummies; prefer aggregated historical.")
    report_lines.append("")
    report_lines.append("9. TARGET TRANSFORMATION")
    report_lines.append(f"   Decision: {'Log-transform recommended (skew > 1.5 or notable outliers).' if target_diag['recommend_log'] else 'Raw target acceptable.'}")

    report_path = output_dir / "delivery_duration_feature_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Report written to {report_path}")

    config = {
        "target": TARGET,
        "numeric": final_numeric,
        "categorical": cat_candidates,
        "target_log_transform": target_diag["recommend_log"],
        "diagnostics": {
            "target_skew": target_diag["skew"],
            "target_mean": target_diag["mean"],
            "outliers_gt_60": target_diag["outliers_gt_60"],
        },
    }
    config_path = output_dir / "delivery_duration_feature_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {config_path}")

    return df, final_numeric, cat_candidates, importance_table, report_lines


if __name__ == "__main__":
    run_full_pipeline()
