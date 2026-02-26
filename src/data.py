"""
Data loading and order-level analytical dataset construction.
Shared with EDA and feature preparation; single source of truth for joins.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_raw_tables():
    """Load all CSV files from ./data. Returns dict name -> DataFrame."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    tables = {}
    for f in sorted(DATA_DIR.glob("*.csv")):
        tables[f.stem] = pd.read_csv(f)
    return tables


def build_order_level_dataset(tables=None):
    """
    Build order-level analytical dataset with documented joins.
    If tables is None, loads from DATA_DIR.
    Returns: df (order-level), and optionally the raw tables if needed.
    """
    if tables is None:
        tables = load_raw_tables()
    orders = tables["olist_orders_dataset"].copy()
    order_items = tables["olist_order_items_dataset"].copy()
    payments = tables["olist_order_payments_dataset"].copy()
    reviews = tables["olist_order_reviews_dataset"].copy()
    products = tables["olist_products_dataset"].copy()
    customers = tables.get("olist_customers_dataset")
    sellers = tables.get("olist_sellers_dataset")

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
        payment_type_list=("payment_type", lambda x: "|".join(sorted(set(x)))),
    ).reset_index()
    pay_type = (
        payments.sort_values("payment_sequential")
        .groupby("order_id")["payment_type"]
        .first()
        .reset_index()
        .rename(columns={"payment_type": "payment_type_primary"})
    )
    pay_agg = pay_agg.merge(pay_type, on="order_id", how="left")

    rev_agg = (
        reviews.groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "count"),
        )
        .reset_index()
    )
    rev_agg["review_score"] = rev_agg["review_score"].round(1)

    df = orders.merge(items_agg, on="order_id", how="left")
    df = df.merge(pay_agg, on="order_id", how="left")
    df = df.merge(rev_agg, on="order_id", how="left")
    df = df.merge(cat_per_order, on="order_id", how="left")
    df = df.merge(seller_per_order, on="order_id", how="left")
    if customers is not None:
        df = df.merge(customers[["customer_id", "customer_zip_code_prefix"]], on="customer_id", how="left")
    if sellers is not None:
        seller_zips = sellers[["seller_id", "seller_zip_code_prefix"]].rename(columns={"seller_id": "primary_seller_id"})
        df = df.merge(seller_zips, on="primary_seller_id", how="left")

    df["total_order_value"] = df["total_order_value"].fillna(df["total_items_price"])
    df["total_freight_value"] = df["total_freight_value"].fillna(0)
    df["item_count"] = df["item_count"].fillna(0).astype(int)

    return df


def add_datetime_and_target_features(df):
    """
    Add datetime parsing, target-related and time-based features.
    - delivery_delay_days, late_delivery_flag (target; only valid for delivered)
    - freight_ratio, approval_delay_hours, purchase_* (safe at approval time)
    - days_until_estimated_delivery (known at approval: estimated - approved)
    """
    df = df.copy()
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in date_cols:
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

    if "order_purchase_timestamp" in df.columns:
        df["purchase_year"] = df["order_purchase_timestamp"].dt.year
        df["purchase_month"] = df["order_purchase_timestamp"].dt.month
        df["purchase_day_of_week"] = df["order_purchase_timestamp"].dt.dayofweek
        df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
        df["purchase_ym"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    if "order_approved_at" in df.columns and "order_purchase_timestamp" in df.columns:
        df["approval_delay_hours"] = (
            df["order_approved_at"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 3600
        df["approval_delay_hours"] = df["approval_delay_hours"].round(2)

    # Known at approval time: promised delivery is set when order is placed/approved
    if "order_approved_at" in df.columns and "order_estimated_delivery_date" in df.columns:
        df["days_until_estimated_delivery"] = (
            df["order_estimated_delivery_date"] - df["order_approved_at"]
        ).dt.total_seconds() / (24 * 3600)
        df["days_until_estimated_delivery"] = df["days_until_estimated_delivery"].round(1)

    return df


def get_geo_zip_centroids(tables=None):
    """Return DataFrame with one row per zip: zip_code_prefix, lat, lng (mean of geolocation points)."""
    if tables is None:
        tables = load_raw_tables()
    geo = tables.get("olist_geolocation_dataset")
    if geo is None:
        return pd.DataFrame(columns=["zip_code_prefix", "lat", "lng"])
    geo = geo.copy()
    geo["zip_code_prefix"] = geo["geolocation_zip_code_prefix"].astype(str)
    centroids = geo.groupby("zip_code_prefix").agg(
        lat=("geolocation_lat", "mean"),
        lng=("geolocation_lng", "mean"),
    ).reset_index()
    return centroids


def load_order_level_dataset():
    """Load and build full order-level dataset with all base + target + time features."""
    tables = load_raw_tables()
    df = build_order_level_dataset(tables)
    df = add_datetime_and_target_features(df)
    return df
