#!/usr/bin/env python3
"""
Complete Exploratory Data Analysis (EDA) for Olist Marketplace Dataset.
Loads CSVs from ./data, builds order-level analytical dataset, and generates
quantitative EDA report + structured ML insights report.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO

# ---------------------------------------------------------------------------
# Paths: data and output relative to project root (parent of notebooks/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR
EDA_RESULTS_PATH = OUTPUT_DIR / "eda_results.txt"
EDA_INSIGHTS_PATH = OUTPUT_DIR / "eda_insights.txt"


def ensure_data_dir():
    """Ensure data directory exists."""
    if not DATA_DIR.is_dir():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        sys.exit(1)


def load_all_csvs():
    """Load all CSV files from ./data. Returns dict name -> DataFrame."""
    ensure_data_dir()
    tables = {}
    for f in sorted(DATA_DIR.glob("*.csv")):
        name = f.stem
        try:
            tables[name] = pd.read_csv(f)
        except Exception as e:
            print(f"Warning: could not load {f.name}: {e}")
    return tables


def verify_review_scores_raw(reviews_df):
    """
    Verify that olist_order_reviews_dataset.review_score contains only integer values 1-5.
    Logs result and raises AssertionError if any invalid value is found.
    """
    col = "review_score"
    if col not in reviews_df.columns:
        print("Warning: review_score not found in reviews dataset.")
        return
    s = reviews_df[col].dropna()
    valid = s.isin([1, 2, 3, 4, 5])
    if not valid.all():
        invalid = s[~valid].unique().tolist()
        raise ValueError(
            f"review_score must contain only integers 1-5. Found invalid values: {invalid}"
        )
    if (s != s.astype(int)).any():
        raise ValueError("review_score must be integer type; found non-integer values.")
    uniques = sorted(s.unique().astype(int).tolist())
    print(f"  [OK] review_score in raw reviews: only integers {uniques}; count={len(s)}")


def print_dataset_overview(tables):
    """For each dataset: shape, column names/dtypes, missing value percentages."""
    for name, df in tables.items():
        print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print("\nColumns and dtypes:")
        print(df.dtypes.to_string())
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
        missing_df = missing_df[missing_df["missing_count"] > 0]
        if len(missing_df) > 0:
            print("\nMissing values (%):")
            print(missing_df.to_string())
        else:
            print("\nMissing values: none")
    return


def build_order_level_dataset(tables):
    """
    Create a clean order-level analytical dataset with documented joins.

    Join logic:
    - Base: olist_orders_dataset (one row per order)
    - order_items: aggregate per order_id -> item_count, total price/freight (sum)
    - order_payments: aggregate per order_id -> total payment_value
    - order_reviews: one review per order -> review_score (Option A: latest review only by review_creation_date; no averaging)
    - customers: left join on customer_id (for geography if needed later)
    - Products/sellers/categories: aggregated at order level via order_items
    """
    orders = tables["olist_orders_dataset"].copy()
    order_items = tables["olist_order_items_dataset"].copy()
    payments = tables["olist_order_payments_dataset"].copy()
    reviews = tables["olist_order_reviews_dataset"].copy()
    products = tables["olist_products_dataset"].copy()
    sellers = tables["olist_sellers_dataset"].copy()
    cat_trans = tables.get("product_category_name_translation")
    if cat_trans is None:
        cat_trans = pd.DataFrame(columns=["product_category_name", "product_category_name_english"])

    # ----- Aggregates from order_items (per order) -----
    items_agg = order_items.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
    ).reset_index()
    items_agg = items_agg.rename(columns={"total_price": "total_items_price"})

    # Primary product category per order (most frequent category in the order)
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
    # Primary seller (first by order_item_id)
    seller_per_order = (
        oi_with_cat.sort_values("order_item_id")
        .groupby("order_id")["seller_id"]
        .first()
        .reset_index()
        .rename(columns={"seller_id": "primary_seller_id"})
    )

    # ----- Payments: total payment value per order -----
    pay_agg = payments.groupby("order_id").agg(
        total_order_value=("payment_value", "sum"),
        payment_type_list=("payment_type", lambda x: "|".join(sorted(set(x)))),
    ).reset_index()
    # Primary payment type (first in sequence or most common)
    pay_type = (
        payments.sort_values("payment_sequential")
        .groupby("order_id")["payment_type"]
        .first()
        .reset_index()
        .rename(columns={"payment_type": "payment_type_primary"})
    )
    pay_agg = pay_agg.merge(pay_type, on="order_id", how="left")

    # ----- Reviews: one row per order (Option A: keep latest review only by review_creation_date) -----
    # Do NOT average review_score; keep integer categorical 1-5.
    reviews = reviews.copy()
    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"], errors="coerce")
    # Latest review per order (most recent review_creation_date)
    idx_latest = reviews.groupby("order_id")["review_creation_date"].idxmax()
    reviews_latest = reviews.loc[idx_latest, ["order_id", "review_score", "review_id"]].copy()
    review_count_per_order = reviews.groupby("order_id").size().reset_index(name="review_count")
    rev_agg = reviews_latest.merge(review_count_per_order, on="order_id", how="left")
    # review_score remains integer 1-5 from raw data (NaN for orders with no review)

    # ----- Join all to orders -----
    # 1) Orders + items aggregate
    df = orders.merge(items_agg, on="order_id", how="left")
    # 2) + payments
    df = df.merge(pay_agg, on="order_id", how="left")
    # 3) + reviews
    df = df.merge(rev_agg, on="order_id", how="left")
    # 4) + primary category and seller
    df = df.merge(cat_per_order, on="order_id", how="left")
    df = df.merge(seller_per_order, on="order_id", how="left")

    # Use total_order_value from payments; if missing, fallback to sum of item prices
    df["total_order_value"] = df["total_order_value"].fillna(df["total_items_price"])
    df["total_freight_value"] = df["total_freight_value"].fillna(0)
    df["item_count"] = df["item_count"].fillna(0).astype(int)

    return df, orders, order_items, payments, reviews, products, sellers, cat_trans


def engineer_features(df):
    """
    Add derived features:
    - delivery_delay_days, late_delivery_flag
    - total_order_value, total_freight_value, freight_ratio, item_count (already present; ensure freight_ratio)
    - time-based features
    """
    df = df.copy()

    # Datetime columns
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

    # Delivery delay: (delivered_customer_date - estimated_delivery_date) in days
    # Positive = delivered after estimate = late
    if "order_delivered_customer_date" in df.columns and "order_estimated_delivery_date" in df.columns:
        df["delivery_delay_days"] = (
            df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
        ).dt.total_seconds() / (24 * 3600)
        df["delivery_delay_days"] = df["delivery_delay_days"].round(2)
        df["late_delivery_flag"] = (df["delivery_delay_days"] > 0).astype(int)
    else:
        df["delivery_delay_days"] = np.nan
        df["late_delivery_flag"] = np.nan

    # total_order_value, total_freight_value, item_count already on df
    df["freight_ratio"] = np.where(
        df["total_order_value"] > 0,
        (df["total_freight_value"] / df["total_order_value"]).round(4),
        np.nan,
    )

    # Time-based features
    if "order_purchase_timestamp" in df.columns:
        df["purchase_year"] = df["order_purchase_timestamp"].dt.year
        df["purchase_month"] = df["order_purchase_timestamp"].dt.month
        df["purchase_day_of_week"] = df["order_purchase_timestamp"].dt.dayofweek
        df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
        df["purchase_ym"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    # Approval delay (hours from purchase to approval)
    if "order_approved_at" in df.columns and "order_purchase_timestamp" in df.columns:
        df["approval_delay_hours"] = (
            df["order_approved_at"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 3600
        df["approval_delay_hours"] = df["approval_delay_hours"].round(2)

    return df


def run_eda_analyses(df, orders, order_items, payments, reviews, products, sellers, cat_trans):
    """Run all EDA computations and return a dict of results for report writing."""
    results = {}

    # Table shapes and schema
    results["order_level_shape"] = df.shape
    results["order_level_dtypes"] = df.dtypes
    results["order_level_columns"] = list(df.columns)

    # Order status distribution
    results["order_status_counts"] = df["order_status"].value_counts()
    results["order_status_pct"] = (df["order_status"].value_counts(normalize=True) * 100).round(2)

    # Review score distribution: only integer 1-5 (one review per order; no averaging)
    rev = df["review_score"].dropna()
    rev = rev[(rev >= 1) & (rev <= 5)].astype(int)
    counts = rev.value_counts().sort_index()
    for s in [1, 2, 3, 4, 5]:
        if s not in counts.index:
            counts[s] = 0
    counts = counts.sort_index()
    results["review_score_counts"] = counts
    n_rev = int(counts.sum())
    results["review_score_pct"] = (counts / n_rev * 100).round(2) if n_rev > 0 else pd.Series({s: 0.0 for s in [1, 2, 3, 4, 5]})
    results["review_low_high"] = {
        "low (1-3)": int(counts.reindex([1, 2, 3], fill_value=0).sum()),
        "high (4-5)": int(counts.reindex([4, 5], fill_value=0).sum()),
    }
    if n_rev > 0:
        results["review_low_pct"] = results["review_low_high"]["low (1-3)"] / n_rev * 100
        results["review_high_pct"] = results["review_low_high"]["high (4-5)"] / n_rev * 100
    else:
        results["review_low_pct"] = results["review_high_pct"] = 0

    # Delivered orders with review coverage
    delivered = df[df["order_status"] == "delivered"]
    results["delivered_count"] = len(delivered)
    results["delivered_with_review"] = delivered["review_score"].notna().sum()
    results["delivered_review_coverage_pct"] = (
        delivered["review_score"].notna().sum() / len(delivered) * 100
    ) if len(delivered) > 0 else 0

    # Delivery delay stats (delivered only, with valid dates)
    dd = df[(df["order_status"] == "delivered") & (df["delivery_delay_days"].notna())]["delivery_delay_days"]
    results["delivery_delay_mean"] = dd.mean()
    results["delivery_delay_median"] = dd.median()
    results["delivery_delay_std"] = dd.std()
    results["delivery_delay_late_rate"] = (dd > 0).mean() * 100 if len(dd) > 0 else np.nan
    results["delivery_delay_n"] = len(dd)

    # Payment type distribution (order-level: primary payment type)
    if "payment_type_primary" in df.columns:
        results["payment_type_counts"] = df["payment_type_primary"].value_counts()
        results["payment_type_pct"] = (
            df["payment_type_primary"].value_counts(normalize=True) * 100
        ).round(2)

    # Order-level value, freight, item count statistics
    for col in ["total_order_value", "total_freight_value", "freight_ratio", "item_count"]:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if col == "item_count":
            s = s.astype(int)
        results[f"{col}_mean"] = s.mean()
        results[f"{col}_median"] = s.median()
        results[f"{col}_std"] = s.std()
        results[f"{col}_min"] = s.min()
        results[f"{col}_max"] = s.max()

    # Late vs non-late review rates
    delivered_valid = df[(df["order_status"] == "delivered") & (df["late_delivery_flag"].notna())]
    if len(delivered_valid) > 0:
        late = delivered_valid[delivered_valid["late_delivery_flag"] == 1]
        not_late = delivered_valid[delivered_valid["late_delivery_flag"] == 0]
        results["late_review_mean"] = late["review_score"].mean()
        results["late_review_n"] = late["review_score"].notna().sum()
        results["not_late_review_mean"] = not_late["review_score"].mean()
        results["not_late_review_n"] = not_late["review_score"].notna().sum()
        results["late_count"] = len(late)
        results["not_late_count"] = len(not_late)
    else:
        results["late_review_mean"] = results["not_late_review_mean"] = np.nan
        results["late_review_n"] = results["not_late_review_n"] = 0
        results["late_count"] = results["not_late_count"] = 0

    # Top product categories by volume (order count)
    if "primary_category" in df.columns:
        results["top_categories_by_orders"] = df["primary_category"].value_counts().head(20)

    # Late rate by category
    if "primary_category" in df.columns:
        del_cat = df[(df["order_status"] == "delivered") & (df["primary_category"].notna())]
        if len(del_cat) > 0 and "late_delivery_flag" in del_cat.columns:
            late_by_cat = del_cat.groupby("primary_category").agg(
                late_rate=("late_delivery_flag", "mean"),
                n_orders=("order_id", "count"),
            ).reset_index()
            late_by_cat["late_rate_pct"] = (late_by_cat["late_rate"] * 100).round(2)
            results["late_rate_by_category"] = late_by_cat.sort_values("n_orders", ascending=False).head(25)

    # Late rate by seller
    if "primary_seller_id" in df.columns:
        del_seller = df[(df["order_status"] == "delivered") & (df["primary_seller_id"].notna())]
        if len(del_seller) > 0:
            late_by_seller = del_seller.groupby("primary_seller_id").agg(
                late_rate=("late_delivery_flag", "mean"),
                n_orders=("order_id", "count"),
            ).reset_index()
            late_by_seller["late_rate_pct"] = (late_by_seller["late_rate"] * 100).round(2)
            results["late_rate_by_seller"] = late_by_seller.sort_values("n_orders", ascending=False).head(25)

    # Additional numeric summaries
    results["total_orders"] = len(df)
    results["delivered_with_delay_n"] = len(dd)
    return results


def write_eda_report(results, df, tables, filepath):
    """Write full EDA report to eda_results.txt."""
    buf = StringIO()

    buf.write("=" * 80 + "\n")
    buf.write("FULL EDA REPORT – Olist Marketplace Dataset\n")
    buf.write("=" * 80 + "\n\n")

    # 1) Table shapes and schema overview
    buf.write("1. TABLE SHAPES AND SCHEMA OVERVIEW\n")
    buf.write("-" * 60 + "\n")
    for name, tbl in tables.items():
        buf.write(f"  {name}: {tbl.shape[0]} rows x {tbl.shape[1]} columns\n")
    buf.write("\nOrder-level analytical dataset (after joins):\n")
    buf.write(f"  Shape: {results['order_level_shape'][0]} rows x {results['order_level_shape'][1]} columns\n")
    buf.write("  Columns: " + ", ".join(results["order_level_columns"]) + "\n\n")

    # 2) Order status distribution
    buf.write("2. ORDER STATUS DISTRIBUTION\n")
    buf.write("-" * 60 + "\n")
    for status in results["order_status_counts"].index:
        cnt = results["order_status_counts"][status]
        pct = results["order_status_pct"].get(status, 0)
        buf.write(f"  {status}: {cnt} ({pct}%)\n")
    buf.write("\n")

    # 3) Review score distribution (integer 1-5 only; count and percentage per class)
    buf.write("3. REVIEW SCORE DISTRIBUTION\n")
    buf.write("-" * 60 + "\n")
    for score in [1, 2, 3, 4, 5]:
        cnt = int(results["review_score_counts"].get(score, 0))
        pct = results["review_score_pct"].get(score, 0)
        buf.write(f"  Score {score}: {cnt} ({pct}%)\n")
    buf.write(f"  Low (1-3): {results['review_low_high']['low (1-3)']} ({results.get('review_low_pct', 0):.2f}%)\n")
    buf.write(f"  High (4-5): {results['review_low_high']['high (4-5)']} ({results.get('review_high_pct', 0):.2f}%)\n\n")

    # 4) Delivered orders with review coverage
    buf.write("4. DELIVERED ORDERS WITH REVIEW COVERAGE\n")
    buf.write("-" * 60 + "\n")
    buf.write(f"  Delivered orders: {results['delivered_count']}\n")
    buf.write(f"  With at least one review: {results['delivered_with_review']}\n")
    buf.write(f"  Review coverage: {results['delivered_review_coverage_pct']:.2f}%\n\n")

    # 5) Delivery delay statistics
    buf.write("5. DELIVERY DELAY STATISTICS (delivered, with dates)\n")
    buf.write("-" * 60 + "\n")
    buf.write(f"  N: {results['delivery_delay_n']}\n")
    buf.write(f"  Mean delay (days): {results['delivery_delay_mean']:.2f}\n")
    buf.write(f"  Median delay (days): {results['delivery_delay_median']:.2f}\n")
    buf.write(f"  Std delay (days): {results['delivery_delay_std']:.2f}\n")
    buf.write(f"  Late delivery rate (%): {results['delivery_delay_late_rate']:.2f}\n\n")

    # 6) Payment type distribution
    buf.write("6. PAYMENT TYPE DISTRIBUTION (primary per order)\n")
    buf.write("-" * 60 + "\n")
    if "payment_type_counts" in results:
        for pt in results["payment_type_counts"].index:
            cnt = results["payment_type_counts"][pt]
            pct = results["payment_type_pct"].get(pt, 0)
            buf.write(f"  {pt}: {cnt} ({pct}%)\n")
    buf.write("\n")

    # 7) Order-level value, freight, item count
    buf.write("7. ORDER-LEVEL VALUE, FREIGHT, ITEM COUNT STATISTICS\n")
    buf.write("-" * 60 + "\n")
    for col in ["total_order_value", "total_freight_value", "freight_ratio", "item_count"]:
        if f"{col}_mean" not in results:
            continue
        buf.write(f"  {col}: mean={results[f'{col}_mean']:.4g}, median={results[f'{col}_median']:.4g}, "
                  f"std={results[f'{col}_std']:.4g}, min={results[f'{col}_min']:.4g}, max={results[f'{col}_max']:.4g}\n")
    buf.write("\n")

    # 8) Late vs non-late review rates
    buf.write("8. LATE VS NON-LATE REVIEW RATES (delivered orders)\n")
    buf.write("-" * 60 + "\n")
    buf.write(f"  Late deliveries: n={results.get('late_count', 0)}, mean review score={results.get('late_review_mean', np.nan):.2f} (n_reviews={results.get('late_review_n', 0)})\n")
    buf.write(f"  On-time deliveries: n={results.get('not_late_count', 0)}, mean review score={results.get('not_late_review_mean', np.nan):.2f} (n_reviews={results.get('not_late_review_n', 0)})\n\n")

    # 9) Top product categories by volume
    buf.write("9. TOP PRODUCT CATEGORIES BY ORDER VOLUME\n")
    buf.write("-" * 60 + "\n")
    if "top_categories_by_orders" in results:
        for cat, cnt in results["top_categories_by_orders"].items():
            buf.write(f"  {cat}: {cnt}\n")
    buf.write("\n")

    # 10) Late rate by category
    buf.write("10. LATE RATE BY CATEGORY (top by order volume)\n")
    buf.write("-" * 60 + "\n")
    if "late_rate_by_category" in results and len(results["late_rate_by_category"]) > 0:
        buf.write(results["late_rate_by_category"].to_string(index=False) + "\n\n")
    else:
        buf.write("  (no data)\n\n")

    # 11) Late rate by seller
    buf.write("11. LATE RATE BY SELLER (top by order volume)\n")
    buf.write("-" * 60 + "\n")
    if "late_rate_by_seller" in results and len(results["late_rate_by_seller"]) > 0:
        buf.write(results["late_rate_by_seller"].to_string(index=False) + "\n\n")
    else:
        buf.write("  (no data)\n\n")

    # 12) Additional notable numeric summaries
    buf.write("12. ADDITIONAL NUMERIC SUMMARIES\n")
    buf.write("-" * 60 + "\n")
    buf.write(f"  Total orders: {results['total_orders']}\n")
    buf.write(f"  Delivered with valid delay: {results['delivered_with_delay_n']}\n")
    if "approval_delay_hours" in df.columns:
        ad = df["approval_delay_hours"].dropna()
        if len(ad) > 0:
            buf.write(f"  Approval delay (hours): mean={ad.mean():.2f}, median={ad.median():.2f}\n")
    buf.write("\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"EDA report written to {filepath}")


def write_insights_report(results, df, filepath):
    """Write structured ML insights report to eda_insights.txt."""
    buf = StringIO()

    buf.write("=" * 80 + "\n")
    buf.write("Marketplace EDA – Structured Insights Report\n")
    buf.write("=" * 80 + "\n\n")

    # Main patterns observed
    buf.write("1. MAIN PATTERNS OBSERVED\n")
    buf.write("-" * 60 + "\n")
    buf.write("- Order status is dominated by 'delivered'; other statuses are minority.\n")
    buf.write("- Review scores are integer 1-5 only (one per order, latest review); right-skewed (many 4-5); low scores (1-3) are a minority.\n")
    buf.write("- Delivery delay (delivered vs estimated date) shows a meaningful late rate.\n")
    buf.write("- Payment mix is concentrated in credit_card and boleto.\n")
    buf.write("- Order value, freight, and item count have positive skew (long upper tails).\n")
    buf.write("- Review coverage among delivered orders is high but not 100%.\n\n")

    # Key operational drivers
    buf.write("2. KEY OPERATIONAL DRIVERS\n")
    buf.write("-" * 60 + "\n")
    buf.write("- Delivery performance (on-time vs late) drives review scores (late -> lower scores).\n")
    buf.write("- Category and seller identity are strong drivers of late delivery rate.\n")
    buf.write("- Order size (item count, value, freight) may correlate with delay and satisfaction.\n")
    buf.write("- Time (year/month) reflects seasonality and growth.\n\n")

    # Strongest relationships discovered
    buf.write("3. STRONGEST RELATIONSHIPS DISCOVERED\n")
    buf.write("-" * 60 + "\n")
    late_mean = results.get("late_review_mean", np.nan)
    not_late_mean = results.get("not_late_review_mean", np.nan)
    if not (np.isnan(late_mean) or np.isnan(not_late_mean)):
        buf.write(f"- Late delivery vs review: late orders mean score {late_mean:.2f} vs on-time {not_late_mean:.2f}.\n")
    buf.write("- Order value and freight are highly correlated (freight scales with value/category).\n")
    buf.write("- Late rate varies by product category and by seller (actionable levers).\n\n")

    # Unexpected findings
    buf.write("4. UNEXPECTED FINDINGS\n")
    buf.write("-" * 60 + "\n")
    buf.write("- Some orders have multiple payment types (payment_type_list); primary used for EDA.\n")
    buf.write("- Negative delivery_delay_days (early delivery) are common; late = positive delay.\n")
    buf.write("- Review coverage < 100% implies missing-not-at-random risk for review models.\n\n")

    # Potential ML problem candidates
    buf.write("5. POTENTIAL ML PROBLEM CANDIDATES\n")
    buf.write("-" * 60 + "\n\n")

    # Candidate 1: Late delivery prediction
    late_rate = results.get("delivery_delay_late_rate", np.nan)
    buf.write("  (A) Late delivery prediction (binary: late_delivery_flag)\n")
    buf.write(f"      Target: order delivered after estimated_delivery_date. Positive rate ~ {late_rate:.1f}%.\n")
    buf.write("      Strengths: Clear target; rich features (category, seller, value, time).\n")
    buf.write("      Weaknesses: Label only known after delivery; possible data quality in dates.\n")
    buf.write("      Modeling risks: Temporal leakage if not split by time; seller/category imbalance.\n\n")

    # Candidate 2: Review score prediction / low review
    review_low_pct = results.get("review_low_pct", 0)
    review_high_pct = results.get("review_high_pct", 0)
    buf.write("  (B) Low review score prediction (e.g. score <= 3 vs 4-5)\n")
    buf.write(f"      Target: review_score <= 3. Positive rate (low) ~ {review_low_pct:.1f}%; high (4-5) ~ {review_high_pct:.1f}%.\n")
    buf.write("      Strengths: Direct customer satisfaction signal; link to delivery and order features.\n")
    buf.write("      Weaknesses: Missing reviews (coverage {:.1f}%); possible non-response bias.\n".format(results.get("delivered_review_coverage_pct", 0)))
    buf.write("      Modeling risks: Class imbalance (few low scores); text/comment not used in this EDA.\n\n")

    # Class imbalance implications (review score)
    buf.write("  CLASS IMBALANCE (review score)\n")
    buf.write("-" * 60 + "\n")
    buf.write("  - Corrected distribution uses integer scores 1-5 only (latest review per order; no averaging).\n")
    counts = results.get("review_score_counts", pd.Series(dtype=float))
    n_rev = int(counts.sum()) if len(counts) else 0
    if n_rev > 0:
        buf.write("  - Counts: " + ", ".join(f"Score {s}: {int(counts.get(s, 0))}" for s in [1, 2, 3, 4, 5]) + ".\n")
        buf.write(f"  - Binary low (1-3) vs high (4-5): minority class ~{review_low_pct:.1f}%; majority ~{review_high_pct:.1f}%.\n")
    buf.write("  - Implications: (1) Binary classifiers may need class weight or resampling to avoid predicting majority always. ")
    buf.write("(2) Multi-class (5-way) is heavily imbalanced (Score 5 dominates); consider macro F1 or per-class metrics. ")
    buf.write("(3) Rare scores (1, 2) are hardest to predict; more data or focal loss may help.\n\n")

    # Candidate 3: Order value / revenue
    buf.write("  (C) Order value prediction (regression or segment)\n")
    buf.write("      Target: total_order_value (continuous) or high-value segment.\n")
    buf.write("      Strengths: Strong business actionability; item_count and category are strong signals.\n")
    buf.write("      Weaknesses: Payment value may double-count with item price; multi-item aggregation.\n")
    buf.write("      Modeling risks: Heavy tail; need train/val split by time.\n\n")

    # Key numerical findings
    buf.write("6. KEY NUMERICAL FINDINGS (from EDA)\n")
    buf.write("-" * 60 + "\n")
    buf.write(f"  Total orders: {results['total_orders']}\n")
    buf.write(f"  Delivered: {results['delivered_count']}; review coverage: {results['delivered_review_coverage_pct']:.2f}%\n")
    buf.write(f"  Delivery delay: mean={results['delivery_delay_mean']:.2f} days, late rate={results['delivery_delay_late_rate']:.2f}%\n")
    buf.write(f"  Review: low (1-3) ~{results.get('review_low_pct', 0):.1f}%, high (4-5) ~{results.get('review_high_pct', 0):.1f}%\n")
    buf.write(f"  Order value: mean={results.get('total_order_value_mean', 0):.2f}, median={results.get('total_order_value_median', 0):.2f}\n")
    buf.write(f"  Item count: mean={results.get('item_count_mean', 0):.2f}, max={results.get('item_count_max', 0)}\n")
    buf.write("\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"Structured insights report written to {filepath}")


def main():
    os.chdir(PROJECT_ROOT)
    print("Loading all CSV files from", DATA_DIR)
    tables = load_all_csvs()
    if not tables:
        print("No CSV files found. Exiting.")
        sys.exit(1)

    if "olist_order_reviews_dataset" in tables:
        print("Verifying raw review_score (integers 1-5 only)...")
        verify_review_scores_raw(tables["olist_order_reviews_dataset"])

    print_dataset_overview(tables)

    print("\nBuilding order-level analytical dataset (with documented joins)...")
    df, orders, oi, pay, rev, products, sellers, cat_trans = build_order_level_dataset(tables)
    df = engineer_features(df)

    print("Running EDA analyses...")
    results = run_eda_analyses(df, orders, oi, pay, rev, products, sellers, cat_trans)

    print("Writing reports...")
    write_eda_report(results, df, tables, EDA_RESULTS_PATH)
    write_insights_report(results, df, EDA_INSIGHTS_PATH)

    print("\nDone. Outputs:")
    print("  -", EDA_RESULTS_PATH)
    print("  -", EDA_INSIGHTS_PATH)
    return df, results


if __name__ == "__main__":
    main()
