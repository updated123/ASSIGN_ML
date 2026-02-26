"""
Time-drift diagnostics for delivery_duration_days.

Compares monthly mean target, train vs test target distribution, and tail frequency over time.
Writes notebooks/delivery_duration_time_drift.txt.
"""

from pathlib import Path
import numpy as np
import pandas as pd

try:
    from src.feature_preparation_delivery_duration import (
        build_delivery_duration_dataset,
        add_bulky_flag,
        add_geo_and_interaction_features,
        get_model_ready_delivery,
        TARGET,
        PROJECT_ROOT,
    )
    from src.data import load_raw_tables
except ImportError:
    from feature_preparation_delivery_duration import (
        build_delivery_duration_dataset,
        add_bulky_flag,
        add_geo_and_interaction_features,
        get_model_ready_delivery,
        TARGET,
        PROJECT_ROOT,
    )
    from data import load_raw_tables

OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "delivery_duration_time_drift.txt"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def run_time_drift_diagnostics():
    tables = load_raw_tables()
    df = build_delivery_duration_dataset(tables)
    df = add_bulky_flag(df)
    df = add_geo_and_interaction_features(df)
    df = get_model_ready_delivery(df)
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    df["_month"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.to_period("M")

    n = len(df)
    t1 = int(n * TRAIN_FRAC)
    t2 = int(n * (TRAIN_FRAC + VAL_FRAC))
    train_df = df.iloc[:t1]
    val_df = df.iloc[t1:t2]
    test_df = df.iloc[t2:]

    lines = []
    lines.append("=" * 70)
    lines.append("DELIVERY DURATION – TIME DRIFT DIAGNOSTICS")
    lines.append("=" * 70)
    lines.append("")

    # 1. Monthly mean delivery_duration_days
    lines.append("1. MONTHLY MEAN delivery_duration_days")
    monthly = df.groupby("_month", observed=True)[TARGET].agg(["mean", "std", "count"])
    for idx, row in monthly.iterrows():
        lines.append(f"   {idx}: mean={row['mean']:.2f} days, std={row['std']:.2f}, n={int(row['count'])}")
    lines.append("")

    # 2. Train vs test target distribution
    lines.append("2. TRAIN vs TEST TARGET DISTRIBUTION")
    for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        s = part[TARGET]
        lines.append(f"   {name}: n={len(s)}, mean={s.mean():.2f}, median={s.median():.2f}, std={s.std():.2f}")
        lines.append(f"         p05={s.quantile(0.05):.1f}, p25={s.quantile(0.25):.1f}, p75={s.quantile(0.75):.1f}, p95={s.quantile(0.95):.1f}, p99={s.quantile(0.99):.1f}")
    lines.append("")

    # 3. Tail frequency over time (% > 30 days, % > p99_train)
    lines.append("3. TAIL FREQUENCY OVER TIME")
    p99_train = train_df[TARGET].quantile(0.99)
    lines.append(f"   Train p99 = {p99_train:.1f} days (used as tail threshold).")
    monthly_gt30 = df.groupby("_month", observed=True)[TARGET].apply(lambda x: (x > 30).mean() * 100)
    monthly_gt99 = df.groupby("_month", observed=True)[TARGET].apply(lambda x: (x > p99_train).mean() * 100)
    lines.append("   % > 30 days by month:")
    for idx in monthly_gt30.index:
        lines.append(f"      {idx}: >30d={monthly_gt30.loc[idx]:.2f}%, >train_p99={monthly_gt99.loc[idx]:.2f}%")
    lines.append("")
    lines.append("   Train % > 30 days: {:.2f}%".format((train_df[TARGET] > 30).mean() * 100))
    lines.append("   Test  % > 30 days: {:.2f}%".format((test_df[TARGET] > 30).mean() * 100))
    lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Time drift diagnostics written to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    run_time_drift_diagnostics()
