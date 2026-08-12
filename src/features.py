"""
FORESIGHT feature engineering
-------------------------------
Aggregates daily sales to weekly, SKU-level series, then builds the
features the forecasting model will use: lags, rolling stats, calendar
signals, and promo/holiday signals.

LEAKAGE RULE (non-negotiable per the brief): every feature for week t
must only use information available *before* week t. Lags and rolling
stats are shifted by 1 week to guarantee this.

Run:
    python src/features.py

Input:  data/processed/analysis_ready.csv
Output: data/processed/features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

LAG_WEEKS = [1, 2, 4, 8]          # how many weeks back to look
ROLLING_WINDOWS = [4, 8]           # rolling mean/std window sizes (in weeks)


def load_clean():
    df = pd.read_csv(PROCESSED_DIR / "analysis_ready.csv", parse_dates=["date", "launch_date"])
    return df


def to_weekly(df):
    """Aggregate the daily analysis-ready table to one row per SKU per week."""
    df = df.copy()
    df["year_week"] = df["date"].dt.to_period("W-MON")  # week ending Monday, matches calendar.csv 'week'
    df["week_start"] = df["year_week"].apply(lambda p: p.start_time)

    weekly = df.groupby(["sku_id", "week_start"]).agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        promo_flag=("promo_flag", "max"),          # 1 if any promo day that week
        is_holiday=("is_holiday", "max"),           # 1 if any holiday that week
        on_hand_units=("on_hand_units", "last"),
        on_order_units=("on_order_units", "last"),
        lead_time_days=("lead_time_days", "last"),
        reorder_point=("reorder_point", "last"),
        category=("category", "first"),
        subcategory=("subcategory", "first"),
        unit_cost=("unit_cost", "first"),
        list_price=("list_price", "first"),
        launch_date=("launch_date", "first"),
    ).reset_index()

    weekly["month"] = weekly["week_start"].dt.month
    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["weeks_since_launch"] = (
        (weekly["week_start"] - weekly["launch_date"]).dt.days // 7
    ).clip(lower=0)

    weekly = weekly.sort_values(["sku_id", "week_start"]).reset_index(drop=True)
    return weekly


def add_features(weekly):
    """Add lag and rolling features, computed strictly on past weeks only."""
    weekly = weekly.copy()
    g = weekly.groupby("sku_id")["units_sold"]

    # --- Lag features: units sold N weeks ago ---
    for lag in LAG_WEEKS:
        weekly[f"lag_{lag}w"] = g.shift(lag)

    # --- Rolling mean/std over PAST weeks (shift(1) first so the current week is excluded) ---
    shifted = g.shift(1)
    for window in ROLLING_WINDOWS:
        weekly[f"rolling_mean_{window}w"] = (
            weekly.groupby("sku_id")["units_sold"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        weekly[f"rolling_std_{window}w"] = (
            weekly.groupby("sku_id")["units_sold"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
        )

    # --- Trend: simple slope of last 4 weeks vs prior 4 weeks (past-only) ---
    weekly["momentum_4w"] = weekly["rolling_mean_4w"] - weekly.groupby("sku_id")["rolling_mean_4w"].shift(4)

    # --- Category as a categorical code (LightGBM handles categoricals natively, but this is a safe default) ---
    weekly["category_code"] = weekly["category"].astype("category").cat.codes

    return weekly


def finalize(weekly):
    """Drop rows where the model has no usable history (first weeks of each SKU's life),
    and clearly document how many were dropped and why — this is a modelling choice,
    not a bug, so it must be visible."""
    before = len(weekly)
    # require at least the longest lag to be available
    required_col = f"lag_{max(LAG_WEEKS)}w"
    weekly_clean = weekly[weekly[required_col].notna()].copy()
    dropped = before - len(weekly_clean)
    print(f"Dropped {dropped} rows ({dropped/before*100:.1f}%) with insufficient history "
          f"(fewer than {max(LAG_WEEKS)} weeks of prior sales) — these SKU-weeks cannot be "
          f"used for training without leaking or fabricating history.")
    return weekly_clean


def run():
    df = load_clean()
    weekly = to_weekly(df)
    print(f"Aggregated to {len(weekly)} SKU-week rows across {weekly['sku_id'].nunique()} SKUs.")

    featured = add_features(weekly)
    final = finalize(featured)

    out_path = PROCESSED_DIR / "features.csv"
    final.to_csv(out_path, index=False)
    print(f"Saved feature-engineered dataset to {out_path}")
    print(f"Final shape: {final.shape}")
    print(f"Columns: {list(final.columns)}")
    return final


if __name__ == "__main__":
    run()
