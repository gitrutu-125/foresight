"""
FORESIGHT data pipeline
------------------------
Ingests the 4 raw extracts, cleans them, and merges them into one
analysis-ready dataset.

This is D1 from the brief: reproducible, coded (not manual), documented,
and re-runnable end-to-end from raw files with a single command.

Run:
    python src/pipeline.py

Output:
    data/processed/analysis_ready.csv
    reports/data_quality_report.md
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Collects one-line notes as we go, so the report always matches the code
log = []


def note(msg):
    print(msg)
    log.append(msg)


# ---------------------------------------------------------------------
# 1. Ingest
# ---------------------------------------------------------------------
def load_raw():
    sales_daily = pd.read_csv(RAW_DIR / "sales_daily.csv", parse_dates=["date"])
    sku_master = pd.read_csv(RAW_DIR / "sku_master.csv", parse_dates=["launch_date"])
    calendar = pd.read_csv(RAW_DIR / "calendar.csv", parse_dates=["date"])
    inventory_snapshots = pd.read_csv(
        RAW_DIR / "inventory_snapshots.csv", parse_dates=["date"]
    )
    note(f"Loaded raw files: sales_daily={len(sales_daily)}, sku_master={len(sku_master)}, "
         f"calendar={len(calendar)}, inventory_snapshots={len(inventory_snapshots)} rows")
    return sales_daily, sku_master, calendar, inventory_snapshots


# ---------------------------------------------------------------------
# 2. Clean sku_master
# ---------------------------------------------------------------------
def clean_sku_master(sku_master):
    before = len(sku_master)

    # Standardize category labels: strip whitespace, title-case
    sku_master["category"] = sku_master["category"].str.strip().str.title()
    n_variants_fixed = sku_master["category"].nunique()
    note(f"sku_master: standardized category labels to {n_variants_fixed} clean categories")

    # Drop exact duplicate SKU rows, keep first
    dupes = sku_master.duplicated(subset=["sku_id"]).sum()
    sku_master = sku_master.drop_duplicates(subset=["sku_id"], keep="first")
    if dupes:
        note(f"sku_master: removed {dupes} duplicate sku_id rows")

    # Drop rows with impossible economics (list_price < unit_cost) — flag, don't silently lose
    bad_econ = sku_master["list_price"] < sku_master["unit_cost"]
    if bad_econ.sum():
        note(f"sku_master: {bad_econ.sum()} SKUs have list_price < unit_cost — kept, flagged for client review")
        sku_master["flag_bad_economics"] = bad_econ
    else:
        sku_master["flag_bad_economics"] = False

    note(f"sku_master: {before} -> {len(sku_master)} rows after cleaning")
    return sku_master


# ---------------------------------------------------------------------
# 3. Clean sales_daily
# ---------------------------------------------------------------------
def clean_sales_daily(sales_daily, valid_skus):
    before = len(sales_daily)

    # Remove exact duplicate rows
    dupes = sales_daily.duplicated().sum()
    sales_daily = sales_daily.drop_duplicates()
    if dupes:
        note(f"sales_daily: removed {dupes} exact duplicate rows")

    # Remove rows referencing unknown SKUs (referential integrity)
    orphan_mask = ~sales_daily["sku_id"].isin(valid_skus)
    if orphan_mask.sum():
        note(f"sales_daily: removed {orphan_mask.sum()} rows with unknown sku_id")
        sales_daily = sales_daily[~orphan_mask]

    # Fix impossible values: negative units_sold -> treat as data-entry error, set to 0
    bad_units = sales_daily["units_sold"] < 0
    if bad_units.sum():
        note(f"sales_daily: {bad_units.sum()} rows had negative units_sold — corrected to 0 "
             f"(logged, not silently dropped)")
        sales_daily.loc[bad_units, "units_sold"] = 0

    # Missing unit_price: fill from same SKU's median price (reasonable, defensible default)
    missing_price = sales_daily["unit_price"].isna()
    if missing_price.sum():
        median_price = sales_daily.groupby("sku_id")["unit_price"].transform("median")
        sales_daily.loc[missing_price, "unit_price"] = median_price[missing_price]
        note(f"sales_daily: filled {missing_price.sum()} missing unit_price values "
             f"with that SKU's median price")

    # Missing revenue: recompute from units_sold * unit_price (self-consistent, more reliable than guessing)
    missing_rev = sales_daily["revenue"].isna()
    if missing_rev.sum():
        sales_daily.loc[missing_rev, "revenue"] = (
            sales_daily.loc[missing_rev, "units_sold"] * sales_daily.loc[missing_rev, "unit_price"]
        )
        note(f"sales_daily: recomputed {missing_rev.sum()} missing revenue values "
             f"as units_sold x unit_price")

    note(f"sales_daily: {before} -> {len(sales_daily)} rows after cleaning")
    return sales_daily


# ---------------------------------------------------------------------
# 4. Clean inventory_snapshots
# ---------------------------------------------------------------------
def clean_inventory(inventory_snapshots, valid_skus):
    before = len(inventory_snapshots)

    orphan_mask = ~inventory_snapshots["sku_id"].isin(valid_skus)
    if orphan_mask.sum():
        note(f"inventory_snapshots: removed {orphan_mask.sum()} rows with unknown sku_id")
        inventory_snapshots = inventory_snapshots[~orphan_mask]

    # Missing lead_time_days: fill with that SKU's most common lead time,
    # fall back to overall median if the SKU has no other snapshots
    missing_lt = inventory_snapshots["lead_time_days"].isna()
    if missing_lt.sum():
        sku_mode = inventory_snapshots.groupby("sku_id")["lead_time_days"].transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        )
        inventory_snapshots.loc[missing_lt, "lead_time_days"] = sku_mode[missing_lt]
        still_missing = inventory_snapshots["lead_time_days"].isna().sum()
        if still_missing:
            overall_median = inventory_snapshots["lead_time_days"].median()
            inventory_snapshots["lead_time_days"] = inventory_snapshots["lead_time_days"].fillna(overall_median)
        note(f"inventory_snapshots: filled {missing_lt.sum()} missing lead_time_days values")

    note(f"inventory_snapshots: {before} -> {len(inventory_snapshots)} rows after cleaning")
    return inventory_snapshots


# ---------------------------------------------------------------------
# 5. Merge everything into one analysis-ready table
# ---------------------------------------------------------------------
def merge_all(sales_daily, sku_master, calendar, inventory_snapshots):
    df = sales_daily.merge(sku_master, on="sku_id", how="left")
    df = df.merge(calendar, on="date", how="left")

    # inventory is a periodic (weekly) snapshot, not daily — merge_asof aligns each
    # sales row with the most recent snapshot on/before that date, per SKU
    df = df.sort_values("date")
    inv_sorted = inventory_snapshots.sort_values("date")

    merged_parts = []
    for sku_id, sku_df in df.groupby("sku_id"):
        sku_inv = inv_sorted[inv_sorted["sku_id"] == sku_id]
        if sku_inv.empty:
            merged_parts.append(sku_df)
            continue
        merged = pd.merge_asof(
            sku_df.sort_values("date"),
            sku_inv[["date", "on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]].sort_values("date"),
            on="date",
            direction="backward",
        )
        merged_parts.append(merged)

    result = pd.concat(merged_parts, ignore_index=True)

    # Rows that predate a SKU's first inventory snapshot have no stock position to
    # attach (merge_asof has nothing earlier to pull from). This is a known, small
    # edge case — documented rather than hidden.
    no_inv = result["on_hand_units"].isna().sum()
    if no_inv:
        note(f"NOTE: {no_inv} rows ({no_inv/len(result)*100:.2f}%) predate their SKU's first "
             f"inventory snapshot and have no stock position — expected edge case, left as NaN, "
             f"excluded from risk scoring in a later step rather than fabricated")

    note(f"Merged into one analysis-ready table: {len(result)} rows, {result.shape[1]} columns")
    return result


# ---------------------------------------------------------------------
# 6. Write data-quality report
# ---------------------------------------------------------------------
def write_report():
    report_path = REPORTS_DIR / "data_quality_report.md"
    with open(report_path, "w") as f:
        f.write("# Data-Quality Report — Project FORESIGHT\n\n")
        f.write("Auto-generated by `src/pipeline.py`. Every line below reflects an actual "
                "cleaning decision made in code.\n\n")
        for line in log:
            f.write(f"- {line}\n")
    note(f"Wrote data-quality report to {report_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run():
    sales_daily, sku_master, calendar, inventory_snapshots = load_raw()

    sku_master = clean_sku_master(sku_master)
    valid_skus = set(sku_master["sku_id"])

    sales_daily = clean_sales_daily(sales_daily, valid_skus)
    inventory_snapshots = clean_inventory(inventory_snapshots, valid_skus)

    analysis_ready = merge_all(sales_daily, sku_master, calendar, inventory_snapshots)

    out_path = PROCESSED_DIR / "analysis_ready.csv"
    analysis_ready.to_csv(out_path, index=False)
    note(f"Saved analysis-ready dataset to {out_path}")

    write_report()
    return analysis_ready


if __name__ == "__main__":
    run()
