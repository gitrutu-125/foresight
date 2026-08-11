"""
FORESIGHT synthetic data generator
-----------------------------------
Produces 4 CSV files matching the brief's schema:
  - sku_master.csv
  - calendar.csv
  - sales_daily.csv
  - inventory_snapshots.csv

Data is DELIBERATELY imperfect (missing values, duplicates, inconsistent
labels) to mimic a real client extract, per Section 05 of the brief.

Run:
    python src/generate_data.py
Outputs land in ./data/
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

N_SKUS = 200
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"   # 2 years of daily history
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "Furniture": ["Chairs", "Tables", "Sofas", "Shelving"],
    "Decor": ["Wall Art", "Vases", "Candles", "Mirrors"],
    "Small Appliances": ["Kettles", "Blenders", "Fans", "Lamps"],
    "Textiles": ["Cushions", "Throws", "Rugs", "Curtains"],
    "Lighting": ["Table Lamps", "Floor Lamps", "String Lights"],
}

# Inconsistent label variants injected later to simulate messy real data
CATEGORY_LABEL_VARIANTS = {
    "Furniture": ["Furniture", "furniture", "FURNITURE"],
    "Decor": ["Decor", "décor", "DECOR"],
    "Small Appliances": ["Small Appliances", "small appliances"],
    "Textiles": ["Textiles", "textiles"],
    "Lighting": ["Lighting", "lighting"],
}

# ---------------------------------------------------------------------
# 1. calendar.csv
# ---------------------------------------------------------------------
dates = pd.date_range(START_DATE, END_DATE, freq="D")
calendar = pd.DataFrame({"date": dates})
calendar["week"] = calendar["date"].dt.isocalendar().week.astype(int)
calendar["month"] = calendar["date"].dt.month
calendar["season"] = calendar["month"] % 12 // 3 + 1
calendar["season"] = calendar["season"].map({1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"})

# simple holiday list (India-relevant + generic retail peaks)
holidays = pd.to_datetime([
    "2024-01-01", "2024-08-15", "2024-10-02", "2024-10-31", "2024-11-01",
    "2024-12-25", "2025-01-01", "2025-08-15", "2025-10-02", "2025-10-20",
    "2025-12-25",
])
calendar["is_holiday"] = calendar["date"].isin(holidays).astype(int)

# promo events: a named sale window a few times a year
promo_windows = [
    ("2024-01-15", "2024-01-22", "New Year Sale"),
    ("2024-06-01", "2024-06-10", "Summer Clearance"),
    ("2024-10-25", "2024-11-05", "Festive Sale"),
    ("2024-12-20", "2024-12-31", "Year End Sale"),
    ("2025-01-15", "2025-01-22", "New Year Sale"),
    ("2025-06-01", "2025-06-10", "Summer Clearance"),
    ("2025-10-25", "2025-11-05", "Festive Sale"),
    ("2025-12-20", "2025-12-31", "Year End Sale"),
]
calendar["promo_event"] = None
for start, end, name in promo_windows:
    mask = (calendar["date"] >= start) & (calendar["date"] <= end)
    calendar.loc[mask, "promo_event"] = name

# ---------------------------------------------------------------------
# 2. sku_master.csv
# ---------------------------------------------------------------------
sku_rows = []
cat_list = list(CATEGORIES.keys())
for i in range(1, N_SKUS + 1):
    sku_id = f"SKU{i:04d}"
    category = np.random.choice(cat_list, p=[0.25, 0.25, 0.2, 0.15, 0.15])
    subcategory = np.random.choice(CATEGORIES[category])
    # most SKUs launch at/near the start; some launch later (new products)
    launch_offset_days = int(np.random.choice(
        [0, 0, 0, 0, 60, 150, 300, 450, 600], p=[0.55,0.05,0.05,0.05,0.08,0.07,0.06,0.05,0.04]
    ))
    launch_date = pd.Timestamp(START_DATE) + pd.Timedelta(days=launch_offset_days)
    unit_cost = round(np.random.uniform(150, 4000), 2)
    margin_mult = np.random.uniform(1.4, 2.3)
    list_price = round(unit_cost * margin_mult, 2)
    sku_rows.append([sku_id, category, subcategory, launch_date, unit_cost, list_price])

sku_master = pd.DataFrame(
    sku_rows,
    columns=["sku_id", "category", "subcategory", "launch_date", "unit_cost", "list_price"],
)

# inject inconsistent category labels into a copy used for sales join later (messiness)
sku_master_messy_labels = sku_master["category"].apply(
    lambda c: np.random.choice(CATEGORY_LABEL_VARIANTS[c])
)

# ---------------------------------------------------------------------
# 3. sales_daily.csv  (the big one — with seasonality, trend, noise)
# ---------------------------------------------------------------------
sales_records = []
promo_dates_set = set(calendar.loc[calendar["promo_event"].notna(), "date"])
holiday_dates_set = set(calendar.loc[calendar["is_holiday"] == 1, "date"])

for _, sku in sku_master.iterrows():
    sku_id = sku["sku_id"]
    base_demand = np.random.uniform(2, 25)          # avg units/day
    trend = np.random.uniform(-0.0005, 0.001)        # slow drift over time
    weekly_amp = np.random.uniform(0.1, 0.4)          # weekend effect
    season_amp = np.random.uniform(0.1, 0.5)
    noise_sd = base_demand * 0.35

    sku_dates = dates[dates >= sku["launch_date"]]
    for d_idx, d in enumerate(sku_dates):
        dow = d.dayofweek
        weekly_factor = 1 + weekly_amp * (1 if dow >= 5 else -0.3)
        season_factor = 1 + season_amp * np.sin(2 * np.pi * d.dayofyear / 365)
        trend_factor = 1 + trend * d_idx
        promo_flag = 1 if d in promo_dates_set else 0
        holiday_boost = 1.5 if d in holiday_dates_set else 1.0
        promo_boost = 1.8 if promo_flag else 1.0

        expected = base_demand * weekly_factor * season_factor * trend_factor * holiday_boost * promo_boost
        units = max(0, int(np.random.normal(expected, noise_sd)))

        unit_price = sku["list_price"] * (0.85 if promo_flag else 1.0)
        revenue = round(units * unit_price, 2)

        sales_records.append([d, sku_id, units, revenue, round(unit_price, 2), promo_flag])

sales_daily = pd.DataFrame(
    sales_records,
    columns=["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"],
)

# ---- inject messiness into sales_daily ----
# 1. missing values in revenue / unit_price (~1.5%)
mask_missing_rev = np.random.rand(len(sales_daily)) < 0.01
sales_daily.loc[mask_missing_rev, "revenue"] = np.nan
mask_missing_price = np.random.rand(len(sales_daily)) < 0.01
sales_daily.loc[mask_missing_price, "unit_price"] = np.nan

# 2. duplicate rows (~0.3%)
dupe_sample = sales_daily.sample(frac=0.003, random_state=SEED)
sales_daily = pd.concat([sales_daily, dupe_sample], ignore_index=True)

# 3. a few negative/garbage units_sold values (data entry errors, ~0.1%)
mask_bad = np.random.rand(len(sales_daily)) < 0.001
sales_daily.loc[mask_bad, "units_sold"] = -1

sales_daily = sales_daily.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle rows

# ---------------------------------------------------------------------
# 4. inventory_snapshots.csv  (weekly snapshot per SKU)
# ---------------------------------------------------------------------
inv_records = []
snapshot_dates = pd.date_range(START_DATE, END_DATE, freq="W-MON")

for _, sku in sku_master.iterrows():
    sku_id = sku["sku_id"]
    lead_time = int(np.random.choice([7, 14, 21, 30], p=[0.3, 0.4, 0.2, 0.1]))
    reorder_point = int(np.random.uniform(20, 150))
    on_hand = int(np.random.uniform(50, 400))

    for d in snapshot_dates:
        if d < sku["launch_date"]:
            continue
        # simple stock walk: drifts down with sales, jumps up on reorder
        weekly_sales_est = np.random.uniform(10, 100)
        on_hand = max(0, on_hand - weekly_sales_est)
        on_order = 0
        if on_hand < reorder_point:
            on_order = int(np.random.uniform(100, 300))
            on_hand += on_order * 0.3  # partial receipt simulation

        inv_records.append([d, sku_id, int(on_hand), int(on_order), lead_time, reorder_point])

inventory_snapshots = pd.DataFrame(
    inv_records,
    columns=["date", "sku_id", "on_hand_units", "on_order_units", "lead_time_days", "reorder_point"],
)

# inject a few missing lead_time values (~1%)
mask_missing_lt = np.random.rand(len(inventory_snapshots)) < 0.01
inventory_snapshots.loc[mask_missing_lt, "lead_time_days"] = np.nan

# ---------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------
sku_master.to_csv(OUT_DIR / "sku_master.csv", index=False)
calendar.to_csv(OUT_DIR / "calendar.csv", index=False)
sales_daily.to_csv(OUT_DIR / "sales_daily.csv", index=False)
inventory_snapshots.to_csv(OUT_DIR / "inventory_snapshots.csv", index=False)

print("Done. Files written to:", OUT_DIR)
print(f"  sku_master:          {len(sku_master):>7,} rows")
print(f"  calendar:            {len(calendar):>7,} rows")
print(f"  sales_daily:         {len(sales_daily):>7,} rows")
print(f"  inventory_snapshots: {len(inventory_snapshots):>7,} rows")
