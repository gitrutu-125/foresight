"""
FORESIGHT risk scoring
-------------------------
Combines the demand forecast with current inventory position to flag
stockout / overstock risk per SKU, with a recommended action and the
rupee value at stake — Section 08 of the brief.

Approach:
1. Pick a reference "today" a few weeks before the end of the dataset
   (so the forecast horizon still has real calendar/promo data).
2. Recursively forecast forward HORIZON_WEEKS for every SKU using the
   final model (trained on all history up to "today").
3. Compare forecasted demand over lead time (stockout) and over the
   forward window (overstock) against current stock position.
4. Classify into the 4 quadrants from Section 08 and attach rupee impact.

Run:
    python src/risk_scoring.py

Output: data/processed/risk_scores.csv   (feeds Power BI + the scoring API)
        reports/risk_summary.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import lightgbm as lgb

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

sys.path.insert(0, str(BASE_DIR / "src"))
from features import load_clean, to_weekly, LAG_WEEKS, ROLLING_WINDOWS
from forecast_model import FEATURE_COLS, TARGET_COL

HORIZON_WEEKS = 6
RISK_THRESHOLD = 0.5   # matches the 0.5 line in the brief's decisioning grid (Figure 6)


# ---------------------------------------------------------------------
# 1. Train the final production model on all history up to "today"
# ---------------------------------------------------------------------
def train_final_model(weekly_features, today):
    train = weekly_features[weekly_features["week_start"] < today]
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=31, min_child_samples=20, random_state=42, verbosity=-1,
    )
    model.fit(train[FEATURE_COLS], train[TARGET_COL])
    print(f"Trained final model on {len(train)} rows (all history before {today.date()})")
    return model


# ---------------------------------------------------------------------
# 2. Recursive multi-week forecast per SKU
# ---------------------------------------------------------------------
def recursive_forecast(weekly_raw, calendar, model, today, sku_ids):
    """weekly_raw: full per-SKU weekly series (sku_id, week_start, units_sold, category_code,
    weeks_since_launch at each week). Extends each SKU's series forward HORIZON_WEEKS using
    the model, feeding each prediction back in as next week's lag input."""
    cal = calendar.set_index("date")
    forecasts = []

    for sku_id in sku_ids:
        series = weekly_raw[(weekly_raw["sku_id"] == sku_id) & (weekly_raw["week_start"] < today)].sort_values("week_start")
        if series.empty or len(series) < max(LAG_WEEKS) + 1:
            continue  # not enough history to forecast this SKU responsibly

        history = series["units_sold"].tolist()
        category_code = series["category_code"].iloc[-1]
        last_week_start = series["week_start"].iloc[-1]
        weeks_since_launch = series["weeks_since_launch"].iloc[-1]

        sku_forecasts = []
        for h in range(1, HORIZON_WEEKS + 1):
            future_week = last_week_start + pd.Timedelta(weeks=h)

            lag_feats = {f"lag_{lag}w": history[-lag] if len(history) >= lag else np.nan for lag in LAG_WEEKS}
            for window in ROLLING_WINDOWS:
                recent = history[-window:] if len(history) >= window else history
                lag_feats[f"rolling_mean_{window}w"] = np.mean(recent)
                lag_feats[f"rolling_std_{window}w"] = np.std(recent) if len(recent) > 1 else 0.0
            lag_feats["momentum_4w"] = lag_feats["rolling_mean_4w"] - (
                np.mean(history[-8:-4]) if len(history) >= 8 else lag_feats["rolling_mean_4w"]
            )

            # pull real promo/holiday flags if the future date exists in calendar, else assume none planned
            if future_week in cal.index:
                promo_flag = int(cal.loc[future_week, "promo_event"] is not None and not pd.isna(cal.loc[future_week, "promo_event"]))
                is_holiday = int(cal.loc[future_week, "is_holiday"]) if not isinstance(cal.loc[future_week, "is_holiday"], pd.Series) else int(cal.loc[future_week, "is_holiday"].max())
            else:
                promo_flag, is_holiday = 0, 0

            row = {
                **lag_feats,
                "month": future_week.month,
                "week_of_year": future_week.isocalendar()[1],
                "weeks_since_launch": weeks_since_launch + h,
                "promo_flag": promo_flag,
                "is_holiday": is_holiday,
                "category_code": category_code,
            }
            X = pd.DataFrame([row])[FEATURE_COLS]
            pred = max(0, model.predict(X)[0])

            sku_forecasts.append({"sku_id": sku_id, "week_start": future_week, "forecast_units": pred, "horizon_week": h})
            history.append(pred)  # feed forward for next step's lags

        forecasts.extend(sku_forecasts)

    return pd.DataFrame(forecasts)


# ---------------------------------------------------------------------
# 3. Risk scoring
# ---------------------------------------------------------------------
def score_risk(forecast_df, latest_inventory, sku_master):
    agg = forecast_df.groupby("sku_id").agg(
        demand_over_horizon=("forecast_units", "sum"),
        avg_weekly_demand=("forecast_units", "mean"),
    ).reset_index()

    df = agg.merge(latest_inventory, on="sku_id", how="left")
    df = df.merge(sku_master[["sku_id", "unit_cost", "list_price", "category"]], on="sku_id", how="left")

    df["lead_time_weeks"] = df["lead_time_days"] / 7
    df["demand_over_lead_time"] = df["avg_weekly_demand"] * df["lead_time_weeks"]
    df["available_stock"] = df["on_hand_units"].fillna(0) + df["on_order_units"].fillna(0)

    # --- stockout risk: how much of lead-time demand is NOT covered by available stock ---
    shortage = (df["demand_over_lead_time"] - df["available_stock"]).clip(lower=0)
    df["stockout_risk"] = (shortage / df["demand_over_lead_time"].replace(0, np.nan)).clip(0, 1).fillna(0)

    # --- overstock risk: how much on-hand stock exceeds forward demand ---
    excess = (df["on_hand_units"] - df["demand_over_horizon"]).clip(lower=0)
    df["overstock_risk"] = (excess / df["on_hand_units"].replace(0, np.nan)).clip(0, 1).fillna(0)

    # --- quadrant classification, matching Section 08 Figure 6 ---
    def classify(row):
        s, o = row["stockout_risk"], row["overstock_risk"]
        if s >= RISK_THRESHOLD and o >= RISK_THRESHOLD:
            return "Watch / Volatile"
        elif s >= RISK_THRESHOLD:
            return "Reorder Now"
        elif o >= RISK_THRESHOLD:
            return "Markdown / Clear"
        else:
            return "Healthy"

    df["risk_quadrant"] = df.apply(classify, axis=1)

    action_map = {
        "Reorder Now": "Raise a replenishment order before stock runs out.",
        "Markdown / Clear": "Promote or discount to free up capital.",
        "Watch / Volatile": "Investigate — demand is erratic; review manually.",
        "Healthy": "No action needed; leave as is.",
    }
    df["recommended_action"] = df["risk_quadrant"].map(action_map)

    # --- rupee value at stake ---
    df["shortage_units"] = shortage
    df["excess_units"] = excess
    df["rupee_at_risk_stockout"] = df["shortage_units"] * df["list_price"]     # lost sales value
    df["rupee_locked_overstock"] = df["excess_units"] * df["unit_cost"]        # capital locked

    def rupee_value(row):
        if row["risk_quadrant"] == "Reorder Now":
            return row["rupee_at_risk_stockout"]
        elif row["risk_quadrant"] == "Markdown / Clear":
            return row["rupee_locked_overstock"]
        elif row["risk_quadrant"] == "Watch / Volatile":
            return max(row["rupee_at_risk_stockout"], row["rupee_locked_overstock"])
        return 0.0

    df["rupee_value_at_stake"] = df.apply(rupee_value, axis=1)

    return df.sort_values("rupee_value_at_stake", ascending=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run():
    raw = load_clean()
    weekly_features = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["week_start"])
    calendar = raw[["date", "is_holiday", "promo_event"]].drop_duplicates(subset="date")

    weekly_raw = to_weekly(raw)  # full series incl. rows dropped from features.csv (need full history for recursion)
    weekly_raw["category_code"] = weekly_raw["category"].astype("category").cat.codes  # same encoding as features.py

    today = weekly_features["week_start"].max() - pd.Timedelta(weeks=HORIZON_WEEKS)
    print(f"Reference 'today' for risk scoring: {today.date()}")

    model = train_final_model(weekly_features, today)

    sku_ids = weekly_features["sku_id"].unique()
    forecast_df = recursive_forecast(weekly_raw, calendar, model, today, sku_ids)
    print(f"Generated {HORIZON_WEEKS}-week forecasts for {forecast_df['sku_id'].nunique()} SKUs")

    latest_inventory = (
        weekly_raw[weekly_raw["week_start"] < today]
        .sort_values("week_start")
        .groupby("sku_id")
        .last()[["on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]]
        .reset_index()
    )
    sku_master = raw[["sku_id", "unit_cost", "list_price", "category"]].drop_duplicates(subset="sku_id")

    risk_df = score_risk(forecast_df, latest_inventory, sku_master)

    out_path = PROCESSED_DIR / "risk_scores.csv"
    risk_df.to_csv(out_path, index=False)
    print(f"\nSaved risk scores to {out_path}")

    quadrant_counts = risk_df["risk_quadrant"].value_counts()
    total_at_risk = risk_df.loc[risk_df["risk_quadrant"] == "Reorder Now", "rupee_value_at_stake"].sum()
    total_locked = risk_df.loc[risk_df["risk_quadrant"] == "Markdown / Clear", "rupee_value_at_stake"].sum()

    print("\nQuadrant breakdown:")
    print(quadrant_counts)
    print(f"\nTotal sales at risk (Reorder Now): Rs {total_at_risk:,.0f}")
    print(f"Total capital locked (Markdown/Clear): Rs {total_locked:,.0f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "risk_summary.md", "w") as f:
        f.write("# Risk Scoring Summary — Project FORESIGHT\n\n")
        f.write(f"Reference date: {today.date()} | Horizon: {HORIZON_WEEKS} weeks\n\n")
        f.write("## Quadrant breakdown\n\n")
        f.write("| Quadrant | SKU count |\n|---|---|\n")
        for q, c in quadrant_counts.items():
            f.write(f"| {q} | {c} |\n")
        f.write(f"\n## Business impact\n\n")
        f.write(f"- **Sales at risk from stockouts:** Rs {total_at_risk:,.0f}\n")
        f.write(f"- **Capital locked in overstock:** Rs {total_locked:,.0f}\n")
        f.write("\n## Top 10 SKUs by rupee value at stake\n\n")
        f.write(risk_df[["sku_id", "category", "risk_quadrant", "recommended_action", "rupee_value_at_stake"]]
                .head(10).to_markdown(index=False))
        f.write("\n")
    print(f"Saved summary to {REPORTS_DIR / 'risk_summary.md'}")

    return risk_df


if __name__ == "__main__":
    run()
