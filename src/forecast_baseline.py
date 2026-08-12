"""
FORESIGHT baseline forecast + accuracy metric
------------------------------------------------
Builds the seasonal-naive baseline (the bar every real model must beat)
and defines WAPE, the primary accuracy metric, evaluated with a proper
rolling-origin backtest (never a random split — see brief Section 07).

Run:
    python src/forecast_baseline.py

Input:  data/processed/analysis_ready.csv (rebuilds weekly series itself)
Output: data/processed/baseline_predictions.csv
        reports/baseline_backtest_results.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

sys.path.insert(0, str(BASE_DIR / "src"))
from features import load_clean, to_weekly  # reuse the same weekly aggregation, single source of truth

SEASONAL_LAG_WEEKS = 52   # "same week last year"
N_BACKTEST_ORIGINS = 6    # how many rolling-origin folds to test
HORIZON_WEEKS = 6         # forecast horizon per the brief (6-8 weeks)


# ---------------------------------------------------------------------
# 1. Metric
# ---------------------------------------------------------------------
def wape(actual, predicted):
    """Weighted Absolute Percentage Error. Robust to low-volume SKUs
    (unlike MAPE, which explodes when actual is near zero)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]
    total_actual = actual.sum()
    if total_actual == 0:
        return np.nan
    return np.abs(actual - predicted).sum() / total_actual


def bias(actual, predicted):
    """Signed mean error — checks systematic over/under-forecasting."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    return (predicted[mask] - actual[mask]).mean()


# ---------------------------------------------------------------------
# 2. Seasonal-naive baseline
# ---------------------------------------------------------------------
def build_seasonal_naive(weekly):
    """Predict week t demand = actual demand the same SKU had
    SEASONAL_LAG_WEEKS ago. For SKUs without a full year of history yet
    (their first year), fall back to that SKU's rolling mean-to-date —
    documented explicitly, not silently substituted."""
    weekly = weekly.sort_values(["sku_id", "week_start"]).copy()

    weekly["seasonal_naive"] = weekly.groupby("sku_id")["units_sold"].shift(SEASONAL_LAG_WEEKS)

    fallback_mask = weekly["seasonal_naive"].isna()
    # fallback: expanding mean of that SKU's own history up to (not including) this week
    expanding_mean = (
        weekly.groupby("sku_id")["units_sold"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    weekly.loc[fallback_mask, "seasonal_naive"] = expanding_mean[fallback_mask]
    weekly["seasonal_naive_is_fallback"] = fallback_mask

    n_fallback = fallback_mask.sum()
    print(f"Seasonal-naive baseline: {n_fallback} rows ({n_fallback/len(weekly)*100:.1f}%) "
          f"lacked a full {SEASONAL_LAG_WEEKS}-week history and used an expanding-mean "
          f"fallback instead — flagged in 'seasonal_naive_is_fallback' column.")
    return weekly


# ---------------------------------------------------------------------
# 3. Rolling-origin backtest
# ---------------------------------------------------------------------
def rolling_origin_backtest(weekly):
    """Evaluate the baseline the way a real forecast would be judged:
    pick a cutoff week, pretend everything after it is unknown, forecast
    forward HORIZON_WEEKS, compare to what actually happened. Repeat at
    several cutoffs moving forward through time."""
    all_weeks = sorted(weekly["week_start"].unique())

    # choose N_BACKTEST_ORIGINS cutoff points spaced through the back half of the series
    # (need enough history before the cutoff, and enough actuals after it, for a fair test)
    usable_weeks = all_weeks[SEASONAL_LAG_WEEKS + 8: -HORIZON_WEEKS]
    if len(usable_weeks) < N_BACKTEST_ORIGINS:
        raise ValueError("Not enough weekly history for the requested number of backtest origins.")
    step = len(usable_weeks) // N_BACKTEST_ORIGINS
    origins = usable_weeks[::step][:N_BACKTEST_ORIGINS]

    results = []
    for origin in origins:
        test_window = [origin + pd.Timedelta(weeks=h) for h in range(HORIZON_WEEKS)]
        test_rows = weekly[weekly["week_start"].isin(test_window)]
        if test_rows.empty:
            continue
        w = wape(test_rows["units_sold"], test_rows["seasonal_naive"])
        b = bias(test_rows["units_sold"], test_rows["seasonal_naive"])
        results.append({"origin_week": origin, "wape": w, "bias": b, "n_rows": len(test_rows)})

    results_df = pd.DataFrame(results)
    return results_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run():
    df = load_clean()
    weekly = to_weekly(df)
    weekly = build_seasonal_naive(weekly)

    backtest_results = rolling_origin_backtest(weekly)

    overall_wape = backtest_results["wape"].mean()
    overall_bias = backtest_results["bias"].mean()

    print("\nRolling-origin backtest results (seasonal-naive baseline):")
    print(backtest_results.to_string(index=False))
    print(f"\nAverage WAPE across {len(backtest_results)} origins: {overall_wape:.3f}")
    print(f"Average bias: {overall_bias:+.2f} units/week")

    # save predictions
    out_path = PROCESSED_DIR / "baseline_predictions.csv"
    weekly.to_csv(out_path, index=False)
    print(f"\nSaved baseline predictions to {out_path}")

    # save backtest report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "baseline_backtest_results.md"
    with open(report_path, "w") as f:
        f.write("# Baseline Backtest Results — Seasonal-Naive\n\n")
        f.write(f"Metric: **WAPE** (Weighted Absolute Percentage Error), secondary check: bias.\n\n")
        f.write(f"Method: rolling-origin backtest, {len(backtest_results)} origins, "
                f"{HORIZON_WEEKS}-week horizon each.\n\n")
        f.write("| Origin week | WAPE | Bias (units/week) | Test rows |\n")
        f.write("|---|---|---|---|\n")
        for _, row in backtest_results.iterrows():
            f.write(f"| {row['origin_week'].date()} | {row['wape']:.3f} | {row['bias']:+.2f} | {int(row['n_rows'])} |\n")
        f.write(f"\n**Average WAPE: {overall_wape:.3f}** ({overall_wape*100:.1f}%)\n")
        f.write(f"\n**Average bias: {overall_bias:+.2f}** units/week ")
        f.write("(positive = over-forecasting, negative = under-forecasting)\n")
        f.write(f"\nThis is the bar the real model (Week 3) must beat on the same backtest setup.\n")
    print(f"Saved backtest report to {report_path}")

    return weekly, backtest_results


if __name__ == "__main__":
    run()
