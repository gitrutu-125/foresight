"""
FORESIGHT demand forecast model
----------------------------------
Trains a LightGBM model on the engineered features and backtests it
against the seasonal-naive baseline using the SAME rolling-origin folds,
so the comparison is fair (brief Section 07: "compare to the baseline on
backtest. If it doesn't win, ship the baseline and say why.").

LEAKAGE RULE: at every origin, the model is trained ONLY on rows whose
week_start is strictly before that origin. No future data ever touches
training.

Run:
    python src/forecast_model.py

Input:  data/processed/features.csv
        data/processed/baseline_predictions.csv
Output: data/processed/model_vs_baseline_predictions.csv
        reports/model_backtest_results.md
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
from forecast_baseline import wape, bias, N_BACKTEST_ORIGINS, HORIZON_WEEKS, SEASONAL_LAG_WEEKS

FEATURE_COLS = [
    "lag_1w", "lag_2w", "lag_4w", "lag_8w",
    "rolling_mean_4w", "rolling_std_4w", "rolling_mean_8w", "rolling_std_8w",
    "momentum_4w", "month", "week_of_year", "weeks_since_launch",
    "promo_flag", "is_holiday", "category_code",
]
TARGET_COL = "units_sold"


def load_data():
    features = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["week_start"])
    baseline = pd.read_csv(
        PROCESSED_DIR / "baseline_predictions.csv", parse_dates=["week_start"]
    )[["sku_id", "week_start", "seasonal_naive", "seasonal_naive_is_fallback"]]

    df = features.merge(baseline, on=["sku_id", "week_start"], how="inner")
    print(f"Merged features + baseline: {len(df)} rows (features had {len(features)}, "
          f"{len(features) - len(df)} dropped for no matching baseline row)")
    return df


def train_and_predict_one_origin(df, origin):
    """Train on everything strictly before `origin`, predict the test window."""
    train = df[df["week_start"] < origin]
    test_window = [origin + pd.Timedelta(weeks=h) for h in range(HORIZON_WEEKS)]
    test = df[df["week_start"].isin(test_window)]

    if train.empty or test.empty:
        return None, None

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # demand can't be negative

    result = test[["sku_id", "week_start", TARGET_COL, "seasonal_naive"]].copy()
    result["model_forecast"] = preds
    return result, model


def rolling_origin_backtest_model(df):
    all_weeks = sorted(df["week_start"].unique())
    usable_weeks = all_weeks[SEASONAL_LAG_WEEKS + 8: -HORIZON_WEEKS]
    step = len(usable_weeks) // N_BACKTEST_ORIGINS
    origins = usable_weeks[::step][:N_BACKTEST_ORIGINS]

    fold_results = []
    all_predictions = []
    last_model = None

    for origin in origins:
        result, model = train_and_predict_one_origin(df, origin)
        if result is None:
            continue
        last_model = model
        model_wape = wape(result[TARGET_COL], result["model_forecast"])
        baseline_wape = wape(result[TARGET_COL], result["seasonal_naive"])
        model_bias = bias(result[TARGET_COL], result["model_forecast"])
        fold_results.append({
            "origin_week": origin,
            "model_wape": model_wape,
            "baseline_wape": baseline_wape,
            "model_bias": model_bias,
            "n_rows": len(result),
        })
        all_predictions.append(result)

    fold_df = pd.DataFrame(fold_results)
    preds_df = pd.concat(all_predictions, ignore_index=True)
    return fold_df, preds_df, last_model


def run():
    df = load_data()
    fold_df, preds_df, model = rolling_origin_backtest_model(df)

    print("\nRolling-origin backtest — model vs baseline:")
    print(fold_df.to_string(index=False))

    avg_model_wape = fold_df["model_wape"].mean()
    avg_baseline_wape = fold_df["baseline_wape"].mean()
    improvement_pct = (avg_baseline_wape - avg_model_wape) / avg_baseline_wape * 100
    beats_baseline = avg_model_wape < avg_baseline_wape

    print(f"\nAverage model WAPE:    {avg_model_wape:.3f}")
    print(f"Average baseline WAPE: {avg_baseline_wape:.3f}")
    print(f"Model {'BEATS' if beats_baseline else 'DOES NOT beat'} the baseline "
          f"({improvement_pct:+.1f}% {'improvement' if beats_baseline else 'change'})")

    # feature importance from the last-trained model (explainability)
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importance (most recent fold's model):")
    print(importance.to_string(index=False))

    # save outputs
    preds_df.to_csv(PROCESSED_DIR / "model_vs_baseline_predictions.csv", index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "model_backtest_results.md"
    with open(report_path, "w") as f:
        f.write("# Model Backtest Results — LightGBM vs Seasonal-Naive Baseline\n\n")
        f.write("Same rolling-origin folds used for both, per brief Section 07.\n\n")
        f.write("| Origin week | Model WAPE | Baseline WAPE | Model bias | Test rows |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in fold_df.iterrows():
            f.write(f"| {row['origin_week'].date()} | {row['model_wape']:.3f} | "
                     f"{row['baseline_wape']:.3f} | {row['model_bias']:+.2f} | {int(row['n_rows'])} |\n")
        f.write(f"\n**Average model WAPE: {avg_model_wape:.3f}**\n")
        f.write(f"\n**Average baseline WAPE: {avg_baseline_wape:.3f}**\n")
        f.write(f"\n**Result: model {'beats' if beats_baseline else 'does NOT beat'} the baseline "
                 f"({improvement_pct:+.1f}%).**\n")
        f.write("\n## Feature importance (most recent fold)\n\n")
        f.write("| Feature | Importance |\n|---|---|\n")
        for _, row in importance.iterrows():
            f.write(f"| {row['feature']} | {row['importance']:.0f} |\n")
    print(f"\nSaved report to {report_path}")

    return fold_df, preds_df, model


if __name__ == "__main__":
    run()
