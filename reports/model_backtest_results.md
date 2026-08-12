# Model Backtest Results — LightGBM vs Seasonal-Naive Baseline

Same rolling-origin folds used for both, per brief Section 07.

| Origin week | Model WAPE | Baseline WAPE | Model bias | Test rows |
|---|---|---|---|---|
| 2025-04-15 | 0.095 | 0.169 | +3.89 | 1111 |
| 2025-05-20 | 0.153 | 0.184 | +6.86 | 1176 |
| 2025-06-24 | 0.126 | 0.215 | +1.12 | 1176 |
| 2025-07-29 | 0.139 | 0.256 | -0.27 | 1176 |
| 2025-09-02 | 0.146 | 0.272 | +0.26 | 1176 |
| 2025-10-07 | 0.174 | 0.218 | -8.66 | 1196 |

**Average model WAPE: 0.139**

**Average baseline WAPE: 0.219**

**Result: model beats the baseline (+36.6%).**

## Feature importance (most recent fold)

| Feature | Importance |
|---|---|
| week_of_year | 1117 |
| momentum_4w | 803 |
| lag_1w | 792 |
| lag_8w | 725 |
| rolling_mean_8w | 680 |
| lag_4w | 620 |
| rolling_mean_4w | 564 |
| rolling_std_4w | 552 |
| lag_2w | 542 |
| rolling_std_8w | 499 |
| weeks_since_launch | 495 |
| month | 298 |
| promo_flag | 260 |
| is_holiday | 167 |
| category_code | 147 |
