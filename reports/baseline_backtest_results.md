# Baseline Backtest Results — Seasonal-Naive

Metric: **WAPE** (Weighted Absolute Percentage Error), secondary check: bias.

Method: rolling-origin backtest, 6 origins, 6-week horizon each.

| Origin week | WAPE | Bias (units/week) | Test rows |
|---|---|---|---|
| 2025-02-18 | 0.179 | -12.87 | 1111 |
| 2025-04-01 | 0.174 | -13.51 | 1176 |
| 2025-05-13 | 0.178 | -12.10 | 1176 |
| 2025-06-24 | 0.215 | -2.17 | 1176 |
| 2025-08-05 | 0.263 | +0.09 | 1192 |
| 2025-09-16 | 0.258 | -1.49 | 1200 |

**Average WAPE: 0.211** (21.1%)

**Average bias: -7.01** units/week (positive = over-forecasting, negative = under-forecasting)

This is the bar the real model (Week 3) must beat on the same backtest setup.
