# Stage 3 — Rolling-origin evaluation and simple benchmarks

## Scope

Stage 3 fixes the common forecast evaluation protocol and evaluates only three benchmarks specified in advance: Historical Mean, Naive, and Seasonal Naive. Forecasts use the original retail_bn series in RMB 100 million (亿元). No SARIMA, ARIMA, ARIMAX, ETS, model-order search, AIC/AICc ranking, CNY predictive test, PMI predictive test, or final forecast is included.

The code reads only data/processed/analysis_quarterly.csv. It does not modify the canonical dataset, raw sources, or submitted proposal.

## Evaluation design

A rolling-origin evaluation preserves the chronological information set at every forecast origin. A random split would mix the time order and would not show how performance changes as the available history expands.

The exact target window is **2005Q1–2026Q2 inclusive** for both horizons, giving **86 targets** per model and horizon:

- h = 2 is the primary horizon.
- h = 1 is retained as a secondary diagnostic.
- For target t and horizon h, origin T = t - h.
- Each expanding training sample is 1994Q1 through T, inclusive.
- All three benchmarks have the same targets at a given horizon. All 86 quarters are scored for all; the pre-specified q1 subgroup contains 22 quarters.

PMI availability does not set the target window. Future models must use this same target window and rolling convention.

## Benchmark definitions

1. **Historical Mean:** arithmetic mean of retail_bn in the training sample.
2. **Naive:** retail_bn observed at the forecast origin.
3. **Seasonal Naive:** retail_bn observed at target quarter t - 4.

For both horizons, t - 4 is at or before the origin, so the Seasonal Naive forecast uses an observation available by that origin. No drift or other benchmark was added.

## Seasonal MASE

For each origin T, the seasonal MASE scale is the mean of |y_t - y_(t-4)| over all valid seasonal differences fully contained in that origin's training sample. The scaled absolute error is the forecast's absolute error divided by that origin-specific scale. MASE is the arithmetic mean of the resulting per-forecast scaled absolute errors. No full-sample denominator is used.

Errors are actual - forecast. Metrics are MAE, RMSE, and MASE, reported for all and the pre-specified Q1 target subgroup. The primary comparison is h = 2, scope all.

## Results

All figures below are in original retail units (亿元); MASE is unit-free. Values are rounded for display. The full-precision result is in [benchmark_metrics.csv](../outputs/tables/benchmark_metrics.csv).

| Horizon | Model | Scope | N | MAE | RMSE | MASE |
|---:|---|---|---:|---:|---:|---:|
| 2 (primary) | Historical Mean | all | 86 | 47,701.43 | 53,655.68 | 14.753 |
| 2 (primary) | Naive | all | 86 | 6,914.05 | 9,376.89 | 2.204 |
| 2 (primary) | Seasonal Naive | all | 86 | 5,991.50 | 7,023.89 | 2.246 |
| 2 (primary) | Historical Mean | q1 | 22 | 45,709.87 | 51,500.66 | 14.312 |
| 2 (primary) | Naive | q1 | 22 | 4,905.25 | 6,476.38 | 1.794 |
| 2 (primary) | Seasonal Naive | q1 | 22 | 6,967.47 | 8,851.87 | 2.488 |
| 1 (secondary) | Historical Mean | all | 86 | 47,207.25 | 53,145.08 | 14.354 |
| 1 (secondary) | Naive | all | 86 | 5,298.71 | 7,938.62 | 1.600 |
| 1 (secondary) | Seasonal Naive | all | 86 | 5,991.50 | 7,023.89 | 2.195 |
| 1 (secondary) | Historical Mean | q1 | 22 | 45,153.51 | 50,923.39 | 13.870 |
| 1 (secondary) | Naive | q1 | 22 | 6,709.55 | 10,341.70 | 1.866 |
| 1 (secondary) | Seasonal Naive | q1 | 22 | 6,967.47 | 8,851.87 | 2.429 |

These are reference results for later models under the fixed protocol. They do not select a future model structure. The primary h = 2, all scores are the main benchmark comparison; the Q1 subgroup and h = 1 scores are reported without changing that priority.

The 2020Q1 disruption and 2022Q2 observation remain in the evaluation. Such unusual periods can affect RMSE, which squares large errors; they were not removed or downweighted.

## Outputs and verification

- [Rolling forecast records](../outputs/forecasts/benchmark_rolling_forecasts.csv) include model, horizon, origin, target, training boundaries and size, actual, forecast, errors, and the origin-specific MASE scale.
- [Metrics table](../outputs/tables/benchmark_metrics.csv) contains the requested model/horizon/scope MAE, RMSE, and MASE.
- [Actual and forecasts, h = 2](../outputs/figures/benchmark_h2_actual_vs_forecasts.png).
- [Forecast errors, h = 2](../outputs/figures/benchmark_h2_errors.png).
- [Machine-readable summary](../outputs/diagnostics/stage3_summary.json) records the window, counts, definitions, metrics, safeguards, scope, and artifacts.

Automated checks verify origin/target alignment, the exact contiguous target window, expanding training boundaries, each benchmark formula, training-only seasonal MASE, common target sets, resistance to target/future perturbations at earlier origins, deterministic artifacts, and canonical-input immutability.

No SARIMA conclusion or model selection is made in Stage 3.
