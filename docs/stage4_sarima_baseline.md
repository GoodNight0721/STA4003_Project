# Stage 4 — Leak-free SARIMA baseline

## Scope and data

This stage fits only a SARIMA baseline to the canonical 1994Q1–2026Q2 quarterly retail series. The response is log_retail; forecasts are retransformed and scored on the original retail level in RMB 100 million (亿元). The model has no exogenous regressors. This stage makes no final future forecast.

## Candidate family and fixed specification

The candidate family contains exactly 46 specifications:

- SARIMA(p,1,q)(P,D,Q)[4].
- p and q are each in {0, 1, 2}; P and Q are each in {0, 1}; D is in {0, 1}.
- p + q + P + Q must not exceed 3.
- d is fixed at 1 for every candidate.
- Every candidate uses trend = n, no exogenous variables, state-space SARIMAX, simple_differencing = False, enforce_stationarity = True, and enforce_invertibility = True.
- The same optimizer, maximum iteration count, deterministic specification, and parameterization are used throughout. The innovation variance is estimated and counted as a parameter.

Stage 2 showed a strongly trending log level and slowly decaying ACF, so d = 1 remains the working regular-difference choice. Seasonal differencing is less settled: the seasonal-difference series alone was not stationary under the stated tests; the combined difference passed those tests, but its lag-4 ACF was -0.492 and raised an over-differencing concern. Accordingly, both D = 0 and D = 1 remain in the pre-set candidate family. The candidate grid is not enlarged or reduced based on these Stage 4 results.

## Per-origin selection and AICc

The Stage 3 window and rolling convention are reused exactly: targets are 2005Q1–2026Q2, h = 2 is primary, h = 1 is secondary, and each training sample is 1994Q1 through origin T = target - h. There are 87 unique origins across both horizons. For each unique origin, the script fits the full candidate grid once using only log_retail available through that origin, then reuses the selected fit for every horizon served by that origin.

Selection minimizes training-sample AICc:

 AICc = AIC + 2k(k+1)/(n-k-1)

Here k is the actual number of estimated parameters reported by the fitted SARIMAX result, including the innovation variance, and n is the state-space training nobs. simple_differencing = False keeps the differencing inside the state-space model so candidates are fit on the same original training sample. A candidate is excluded for a fit exception, non-convergence, non-finite likelihood/AIC/AICc, an invalid AICc denominator, or an nobs mismatch. Ties are resolved deterministically by the order tuple.

No target actual or rolling OOS error is passed to the selector. Selecting orders from the evaluation errors would use the scored outcomes to tune the model and make the reported comparison optimistic. OOS metrics are used only to describe the performance of this fixed training-only selection procedure.

## Rolling selection results

Across 87 unique rolling origins, 4,002 candidate fits were attempted. **3,894** converged with finite likelihood and AICc and **108** were excluded. These totals are the sums across origins; an individual candidate may be successful at one origin and fail at another.

| Selected seasonal difference | Origins | Share |
|---:|---:|---:|
| D = 0 | 87 | 100.0% |
| D = 1 | 0 | 0.0% |

The most frequent complete orders were:

| Nonseasonal order | Seasonal order | Origins | Share |
|---|---|---:|---:|
| (0,1,1) | (1,0,1,4) | 39 | 44.8% |
| (1,1,0) | (1,0,1,4) | 38 | 43.7% |
| (0,1,0) | (1,0,0,4) | 4 | 4.6% |
| (1,1,0) | (1,0,0,4) | 4 | 4.6% |
| (0,1,0) | (1,0,1,4) | 1 | 1.1% |
| (0,1,1) | (1,0,0,4) | 1 | 1.1% |

The rolling AICc selection is stable on D = 0 but alternates between two dominant nonseasonal orders, with a few other orders at early origins. This is training-only selection behavior, not a claim that D = 1 can never be useful. In particular, the Stage 2 negative lag-4 ACF remains a reason for caution about seasonal differencing. The OOS comparison was not used to alter the grid or select a single fixed order.

## Full-sample descriptive fit and residual diagnostics

A separate 1994Q1–2026Q2 AICc selection chose SARIMA(0,1,1)(1,0,1,4), with D = 0, AIC = -415.864 and AICc = -415.544. Of the 46 full-sample candidates, 44 converged with finite AICc and 2 were excluded. Its parameter estimates were:

| Parameter | Estimate |
|---|---:|
| MA(1) | -0.3664 |
| Seasonal AR(1), lag 4 | 0.9824 |
| Seasonal MA(1), lag 4 | -0.6132 |
| Innovation variance | 0.002057 |

After excluding the state-space likelihood burn, residual mean was -0.00057 and sample standard deviation was 0.05056. The lag-8 Ljung–Box statistic was 2.613 with p = 0.759 (model_df = p + q + P + Q = 3; 129 residuals). The residual ACF stays within its displayed approximate confidence bands, while the Q-Q plot shows tail departures, including several large residuals. Across the 87 rolling selected fits, lag-8 Ljung–Box p-values ranged from 0.506 to 1.000, had median 0.797, and none were below 0.05. The test uses model_df = p + q + P + Q at lag 8; it is a residual check, not proof of independent errors.

The full-sample fit is descriptive only. It supplies residual diagnostics and interpretation, and is not used for any historical rolling forecast. See [sarima_full_sample_diagnostics.csv](../outputs/tables/sarima_full_sample_diagnostics.csv) and [full-sample residual diagnostics](../outputs/figures/sarima_full_sample_residual_diagnostics.png).

## Log-to-level forecast convention

At each horizon, statsmodels provides the conditional log forecast mean mu and forecast variance v. The saved median is exp(mu). The primary level point forecast is the lognormal conditional mean:

 forecast = exp(mu + 0.5v)

All reported MAE, RMSE, and MASE use this bias-corrected conditional-mean level forecast. The median is saved for transparency but is not used for primary scoring. Forecast error is actual minus forecast. Seasonal MASE scales are taken directly from the Stage 3 rolling benchmark implementation, with a separate training-only denominator at each origin and seasonal period 4.

## Rolling evaluation and Stage 3 comparison

Both horizons contain 86 forecasts per model; the pre-specified Q1 subgroup contains 22 targets. Values below are in original retail units (亿元), except unit-free MASE. The primary reading is h = 2, all:

| Model | All MAE | All RMSE | All MASE | Q1 MAE | Q1 RMSE | Q1 MASE |
|---|---:|---:|---:|---:|---:|---:|
| SARIMA | 2,970.24 | 5,115.52 | 0.948 | 3,824.04 | 7,159.17 | 1.109 |
| Historical Mean | 47,701.43 | 53,655.68 | 14.753 | 45,709.87 | 51,500.66 | 14.312 |
| Naive | 6,914.05 | 9,376.89 | 2.204 | 4,905.25 | 6,476.38 | 1.794 |
| Seasonal Naive | 5,991.50 | 7,023.89 | 2.246 | 6,967.47 | 8,851.87 | 2.488 |

At h = 2 across all targets, this SARIMA selection procedure has lower MAE, RMSE, and MASE than each Stage 3 benchmark. In Q1 it has lower MAE and MASE than the three benchmarks, while its RMSE is higher than the Naive benchmark's. These are comparisons of the per-origin AICc selection procedure on the fixed evaluation window; they do not justify choosing orders from OOS performance or changing the pre-specified family.

Full-precision results for h = 1 and h = 2 and both scopes are in [sarima_metrics.csv](../outputs/tables/sarima_metrics.csv) and [model_comparison_stage4.csv](../outputs/tables/model_comparison_stage4.csv).

## Limitations and outputs

The series has only 130 quarterly observations, and early rolling training samples are shorter still. AICc selection can vary with the origin; the two dominant nonseasonal orders are close in frequency. Convergence filtering means not every candidate contributes at every origin. The lognormal correction uses the fitted conditional forecast variance and does not add parameter-estimation uncertainty. Ljung–Box is reported at one pre-set lag only. Results are for nominal retail levels and do not establish a final model or a future forecast.

Generated artifacts:

- [Per-target SARIMA forecasts](../outputs/forecasts/sarima_rolling_forecasts.csv).
- [Per-origin selections and Ljung–Box diagnostics](../outputs/tables/sarima_origin_selection.csv).
- [Selected-order and D frequencies](../outputs/tables/sarima_order_frequency.csv).
- [Full-sample selected-fit diagnostics and parameters](../outputs/tables/sarima_full_sample_diagnostics.csv).
- [SARIMA scores](../outputs/tables/sarima_metrics.csv).
- [Stage 3 benchmark comparison](../outputs/tables/model_comparison_stage4.csv).
- [h = 2 actual and forecast](../outputs/figures/sarima_h2_actual_vs_forecast.png).
- [h = 2 forecast errors](../outputs/figures/sarima_h2_errors.png).
- [Full-sample residual diagnostics](../outputs/figures/sarima_full_sample_residual_diagnostics.png).
- [Machine-readable summary](../outputs/diagnostics/stage4_summary.json).

No CNY or PMI regressors are used. Stage 4 does not proceed to Stage 5.
