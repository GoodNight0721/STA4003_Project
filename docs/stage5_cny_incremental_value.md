# Stage 5 — Incremental Lunar New Year Forecasting Value

## Research question and scope

Stage 5 asks whether deterministic Lunar New Year timing improves quarterly retail forecasts beyond the Stage 4 SARIMA baseline. The comparison is between the Stage 4 forecast and the same SARIMA error structure with one CNY timing regressor. Improvement is not assumed. The primary endpoint remains h = 2 across all 86 common targets; h = 2 Q1, with 22 targets, is the prespecified subgroup because the regressor is active only in Q1. h = 1 is secondary.

There is no order search, PMI regressor, ETS/Theta model, OOS-based CNY redesign, or final future forecast in this stage.

## Frozen SARIMA structure and rolling fit

For each of the 87 unique Stage 4 rolling origins, the script reads the already-selected `(p,d,q)(P,D,Q)[4]` from `outputs/tables/sarima_origin_selection.csv`. It verifies that the file covers precisely the Stage 4 origin set and does not call the Stage 4 candidate search. The same order is used for all horizons served by an origin.

The response is `log_retail`. Estimation uses statsmodels state-space SARIMAX with the frozen regular and seasonal orders, `trend="n"`, `simple_differencing=False`, stationarity and invertibility enforcement, no scale concentration, and the Stage 4 L-BFGS settings (`maxiter=100`). One `cny_regressor` is added as exogenous input. Parameters, including the CNY coefficient, are re-estimated at each origin.

Two first attempts reported non-convergence, at origins 2004Q4 and 2007Q2. Each was continued once from its optimizer terminal parameters using the same L-BFGS settings. Both retries converged; the final count is **87 successful origin fits and 0 unresolved failures**, across **89 optimizer attempts**. No origin fell back to the SARIMA forecast. A repeated failure stops the run with an origin-specific error.

## CNY regressor and information set

For an origin `T`, only Q1 rows in the training sample through `T` enter the center:

`center_T = mean(cny_position_fraction for Q1 observations at or before T)`

For training and forecast rows:

`cny_regressor_t = is_q1_t * (cny_position_fraction_t - center_T)`

The regressor is exactly zero outside Q1. Each origin has its own training-only center, and forecast-step exog reuses that same center. The CNY date, Q1 indicator, and fractional calendar position for future quarters are deterministic calendar information known at the forecast origin. Future retail and PMI values are not supplied to fitting or forecasting.

The coefficient describes the conditional log-retail association with a Q1 timing-fraction deviation from the training Q1 mean under the fixed SARIMA dynamics. It is not causal evidence. A full-fraction change is the coefficient scale; smaller timing differences imply proportionally smaller log-scale effects under the fitted linear term.

## Forecasting and scoring protocol

Targets, origins, horizons, and the expanding training sample are exactly those in Stages 3 and 4. The same fitted SARIMA+CNY model supplies h = 1 and h = 2 when an origin serves both. The saved log-scale mean `mu` and variance `v` produce the median `exp(mu)` and the primary level forecast `exp(mu + 0.5 v)`. MAE, RMSE, and MASE use the bias-corrected level forecast.

For every horizon and target, the Stage 4 saved seasonal MASE scale is reused exactly. It is an origin-specific period-4 scale calculated from training data only. Forecasts are paired to Stage 4 on the exact same 86 targets at each horizon.

## Rolling results

MAE and RMSE are in RMB 100 million (亿元); MASE is unit-free. `Delta` is SARIMA+CNY minus SARIMA, and percent change divides that delta by the SARIMA value. Negative deltas indicate improvement.

### Primary endpoint: h = 2, all targets

| Model | n | MAE | RMSE | MASE |
|---|---:|---:|---:|---:|
| SARIMA | 86 | 2,970.24 | 5,115.52 | 0.947893 |
| SARIMA+CNY | 86 | 2,965.73 | 5,021.59 | 0.946407 |
| Delta |  | -4.51 (-0.15%) | -93.93 (-1.84%) | -0.001486 (-0.16%) |

The CNY model's MAE and MASE reductions are very small. RMSE falls more because a few large target errors differ between the paired models. The fixed-protocol result does not show a material overall forecasting gain from the CNY regressor.

### Prespecified subgroup: h = 2, Q1 targets

| Model | n | MAE | RMSE | MASE |
|---|---:|---:|---:|---:|
| SARIMA | 22 | 3,824.04 | 7,159.17 | 1.108583 |
| SARIMA+CNY | 22 | 3,852.97 | 7,044.62 | 1.108537 |
| Delta |  | +28.93 (+0.76%) | -114.55 (-1.60%) | -0.000046 (-0.004%) |

Q1 RMSE improves, but MAE deteriorates and MASE is almost unchanged. Thus the subgroup does not show consistent incremental improvement either. The primary endpoint remains h = 2, all targets.

At h = 1, all-target MAE/RMSE/MASE change by -2.66%/-1.53%/-1.78%; Q1 changes by -2.25%/-0.82%/-1.89%. These secondary results do not change the h = 2 primary conclusion.

## Paired target behavior

The paired absolute-error difference is `CNY absolute error - SARIMA absolute error`; negative values favor CNY. At h = 2, SARIMA+CNY has lower absolute error on **44/86 targets (51.2%) overall** and **13/22 Q1 targets (59.1%)**. The majority of Q1 targets favor CNY by absolute error, while the Q1 aggregate MAE is still higher; the loss differences are uneven in size. These are descriptive paired counts, not significance tests. No DM, Clark-West, or other test search was performed.

## Rolling coefficient stability

The 87 rolling coefficients have mean **0.0436**, median **0.0304**, and range **-0.0255 to 0.1364**. There are 73 positive and 14 negative estimates, with 7 adjacent sign changes in origin order. The coefficient changes materially across the sample and is not sign-stable. Coefficient p-values are reported for transparency, not as the primary evidence of forecasting value.

## Separate full-sample descriptive fit

The full-sample fit uses the Stage 4 full-sample selected SARIMA(0,1,1)(1,0,1)[4] structure without re-selection and centers CNY timing on all 33 Q1 observations. Its coefficient is **0.0992** (SE **0.0533**, p = **0.0625**). The CNY model's AICc is **-416.431**, compared with the Stage 4 baseline AICc of **-415.544**, so delta AICc is **-0.887**. The lag-8 Ljung–Box p-value is **0.688**; residual mean is -0.00050 and sample SD is 0.05016 over 129 post-burn residuals.

This fit is descriptive only. Its coefficient p-value and AICc do not establish forecasting value and are not used in the rolling comparison.

## Limitations and interpretation

- The evaluation has 86 targets per horizon and 22 Q1 targets in the prespecified subgroup.
- The CNY fraction is deterministic calendar information, not an observed economic predictor. Its coefficients can vary as the expanding training sample and frozen Stage 4 order vary across origins.
- Rolling beta and paired-win summaries are descriptive; there is no causal interpretation or significance-test fishing.
- The experiment evaluates only the requested single regressor on the frozen Stage 4 structure. It does not establish that other CNY encodings, orders, predictors, or horizons would behave similarly.
- The very small h = 2 all-target changes and mixed Q1 metrics do not support a material or consistent incremental forecasting claim under this protocol.

## Reproduction and outputs

Run `python scripts/modeling/backtest_cny.py`, then `pytest -q` and `python -m compileall scripts tests`.

- Forecasts: [sarima_cny_rolling_forecasts.csv](../outputs/forecasts/sarima_cny_rolling_forecasts.csv)
- Metrics: [cny_metrics.csv](../outputs/tables/cny_metrics.csv)
- Paired comparison: [cny_incremental_comparison.csv](../outputs/tables/cny_incremental_comparison.csv)
- Paired target losses: [cny_paired_errors.csv](../outputs/tables/cny_paired_errors.csv)
- Rolling coefficients: [cny_coefficient_stability.csv](../outputs/tables/cny_coefficient_stability.csv)
- Full-sample diagnostics: [cny_full_sample_diagnostics.csv](../outputs/tables/cny_full_sample_diagnostics.csv)
- Machine-readable summary: [stage5_summary.json](../outputs/diagnostics/stage5_summary.json)
- Figures: [h = 2 Q1 forecasts](../outputs/figures/cny_h2_q1_actual_vs_forecasts.png), [paired h = 2 errors](../outputs/figures/cny_h2_paired_absolute_error_difference.png), [rolling beta](../outputs/figures/cny_rolling_beta.png)
