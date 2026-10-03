# Stage 6A — Leak-Free PMI Extension

## Scope and information set

Stage 6A evaluates one pre-specified PMI regressor on a secondary window. For quarter t, the regressor is **pmi_lag2[t] = PMI[t-2]**, centered at the PMI expansion/contraction threshold:

**pmi_exog[t] = pmi_lag2[t] - 50**

At forecast origin T, the h=1 target is T+1 and its lag-2 PMI value is PMI[T-1]. The h=2 target is T+2 and its lag-2 value is PMI[T]. Both source quarters are at or before the origin. Contemporaneous PMI at T+1 or T+2 would be unknown when those forecasts are made and is never supplied to the model. No other lag or full-sample correlation search is used.

The canonical lag-2 series is available from 2005Q3 through 2026Q2. The fixed secondary common target window is **2010Q4–2026Q2**, containing **63 targets for h=1 and 63 for h=2**. The first h=2 origin is 2010Q2. Its complete lag-2 PMI training rows run from 2005Q3 through 2010Q2, giving exactly 20 observations. There are 64 unique origins across both horizons. This secondary window does not alter the Stage 3/5 primary target window.

## Models and rolling fits

At every unique origin, the model order is read directly from outputs/tables/sarima_origin_selection.csv; Stage 6A does no order search. Three models are fitted to the same response observations, log_retail from 2005Q3 through the origin:

1. **SARIMA-common** — frozen Stage 4 order, without exogenous variables.
2. **SARIMA+PMI** — the same order and observations, with pmi_lag2 - 50.
3. **SARIMA+CNY+PMI** — supplementary model with the same order and observations, plus the Stage 5 CNY regressor.

The common SARIMA model is refitted on the restricted sample so that the PMI comparison does not confound the regressor with Stage 4's longer 1994-start training period. The combined model reuses the Stage 5 definition: **is_q1 * (cny_position_fraction - center_T)**, where center_T is the mean Q1 CNY fraction in the restricted training sample through the origin. Non-Q1 values are zero and the same training-only center is used for future exogenous values.

All models use the Stage 4 state-space SARIMAX specification and L-BFGS settings (maxiter=100). Fits are run once per model and unique origin and reused for horizons served by that origin. Same-order rolling fits use the previous origin's fitted parameters as starting values when available. Otherwise, an extended model starts from its same-origin nested model, with added exogenous coefficients set to zero. An unconverged fit is continued once from its terminal parameters; if necessary, one final deterministic order-based starting value is tried with the same optimizer and settings. This affected two of the 192 origin-model fits and both ultimately converged. There were **192 successful fits, 0 unresolved failures, and 195 L-BFGS attempts**.

At each origin, the period-4 MASE scale is calculated from retail levels in the same restricted 2005Q3-to-origin training sample. The scored level forecast is the lognormal conditional mean, **exp(mu_log + 0.5 * forecast_log_variance)**. All comparison deltas are extended model minus reference model; a negative value is an improvement.

## Forecast value

The primary Stage 6A comparison is h=2 across all 63 secondary-window targets:

| Model | MAE | RMSE | MASE |
|---|---:|---:|---:|
| SARIMA-common | 3,459.96 | 5,906.20 | 0.549654 |
| SARIMA+PMI | 3,512.17 | 5,926.59 | 0.558122 |
| SARIMA+CNY+PMI | 3,523.10 | 5,707.31 | 0.560432 |

For **SARIMA+PMI versus SARIMA-common**, the h=2 changes are MAE **+52.22 (+1.51%)**, RMSE **+20.39 (+0.35%)**, and MASE **+0.008467 (+1.54%)**. All-target h=2 metrics are worse with PMI under this fixed protocol.

For the supplementary **SARIMA+CNY+PMI versus SARIMA+PMI**, h=2 changes are MAE **+10.93 (+0.31%)**, RMSE **−219.28 (−3.70%)**, and MASE **+0.002310 (+0.41%)**. RMSE improves, while MAE and MASE worsen; the combined model does not show consistent incremental improvement.

Secondary diagnostics also vary by metric. At h=1 across all targets, PMI changes versus common SARIMA are +2.85% MAE, +1.06% RMSE, and +3.03% MASE. At h=2 within the 16 Q1 targets, PMI changes are −0.65% MAE, −0.03% RMSE, and −1.13% MASE; adding CNY to PMI changes Q1 MAE by +5.23%, RMSE by −3.08%, and MASE by +4.90%. These results do not change the h=2 all-target endpoint.

## Rolling coefficient stability

The 64 PMI-only coefficients have mean **0.001700**, median **0.001561**, and range **0.000991 to 0.002354**; all 64 are positive, with no adjacent sign changes. In the combined model, PMI coefficients have mean **0.001281**, median **0.001268**, and range **−0.000069 to 0.002223**; 63 are positive and one is negative, with two adjacent sign changes. The 64 combined-model CNY coefficients have mean **0.095481**, median **0.058588**, and range **0.008249 to 0.247393**; all are positive, with no adjacent sign changes.

These coefficients describe conditional associations under changing rolling samples and locked, origin-specific SARIMA orders. They are not causal effects. Standard errors and coefficient p-values are retained in the forecast and stability outputs for description only; they are not evidence of forecasting value.

## Interpretation and limitations

The fixed lag-2 PMI extension does not improve the primary h=2 all-target MAE, RMSE, or MASE against the restricted-sample common SARIMA model. The supplementary CNY+PMI model has a lower RMSE than PMI alone but slightly higher MAE and MASE. These are results for a 63-quarter secondary window, not replacements for the Stage 3/5 primary-window evidence or the Stage 5 conclusion.

The experiment relies on the specified quarterly information set and does not model PMI release calendars, revisions, or historical vintages. It tests one lag and one fixed threshold-centering rule only. No causal claim, alternative predictor search, ETS/Theta comparison, shock robustness, or final forecast is included.

## Reproduction and outputs

Run:

    python scripts/modeling/backtest_pmi.py
    pytest -q
    python -m compileall scripts tests

- [Rolling forecasts](../outputs/forecasts/pmi_rolling_forecasts.csv)
- [Metrics](../outputs/tables/pmi_metrics.csv)
- [Incremental comparisons](../outputs/tables/pmi_incremental_comparison.csv)
- [Rolling coefficients and stability summaries](../outputs/tables/pmi_coefficient_stability.csv)
- [Machine-readable summary](../outputs/diagnostics/stage6a_summary.json)
