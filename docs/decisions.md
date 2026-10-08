# Decision log

## 2026-10-03 — Stage 0 repository bootstrap

- Established the repository structure for proposal materials, raw and derived data, scripts, outputs, and documentation.
- Treat raw data as immutable; keep existing files byte-for-byte unchanged while relocating them.
- Separate generated outputs from source data.
- Divide future work into explicit project stages and stop after each requested stage.
- Do not build models or perform new analysis during Stage 0.

## 2026-10-03 — Stage 1 data audit and canonical dataset

- Adopt `data/processed/analysis_quarterly.csv` as the canonical quarterly input, with one row per quarter from 1994Q1 through 2026Q2. Keep `data/raw/*` and the submitted proposal immutable; Stage 1 performs no economic-data imputation.
- Retain retail in its supplied units, RMB 100 million (亿元), and define `log_retail` as the natural log of that unchanged quarterly level.
- Define `cny_position_fraction` as `(CNY date - January 1).days / (April 1 - January 1).days`. The measure is zero-based (January 1 is zero); the denominator is 90 days in ordinary years and 91 in leap years. CNY timing is annual metadata repeated across quarters; any later CNY regressor effect is restricted to Q1 and is not centered using the full sample.
- Align `pmi_lag1` and `pmi_lag2` to the full canonical quarterly index, so their values at quarter `t` are PMI at `t-1` and `t-2`; leave unavailable values missing. Actual contemporaneous future PMI at `T+1` or `T+2` is prohibited as an input at rolling forecast origin `T` because it is not then observed. Preserve lagged candidates without claiming predictive benefit.
- Treat the exact post-2000 annual equality between quarterly sums and December cumulative retail as internal accounting consistency from differencing, not independent validation. Retain and document the current-period versus cumulative vintage discrepancy without interpolation or reconciliation.
- Correct the interpretation of the approximately 0.909 and 0.949 diagnostics: they are February/January retail ratios, not February shares of combined January-February retail.
- Stage 1 is limited to data audit, reproducibility, canonical data construction, and documentation; it performs no EDA or model selection.

## 2026-10-03 — Stage 2 EDA and stationarity diagnostics

- Use `log_retail` as the working response for subsequent time-series analysis because the level-scale dispersion grows with the series and the log scale expresses proportional change more usefully. Retain the original level for interpretation; the log transform does not establish stationarity.
- Fix quarterly seasonal period at `s = 4`. Complete-year centered log effects show a consistent Q4-high and Q1/Q2-low pattern; these sample averages are descriptive, not a forecast model.
- Pre-specify ADF with fixed max lag 8 (`ct` for log levels, `c` for differences) and KPSS with automatic bandwidth (`ct` for log levels, `c` for differences). Report KPSS table-bound p-values as bounds.
- Treat `d = 1` as a cautious provisional starting point because of the log-level trend, while acknowledging inconclusive ADF/KPSS results on the first difference. Do not establish `D = 1`: seasonal differencing alone remains non-stationary, and the combined difference's negative lag-4 ACF (-0.492) raises a possible seasonal over-differencing concern. Leave the final differencing choice unresolved for later diagnostics.
- Stage 2 ACF/PACF plots are structural diagnostics only; no ARMA orders are selected. No forecasting model is fitted or evaluated.
- The Q1 CNY-position versus Q1 year-over-year log-growth check is descriptive only (32 pairs; sample correlation 0.215); it is not causal or predictive evidence. PMI is described only by coverage; its predictive value is not tested.

## 2026-10-03 — Stage 3 rolling-origin evaluation and benchmarks

- Fix one expanding-window rolling-origin protocol for all future forecasts: the same target window of 2005Q1–2026Q2 (86 quarters) for both horizons, with origin t-h and training from 1994Q1 through the origin inclusive.
- Use h = 2 as the primary horizon and h = 1 only as a secondary diagnostic. Do not change the primary target window based on PMI availability.
- Evaluate exactly the pre-specified Historical Mean, Naive, and Seasonal Naive benchmarks, all on original retail_bn levels in RMB 100 million; do not add drift or other benchmarks.
- Calculate seasonal MASE with period 4 and an origin-specific denominator using only seasonal differences within that origin's training sample. No full-sample denominator is allowed.
- Report all and the pre-specified Q1 target subgroup; the primary comparison is h = 2, scope all.
- Stage 3 establishes reference performance only. It performs no SARIMA/model-order selection, AIC/AICc ranking, CNY or PMI predictive test, or final forecast.

## 2026-10-03 — Stage 4 leak-free SARIMA baseline

- Fix the SARIMA candidate family to SARIMA(p,1,q)(P,D,Q)[4], with p,q in {0,1,2}, P,Q,D in {0,1}, and p+q+P+Q <= 3; use d = 1 and no trend/drift or exogenous variables.
- Select one candidate at each unique Stage 3 origin using only that origin's training sample and minimum AICc. Do not use rolling OOS error to tune p,q,P,Q,D or the candidate family.
- Use statsmodels state-space SARIMAX with simple_differencing = False and the same deterministic specification for all candidates.
- Score level forecasts using exp(mu_log + 0.5 * forecast_variance_log); save exp(mu_log) as the median but do not use it for primary scoring.
- Reuse Stage 3 targets, horizons, training-only seasonal MASE scales, and metric implementation.
- Treat the 1994Q1–2026Q2 selected fit as descriptive residual diagnostics only; do not use it for rolling forecasts.
- Stage 4 uses no CNY or PMI regressors, ARIMAX, ETS/Theta, or final future forecast.

## 2026-10-03 — Stage 5 Lunar New Year incremental-value analysis

- Freeze the exact Stage 4 AICc-selected SARIMA order separately at each of its 87 unique rolling origins. Stage 5 performs no order search and does not change the Stage 4 candidate family.
- Add exactly one exogenous variable to the same state-space SARIMAX error structure: `is_q1 * (cny_position_fraction - center_T)`, where `center_T` is the mean CNY fraction among Q1 observations in that origin's training sample. Non-Q1 values are exactly zero; use the same center for forecast-step exog.
- Future `is_q1` and CNY-position values are allowed because the calendar is deterministic and known at the forecast origin. Do not use future retail or PMI observations. Do not add PMI or redesign the CNY variable based on OOS results.
- Preserve the Stage 3/4 targets, horizons, bias-corrected lognormal level forecast, and exact Stage 4 origin-specific training-only seasonal MASE scales. Keep h = 2, all targets as the primary endpoint and h = 2, Q1 as the prespecified subgroup.
- If L-BFGS does not report convergence at its first Stage 4-configured attempt, permit one continuation from that attempt's terminal parameters with the same method and settings. Report any retry; do not fall back to the baseline. Stop with an origin-specific failure if it still does not converge.
- Treat rolling coefficient estimates, paired target wins, coefficient p-values, and the separate full-sample AICc/residual diagnostics as descriptive. Judge forecasting value only from the fixed-protocol OOS comparison; make no causal claim.
- Stage 5 adds no PMI, alternate SARIMA orders, other forecasting models, or final future forecast.

## 2026-10-03 — Stage 6A leak-free PMI extension

- Fix the only PMI predictor at **pmi_lag2[t] = PMI[t-2]**, with regressor **pmi_lag2 - 50**. For origin T, the h=1 and h=2 target values therefore use PMI[T-1] and PMI[T], respectively; PMI after T is prohibited. Do not test other PMI lags or choose lags from full-sample correlations.
- Keep the Stage 3/5 primary target window unchanged. Define a secondary common h=1/h=2 target window of 2010Q4–2026Q2 (63 quarters per horizon). The first h=2 origin is 2010Q2, where the restricted training sample 2005Q3–2010Q2 has exactly 20 complete lag-2 PMI rows.
- At each of the 64 unique secondary-study origins, read the exact Stage 4 order and refit SARIMA-common, SARIMA+PMI, and supplementary SARIMA+CNY+PMI on the identical 2005Q3-to-origin log_retail sample. Do not use Stage 4's 1994-start forecasts as the incremental PMI reference.
- Preserve the Stage 5 CNY definition in the combined model: **is_q1 * (cny_position_fraction - mean Q1 cny_position_fraction in the restricted training sample through origin)**. Use that same center for future CNY exog.
- Use Stage 4 state-space SARIMAX settings and L-BFGS (maxiter=100) for all models. To handle numerical starting values without changing the estimator, seed same-order rolling fits from their prior-origin fit when available; otherwise seed nested exogenous models from the same-origin parent fit with added exogenous coefficients at zero. If a fit is non-converged, continue once from terminal parameters and then, if needed, make one final attempt from a deterministic order-based start with the same optimizer/settings. Record attempts and stop if the final attempt still fails.
- Calculate period-4 MASE using only retail levels from the same restricted 2005Q3-to-origin training sample. Score levels using exp(mu_log + 0.5 * forecast_variance_log). Report negative extended-minus-reference deltas as improvements; coefficient significance is not forecasting-value evidence.
- Treat Stage 6A as secondary predictor evidence. Do not replace Stage 5's primary conclusion, start Stage 6B, add ETS/Theta or shock robustness, or produce a final forecast as part of this stage.

## 2026-10-03 — Stage 6B ETS and target-shock robustness

- Add one fixed statsmodels ETS(A,Ad,A) model for `log_retail` with additive error, additive damped trend, additive seasonality, and period 4. Do not search ETS components or select the model from out-of-sample performance.
- Reuse the exact Stage 3 2005Q1–2026Q2 targets, h=1/h=2 origins, expanding 1994Q1-to-origin training samples, and origin-specific seasonal MASE scales. Retransform each forecast as `exp(log forecast) * mean(exp(training residuals))`, with residuals from that origin's fitted training sample only.
- Treat 2020Q1 and 2022Q2 as preset score-sensitivity targets. Keep all observations in every fitted sample and all forecasts in the raw forecast records; change only which target rows enter the secondary full-versus-excluded metric aggregates.
- Keep h=2, all targets as the primary endpoint. The shock sensitivity is not a replacement endpoint, and neither full-sample descriptive fit statistics nor the sensitivity results tune or refit any model.
- The fixed ETS fit converged at all 87 unique origins. It is behind both Stage 4 SARIMA and Stage 5 SARIMA+CNY on h=2 all-target MAE, RMSE, and MASE. Excluding the two shock targets lowers h=2 all-target RMSE for SARIMA, SARIMA+CNY, and ETS; the already small CNY deltas remain mixed, so this sensitivity does not support a consistent incremental-value claim.
- Stage 6B produces a full-sample descriptive ETS fit only, with no full-sample forecast; no PMI, alternate ETS family, or final future forecast is added.

## 2026-10-03 — Stage 7 final forecast and report

- Retain plain SARIMA as the final primary forecaster because it beats the three simple benchmarks and fixed ETS on the primary h=2 all-target comparison; CNY's small gains are mixed and do not survive the preset-shock sensitivity consistently; lag-2 PMI does not improve its distinct secondary-window comparison.
- Freeze the final primary order at Stage 4's full-sample selection, SARIMA(0,1,1)(1,0,1)[4]. Refit that order on `log_retail` from 1994Q1 through 2026Q2 (130 observations). Stage 7 performs no candidate search and supplies no CNY or PMI exogenous variable to the primary model.
- Set the final forecast origin to 2026Q2 and the only targets to 2026Q3 (h=1) and 2026Q4 (h=2). Do not extend the primary horizon beyond those quarters.
- Report the lognormal median as `exp(mu)`, the conditional-mean point forecast as `exp(mu + 0.5 * v)`, and the 80%/95% predictive quantiles as `exp(mu +/- z * sqrt(v))`; the half-variance adjustment applies only to the conditional-mean point forecast.
- Refit SARIMA+CNY with the Stage 5 training-centered Q1 timing definition and fixed ETS(A,Ad,A) with period 4 only as supplementary forecasts. The target CNY regressor is zero for both non-Q1 forecast quarters. Do not ensemble or reselect the primary model.
- Keep the PMI result labeled as the 2010Q4–2026Q2 secondary window. Do not use the Stage 5 full-sample CNY coefficient as out-of-sample evidence, describe the February/January retail ratio as a share, or promote shock-excluded scores to the primary endpoint.
- Freeze SHA-256 hashes for the Stage 0–6B tracked data and outputs in `docs/stage7_protected_artifact_sha256.json`; Stage 7 verifies that these artifacts remain unchanged.
- Stage 7 completes the future forecast, final report, supplementary model forecasts, diagnostics, summary outputs, and reproducibility checks. No post-hoc model search or additional research stage is authorized by this decision.

## 2026-10-08 — User-authorized optimization research branch

- The user explicitly authorized experiments, analysis, and a GitHub optimization branch. This supersedes the prior no-extension boundary only for the separate research round described in `docs/optimization_protocol.md`; Stage 1–7 results and the submitted proposal remain preserved.
- Create `research/forecast-optimization-20261008` from `4b03bdc3b4602b4419478584b170da2e8b5dd927`. Keep the original h=2, all-target endpoint and archived origin-specific MASE scales. Preserve each origin's selected order; never use the full-sample final order for earlier forecasts.
- Freeze W40/W60 windows, paired circular-block bootstrap (5,000 replicates, seed 20261008, blocks 4/8), 80%/95% interval evaluation, realized-error-only supplementary calibration (last 40, minimum 20), a fixed equal-weight SARIMA/ETS point combination, separate mean/median scores, and inclusive CNY windows [-14,+7]/[-30,+7] before computing the new results.
- Persist per-origin successes, failures, retries and exact resumability inputs. All 174 window fits succeeded; two origins required the predefined retry sequence. No unsuccessful origin is silently replaced with another forecast.
- After initial results, apply target-only scoring sensitivity to the original Stage 6B preset 2020Q1 and 2022Q2 across complete mean forecast models. No new shock dates, fitting, or primary-endpoint changes are introduced; this follow-up is supplementary.
- Preserve the existing manifest's mixed serialized line endings with `.gitattributes`. Fix the existing Stage 7 determinism test to compare repeated local fits in temporary paths and verify archives untouched; retain cross-runtime numerical differences as a limitation rather than rewriting the archived forecast.
- Treat W60's historical improvements as an exploratory candidate. Paired percentile intervals still cross zero and improvements are concentrated after 2020. W40, the fixed combination, and the tested interval calibration retain their negative results. No merge to main, main-model replacement, or new final forecast is authorized or performed in this round.
