# Forecast optimization research contract — 2026-10-08

## Authorization and scope

The user authorized experiments, analysis, and a GitHub optimization research branch on 2026-10-08. This is a separate exploratory extension of baseline commit `4b03bdc3b4602b4419478584b170da2e8b5dd927`, not a replacement for Stage 1–7. No new final forecast, primary-model adoption, causal claim, or merge to main is part of this round.

## Frozen evaluation

- Canonical data: 1994Q1–2026Q2; preserve all existing data, proposal, and Stage 1–7 outputs.
- Main endpoint: h=2, all 86 targets, 2005Q1–2026Q2. h=1 and Q1 are secondary.
- Models share target/origin pairs, original retail-level units, and the archived Stage 4 origin-specific MASE denominator. A window-model denominator is not substituted.
- At each origin use that origin's archived Stage 4 order, never the full-sample final order. Do not search orders, windows, weights, holiday encodings, or calibration settings against the scored targets.
- All losses and bootstrap deltas are extended minus baseline; negative favors the extension. All experiments are exploratory because historical benchmark results were already viewed.
- Any historical improvement is a candidate for independent future evaluation, not confirmed generalization. No significance or zero-effect claim follows from a percentile interval alone.

## Experiment families fixed before new results

1. **Historical paired losses**: CNY versus SARIMA; paired chronological circular-block bootstrap, 5,000 replicates, seed 20261008, block lengths 4 (primary) and 8 (sensitivity). Report MAE/RMSE/MASE deltas and percentile 95% intervals. Blocks count observations; a Q1 block of 4 covers four years. This is descriptive uncertainty conditional on archived fitted procedures, with no model refitting or multiple-testing-adjusted claim.
2. **Training windows**: SARIMA-W40 and SARIMA-W60, using the last min(W, n_available) quarters through each origin. Frozen origin order and Stage 4 estimator settings. One initial L-BFGS attempt, one terminal-parameter continuation, then one deterministic start attempt, each maxiter=100; never change the order or fall back to another model. Persist every origin result and fit attempt, including failures. Resume only entries matching current input/specification. Keep incomplete studies visibly incomplete; aggregate scores only if all requested fits succeed.
3. **Prediction intervals**: derive 80%/95% lognormal intervals from saved SARIMA/CNY/window moments; record coverage, mean width, and Winkler score. Interval endpoints receive no half-variance mean adjustment. Supplementary calibration uses prior same-model/same-horizon absolute standardized log errors whose target is <= current origin, last 40 eligible errors, at least 20, empirical quantile (method=higher). Before 20 errors retain parametric intervals and flag fallback; report calibrated-only and all-target scores separately. This is an exploratory empirical scale calibration, without a coverage guarantee.
4. **Fixed point combination**: 0.5 SARIMA + 0.5 ETS, aligned by origin/horizon/target, with unchanged baseline MASE scales. No weight search and no averaging of interval endpoints.
5. **Point-functional sensitivity**: score archived lognormal medians separately from the original conditional-mean forecasts, for SARIMA/CNY/windows. Do not replace original scores or mix point conventions within a comparison.
6. **Calendar aggregation diagnostic**: use inclusive windows [-14,+7] and [-30,+7] days around CNY to measure Q1 and preceding-Q4 allocation. Report variation and constant-regressor risk without fitting a new holiday model. No economic-data imputation, external data acquisition, or causal effect estimate.

## Reporting and acceptance

Baseline portability repairs are confined to explicit `.gitattributes` rules for the mixed serialized line endings in the existing manifest, and a Stage 7 determinism test that compares two same-runtime fits in temporary paths. The original test compared a fresh local fit with archived numerical optimization bytes and overwrote the archives. On this runtime the original Q4 forecast differs by about 1.56 亿元 (0.00112%); archive forecasts remain unchanged and are used as the research reference. The repair does not claim exact numerical portability.

Report pooled results, Q1 results, descriptive periods 2005–2011, 2012–2019, 2020–2026, fit coverage/failures, numerical/runtime limitations, and all negative results. Periods were defined after baseline inspection and remain supplementary. Plot h=2 paired cumulative loss, window forecast/error comparison, and interval coverage. Record software versions and commands. Newly generated artifacts live exclusively under `outputs/optimization/round1/`.

Use focused leakage, pairing, formula, failure and resume tests plus the existing suite. Verify original tracked artifacts remain byte-identical to baseline. Publish code, protocol, tests, results, English machine-readable tables, and a Chinese research report to `research/forecast-optimization-20261008`.

## Sources informing the design

- [Forecast evaluation](https://otexts.com/fpp3/accuracy.html)
- [Distributional evaluation](https://otexts.com/fpp3/distaccuracy.html)
- [Time-series cross-validation](https://otexts.com/fpp3/tscv.html)
- [Forecast combinations](https://otexts.com/fpp3/combinations.html)
- [Holiday-regressor allocation](https://www.census.gov/data/software/x13as/genhol/holiday-regressors.html)
- [Diebold–Mariano original paper](https://doi.org/10.1080/07350015.1995.10524599)
