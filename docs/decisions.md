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
