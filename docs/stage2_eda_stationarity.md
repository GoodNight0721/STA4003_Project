# Stage 2 — EDA, transformation, and stationarity diagnostics

## Scope and data

All analyses in this stage read `data/processed/analysis_quarterly.csv`, the canonical Stage 1 dataset. The sample is 130 quarterly observations from 1994Q1 through 2026Q2. The quarterly seasonal period is fixed at `s = 4`. No source series or canonical values were changed.

This stage is descriptive. It fits no ARIMA, SARIMA, ARIMAX, or ETS model; searches no model orders; evaluates no forecasts; and tests no CNY or PMI predictive value.

## Level, log transform, and unusual observations

The level plot shows a strong long-run rise in retail sales. Its absolute rolling standard deviation increases substantially with the level: across 20-quarter windows it ranges from about 922 to 13,439 RMB 100 million. The level series also has a clear within-year pattern and high persistence. The 2020Q1 downturn and the 2022Q2 observation are marked for context; both remain unchanged in the series.

The log plot compresses the growth in level and makes proportional changes easier to compare over time. Its rolling standard deviation is on a relative scale and tracks the level coefficient of variation closely (median 20-quarter log standard deviation 0.175 versus median level coefficient of variation 0.173). Both measures vary and decline in later years, so the log transform improves scale interpretation but does not make volatility constant or establish stationarity.

**Transformation decision:** keep `log_retail` as the working response for later time-series analysis, with level retail retained for economic interpretation. This is consistent with the proposal and with the multiplicative growth pattern visible in the series; it is not a stationarity conclusion.

Figures: [retail level](../outputs/figures/retail_level.png), [log retail](../outputs/figures/log_retail.png), and [20-quarter rolling scale comparison](../outputs/figures/rolling_scale_comparison.png).

## Quarterly seasonality

Seasonal summaries use `log_retail`. To separate the within-year pattern from the long-run trend, the seasonal subseries figure and centered effects use the 32 complete years 1994–2025; 2026 is omitted from that centered comparison because only Q1 and Q2 are present. Across the complete years, Q4 is above its annual mean in all 32 years, while Q1 and Q2 are below it in all 32 years. Q3 is below its annual mean in 30 of 32 years. The average annual-mean-centered effects are:

| Quarter | Mean centered log deviation | Median | Standard deviation | Years above annual mean |
|---|---:|---:|---:|---:|
| Q1 | -0.0501 | -0.0484 | 0.0396 | 0 / 32 |
| Q2 | -0.0635 | -0.0595 | 0.0190 | 0 / 32 |
| Q3 | -0.0215 | -0.0164 | 0.0218 | 2 / 32 |
| Q4 | 0.1351 | 0.1241 | 0.0412 | 32 / 32 |

The seasonal pattern is clear and fairly consistent on the log scale, especially the higher Q4 and lower Q1/Q2. Its magnitude varies by year, including the 2020 disruption; these sample effects are descriptive averages, not forecasting factors. The broader quarter-by-quarter distributions include the long-run trend and should not be read as a trend-adjusted seasonality test.

See [seasonal subseries](../outputs/figures/seasonal_subseries_log.png), [quarter distributions](../outputs/figures/seasonal_distribution_log.png), and [seasonal summary table](../outputs/tables/seasonal_summary.csv).

## ADF and KPSS tests

The null hypotheses differ: ADF tests a unit root, while KPSS tests stationarity. At the 5% level, a failure to reject either null is inconclusive; the two tests are interpreted together with the plots and dependence diagnostics.

Specifications were fixed before interpreting the results. ADF uses a fixed eight-quarter augmentation lag with no automatic lag search (`autolag=None`): `ct` (constant plus linear trend) for log retail and `c` (constant) for differenced series. KPSS uses `nlags="auto"` for its automatic HAC bandwidth, with `ct` for log retail and `c` for differenced series. The actual KPSS bandwidths are listed below. KPSS table probabilities at 0.01 and 0.10 are boundary reports: the actual probability is below 0.01 or above 0.10, respectively.

| Transformation | N | ADF statistic | ADF p-value | KPSS statistic | KPSS p-value | KPSS bandwidth | Combined reading |
|---|---:|---:|---:|---:|---:|---:|---|
| `log_retail` | 130 | 0.1186 | 0.9953 | 0.3539 | <0.01 | 6 | Evidence of non-stationarity under the selected trend specifications |
| `Δ log_retail` | 129 | -1.9814 | 0.2948 | 0.4153 | 0.0705 | 23 | Inconclusive: neither null is rejected at 5% |
| `Δ4 log_retail` | 126 | -1.4708 | 0.5479 | 0.8932 | <0.01 | 5 | Evidence of non-stationarity under the selected constant specifications |
| `Δ Δ4 log_retail` | 125 | -5.2041 | 0.0000086 | 0.0801 | >0.10 | 10 | Both tests are consistent with stationarity under the selected specifications |

The full results, including the deterministic-term choices, effective ADF observations, and KPSS boundary notes, are in [stationarity_tests.csv](../outputs/tables/stationarity_tests.csv).

## ACF and PACF dependence diagnostics

The log-level ACF decays slowly: it is 0.976 at lag 1 and remains 0.599 at lag 20. The seasonal-lag correlations are also large (0.919, 0.842, 0.764, 0.684, and 0.599 at lags 4, 8, 12, 16, and 20). This is consistent with strong persistence and recurring quarterly dependence, not stationarity.

After a regular difference, the seasonal-lag ACF remains high: 0.795 at lag 4, 0.768 at lag 8, and 0.634 at lag 20. Seasonal differencing alone reduces some annual-cycle dependence, but its ACF still has a positive lag-1 value of 0.665 and decays gradually. With both differences, most later ACF values are small, but lag 1 is -0.332 and lag 4 is -0.492. The combined-difference chart shows both as notable negative spikes, especially at the seasonal lag.

The negative lag-1 value is moderate rather than extreme, so it is not conclusive proof of over-differencing. The pronounced negative lag-4 correlation is a warning that seasonal differencing may remove too much of the annual pattern. PACF and ACF plots are used only for structural diagnosis; no `p`, `q`, `P`, or `Q` order is selected.

Figures: [log level](../outputs/figures/acf_pacf_log_retail.png), [first difference](../outputs/figures/acf_pacf_first_difference.png), [seasonal difference](../outputs/figures/acf_pacf_seasonal_difference.png), and [combined difference](../outputs/figures/acf_pacf_combined_difference.png). Each displays lags through 20 quarters.

## Differencing conclusion

- **Regular difference (`d=1`):** a cautious working recommendation for later modeling because the log level trends strongly and its ACF decays slowly. However, the first-difference ADF/KPSS pair is inconclusive at 5%, so `d=1` is not established by these tests alone.
- **Seasonal difference (`D=1`, lag 4):** not established as necessary. Seasonal differencing alone remains non-stationary under the selected tests. The combined difference passes both tests, but its negative lag-4 ACF raises a seasonal over-differencing concern.
- **Overall:** use `log_retail`; treat `d=1` as a provisional starting point and leave `D` unresolved for later modeling diagnostics. Do not assume `d=1, D=1` is required simply because its stationarity tests pass. The evidence supports ambiguity rather than a unique differencing choice.

No model was fit to settle this ambiguity. The detailed lag correlations and recommendation are recorded in [stage2_summary.json](../outputs/diagnostics/stage2_summary.json).

## Descriptive CNY check and PMI coverage

For Q1 only, CNY position was compared descriptively with same-year Q1 year-over-year log growth for the 32 available pairs from 1995–2026. The Pearson correlation is `0.215`, a modest positive sample association. The scatter is wide and includes unusual growth observations. There is no significance test or causal interpretation; this is not out-of-sample evidence and does not show that CNY timing improves forecasts. See the single [Q1 CNY descriptive plot](../outputs/figures/cny_q1_descriptive.png) and [summary table](../outputs/tables/cny_q1_descriptive.csv).

PMI coverage is 86 quarterly observations from 2005Q1 through 2026Q2. Stage 2 performs no PMI regression, cross-correlation search, or predictive analysis.

## Remaining caveats

- The sample contains only 130 quarters and includes changes in growth dynamics and unusual economic periods. ADF/KPSS results can be sensitive to deterministic terms and structural breaks; this stage used one pre-specified test setup and did not run structural-break tests.
- KPSS probabilities are table-bounded at the reported extremes, as noted above.
- Retail is analyzed in the canonical nominal 亿元 units. The log transform changes interpretation to proportional growth; it does not remove seasonal dependence or guarantee constant variance.
- Stage 2 creates no fitted model, order search, forecast, prediction interval, rolling-origin result, forecast error comparison, or predictive-value conclusion for CNY or PMI.
