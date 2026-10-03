# Stage 1 data audit

## Scope and provenance

This audit covers the committed source files, the committed processed quarterly series, and the canonical analysis dataset. It performs data-integrity checks and reconstruction comparisons only; it does not perform time-series exploration or model selection.

The monthly retail and manufacturing PMI files are economic series attributed to the National Bureau of Statistics (NBS) in the supplied project materials. The file `data/meta/lunar_new_year.csv` is deterministic calendar metadata, not an NBS economic series. Its dates were checked for internal consistency; no alternative calendar source was fetched.

No file under `data/raw/` was changed. The submitted `proposal/Proposal_gpX.docx` was also left unchanged. Retail amounts retain the supplied unit, RMB 100 million (亿元); no rescaling was applied.

## Raw source checks

All four monthly files have the expected columns (`yyyym`, `value`, `dt_name`, `unit`), complete ordered month keys over their stated row ranges, valid dates and finite numeric values wherever populated. No file has duplicate month keys. The validator checks the exact documented blank pattern, so a new blank or an unexpectedly populated documented gap fails validation.

| Source | Rows and row-key coverage | Observed values | Missing values | Duplicate months | Notes |
|---|---:|---:|---:|---:|---|
| `data/raw/retail_sales_monthly.csv` | 392; 1994-01–2026-08 | 362 | 30 | 0 | Separate January and February values are unavailable from 2012 onward because NBS reports Jan-Feb jointly (15 January-February pairs). |
| `data/raw/retail_sales_cumulative.csv` | 392; 1994-01–2026-08 | 305 | 87 | 0 | Values are unavailable before 2000 (72 months), and January cumulative values are unavailable from 2012 onward (15 months). This cumulative series is used to construct quarterly retail from 2000 onward. |
| `data/raw/retail_sales_yoy.csv` | 392; 1994-01–2026-08 | 290 | 102 | 0 | Values are unavailable before 2000 (72 months); Jan-Feb growth observations are unavailable from 2012 onward (30 months). This series is audited but is not used to reconstruct the target. |
| `data/raw/manuf_pmi.csv` | 260; 2005-01–2026-08 | 260 | 0 | 0 | No missing monthly PMI values occur within the observed range; values are finite and within 0–100. |

The NBS Jan-Feb reporting convention is expected missingness, not accidental loss of separate January and February observations. Quarterly retail after 2011 remains constructible from available quarter-end cumulative values; no monthly values were filled in.

## Retail vintage comparison

The supplied current-period and cumulative retail series have a material, documented vintage difference. The annual sums below are shown without reconciling, interpolating, or replacing either source:

| Year | Sum of current-period monthly retail | December cumulative retail | Cumulative minus current-period | Difference as % of cumulative | Difference as % of current-period |
|---:|---:|---:|---:|---:|---:|
| 2005 | 63,686.6 | 67,176.6 | 3,490.0 | 5.1953% | 5.4800% |
| 2011 | 180,910.1 | 181,225.8 | 315.7 | 0.1742% | 0.1745% |

All amounts are RMB 100 million (亿元). The quarterly target follows the existing construction choice: monthly current-period sums in 1994–1999 and differences in the cumulative series from 2000 onward. Consequently the cumulative-vintage discrepancy remains a source limitation and is not resolved by Stage 1.

## Processed quarterly series and reconstruction

`data/processed/retail_quarterly.csv` contains 130 consecutive observations from 1994Q1 through 2026Q2. There are no duplicate or missing quarters, no missing or non-positive retail values, and every stored log is finite. Its `log_value` agrees with the natural logarithm of `value_bn` within the six-decimal storage tolerance; the maximum absolute rounding difference is approximately `4.998e-7`.

The validator independently reconstructs retail from the raw files:

- 1994–1999: sum the three monthly current-period values in each quarter.
- 2000 onward: use quarter-end cumulative retail; Q1 is March cumulative, and Q2–Q4 are the difference between the current and preceding quarter-end cumulative values. A quarter is formed only when the required cumulative observations exist.

The reconstructed sequence has the same 130 quarters as the processed file. The maximum absolute difference is `4.4e-11` RMB 100 million, within the one-decimal source-rounding tolerance of `0.0500001`.

This comparison confirms that the committed processed series is reproducible from the committed raw series under the stated rules. It does not independently validate the underlying source vintages. In particular, from 2000 onward quarterly values are differences of the same cumulative series, so `Q1 + Q2 + Q3 + Q4 = December cumulative` is primarily an internal accounting consistency property of that construction, not independent external evidence for the quarterly values.

`data/processed/pmi_quarterly.csv` contains 86 consecutive observations from 2005Q1 through 2026Q2, with no duplicate quarters or missing values. Independent arithmetic means of the three monthly PMI values reproduce the processed quarterly PMI series; the maximum absolute difference is `0.003333333333`, within the two-decimal storage tolerance of `0.0050001`.

## Lunar New Year metadata

`data/meta/lunar_new_year.csv` contains one ordered row for each year from 1994 through 2026 (33 unique years), with no duplicate year or date. Dates parse as ISO dates, each date's year matches its row, all dates fall in January or February, and `cny_month` agrees with the parsed date. This is an internal consistency audit only; Stage 1 did not fetch or compare alternative calendars.

## Canonical analysis dataset

`data/processed/analysis_quarterly.csv` has exactly 130 rows, one for every quarter from 1994Q1 through 2026Q2. Its 12 columns are `quarter`, `year`, `quarter_num`, `retail_bn`, `log_retail`, `is_q1`, `cny_date`, `cny_month`, `cny_position_fraction`, `pmi`, `pmi_lag1`, and `pmi_lag2`. Retail and log values agree with the committed processed retail file; quarter and calendar fields are deterministic derivations. The CNY date, month, and position fraction are annual calendar metadata repeated across the year's four quarterly rows.

The zero-based CNY position fraction is:

```text
(CNY date - January 1).days / (April 1 - January 1).days
```

Thus January 1 maps to zero; the Q1 denominator is 90 days in an ordinary year and 91 days in a leap year. No economic variable is imputed. PMI remains blank before 2005Q1 (44 quarters); `pmi_lag1` is blank for 45 rows and `pmi_lag2` for 46 rows because unavailable source quarters are preserved as missing.

The PMI lags use the complete canonical quarter index: `pmi_lag1[t] = pmi[t-1]` and `pmi_lag2[t] = pmi[t-2]`. Explicit checks confirm `2005Q2.pmi_lag1 == 2005Q1.pmi` and `2005Q3.pmi_lag2 == 2005Q1.pmi`. For the planned two-quarter forecast horizon, realized contemporaneous PMI at `T+1` or `T+2` is unavailable at forecast origin `T` and must not be supplied to a genuine out-of-sample forecast. Lagged PMI columns preserve candidates that are observable at the origin; Stage 1 makes no claim that PMI improves forecasts and selects no PMI model.

## Manual spot checks

The following rows were inspected directly in the generated canonical file. Fractions are displayed to 12 decimal places; `—` denotes a genuinely blank value.

| Quarter | Retail (亿元) | Log retail | CNY date | CNY month | CNY fraction | PMI | PMI lag 1 | PMI lag 2 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1994Q1 | 3,569.6 | 8.180209 | 1994-02-10 | 2 | 0.444444444444 | — | — | — |
| 2005Q1 | 15,112.2 | 9.623258 | 2005-02-09 | 2 | 0.433333333333 | 55.70 | — | — |
| 2005Q2 | 14,497.5 | 9.581731 | 2005-02-09 | 2 | 0.433333333333 | 53.77 | 55.70 | — |
| 2005Q3 | 15,470.9 | 9.646716 | 2005-02-09 | 2 | 0.433333333333 | 52.93 | 53.77 | 55.70 |
| 2012Q1 | 49,318.8 | 10.806061 | 2012-01-23 | 1 | 0.241758241758 | 51.53 | 49.90 | 50.93 |
| 2020Q1 | 78,579.7 | 11.271869 | 2020-01-25 | 1 | 0.263736263736 | 45.90 | 49.90 | 49.67 |
| 2022Q2 | 101,772.6 | 11.530496 | 2022-02-01 | 2 | 0.344444444444 | 49.07 | 49.93 | 49.87 |
| 2026Q1 | 127,694.9 | 11.757399 | 2026-02-17 | 2 | 0.522222222222 | 49.57 | 49.43 | 49.50 |
| 2026Q2 | 121,027.2 | 11.703771 | 2026-02-17 | 2 | 0.522222222222 | 50.20 | 49.57 | 49.43 |

The 2020Q1 and 2022Q2 retail observations are retained as genuine economic observations; neither is removed or adjusted in Stage 1. The approximately `0.909` and `0.949` diagnostics in the project refer to February retail divided by January retail (Feb/Jan ratios), not February's share of the combined January-February total.

## Remaining limitations and scope

- The 2005 and 2011 source-vintage differences above remain unresolved; no attempt was made to hide or interpolate them.
- Separate January and February current-period retail observations are not published from 2012 onward under the documented NBS convention.
- PMI is unavailable before 2005Q1 and remains missing in the canonical dataset before that quarter.
- The calendar metadata was checked internally but not independently sourced in this stage.
- The latest complete quarterly observations are through 2026Q2. No economic variable is imputed.
- Stage 1 creates no EDA plots, stationarity tests, ACF/PACF analysis, forecasting models, model-selection results, forecasts, or forecast-accuracy comparisons.
