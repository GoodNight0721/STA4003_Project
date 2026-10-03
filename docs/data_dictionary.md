# Data dictionary

## Source and processed files

The four raw economic series below are NBS National Data exports for the national aggregate. Lunar New Year dates are deterministic calendar metadata, not an NBS economic series. Retail amounts retain the NBS unit `亿元` (RMB 100 million); the existing `value_bn` name does not indicate a rescaling to billions.

| Filename | Source / purpose | Frequency and coverage | Major variables | Status |
|---|---|---|---|---|
| `data/raw/retail_sales_monthly.csv` | NBS retail-sales current-period series | Monthly, 1994-01–2026-08 rows | `yyyym`, `value`, `dt_name`, `unit` | Raw; 362 values, 30 expected Jan-Feb reporting gaps since 2012 |
| `data/raw/retail_sales_cumulative.csv` | NBS cumulative retail-sales series | Monthly, 1994-01–2026-08 rows | `yyyym`, `value`, `dt_name`, `unit` | Raw; 305 values, 72 pre-2000 blanks and 15 expected January gaps from 2012 |
| `data/raw/retail_sales_yoy.csv` | NBS retail-sales year-on-year growth | Monthly, 1994-01–2026-08 rows | `yyyym`, `value`, `dt_name`, `unit` | Raw; 290 values, 72 pre-2000 blanks and 30 expected Jan-Feb gaps from 2012 |
| `data/raw/manuf_pmi.csv` | NBS manufacturing PMI | Monthly, 2005-01–2026-08 | `yyyym`, `value`, `dt_name`, `unit` | Raw; 260 values, no missing months |
| `data/processed/retail_quarterly.csv` | Quarterly retail sales: monthly sums for 1994–1999, cumulative differences from 2000 onward | Quarterly, 1994Q1–2026Q2 | `quarter`, `value_bn`, `log_value` | Processed; 130 values, retail in 亿元 |
| `data/processed/pmi_quarterly.csv` | Arithmetic mean of the three monthly PMI observations | Quarterly, 2005Q1–2026Q2 | `quarter`, `pmi` | Processed; 86 values |
| `data/meta/lunar_new_year.csv` | Annual Lunar New Year calendar dates | Annual, 1994–2026 | `year`, `cny_date`, `cny_month` | Deterministic calendar metadata; 33 rows |
| `data/meta/FETCH_REPORT.txt` | NBS retrieval row counts, coverage, and gaps | Per retrieval | Text summary | Retrieval metadata |

Quarter sums from 2000 onward telescope to the December cumulative value because they are differences of that same cumulative series. This is internal accounting consistency, not independent validation. The current-period and cumulative vintages differ in some years; the project uses the cumulative vintage from 2000 onward and does not reconcile or alter the source observations.

## Canonical dataset: `data/processed/analysis_quarterly.csv`

The dataset has one row for every quarter from 1994Q1 through 2026Q2 (130 rows). Rows are in canonical chronological order. `cny_date`, `cny_month`, and `cny_position_fraction` repeat for each quarter of a calendar year. No economic values are imputed.

| Column | Type | Unit | Definition | Availability / missingness | Kind |
|---|---|---|---|---|---|
| `quarter` | string | — | Quarter label, such as `1994Q1` | Complete sequence 1994Q1–2026Q2 | Index |
| `year` | integer | year | Calendar year parsed from `quarter` | Every row | Derived index |
| `quarter_num` | integer | quarter | Quarter number, 1–4 | Every row | Derived index |
| `retail_bn` | float | RMB 100 million (`亿元`) | Quarterly retail amount, unchanged from `value_bn` | Every row; strictly positive | Economic data, derived |
| `log_retail` | float | Natural log of the amount expressed in 亿元 | Natural logarithm of `retail_bn`; stored to six decimals to match `log_value` | Every row; finite | Derived |
| `is_q1` | integer | 0/1 indicator | 1 when `quarter_num` is 1, otherwise 0 | Every row | Derived calendar field |
| `cny_date` | ISO date string | date | Lunar New Year date for the row's calendar year | All years 1994–2026; repeated within year | Deterministic calendar metadata |
| `cny_month` | integer | month number | Month component of `cny_date` (1 or 2) | Every row | Deterministic calendar metadata |
| `cny_position_fraction` | float | unitless fraction | `(cny_date - Jan 1).days / (Apr 1 - Jan 1).days`; Jan 1 is zero, with a 90-day ordinary-year or 91-day leap-year denominator | Every row; repeated within year; not centered or standardized | Deterministic calendar metadata |
| `pmi` | float | PMI index value (source unit `%`) | Quarterly arithmetic mean of monthly manufacturing PMI | Available from 2005Q1; blank before then | Economic data, derived |
| `pmi_lag1` | float | PMI index value | `pmi` from the immediately preceding canonical quarter | Blank until the prior quarter has an observed PMI; first available 2005Q2 | Economic data, lagged derived |
| `pmi_lag2` | float | PMI index value | `pmi` from two quarters earlier on the canonical quarterly index | Blank until the two-quarter lag has observed PMI; first available 2005Q3 | Economic data, lagged derived |

For a two-quarter forecast from origin `T`, realized `PMI[T+1]` and `PMI[T+2]` are unknown and cannot be supplied as contemporaneous predictors. `pmi_lag2` is available for both forecast horizons using information observed by `T`. `pmi_lag1` may require `PMI[T+1]` when predicting `T+2`, so it is not by itself safe at that horizon without an observed or separately forecast predictor. Stage 1 does not select a PMI model or form the later Q1-only, training-centered Lunar New Year regressor.

## Audit records

`docs/data_audit.md` summarizes the Stage 1 audit. `outputs/diagnostics/data_audit.json` contains compact deterministic validation results. `outputs/diagnostics/VERIFICATION.txt` is a pre-existing construction report; its wording was corrected, but its descriptive summaries were not recalculated in Stage 1.
