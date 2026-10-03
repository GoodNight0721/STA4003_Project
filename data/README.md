# Data folder (STA4003 project - retail sales & PMI)

The raw retail-sales and manufacturing-PMI economic series are from the National
Bureau of Statistics of China (NBS) National Data platform (`https://data.stats.gov.cn`),
national aggregate. The Lunar New Year table is separate deterministic calendar
metadata; it is not an NBS economic series. Stage 1 checks its internal date/year/month
consistency and does not fetch alternate calendar dates.

## Layout
- `raw/` - monthly series as downloaded from the NBS API (see `scripts/data/fetch_nbs_data.py`).
  - `retail_sales_monthly.csv` - total retail sales of consumer goods, current-month
    value, RMB 100 million (当期值, 亿元). 1994-01 .. 2026-08. Jan & Feb are **empty
    from 2012 onward** because NBS publishes them only as a combined "Jan-Feb" total.
  - `retail_sales_cumulative.csv` - same series, year-to-date cumulative (累计值),
    2000-01 .. 2026-08. From 2012 onward the January row is empty and the February row
    holds the combined Jan-Feb cumulative total.
  - `retail_sales_yoy.csv` - year-on-year growth, % (from 2000).
  - `manuf_pmi.csv` - manufacturing PMI, %, 2005-01 .. 2026-08.
- `meta/lunar_new_year.csv` - Chinese New Year date per year 1994-2026. Calendar
  metadata is kept separate from NBS economic observations.
- `meta/FETCH_REPORT.txt` - retrieval metadata, gaps and duplicates check.
- `processed/` - derived analysis-ready series.
  - `retail_quarterly.csv` - quarterly total retail sales in RMB 100 million (`亿元`),
    1994Q1-2026Q2, N=130, no missing. The legacy `value_bn` column is not rescaled:
    its values remain in `亿元`. 1994-1999 use monthly current-period sums; 2000 onward
    use differences of the cumulative series. Quarter sums from 2000 onward telescope
    to December cumulative values by construction, an internal accounting identity and
    not independent validation.
  - `pmi_quarterly.csv` - quarterly mean PMI, 2005Q1-2026Q2, N=86.
- `processed/analysis_quarterly.csv` - canonical 130-quarter analysis input, including
  retail, deterministic CNY calendar fields, PMI, and quarterly PMI lags. PMI fields
  remain blank where source values or lagged quarters are unavailable.
- The earlier construction report is preserved at `outputs/diagnostics/VERIFICATION.txt`.

## Reproduce
1. `python scripts/data/validate_data.py`
2. `python scripts/data/build_analysis_dataset.py`
3. `python scripts/data/validate_data.py`
4. `pytest -q`

## Known data-quality issues (see proposal section 4)
- Since 2012 NBS publishes Jan & Feb retail as a single combined total.
- The current-period and cumulative vintages disagree in 2005 and 2011. In 2005 the
  current-period annual sum is 63,686.6 亿元 and December cumulative is 67,176.6 亿元
  (5.195% of the cumulative value; 5.48% of the current-period sum). In 2011 the
  difference is 0.174% of the cumulative value. The project uses cumulative values
  from 2000 onward; this choice does not erase the vintage discrepancy.
- 2020Q1 and 2022Q2 are genuine COVID-19 outliers, kept in the series.
- The retail series was revised under the 5th Economic Census; growth rates since
  2025 are on a comparable basis.
- PMI is unavailable before 2005Q1. No economic values are imputed in the canonical
  dataset, and future contemporaneous PMI must not be used in rolling forecasts.
