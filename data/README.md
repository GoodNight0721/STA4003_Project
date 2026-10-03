# Data folder (STA4003 project - retail sales & PMI)

All data downloaded from the official National Bureau of Statistics of China
"National Data" platform (`https://data.stats.gov.cn`), national aggregate.

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
- `meta/lunar_new_year.csv` - Chinese New Year date per year 1994-2026 (computed with
  the `sxtwl` calendar library, spot-checked against official dates).
- `meta/FETCH_REPORT.txt` - retrieval metadata, gaps and duplicates check.
- `processed/` - derived analysis-ready series.
  - `retail_quarterly.csv` - quarterly total retail sales (RMB 100 million), 1994Q1-2026Q2,
    N=130, no missing. Q1 2012+ built from the published Jan-Feb combined total + March;
    all quarters 2000+ recovered by differencing the official cumulative series (verified
    to match published annual totals exactly). 1994-1999 from monthly sums.
  - `pmi_quarterly.csv` - quarterly mean PMI, 2005Q1-2026Q2, N=86.
- The construction report is preserved at `outputs/diagnostics/VERIFICATION.txt`.

## Reproduce
1. `python scripts/data/fetch_nbs_data.py` (needs internet; pulls raw CSVs from the NBS API)
2. `python scripts/data/build_quarterly.py` (raw -> processed + diagnostics report)

## Known data-quality issues (see proposal section 4)
- Since 2012 NBS publishes Jan & Feb retail as a single combined total.
- The current-month and cumulative vintages disagree in a few years (2005 ~5.2%,
  2011 ~0.2%); the cumulative vintage matches official annual totals and is used.
- 2020Q1 and 2022Q2 are genuine COVID-19 outliers, kept in the series.
- The retail series was revised under the 5th Economic Census; growth rates since
  2025 are on a comparable basis.
