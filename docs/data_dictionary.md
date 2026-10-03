# Data dictionary

This is an inventory of supplied files, not a full data audit. Source and variable descriptions follow the accompanying NBS export fields and project notes.

| Filename | Source / purpose | Frequency | Major variables | Status |
|---|---|---|---|---|
| `data/raw/retail_sales_monthly.csv` | NBS total retail-sales current-period series | Monthly | `yyyym`, `value`, `dt_name`, `unit` | Raw |
| `data/raw/retail_sales_cumulative.csv` | NBS cumulative retail-sales series | Monthly, year-to-date | `yyyym`, `value`, `dt_name`, `unit` | Raw |
| `data/raw/retail_sales_yoy.csv` | NBS retail-sales year-on-year growth series | Monthly | `yyyym`, `value`, `dt_name`, `unit` | Raw |
| `data/raw/manuf_pmi.csv` | NBS manufacturing PMI series | Monthly | `yyyym`, `value`, `dt_name`, `unit` | Raw |
| `data/processed/retail_quarterly.csv` | Quarterly retail-sales series derived from raw NBS exports | Quarterly | `quarter`, `value_bn`, `log_value` | Processed |
| `data/processed/pmi_quarterly.csv` | Quarterly mean PMI derived from raw monthly PMI | Quarterly | `quarter`, `pmi` | Processed |
| `data/meta/lunar_new_year.csv` | Lunar New Year date metadata used by the project | Annual | `year`, `cny_date`, `cny_month` | Metadata |
| `data/meta/FETCH_REPORT.txt` | Retrieval summary for the NBS exports | Per retrieval | Series row counts, ranges, gaps, and duplicates | Metadata |

`data/README.md` contains the supplied source notes and known reporting details. `outputs/diagnostics/VERIFICATION.txt` is a pre-existing construction report, preserved as an output; Stage 0 did not rerun its calculations.
