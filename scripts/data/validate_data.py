"""Validate committed source data, processed series, and the canonical dataset."""

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
META_DIR = PROJECT_ROOT / "data" / "meta"
DIAGNOSTICS_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
ANALYSIS_PATH = PROCESSED_DIR / "analysis_quarterly.csv"

MONTHLY_COLUMNS = ["yyyym", "value", "dt_name", "unit"]
RETAIL_QUARTERLY_COLUMNS = ["quarter", "value_bn", "log_value"]
PMI_QUARTERLY_COLUMNS = ["quarter", "pmi"]
ANALYSIS_COLUMNS = [
    "quarter",
    "year",
    "quarter_num",
    "retail_bn",
    "log_retail",
    "is_q1",
    "cny_date",
    "cny_month",
    "cny_position_fraction",
    "pmi",
    "pmi_lag1",
    "pmi_lag2",
]

RETAIL_TOLERANCE = 0.0500001  # committed processed values are rounded to one decimal
PMI_TOLERANCE = 0.0050001  # committed quarterly PMI values are rounded to two decimals
LOG_TOLERANCE = 0.00000051  # committed log values are rounded to six decimals


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_table(path, expected_columns):
    require(path.is_file(), f"Missing required file: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames == expected_columns,
            f"{path.relative_to(PROJECT_ROOT)}: expected columns {expected_columns}, got {reader.fieldnames}",
        )
        rows = list(reader)
    for row_number, row in enumerate(rows, start=2):
        require(
            None not in row and all(value is not None for value in row.values()),
            f"{path.relative_to(PROJECT_ROOT)}:{row_number}: malformed row width",
        )
    return rows


def parse_number(value, label, allow_blank=False):
    value = value.strip()
    if not value:
        require(allow_blank, f"{label}: value is unexpectedly blank")
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{label}: expected a number, got {value!r}") from exc
    require(math.isfinite(number), f"{label}: value must be finite, got {value!r}")
    return number


def parse_month_key(key, label):
    require(re.fullmatch(r"\d{6}", key) is not None, f"{label}: invalid YYYYMM key {key!r}")
    year, month = int(key[:4]), int(key[4:])
    try:
        return date(year, month, 1)
    except ValueError as exc:
        raise ValueError(f"{label}: invalid month key {key!r}") from exc


def month_key(value):
    return f"{value.year:04d}{value.month:02d}"


def month_sequence(start, end):
    first = parse_month_key(start, "month range")
    last = parse_month_key(end, "month range")
    require(first <= last, f"Invalid month range {start}..{end}")
    result = []
    current = first
    while current <= last:
        result.append(month_key(current))
        current = date(current.year + (current.month == 12), 1 if current.month == 12 else current.month + 1, 1)
    return result


def quarter_parts(key, label):
    match = re.fullmatch(r"(\d{4})Q([1-4])", key)
    require(match is not None, f"{label}: invalid quarter key {key!r}")
    return int(match.group(1)), int(match.group(2))


def quarter_sequence(start, end):
    start_year, start_quarter = quarter_parts(start, "quarter range")
    end_year, end_quarter = quarter_parts(end, "quarter range")
    result = []
    year, quarter = start_year, start_quarter
    while (year, quarter) <= (end_year, end_quarter):
        result.append(f"{year}Q{quarter}")
        if quarter == 4:
            year, quarter = year + 1, 1
        else:
            quarter += 1
    return result


def expected_missing_months(series_name, keys):
    missing = set()
    for key in keys:
        year, month = int(key[:4]), int(key[4:])
        if series_name in {"retail_sales_cumulative", "retail_sales_yoy"} and year < 2000:
            missing.add(key)
        elif year >= 2012 and (
            month == 1 if series_name == "retail_sales_cumulative" else month in (1, 2)
        ) and series_name in {"retail_sales_monthly", "retail_sales_cumulative", "retail_sales_yoy"}:
            missing.add(key)
    return missing


RAW_SPECS = {
    "retail_sales_monthly": {
        "filename": "retail_sales_monthly.csv",
        "start": "199401",
        "end": "202608",
        "unit": "亿元",
        "positive": True,
        "missing_note": "Separate January and February values are not published from 2012; NBS reports them jointly.",
    },
    "retail_sales_cumulative": {
        "filename": "retail_sales_cumulative.csv",
        "start": "199401",
        "end": "202608",
        "unit": "亿元",
        "positive": True,
        "missing_note": "Values begin in 2000; January cumulative rows are unavailable from 2012 under the Jan-Feb convention.",
    },
    "retail_sales_yoy": {
        "filename": "retail_sales_yoy.csv",
        "start": "199401",
        "end": "202608",
        "unit": "%",
        "positive": False,
        "missing_note": "Values begin in 2000; Jan-Feb growth values are unavailable from 2012 under the combined reporting convention.",
    },
    "manuf_pmi": {
        "filename": "manuf_pmi.csv",
        "start": "200501",
        "end": "202608",
        "unit": "%",
        "positive": True,
        "missing_note": "No missing monthly values are expected within the observed PMI range.",
    },
}


def audit_raw_series(series_name, spec):
    path = RAW_DIR / spec["filename"]
    rows = read_table(path, MONTHLY_COLUMNS)
    keys = [row["yyyym"].strip() for row in rows]
    parsed_dates = [parse_month_key(key, f"{path.name} row {index + 2}") for index, key in enumerate(keys)]
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    require(not duplicate_keys, f"{path.name}: duplicate months {duplicate_keys[:8]}")
    require(keys == sorted(keys), f"{path.name}: months are not in chronological order")
    expected_keys = month_sequence(spec["start"], spec["end"])
    require(keys == expected_keys, f"{path.name}: row coverage must be {spec['start']}..{spec['end']} with every month represented")

    values = {}
    observed_units = set()
    for index, row in enumerate(rows, start=2):
        key = row["yyyym"].strip()
        value = parse_number(row["value"], f"{path.name}:{index} ({key})", allow_blank=True)
        values[key] = value
        if value is not None:
            if spec["positive"]:
                require(value > 0, f"{path.name}:{index} ({key}): expected a positive value")
            if series_name == "manuf_pmi":
                require(0 <= value <= 100, f"{path.name}:{index} ({key}): PMI must lie between 0 and 100")
            observed_units.add(row["unit"].strip())
    require(observed_units == {spec["unit"]}, f"{path.name}: expected unit {spec['unit']!r}, found {sorted(observed_units)}")

    actual_missing = {key for key, value in values.items() if value is None}
    expected_missing = expected_missing_months(series_name, keys)
    require(
        actual_missing == expected_missing,
        f"{path.name}: missingness differs from the documented pattern; unexpected={sorted(actual_missing - expected_missing)[:8]}, "
        f"unexpectedly_present={sorted(expected_missing - actual_missing)[:8]}",
    )
    observed = [key for key, value in values.items() if value is not None]
    summary = {
        "filename": spec["filename"],
        "row_count": len(rows),
        "row_start": keys[0],
        "row_end": keys[-1],
        "non_missing_count": len(observed),
        "missing_count": len(actual_missing),
        "observed_start": observed[0] if observed else None,
        "observed_end": observed[-1] if observed else None,
        "duplicate_count": len(duplicate_keys),
        "unit": spec["unit"],
        "documented_missingness": spec["missing_note"],
    }
    if series_name in {"retail_sales_monthly", "retail_sales_cumulative", "retail_sales_yoy"}:
        summary["documented_gap_count"] = len(expected_missing)
    return summary, values


def read_processed_retail():
    path = PROCESSED_DIR / "retail_quarterly.csv"
    rows = read_table(path, RETAIL_QUARTERLY_COLUMNS)
    quarters = [row["quarter"].strip() for row in rows]
    for key in quarters:
        quarter_parts(key, path.name)
    duplicate_keys = sorted(key for key, count in Counter(quarters).items() if count > 1)
    require(not duplicate_keys, f"{path.name}: duplicate quarters {duplicate_keys[:8]}")
    require(quarters == sorted(quarters), f"{path.name}: quarters are not in chronological order")
    expected = quarter_sequence("1994Q1", "2026Q2")
    require(quarters == expected, f"{path.name}: expected 130 consecutive quarters from 1994Q1 through 2026Q2")

    result = {}
    max_log_difference = 0.0
    for index, row in enumerate(rows, start=2):
        key = row["quarter"].strip()
        value = parse_number(row["value_bn"], f"{path.name}:{index} ({key})")
        log_value = parse_number(row["log_value"], f"{path.name}:{index} ({key})")
        require(value > 0, f"{path.name}:{index} ({key}): retail must be positive")
        difference = abs(log_value - math.log(value))
        max_log_difference = max(max_log_difference, difference)
        require(difference <= LOG_TOLERANCE, f"{path.name}:{index} ({key}): log_value differs from log(value_bn) by {difference}")
        result[key] = {"retail": value, "log_retail": log_value}
    return result, {
        "filename": path.name,
        "row_count": len(rows),
        "first_quarter": quarters[0],
        "last_quarter": quarters[-1],
        "duplicate_count": len(duplicate_keys),
        "missing_value_count": 0,
        "retail_unit": "RMB 100 million (亿元), unchanged from the source series",
        "max_log_rounding_difference": round(max_log_difference, 12),
    }


def read_processed_pmi():
    path = PROCESSED_DIR / "pmi_quarterly.csv"
    rows = read_table(path, PMI_QUARTERLY_COLUMNS)
    quarters = [row["quarter"].strip() for row in rows]
    for key in quarters:
        quarter_parts(key, path.name)
    duplicate_keys = sorted(key for key, count in Counter(quarters).items() if count > 1)
    require(not duplicate_keys, f"{path.name}: duplicate quarters {duplicate_keys[:8]}")
    require(quarters == sorted(quarters), f"{path.name}: quarters are not in chronological order")
    expected = quarter_sequence("2005Q1", "2026Q2")
    require(quarters == expected, f"{path.name}: expected 86 consecutive quarters from 2005Q1 through 2026Q2")
    result = {}
    for index, row in enumerate(rows, start=2):
        key = row["quarter"].strip()
        value = parse_number(row["pmi"], f"{path.name}:{index} ({key})")
        require(0 <= value <= 100, f"{path.name}:{index} ({key}): PMI must lie between 0 and 100")
        result[key] = value
    return result, {
        "filename": path.name,
        "row_count": len(rows),
        "first_quarter": quarters[0],
        "last_quarter": quarters[-1],
        "duplicate_count": len(duplicate_keys),
        "missing_value_count": 0,
        "unit": "PMI index value (%)",
    }


def reconstruct_retail(monthly, cumulative):
    reconstructed = {}
    for year in range(1994, 2027):
        for quarter in range(1, 5):
            key = f"{year}Q{quarter}"
            if year < 2000:
                months = [f"{year}{month:02d}" for month in range(quarter * 3 - 2, quarter * 3 + 1)]
                values = [monthly.get(month) for month in months]
                if all(value is not None for value in values):
                    reconstructed[key] = sum(values)
                continue
            end_month = quarter * 3
            end_value = cumulative.get(f"{year}{end_month:02d}")
            if quarter == 1:
                if end_value is not None:
                    reconstructed[key] = end_value
                continue
            previous_value = cumulative.get(f"{year}{end_month - 3:02d}")
            if end_value is not None and previous_value is not None:
                reconstructed[key] = end_value - previous_value
    return reconstructed


def reconstruct_pmi(monthly):
    result = {}
    for year in range(2005, 2027):
        for quarter in range(1, 5):
            months = [f"{year}{month:02d}" for month in range(quarter * 3 - 2, quarter * 3 + 1)]
            values = [monthly.get(month) for month in months]
            if all(value is not None for value in values):
                result[f"{year}Q{quarter}"] = sum(values) / 3
    return result


def read_cny_metadata():
    path = META_DIR / "lunar_new_year.csv"
    rows = read_table(path, ["year", "cny_date", "cny_month"])
    years = []
    dates = []
    result = {}
    for index, row in enumerate(rows, start=2):
        label = f"{path.name}:{index}"
        require(re.fullmatch(r"\d{4}", row["year"].strip()) is not None, f"{label}: invalid year")
        year = int(row["year"])
        try:
            cny_date = date.fromisoformat(row["cny_date"].strip())
        except ValueError as exc:
            raise ValueError(f"{label}: invalid ISO date {row['cny_date']!r}") from exc
        cny_month = parse_number(row["cny_month"], f"{label} cny_month")
        require(cny_date.year == year, f"{label}: cny_date year does not match year")
        require(cny_date.month in (1, 2), f"{label}: CNY must fall in January or February")
        require(cny_month == int(cny_date.month), f"{label}: cny_month does not match cny_date")
        years.append(year)
        dates.append(cny_date)
        result[year] = cny_date
    duplicate_years = sorted(year for year, count in Counter(years).items() if count > 1)
    duplicate_dates = sorted(day.isoformat() for day, count in Counter(dates).items() if count > 1)
    require(not duplicate_years, f"{path.name}: duplicate years {duplicate_years}")
    require(not duplicate_dates, f"{path.name}: duplicate dates {duplicate_dates}")
    expected_years = list(range(1994, 2027))
    require(years == expected_years, f"{path.name}: expected exactly one ordered row for each year 1994..2026")
    return result, {
        "filename": path.name,
        "row_count": len(rows),
        "unique_year_count": len(set(years)),
        "first_year": years[0],
        "last_year": years[-1],
        "duplicate_year_count": len(duplicate_years),
        "duplicate_date_count": len(duplicate_dates),
        "valid_months": [1, 2],
        "source_note": "Deterministic Lunar New Year calendar metadata; not an NBS economic series.",
    }


def load_validated_inputs():
    raw_values = {}
    raw_summary = {}
    for name, spec in RAW_SPECS.items():
        raw_summary[name], raw_values[name] = audit_raw_series(name, spec)

    vintage_comparison = {}
    monthly_values = raw_values["retail_sales_monthly"]
    cumulative_values = raw_values["retail_sales_cumulative"]
    for year in (2005, 2011):
        current_months = [monthly_values.get(f"{year}{month:02d}") for month in range(1, 13)]
        december_cumulative = cumulative_values.get(f"{year}12")
        require(
            all(value is not None for value in current_months) and december_cumulative is not None,
            f"Cannot compare current-period and cumulative retail vintages for {year}",
        )
        current_annual = sum(current_months)
        difference = december_cumulative - current_annual
        vintage_comparison[str(year)] = {
            "unit": "RMB 100 million (亿元)",
            "current_period_annual_total": round(current_annual, 1),
            "december_cumulative": round(december_cumulative, 1),
            "cumulative_minus_current": round(difference, 1),
            "difference_pct_of_cumulative": round(100 * difference / december_cumulative, 4),
            "difference_pct_of_current": round(100 * difference / current_annual, 4),
        }

    cny, cny_summary = read_cny_metadata()
    retail_processed, retail_summary = read_processed_retail()
    pmi_processed, pmi_summary = read_processed_pmi()

    retail_rebuilt = reconstruct_retail(
        raw_values["retail_sales_monthly"], raw_values["retail_sales_cumulative"]
    )
    expected_retail_quarters = list(retail_processed)
    require(
        list(retail_rebuilt) == expected_retail_quarters,
        "Raw retail reconstruction quarter sequence does not match data/processed/retail_quarterly.csv",
    )
    retail_differences = {
        quarter: abs(retail_rebuilt[quarter] - retail_processed[quarter]["retail"])
        for quarter in expected_retail_quarters
    }
    max_retail_difference = max(retail_differences.values())
    require(
        max_retail_difference <= RETAIL_TOLERANCE,
        f"Raw retail reconstruction differs from processed retail by up to {max_retail_difference:.9g}; "
        f"tolerance is {RETAIL_TOLERANCE}",
    )

    pmi_rebuilt = reconstruct_pmi(raw_values["manuf_pmi"])
    require(
        list(pmi_rebuilt) == list(pmi_processed),
        "Raw PMI reconstruction quarter sequence does not match data/processed/pmi_quarterly.csv",
    )
    pmi_differences = {
        quarter: abs(pmi_rebuilt[quarter] - pmi_processed[quarter]) for quarter in pmi_processed
    }
    max_pmi_difference = max(pmi_differences.values())
    require(
        max_pmi_difference <= PMI_TOLERANCE,
        f"Raw PMI reconstruction differs from processed PMI by up to {max_pmi_difference:.9g}; "
        f"tolerance is {PMI_TOLERANCE}",
    )

    report = {
        "raw": raw_summary,
        "known_retail_vintage_comparisons": vintage_comparison,
        "calendar_metadata": cny_summary,
        "processed": {
            "retail_quarterly": {
                **retail_summary,
                "max_abs_reconstruction_difference": round(max_retail_difference, 12),
                "reconstruction_tolerance": RETAIL_TOLERANCE,
                "reconstruction_pass": True,
            },
            "pmi_quarterly": {
                **pmi_summary,
                "max_abs_reconstruction_difference": round(max_pmi_difference, 12),
                "reconstruction_tolerance": PMI_TOLERANCE,
                "reconstruction_pass": True,
            },
        },
    }
    bundle = {
        "raw": raw_values,
        "cny": cny,
        "retail_processed": retail_processed,
        "pmi_processed": pmi_processed,
        "retail_reconstructed": retail_rebuilt,
        "pmi_reconstructed": pmi_rebuilt,
    }
    return report, bundle


def cny_position_fraction(cny_date):
    q1_start = date(cny_date.year, 1, 1)
    q1_end = date(cny_date.year, 4, 1)
    return (cny_date - q1_start).days / (q1_end - q1_start).days


def validate_analysis_dataset(bundle):
    rows = read_table(ANALYSIS_PATH, ANALYSIS_COLUMNS)
    quarters = [row["quarter"].strip() for row in rows]
    for key in quarters:
        quarter_parts(key, ANALYSIS_PATH.name)
    duplicate_quarters = sorted(key for key, count in Counter(quarters).items() if count > 1)
    require(not duplicate_quarters, f"{ANALYSIS_PATH.name}: duplicate quarters {duplicate_quarters[:8]}")
    expected_quarters = quarter_sequence("1994Q1", "2026Q2")
    require(quarters == expected_quarters, f"{ANALYSIS_PATH.name}: expected exactly 130 consecutive quarters")

    missing_pmi = 0
    missing_lag1 = 0
    missing_lag2 = 0
    for index, row in enumerate(rows):
        quarter = quarters[index]
        year, quarter_num = quarter_parts(quarter, ANALYSIS_PATH.name)
        label = f"{ANALYSIS_PATH.name}:{index + 2} ({quarter})"
        require(parse_number(row["year"], f"{label} year") == year, f"{label}: incorrect year")
        require(parse_number(row["quarter_num"], f"{label} quarter_num") == quarter_num, f"{label}: incorrect quarter_num")
        is_q1 = parse_number(row["is_q1"], f"{label} is_q1")
        require(is_q1 == int(quarter_num == 1), f"{label}: incorrect is_q1")

        retail_source = bundle["retail_processed"][quarter]
        retail = parse_number(row["retail_bn"], f"{label} retail_bn")
        log_retail = parse_number(row["log_retail"], f"{label} log_retail")
        require(retail > 0, f"{label}: retail_bn must be positive")
        require(abs(retail - retail_source["retail"]) <= 1e-9, f"{label}: retail_bn differs from processed retail")
        require(abs(log_retail - math.log(retail)) <= LOG_TOLERANCE, f"{label}: log_retail differs from log(retail_bn)")
        require(abs(log_retail - retail_source["log_retail"]) <= LOG_TOLERANCE, f"{label}: log_retail differs from processed log_value")

        cny_date = bundle["cny"][year]
        require(row["cny_date"].strip() == cny_date.isoformat(), f"{label}: incorrect cny_date")
        require(parse_number(row["cny_month"], f"{label} cny_month") == cny_date.month, f"{label}: incorrect cny_month")
        position = parse_number(row["cny_position_fraction"], f"{label} cny_position_fraction")
        require(
            abs(position - cny_position_fraction(cny_date)) <= 5.1e-13,
            f"{label}: cny_position_fraction does not follow the documented formula",
        )

        expected_pmi = bundle["pmi_processed"].get(quarter)
        pmi_value = parse_number(row["pmi"], f"{label} pmi", allow_blank=True)
        if expected_pmi is None:
            missing_pmi += 1
            require(pmi_value is None, f"{label}: PMI must remain missing when no quarterly source exists")
        else:
            require(pmi_value is not None and abs(pmi_value - expected_pmi) <= 1e-9, f"{label}: incorrect pmi")

        for column, lag in (("pmi_lag1", 1), ("pmi_lag2", 2)):
            expected_lag = None
            if index >= lag:
                source_quarter = quarters[index - lag]
                expected_lag = bundle["pmi_processed"].get(source_quarter)
            lag_value = parse_number(row[column], f"{label} {column}", allow_blank=True)
            if expected_lag is None:
                if column == "pmi_lag1":
                    missing_lag1 += 1
                else:
                    missing_lag2 += 1
                require(lag_value is None, f"{label}: {column} must be missing without an observed lagged quarter")
            else:
                require(lag_value is not None and abs(lag_value - expected_lag) <= 1e-9, f"{label}: incorrect {column}")

    return {
        "filename": ANALYSIS_PATH.name,
        "present": True,
        "row_count": len(rows),
        "first_quarter": quarters[0],
        "last_quarter": quarters[-1],
        "duplicate_count": len(duplicate_quarters),
        "pmi_missing_rows": missing_pmi,
        "pmi_lag1_missing_rows": missing_lag1,
        "pmi_lag2_missing_rows": missing_lag2,
        "economic_variable_imputations": 0,
        "sha256": hashlib.sha256(ANALYSIS_PATH.read_bytes()).hexdigest(),
    }


def validate_all():
    report, bundle = load_validated_inputs()
    if ANALYSIS_PATH.is_file():
        report["analysis_dataset"] = validate_analysis_dataset(bundle)
    else:
        report["analysis_dataset"] = {"filename": ANALYSIS_PATH.name, "present": False}
    return report


def main():
    report = validate_all()
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DIAGNOSTICS_DIR / "data_audit.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Data validation passed.")
    for name, details in report["raw"].items():
        print(
            f"{name}: rows={details['row_count']}, observed={details['observed_start']}..{details['observed_end']}, "
            f"nonmissing={details['non_missing_count']}, missing={details['missing_count']}, duplicates={details['duplicate_count']}"
        )
    for name, details in report["processed"].items():
        print(
            f"{name}: rows={details['row_count']}, range={details['first_quarter']}..{details['last_quarter']}, "
            f"max reconstruction difference={details['max_abs_reconstruction_difference']}"
        )
    analysis = report["analysis_dataset"]
    if analysis["present"]:
        print(
            f"analysis_quarterly: rows={analysis['row_count']}, "
            f"PMI missing={analysis['pmi_missing_rows']}, "
            f"sha256={analysis['sha256']}"
        )
    else:
        print("analysis_quarterly: not built yet")
    print(f"Audit summary: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
