"""Offline checks for the supplied and derived quarterly data pipeline."""

from datetime import date
import math

import pytest

from scripts.data import validate_data


@pytest.fixture(scope="module")
def validated():
    return validate_data.load_validated_inputs()


@pytest.fixture(scope="module")
def analysis_rows():
    return validate_data.read_table(validate_data.ANALYSIS_PATH, validate_data.ANALYSIS_COLUMNS)


def test_raw_month_keys_are_unique_ordered_and_have_documented_missingness(validated):
    _, bundle = validated
    for name, values in bundle["raw"].items():
        keys = list(values)
        assert len(keys) == len(set(keys))
        assert keys == sorted(keys)
        missing = {key for key, value in values.items() if value is None}
        expected = set()
        for key in keys:
            year, month = int(key[:4]), int(key[4:])
            if name in {"retail_sales_cumulative", "retail_sales_yoy"} and year < 2000:
                expected.add(key)
            elif year >= 2012 and (
                (name == "retail_sales_cumulative" and month == 1)
                or (name in {"retail_sales_monthly", "retail_sales_yoy"} and month in (1, 2))
            ):
                expected.add(key)
        assert missing == expected


def test_raw_coverage_and_missing_counts(validated):
    report, _ = validated
    raw = report["raw"]
    assert (raw["retail_sales_monthly"]["row_count"], raw["retail_sales_monthly"]["non_missing_count"]) == (392, 362)
    assert raw["retail_sales_monthly"]["missing_count"] == 30
    assert (raw["retail_sales_cumulative"]["row_count"], raw["retail_sales_cumulative"]["non_missing_count"]) == (392, 305)
    assert raw["retail_sales_cumulative"]["missing_count"] == 87
    assert (raw["retail_sales_yoy"]["row_count"], raw["retail_sales_yoy"]["non_missing_count"]) == (392, 290)
    assert raw["retail_sales_yoy"]["missing_count"] == 102
    assert (raw["manuf_pmi"]["row_count"], raw["manuf_pmi"]["non_missing_count"]) == (260, 260)


def test_known_current_period_and_cumulative_vintage_differences_are_visible(validated):
    report, _ = validated
    comparisons = report["known_retail_vintage_comparisons"]
    assert comparisons["2005"]["difference_pct_of_cumulative"] == pytest.approx(5.195, abs=0.002)
    assert comparisons["2005"]["difference_pct_of_current"] == pytest.approx(5.48, abs=0.002)
    assert comparisons["2011"]["difference_pct_of_cumulative"] == pytest.approx(0.174, abs=0.002)


def test_processed_retail_has_expected_complete_range_and_positive_finite_logs(validated):
    _, bundle = validated
    retail = bundle["retail_processed"]
    quarters = list(retail)
    assert len(quarters) == 130
    assert quarters == validate_data.quarter_sequence("1994Q1", "2026Q2")
    for row in retail.values():
        assert row["retail"] > 0
        assert row["log_retail"] == pytest.approx(math.log(row["retail"]), abs=validate_data.LOG_TOLERANCE)


def test_processed_pmi_has_expected_complete_range(validated):
    _, bundle = validated
    pmi = bundle["pmi_processed"]
    quarters = list(pmi)
    assert len(quarters) == 86
    assert quarters == validate_data.quarter_sequence("2005Q1", "2026Q2")
    assert all(0 <= value <= 100 for value in pmi.values())


def test_cny_metadata_has_33_unique_january_or_february_dates(validated):
    report, bundle = validated
    cny = bundle["cny"]
    assert report["calendar_metadata"]["row_count"] == 33
    assert list(cny) == list(range(1994, 2027))
    assert len({value.isoformat() for value in cny.values()}) == 33
    assert all(year == value.year and value.month in (1, 2) for year, value in cny.items())


def test_analysis_dataset_has_exact_quarter_sequence_and_derived_fields(analysis_rows):
    quarters = validate_data.quarter_sequence("1994Q1", "2026Q2")
    assert len(analysis_rows) == 130
    assert [row["quarter"] for row in analysis_rows] == quarters
    for row in analysis_rows:
        year, quarter_num = validate_data.quarter_parts(row["quarter"], "analysis test")
        assert int(row["year"]) == year
        assert int(row["quarter_num"]) == quarter_num
        assert int(row["is_q1"]) == int(quarter_num == 1)


def test_analysis_retail_and_log_values_match_processed_source(validated, analysis_rows):
    _, bundle = validated
    for row in analysis_rows:
        source = bundle["retail_processed"][row["quarter"]]
        retail = float(row["retail_bn"])
        log_retail = float(row["log_retail"])
        assert retail == pytest.approx(source["retail"], abs=1e-9)
        assert log_retail == pytest.approx(source["log_retail"], abs=validate_data.LOG_TOLERANCE)
        assert log_retail == pytest.approx(math.log(retail), abs=validate_data.LOG_TOLERANCE)


def test_cny_fields_and_zero_based_position_formula(validated, analysis_rows):
    _, bundle = validated
    for row in analysis_rows:
        year = int(row["year"])
        cny = bundle["cny"][year]
        assert row["cny_date"] == cny.isoformat()
        assert int(row["cny_month"]) == cny.month
        q1_start = date(year, 1, 1)
        q1_end = date(year, 4, 1)
        expected_fraction = (cny - q1_start).days / (q1_end - q1_start).days
        assert float(row["cny_position_fraction"]) == pytest.approx(
            expected_fraction, abs=5.1e-13
        )
    assert (date(2020, 4, 1) - date(2020, 1, 1)).days == 91
    assert (date(2021, 4, 1) - date(2021, 1, 1)).days == 90


def test_pmi_is_missing_before_2005q1_and_not_imputed(validated, analysis_rows):
    _, bundle = validated
    for row in analysis_rows:
        expected = bundle["pmi_processed"].get(row["quarter"])
        if expected is None:
            assert row["pmi"] == ""
        else:
            assert float(row["pmi"]) == pytest.approx(expected, abs=1e-9)
    assert all(row["pmi"] == "" for row in analysis_rows if row["quarter"] < "2005Q1")


def test_pmi_lags_align_to_canonical_quarter_index(validated, analysis_rows):
    by_quarter = {row["quarter"]: row for row in analysis_rows}
    assert float(by_quarter["2005Q2"]["pmi_lag1"]) == pytest.approx(float(by_quarter["2005Q1"]["pmi"]))
    assert float(by_quarter["2005Q3"]["pmi_lag2"]) == pytest.approx(float(by_quarter["2005Q1"]["pmi"]))

    for index, row in enumerate(analysis_rows):
        for column, lag in (("pmi_lag1", 1), ("pmi_lag2", 2)):
            expected = None
            if index >= lag:
                expected = validated[1]["pmi_processed"].get(analysis_rows[index - lag]["quarter"])
            if expected is None:
                assert row[column] == ""
            else:
                assert float(row[column]) == pytest.approx(expected, abs=1e-9)


def test_independent_raw_reconstruction_matches_processed_quarterly_series(validated):
    report, bundle = validated
    assert report["processed"]["retail_quarterly"]["reconstruction_pass"]
    assert report["processed"]["retail_quarterly"]["max_abs_reconstruction_difference"] <= validate_data.RETAIL_TOLERANCE
    assert report["processed"]["pmi_quarterly"]["reconstruction_pass"]
    assert report["processed"]["pmi_quarterly"]["max_abs_reconstruction_difference"] <= validate_data.PMI_TOLERANCE
    monthly = bundle["raw"]["retail_sales_monthly"]
    cumulative = bundle["raw"]["retail_sales_cumulative"]
    retail_rebuilt = {}
    for year in range(1994, 2027):
        for quarter_num in range(1, 5):
            quarter = f"{year}Q{quarter_num}"
            end_month = quarter_num * 3
            if year < 2000:
                month_keys = [f"{year}{month:02d}" for month in range(end_month - 2, end_month + 1)]
                values = [monthly.get(key) for key in month_keys]
                if all(value is not None for value in values):
                    retail_rebuilt[quarter] = sum(values)
            elif quarter_num == 1:
                value = cumulative.get(f"{year}{end_month:02d}")
                if value is not None:
                    retail_rebuilt[quarter] = value
            else:
                current = cumulative.get(f"{year}{end_month:02d}")
                previous = cumulative.get(f"{year}{end_month - 3:02d}")
                if current is not None and previous is not None:
                    retail_rebuilt[quarter] = current - previous
    assert list(retail_rebuilt) == list(bundle["retail_processed"])
    for quarter, value in retail_rebuilt.items():
        assert value == pytest.approx(bundle["retail_processed"][quarter]["retail"], abs=validate_data.RETAIL_TOLERANCE)

    raw_pmi = bundle["raw"]["manuf_pmi"]
    pmi_rebuilt = {}
    for year in range(2005, 2027):
        for quarter_num in range(1, 5):
            month_keys = [f"{year}{month:02d}" for month in range(quarter_num * 3 - 2, quarter_num * 3 + 1)]
            values = [raw_pmi.get(key) for key in month_keys]
            if all(value is not None for value in values):
                pmi_rebuilt[f"{year}Q{quarter_num}"] = sum(values) / 3
    assert list(pmi_rebuilt) == list(bundle["pmi_processed"])
    for quarter, value in pmi_rebuilt.items():
        assert value == pytest.approx(bundle["pmi_processed"][quarter], abs=validate_data.PMI_TOLERANCE)
