"""Stage 2 transformation, diagnostics, reproducibility, and immutability tests."""

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import eda_stationarity


@pytest.fixture(scope="module")
def canonical():
    return eda_stationarity.load_canonical()


@pytest.fixture(scope="module")
def transformations(canonical):
    return eda_stationarity.make_transformations(canonical)


def test_transformation_lengths_and_indexes(transformations):
    assert len(transformations["log_retail"]) == 130
    assert len(transformations["first_difference"]) == 129
    assert len(transformations["seasonal_difference"]) == 126
    assert len(transformations["combined_difference"]) == 125
    assert str(transformations["first_difference"].index[0]) == "1994Q2"
    assert str(transformations["seasonal_difference"].index[0]) == "1995Q1"
    assert str(transformations["combined_difference"].index[0]) == "1995Q2"


def test_regular_difference_alignment_is_exact(canonical, transformations):
    y = canonical["log_retail"]
    quarter = pd.Period("1994Q2", freq="Q-DEC")
    expected = y.loc[quarter] - y.loc[pd.Period("1994Q1", freq="Q-DEC")]
    assert transformations["first_difference"].loc[quarter] == pytest.approx(expected, abs=1e-12)


def test_seasonal_difference_uses_four_quarters(canonical, transformations):
    y = canonical["log_retail"]
    quarter = pd.Period("1995Q1", freq="Q-DEC")
    expected = y.loc[quarter] - y.loc[pd.Period("1994Q1", freq="Q-DEC")]
    assert transformations["seasonal_difference"].loc[quarter] == pytest.approx(expected, abs=1e-12)


def test_combined_difference_is_regular_difference_of_seasonal_difference(canonical, transformations):
    y = canonical["log_retail"]
    quarter = pd.Period("1995Q2", freq="Q-DEC")
    previous = pd.Period("1995Q1", freq="Q-DEC")
    year_ago = pd.Period("1994Q2", freq="Q-DEC")
    prior_year_ago = pd.Period("1994Q1", freq="Q-DEC")
    expected = (y.loc[quarter] - y.loc[previous]) - (y.loc[year_ago] - y.loc[prior_year_ago])
    assert transformations["combined_difference"].loc[quarter] == pytest.approx(expected, abs=1e-12)


def test_stationarity_table_schema_and_results_are_finite():
    path = eda_stationarity.PROJECT_ROOT / "outputs" / "tables" / "stationarity_tests.csv"
    table = pd.read_csv(path)
    assert table.columns.tolist() == eda_stationarity.STATIONARITY_COLUMNS
    assert table["transformation"].tolist() == [
        "log_retail",
        "first_difference",
        "seasonal_difference",
        "combined_difference",
    ]
    for column in ("n_obs", "adf_statistic", "adf_pvalue", "kpss_statistic", "kpss_pvalue"):
        values = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
        assert np.isfinite(values).all()
    assert table["adf_pvalue"].between(0, 1).all()
    assert table["kpss_pvalue"].between(0, 1).all()


def test_generated_summary_records_fixed_seasonality_and_no_model_search():
    path = eda_stationarity.PROJECT_ROOT / "outputs" / "diagnostics" / "stage2_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["seasonal_period"] == 4
    assert summary["n_obs"] == 130
    assert summary["scope"]["models_fitted"] is False
    assert summary["scope"]["arma_orders_searched"] is False
    assert summary["scope"]["forecasts_or_evaluation_performed"] is False
    assert summary["scope"]["cny_predictive_value_tested"] is False
    assert summary["scope"]["pmi_predictive_value_tested"] is False


def test_script_outputs_are_deterministic_and_canonical_input_is_unchanged():
    source = eda_stationarity.DATA_PATH
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="sta4003_stage2_") as temporary:
        output_root = Path(temporary)
        first = eda_stationarity.run_analysis(output_root=output_root)
        first_hashes = {
            str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((output_root / "outputs").rglob("*"))
            if path.is_file()
        }
        second = eda_stationarity.run_analysis(output_root=output_root)
        second_hashes = {
            str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((output_root / "outputs").rglob("*"))
            if path.is_file()
        }
    assert first["generated_artifacts"] == second["generated_artifacts"]
    assert first_hashes == second_hashes
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
