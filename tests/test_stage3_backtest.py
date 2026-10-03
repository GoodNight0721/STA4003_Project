"""Stage 3 evaluation-window, benchmark, leakage, and reproducibility tests."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from scripts.modeling import backtest_benchmarks


@pytest.fixture(scope="module")
def canonical():
    return backtest_benchmarks.load_canonical()


@pytest.fixture(scope="module")
def forecasts(canonical):
    return backtest_benchmarks.build_rolling_forecasts(canonical)


def _record(forecasts, model, horizon, target):
    rows = forecasts.loc[
        (forecasts["model"] == model)
        & (forecasts["horizon"] == horizon)
        & (forecasts["target"] == str(target))
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_target_window_is_exact_contiguous_and_duplicate_free(forecasts):
    expected = pd.period_range("2005Q1", "2026Q2", freq="Q-DEC")
    assert len(expected) == 86
    for horizon in backtest_benchmarks.HORIZONS:
        for model in backtest_benchmarks.BENCHMARKS:
            subset = forecasts.loc[
                (forecasts["model"] == model) & (forecasts["horizon"] == horizon)
            ]
            targets = pd.PeriodIndex(subset["target"], freq="Q-DEC")
            assert targets.equals(expected)
            assert targets.is_unique
            assert np.all(np.diff(targets.asi8) == 1)
            origins = pd.PeriodIndex(subset["origin"], freq="Q-DEC")
            assert (targets - horizon).equals(origins)
            assert len(subset) == 86


def test_expanding_training_window_ends_at_each_origin(forecasts):
    for row in forecasts.itertuples(index=False):
        origin = pd.Period(row.origin, freq="Q-DEC")
        assert row.training_start == "1994Q1"
        assert row.training_end == str(origin)
        assert origin == pd.Period(row.target, freq="Q-DEC") - row.horizon
        expected_n = origin.ordinal - backtest_benchmarks.SAMPLE_START.ordinal + 1
        assert row.n_train == expected_n


def test_historical_mean_uses_only_the_origin_training_values(canonical, forecasts):
    target = pd.Period("2019Q4", freq="Q-DEC")
    row = _record(forecasts, "Historical Mean", 2, target)
    origin = pd.Period(row.origin, freq="Q-DEC")
    expected = canonical.loc[:origin, "retail_bn"].mean()
    assert row.forecast == pytest.approx(expected, abs=1e-10)


def test_naive_forecast_is_the_observed_origin_value(canonical, forecasts):
    for horizon in backtest_benchmarks.HORIZONS:
        target = pd.Period("2019Q4", freq="Q-DEC")
        row = _record(forecasts, "Naive", horizon, target)
        origin = pd.Period(row.origin, freq="Q-DEC")
        assert row.forecast == pytest.approx(canonical.loc[origin, "retail_bn"])


def test_seasonal_naive_uses_target_four_quarters_earlier(canonical, forecasts):
    for horizon in backtest_benchmarks.HORIZONS:
        target = pd.Period("2019Q4", freq="Q-DEC")
        row = _record(forecasts, "Seasonal Naive", horizon, target)
        source = target - backtest_benchmarks.SEASONAL_PERIOD
        origin = pd.Period(row.origin, freq="Q-DEC")
        assert source <= origin
        assert row.forecast == pytest.approx(canonical.loc[source, "retail_bn"])


def test_error_columns_follow_actual_minus_forecast(forecasts):
    assert np.allclose(forecasts["error"], forecasts["actual"] - forecasts["forecast"])
    assert np.allclose(forecasts["absolute_error"], np.abs(forecasts["error"]))
    assert np.allclose(forecasts["squared_error"], forecasts["error"] ** 2)


def test_mase_scale_is_seasonal_and_origin_specific(canonical, forecasts):
    target = pd.Period("2019Q4", freq="Q-DEC")
    for horizon in backtest_benchmarks.HORIZONS:
        row = _record(forecasts, "Naive", horizon, target)
        origin = pd.Period(row.origin, freq="Q-DEC")
        training = canonical.loc[:origin, "retail_bn"].to_numpy(dtype=float)
        expected_scale = np.abs(training[4:] - training[:-4]).mean()
        assert row.mase_scale == pytest.approx(expected_scale, abs=1e-10)
        assert row.scaled_absolute_error == pytest.approx(row.absolute_error / expected_scale)


def test_q1_metrics_are_a_predefined_subset_and_counts_are_exact(forecasts):
    metrics = backtest_benchmarks.compute_metrics(forecasts)
    assert set(metrics["scope"]) == {"all", "q1"}
    assert len(metrics) == 12
    assert (metrics.loc[metrics["scope"] == "all", "n_forecasts"] == 86).all()
    assert (metrics.loc[metrics["scope"] == "q1", "n_forecasts"] == 22).all()
    for model in backtest_benchmarks.BENCHMARKS:
        for horizon in backtest_benchmarks.HORIZONS:
            q1 = forecasts.loc[
                (forecasts["model"] == model)
                & (forecasts["horizon"] == horizon)
                & (pd.PeriodIndex(forecasts["target"], freq="Q-DEC").quarter == 1)
            ]
            row = metrics.loc[
                (metrics["model"] == model)
                & (metrics["horizon"] == horizon)
                & (metrics["scope"] == "q1")
            ].iloc[0]
            assert row.MAE == pytest.approx(q1["absolute_error"].mean())
            assert row.RMSE == pytest.approx(np.sqrt(q1["squared_error"].mean()))
            assert row.MASE == pytest.approx(q1["scaled_absolute_error"].mean())


def test_all_benchmarks_share_exact_targets_for_each_horizon(forecasts):
    for horizon in backtest_benchmarks.HORIZONS:
        target_sets = [
            tuple(
                forecasts.loc[
                    (forecasts["model"] == model) & (forecasts["horizon"] == horizon), "target"
                ]
            )
            for model in backtest_benchmarks.BENCHMARKS
        ]
        assert target_sets[0] == target_sets[1] == target_sets[2]


def test_target_and_future_changes_do_not_change_an_earlier_origin_forecast(canonical, forecasts):
    target = pd.Period("2019Q4", freq="Q-DEC")
    origin = target - 2
    post_origin = origin + 1
    future = pd.Period("2020Q1", freq="Q-DEC")
    changed = canonical.copy()
    changed.loc[target, "retail_bn"] *= 1.7
    changed.loc[post_origin, "retail_bn"] *= 1.9
    changed.loc[future, "retail_bn"] *= 2.2
    changed_forecasts = backtest_benchmarks.build_rolling_forecasts(changed)

    for model in backtest_benchmarks.BENCHMARKS:
        original = _record(forecasts, model, 2, target)
        mutated = _record(changed_forecasts, model, 2, target)
        assert mutated.forecast == pytest.approx(original.forecast, abs=1e-10)
        assert mutated.mase_scale == pytest.approx(original.mase_scale, abs=1e-10)
    assert _record(changed_forecasts, "Naive", 2, target).actual != _record(
        forecasts, "Naive", 2, target
    ).actual


def test_benchmark_outputs_are_deterministic(tmp_path):
    first_summary = backtest_benchmarks.run_backtest(output_root=tmp_path)
    artifact_paths = [
        tmp_path / relative_path for relative_path in first_summary["generated_artifacts"]
    ]
    first_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact_paths}

    second_summary = backtest_benchmarks.run_backtest(output_root=tmp_path)
    second_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact_paths}

    assert first_summary == second_summary
    assert first_hashes == second_hashes
    forecast_table = pd.read_csv(tmp_path / "outputs" / "forecasts" / "benchmark_rolling_forecasts.csv")
    metric_table = pd.read_csv(tmp_path / "outputs" / "tables" / "benchmark_metrics.csv")
    assert forecast_table.columns.tolist() == backtest_benchmarks.FORECAST_COLUMNS
    assert len(forecast_table) == 3 * 2 * 86
    assert metric_table.columns.tolist() == backtest_benchmarks.METRIC_COLUMNS
    assert len(metric_table) == 3 * 2 * 2
    assert second_summary["target_count_per_model_horizon"] == {"1": 86, "2": 86}
    assert second_summary["primary_horizon"] == 2
    assert second_summary["secondary_horizon"] == 1
    assert second_summary["primary_scope"] == "all"
    assert "Pre-specified" in second_summary["reporting_scopes"]["q1"]
    assert second_summary["scope"]["sarima_or_other_fitted_model"] is False
    assert second_summary["scope"]["model_order_search"] is False
    assert second_summary["scope"]["cny_predictive_test"] is False
    assert second_summary["scope"]["pmi_predictive_test"] is False


def test_canonical_dataset_is_unchanged_by_backtest(canonical):
    del canonical
    source = backtest_benchmarks.DATA_PATH
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    backtest_benchmarks.build_rolling_forecasts(backtest_benchmarks.load_canonical())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
