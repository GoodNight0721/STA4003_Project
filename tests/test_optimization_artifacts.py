"""Independently reconcile saved research tables with their source forecasts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/optimization/round1"


def source_forecasts():
    frames = [pd.read_csv(ROOT / "outputs/forecasts" / name) for name in (
        "sarima_rolling_forecasts.csv", "sarima_cny_rolling_forecasts.csv", "ets_rolling_forecasts.csv"
    )]
    frames.append(pd.read_csv(OUTPUT / "window_forecasts.csv"))
    keys = ["origin", "target", "horizon"]
    joined = frames[0].merge(frames[2], on=keys, suffixes=("", "_ets"), validate="one_to_one")
    combination = joined[keys + ["actual", "mase_scale"]].copy()
    combination["model"] = "SARIMA+ETS-50:50"
    combination["forecast"] = (joined.forecast + joined.forecast_ets) / 2
    frames.append(combination)
    return pd.concat(frames, ignore_index=True, sort=False)


def test_saved_point_scores_reconcile_with_source_predictions():
    source = source_forecasts()
    for row in pd.read_csv(OUTPUT / "point_metrics.csv").itertuples():
        subset = source.loc[(source.model == row.model) & (source.horizon == row.horizon)]
        if row.scope == "Q1":
            subset = subset.loc[pd.PeriodIndex(subset.target, freq="Q-DEC").quarter == 1]
        column = "forecast_level_median" if row.point_convention == "median" else "forecast"
        errors = subset.actual.to_numpy() - subset[column].to_numpy()
        assert row.n == len(errors)
        assert row.MAE == pytest.approx(np.mean(np.abs(errors)), rel=1e-11, abs=1e-8)
        assert row.RMSE == pytest.approx(np.sqrt(np.mean(errors ** 2)), rel=1e-11, abs=1e-8)
        assert row.MASE == pytest.approx(np.mean(np.abs(errors) / subset.mase_scale), rel=1e-11, abs=1e-10)


def test_window_rows_reconcile_frozen_orders_scales_and_training_boundaries():
    windows = pd.read_csv(OUTPUT / "window_forecasts.csv")
    archived = pd.read_csv(ROOT / "outputs/forecasts/sarima_rolling_forecasts.csv")
    canonical = pd.read_csv(ROOT / "data/processed/analysis_quarterly.csv")
    quarters = pd.PeriodIndex(canonical.quarter, freq="Q-DEC")
    keys = ["origin", "target", "horizon"]
    for model, group in windows.groupby("model"):
        paired = group.merge(archived, on=keys, suffixes=("", "_archive"), validate="one_to_one")
        assert len(paired) == len(archived)
        assert not group.duplicated(keys).any()
        width = int(model.removeprefix("SARIMA-W"))
        for row in paired.itertuples():
            training = quarters[quarters <= pd.Period(row.origin, freq="Q-DEC")][-width:]
            assert row.training_start == str(training[0])
            assert row.training_end == row.origin
            assert row.n_train == len(training)
            for name in ("p", "d", "q", "P", "D", "Q"):
                assert getattr(row, name) == getattr(row, name + "_archive")
            assert row.mase_scale == pytest.approx(row.mase_scale_archive, rel=1e-12)


def test_saved_calibrators_use_only_realized_errors_and_exact_last_40_quantile():
    source = source_forecasts()
    records = pd.read_csv(OUTPUT / "interval_records.csv")
    calibrated = records.loc[(records.method == "calibrated") & ~records.fallback]
    assert not calibrated.empty
    for row in calibrated.itertuples():
        eligible = source.loc[(source.model == row.model) & (source.horizon == row.horizon)]
        targets = pd.PeriodIndex(eligible.target, freq="Q-DEC")
        eligible = eligible.loc[targets <= pd.Period(row.origin, freq="Q-DEC")].sort_values("target").tail(40)
        scores = np.abs(np.log(eligible.actual) - eligible.forecast_log_mean) / np.sqrt(eligible.forecast_log_variance)
        expected = np.sort(scores)[int(np.ceil((len(scores) - 1) * row.coverage))]
        assert row.n_calibration == len(scores)
        assert 20 <= row.n_calibration <= 40
        assert row.calibration_last_target == eligible.target.iloc[-1]
        assert row.critical_value == pytest.approx(expected, rel=1e-8, abs=1e-8)


def test_saved_interval_scores_reconcile_width_coverage_and_tail_penalties():
    records = pd.read_csv(OUTPUT / "interval_records.csv")
    width = records.upper - records.lower
    outside = np.maximum(records.lower - records.actual, 0) + np.maximum(records.actual - records.upper, 0)
    expected = width + 2 / (1 - records.coverage) * outside
    np.testing.assert_allclose(records.width, width, rtol=1e-10, atol=1e-7)
    np.testing.assert_allclose(records.winkler_score, expected, rtol=1e-10, atol=1e-7)
    assert records.covered.equals((records.actual >= records.lower) & (records.actual <= records.upper))
