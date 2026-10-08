"""Hand-checked research-scoring, pairing, calibration and calendar tests."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.optimization import analysis


def forecasts(values=(12.0, 16.0), model="SARIMA", horizon=1):
    targets = pd.period_range("2005Q1", periods=len(values), freq="Q-DEC")
    return pd.DataFrame({
        "model": model, "horizon": horizon,
        "origin": [str(t - horizon) for t in targets],
        "target": targets.astype(str), "actual": 10.0,
        "forecast": values, "mase_scale": [2.0] * len(values),
    })


# Pairing and bootstrap tests.


def test_paired_signed_metrics_use_extended_minus_reference():
    reference = forecasts()
    extended = forecasts((11.0, 13.0), "extension").iloc[::-1]
    paired = analysis.paired_losses(reference, extended)
    assert paired["target"].tolist() == ["2005Q1", "2005Q2"]
    assert paired["absolute_error_delta"].tolist() == [-1.0, -3.0]
    assert paired["squared_error_delta"].tolist() == [-3.0, -27.0]
    assert paired["scaled_absolute_error_delta"].tolist() == [-0.5, -1.5]
    result = analysis.bootstrap_deltas(paired, 1, 100, 7).set_index("metric")
    assert result.loc["MAE", "delta"] == -2.0
    assert result.loc["RMSE", "delta"] == pytest.approx(np.sqrt(5) - np.sqrt(20))
    assert result.loc["MASE", "delta"] == -1.0


@pytest.mark.parametrize("change", ["target", "duplicate", "actual", "scale", "origin"])
def test_pairing_rejects_incomplete_or_inconsistent_keys(change):
    reference = forecasts()
    extended = forecasts((11.0, 13.0), "extension")
    if change == "target":
        extended = extended.iloc[:1]
    elif change == "duplicate":
        extended = pd.concat([extended, extended.iloc[:1]], ignore_index=True)
    elif change == "actual":
        extended.loc[0, "actual"] += 1
    elif change == "scale":
        extended.loc[0, "mase_scale"] += 1
    else:
        extended.loc[0, "origin"] = "2004Q3"
    with pytest.raises(ValueError):
        analysis.paired_losses(reference, extended)


def test_point_scoring_rejects_duplicate_keys_after_horizon_normalization():
    frame = forecasts((12.0,))
    duplicate = frame.copy()
    duplicate["horizon"] = "1"
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.point_metrics(pd.concat([frame, duplicate], ignore_index=True))


def test_identical_forecasts_have_exact_zero_bootstrap_and_repeatable_output():
    frame = forecasts(tuple(range(11, 31)))
    paired = analysis.paired_losses(frame, frame.assign(model="identical"))
    result = analysis.bootstrap_deltas(paired, 4, 5000, 20261008)
    assert (result[["delta", "lower_95", "upper_95"]] == 0).all().all()
    pd.testing.assert_frame_equal(result, analysis.bootstrap_deltas(paired, 4, 5000, 20261008))


def test_bootstrap_resamples_chronological_circular_blocks():
    reference = forecasts((11.0, 12.0, 13.0, 14.0))
    extended = forecasts((10.0,) * 4, "extension")
    paired = analysis.paired_losses(reference, extended)
    # Every circular block of all four observations has the same mean loss.
    result = analysis.bootstrap_deltas(paired.iloc[::-1], 4, 50, 3).set_index("metric")
    assert result.loc["MAE", "lower_95"] == -2.5
    assert result.loc["MAE", "upper_95"] == -2.5
    assert result.loc["RMSE", "lower_95"] == pytest.approx(-np.sqrt(7.5))


# Distributional and calibration tests.


def log_forecasts(n=2, model="SARIMA", horizon=1):
    frame = forecasts((float(np.exp(0.02)),) * n, model, horizon)
    frame["forecast_log_mean"] = 0.0
    frame["forecast_log_variance"] = 0.04
    frame["forecast_level_median"] = 1.0
    frame["actual"] = 1.0
    return frame


def test_parametric_quantiles_exclude_mean_adjustment_and_winkler_penalizes_misses():
    frame = log_forecasts(1)
    frame["actual"] = 2.0
    records = analysis.interval_records(frame).set_index("coverage")
    row = records.loc[0.8]
    assert row["lower"] == pytest.approx(0.773901779660)
    assert row["upper"] == pytest.approx(1.292153637945)
    assert not row["covered"]
    assert row["winkler_score"] == pytest.approx(
        1.292153637945 - 0.773901779660 + 10 * (2 - 1.292153637945)
    )
    assert records.loc[0.95, "lower"] == pytest.approx(0.675708981137)
    assert records.loc[0.95, "upper"] == pytest.approx(1.47992705131)


def test_winkler_inside_interval_is_width():
    row = analysis.interval_records(log_forecasts(1)).iloc[0]
    assert row["covered"]
    assert row["winkler_score"] == pytest.approx(row["upper"] - row["lower"])


def calibration_forecasts(n=52):
    frame = log_forecasts(n, horizon=2)
    frame["forecast_log_variance"] = 0.01
    frame["actual"] = np.exp(np.arange(1, n + 1) * 0.1)
    return frame


def test_calibration_uses_observed_targets_at_origin_and_higher_quantile():
    frame = calibration_forecasts()
    records = analysis.interval_records(frame, calibrate=True)
    early = records.loc[(records["target"] == frame.loc[20, "target"]) & (records["coverage"] == 0.8)].iloc[0]
    assert early["n_calibration"] == 19
    assert early["fallback"]
    first = records.loc[(records["target"] == frame.loc[21, "target"]) & (records["coverage"] == 0.8)].iloc[0]
    assert first["n_calibration"] == 20
    assert not first["fallback"]
    assert first["calibration_last_target"] == frame.loc[19, "target"]
    assert first["critical_value"] == pytest.approx(17.0)
    last = records.loc[(records["target"] == frame.loc[51, "target"]) & (records["coverage"] == 0.8)].iloc[0]
    assert last["n_calibration"] == 40
    assert last["critical_value"] == pytest.approx(43.0)


def test_calibration_future_changes_do_not_change_endpoints_and_models_are_separate():
    frame = calibration_forecasts(25)
    other = frame.assign(model="other", actual=1.0)
    original = analysis.interval_records(pd.concat([frame, other]), calibrate=True)
    changed = pd.concat([frame, other], ignore_index=True)
    changed.loc[(changed["model"] == "SARIMA") & (changed.index >= 20), "actual"] *= 100
    mutated = analysis.interval_records(changed, calibrate=True)
    keys = (original["model"] == "SARIMA") & (original["target"] == frame.loc[21, "target"])
    pd.testing.assert_frame_equal(original.loc[keys, ["lower", "upper", "n_calibration"]], mutated.loc[keys, ["lower", "upper", "n_calibration"]])
    other_row = original.loc[(original["model"] == "other") & (original["target"] == frame.loc[21, "target"])].iloc[0]
    assert other_row["critical_value"] == 0.0


def test_calibration_zero_variance_pool_explicitly_falls_back():
    frame = calibration_forecasts(25)
    frame.loc[0, "forecast_log_variance"] = 0.0
    record = analysis.interval_records(frame, calibrate=True)
    row = record.loc[record["target"] == frame.loc[21, "target"]].iloc[0]
    assert row["fallback"]
    assert row["fallback_reason"] == "invalid_calibration_errors"


@pytest.mark.parametrize("column,value", [("actual", 0), ("actual", -1), ("forecast_log_mean", np.nan), ("forecast_log_variance", -0.1)])
def test_interval_records_reject_invalid_actual_or_log_moments(column, value):
    frame = log_forecasts(1)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        analysis.interval_records(frame)


# Point combination, calendar and integration tests.


def test_point_scoring_separates_median_and_preserves_origin_mase_scale():
    frame = forecasts()
    frame["forecast_level_median"] = [11.0, 13.0]
    frame["mase_scale"] = [1.0, 3.0]
    result = analysis.point_metrics(frame)
    mean = result.loc[(result["scope"] == "all") & (result["point_convention"] == "mean")].iloc[0]
    median = result.loc[(result["scope"] == "all") & (result["point_convention"] == "median")].iloc[0]
    assert mean["MASE"] == 2.0
    assert mean["RMSE"] == pytest.approx(np.sqrt(20))
    assert median["MAE"] == 2.0
    assert median["MASE"] == 1.0


def test_combination_is_exact_half_weight_and_does_not_infer_distribution():
    combined = analysis.combine_forecasts(forecasts(), forecasts((8.0, 20.0), "ETS").iloc[::-1])
    assert combined["forecast"].tolist() == [10.0, 18.0]
    assert combined["mase_scale"].tolist() == [2.0, 2.0]
    assert combined["error"].tolist() == [0.0, -8.0]
    assert "forecast_level_median" not in combined
    assert "forecast_log_variance" not in combined
    assert set(analysis.point_metrics(combined)["point_convention"]) == {"mean"}


def test_calendar_windows_are_inclusive_and_partition_previous_q4_and_q1():
    calendar = pd.DataFrame({"year": [2020, 2021], "cny_date": ["2020-01-25", "2021-02-12"]})
    allocations, variation = analysis.calendar_diagnostic(calendar)
    short = allocations.loc[allocations["window"] == "[-14,+7]"]
    assert short["total_days"].tolist() == [22, 22]
    assert short["q1_days"].tolist() == [22, 22]
    assert short["previous_q4_days"].tolist() == [0, 0]
    long = allocations.loc[(allocations["year"] == 2020) & (allocations["window"] == "[-30,+7]")].iloc[0]
    assert long["total_days"] == 38
    assert long["previous_q4_days"] == 6
    assert long["q1_days"] == 32
    assert (allocations["q1_days"] + allocations["previous_q4_days"] + allocations["other_days"]).equals(allocations["total_days"])
    assert variation.loc[variation["window"] == "[-14,+7]", "q1_constant_regressor_risk"].item()


def test_run_analysis_temp_outputs_match_archived_point_tables(tmp_path):
    summary = analysis.run_analysis(tmp_path)
    assert summary["evidence_status"] == "exploratory"
    assert not summary["window_completeness"]["SARIMA-W40"]["complete"]
    json.dumps(summary, allow_nan=False)
    metrics = pd.read_csv(tmp_path / "point_metrics.csv")
    for filename in ("sarima_metrics.csv", "cny_metrics.csv"):
        archived = pd.read_csv(analysis.PROJECT_ROOT / "outputs" / "tables" / filename)
        for record in archived.itertuples(index=False):
            row = metrics.loc[(metrics["model"] == record.model) & (metrics["horizon"] == record.horizon) & (metrics["scope"].str.lower() == record.scope) & (metrics["point_convention"] == "mean")].iloc[0]
            np.testing.assert_allclose(row[["MAE", "RMSE", "MASE"]].to_numpy(dtype=float), [record.MAE, record.RMSE, record.MASE], rtol=1e-12, atol=1e-8)
    interval = pd.read_csv(tmp_path / "interval_metrics.csv")
    assert {"all", "calibrated_only"} <= set(interval["subset"])
    assert (tmp_path / "calendar_allocations.csv").is_file()
    assert (tmp_path / "paired_bootstrap.csv").is_file()
