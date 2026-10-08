"""Fixed-window fitting, leakage and checkpoint contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.backtest_sarima import SarimaCandidate
from scripts.optimization import window_backtest as windows


@pytest.fixture
def small_study(monkeypatch):
    index = pd.period_range("2000Q1", periods=32, freq="Q-DEC")
    rng = np.random.default_rng(23)
    logs = 5 + np.cumsum(rng.normal(0.01, 0.025, len(index)))
    frame = pd.DataFrame({"log_retail": logs, "retail_bn": np.exp(logs)}, index=index)
    orders = {index[25]: SarimaCandidate(0, 1, 0, 0, 0, 0),
              index[26]: SarimaCandidate(0, 1, 1, 0, 0, 0)}
    rows = []
    for origin in orders:
        for horizon in (1, 2):
            rows.append({"model": "SARIMA", "origin": str(origin), "horizon": horizon,
                         "target": str(origin + horizon),
                         "actual": float(frame.loc[origin + horizon, "retail_bn"]),
                         "mase_scale": 7.25 + horizon})
    reference = pd.DataFrame(rows)
    monkeypatch.setattr(windows, "load_inputs", lambda: (frame, orders, reference))
    return frame, orders, reference


def test_window_slice_is_inclusive_and_uses_only_last_available_quarters():
    frame = pd.DataFrame({"log_retail": range(12)},
                         index=pd.period_range("2000Q1", periods=12, freq="Q-DEC"))
    actual = windows.window_training(frame, pd.Period("2001Q2", freq="Q-DEC"), 4)
    assert actual.index.astype(str).tolist() == ["2000Q3", "2000Q4", "2001Q1", "2001Q2"]
    assert actual.log_retail.tolist() == [2, 3, 4, 5]
    assert len(windows.window_training(frame, frame.index[2], 40)) == 3
    with pytest.raises(ValueError):
        windows.window_training(frame, frame.index[-1] + 1, 4)
    with pytest.raises(ValueError):
        windows.window_training(frame, frame.index[-1], 0)


def test_real_fit_future_changes_leave_training_and_forecast_unchanged(small_study):
    frame, orders, _ = small_study
    origin = frame.index[25]
    changed = frame.copy()
    changed.loc[changed.index > origin, "log_retail"] += 8
    changed.loc[changed.index > origin, "retail_bn"] *= 100
    first = windows.fit_window(windows.window_training(frame, origin, 20), orders[origin])
    second = windows.fit_window(windows.window_training(changed, origin, 20), orders[origin])
    assert first.converged and second.converged
    left, right = first.result.get_forecast(steps=2), second.result.get_forecast(steps=2)
    np.testing.assert_array_equal(left.predicted_mean, right.predicted_mean)
    np.testing.assert_array_equal(left.var_pred_mean, right.var_pred_mean)
    assert np.isfinite(left.var_pred_mean).all() and (left.var_pred_mean >= 0).all()


def test_run_uses_one_real_fit_per_origin_window_and_archived_scales(tmp_path, small_study, monkeypatch):
    _, orders, reference = small_study
    real_fit = windows.fit_window
    calls = []
    def counted(training, candidate):
        calls.append((str(training.index[-1]), candidate))
        return real_fit(training, candidate)
    monkeypatch.setattr(windows, "fit_window", counted)
    summary = windows.run_windows(tmp_path)
    assert summary["complete"] and summary["successful_fits"] == 4
    assert len(calls) == 4
    forecasts = pd.read_csv(tmp_path / "window_forecasts.csv")
    assert len(forecasts) == 8
    for row in forecasts.itertuples(index=False):
        order = orders[pd.Period(row.origin, freq="Q-DEC")]
        assert (row.p, row.d, row.q, row.P, row.D, row.Q) == (*order.order, *order.seasonal_order[:3])
        original = reference.loc[(reference.origin == row.origin) & (reference.horizon == row.horizon)].iloc[0]
        assert row.mase_scale == original.mase_scale
        assert row.forecast == pytest.approx(np.exp(row.forecast_log_mean + row.forecast_log_variance / 2))
        assert row.scaled_absolute_error == pytest.approx(abs(row.actual - row.forecast) / original.mase_scale)
    assert set(forecasts.model) == {"SARIMA-W40", "SARIMA-W60"}


def test_completed_checkpoints_reuse_and_invalidate_exact_training_order_and_scale(tmp_path, small_study, monkeypatch):
    frame, orders, reference = small_study
    windows.run_windows(tmp_path)
    before = (tmp_path / "window_forecasts.csv").read_bytes()
    real_fit = windows.fit_window
    calls = []
    def counted(training, candidate):
        calls.append(str(training.index[-1]))
        return real_fit(training, candidate)
    monkeypatch.setattr(windows, "fit_window", counted)
    windows.run_windows(tmp_path)
    assert not calls
    assert (tmp_path / "window_forecasts.csv").read_bytes() == before
    frame.loc[frame.index[25], "log_retail"] = np.nextafter(frame.loc[frame.index[25], "log_retail"], np.inf)
    windows.run_windows(tmp_path)
    assert len(calls) == 4
    calls.clear()
    origin = next(iter(orders))
    orders[origin] = SarimaCandidate(0, 1, 0, 0, 1, 0)
    windows.run_windows(tmp_path)
    assert calls == [str(origin), str(origin)]
    calls.clear()
    reference.loc[reference.origin == str(origin), "mase_scale"] += 1
    windows.run_windows(tmp_path)
    assert calls == [str(origin), str(origin)]


def test_failed_checkpoint_is_explicit_and_only_retries_on_request(tmp_path, small_study, monkeypatch):
    origin = str(next(iter(small_study[1])))
    real_fit = windows.fit_window
    def failure(training, candidate):
        if str(training.index[-1]) == origin:
            return windows.WindowFit(None, [{"attempt": 1, "converged": False}], "test numerical failure")
        return real_fit(training, candidate)
    monkeypatch.setattr(windows, "fit_window", failure)
    first = windows.run_windows(tmp_path)
    assert not first["complete"] and first["failed_fits"] == 2
    assert len(pd.read_csv(tmp_path / "window_forecasts.csv")) == 4
    fits = pd.read_csv(tmp_path / "window_fits.csv")
    assert fits.loc[fits.origin == origin, "status"].eq("failed").all()
    monkeypatch.setattr(windows, "fit_window", lambda *args: pytest.fail("checkpoint silently refitted"))
    assert windows.run_windows(tmp_path)["failed_fits"] == 2
    monkeypatch.setattr(windows, "fit_window", real_fit)
    assert windows.run_windows(tmp_path, retry_failed=True)["complete"]


def test_invalid_prediction_moments_are_failed_and_persisted(tmp_path, small_study, monkeypatch):
    class InvalidResult:
        def get_forecast(self, steps):
            return SimpleNamespace(predicted_mean=np.full(steps, 5.0), var_pred_mean=np.full(steps, -0.1))
    monkeypatch.setattr(windows, "fit_window", lambda *args: windows.WindowFit(
        InvalidResult(), [{"attempt": 1, "converged": True}], None))
    result = windows.run_windows(tmp_path)
    assert not result["complete"] and result["failed_fits"] == 4
    assert pd.read_csv(tmp_path / "window_forecasts.csv").empty
    for path in (tmp_path / "checkpoints").glob("*.json"):
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["status"] == "failed" and saved["attempts"]
        assert "variance" in saved["failure_reason"].lower()


def test_optimizer_retries_terminal_then_deterministic_without_changing_estimator(monkeypatch, small_study):
    frame, orders, _ = small_study
    candidate = next(iter(orders.values()))
    calls = []
    class Model:
        param_names = ["sigma2"]
        start_params = np.array([0.025])
        def __init__(self, training, **kwargs):
            assert kwargs == {"order": candidate.order, "seasonal_order": candidate.seasonal_order,
                              "trend": "n", "simple_differencing": False,
                              "enforce_stationarity": True, "enforce_invertibility": True,
                              "concentrate_scale": False}
        def fit(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(params=np.array([0.01]), llf=-1, aic=4,
                                   mle_retvals={"converged": len(calls) == 3, "iterations": 100})
    monkeypatch.setattr(windows, "SARIMAX", Model)
    result = windows.fit_window(frame, candidate)
    assert result.converged and len(result.attempts) == 3
    assert calls[0] == {"method": "lbfgs", "disp": False, "maxiter": 100}
    np.testing.assert_array_equal(calls[1]["start_params"], [0.01])
    np.testing.assert_array_equal(calls[2]["start_params"], [0.025])
    assert all(call["method"] == "lbfgs" and call["maxiter"] == 100 for call in calls)
