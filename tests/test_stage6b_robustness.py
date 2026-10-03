"""Fixed ETS specification, target-only shock scoring, and frozen-input checks."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from scripts.modeling import backtest_benchmarks as evaluation
from scripts.modeling import backtest_ets


@pytest.fixture(scope="module")
def canonical():
    return backtest_ets.load_analysis_data()


class FakeResult:
    def __init__(self, training):
        self.resid = np.linspace(-0.04, 0.04, len(training), dtype=float)
        self.mle_retvals = {"converged": True}
        self.sse = float(self.resid @ self.resid)
        self.llf = -12.5
        self.aic = 43.0
        self.last_value = float(training.iloc[-1])
        self.forecast_calls = 0

    def forecast(self, steps):
        self.forecast_calls += 1
        return pd.Series(self.last_value + 0.01 * np.arange(1, steps + 1, dtype=float))


def fake_fit(training):
    return backtest_ets.ETSFit(FakeResult(training), converged=True)


def build(canonical, fit_function=fake_fit):
    return backtest_ets.build_rolling_forecasts(canonical, fit_function=fit_function)


def _row(forecasts, horizon, target):
    return forecasts.loc[(forecasts["horizon"] == horizon) & (forecasts["target"] == str(target))].iloc[0]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_ets_spec_is_exact_and_fit_has_no_component_search(monkeypatch, canonical):
    calls = []

    class StubETSModel:
        def __init__(self, training, **kwargs):
            calls.append((training.copy(), kwargs))

        def fit(self, **kwargs):
            assert kwargs == {"disp": False}
            return FakeResult(calls[-1][0])

    monkeypatch.setattr(backtest_ets, "ETSModel", StubETSModel)
    training = canonical.loc[: pd.Period("2004Q3", freq="Q-DEC"), "log_retail"]
    fitted = backtest_ets.fit_fixed_ets(training)

    assert fitted.result is not None
    assert len(calls) == 1
    assert calls[0][1] == {
        "error": "add",
        "trend": "add",
        "damped_trend": True,
        "seasonal": "add",
        "seasonal_periods": 4,
    }
    assert backtest_ets.FIXED_ETS_SPEC == calls[0][1]


def test_rolling_targets_origins_and_expanding_training_match_stage3(canonical):
    forecasts, diagnostics = build(canonical)
    expected_targets = pd.period_range("2005Q1", "2026Q2", freq="Q-DEC")
    assert len(diagnostics) == 87
    assert len(backtest_ets.origin_horizons()) == 87
    for horizon in (1, 2):
        subset = forecasts.loc[forecasts["horizon"] == horizon].sort_values("target", kind="stable")
        targets = pd.PeriodIndex(subset["target"], freq="Q-DEC")
        origins = pd.PeriodIndex(subset["origin"], freq="Q-DEC")
        assert len(subset) == 86
        assert targets.equals(expected_targets)
        assert (targets - horizon).equals(origins)
        assert (subset["training_end"].to_numpy() == subset["origin"].to_numpy()).all()
        assert (subset["training_start"] == "1994Q1").all()
        assert (subset["n_train"].to_numpy() == origins.asi8 - pd.Period("1994Q1", freq="Q-DEC").ordinal + 1).all()
    assert diagnostics["fit_success"].all()
    assert diagnostics["converged"].all()


def test_future_target_perturbation_cannot_change_earlier_ets_forecast(canonical):
    target = pd.Period("2019Q4", freq="Q-DEC")
    origin = target - 2
    original, _ = build(canonical)
    changed = canonical.copy()
    changed.loc[target, "retail_bn"] *= 1.8
    changed.loc[target, "log_retail"] += 0.35
    after_origin = changed.index > origin
    changed.loc[after_origin, "retail_bn"] *= 1.25
    changed.loc[after_origin, "log_retail"] += 0.08
    perturbed, _ = build(changed)

    before_row = _row(original, 2, target)
    after_row = _row(perturbed, 2, target)
    assert after_row["forecast_log"] == pytest.approx(before_row["forecast_log"], abs=1e-12)
    assert after_row["smearing_factor"] == pytest.approx(before_row["smearing_factor"], abs=1e-12)
    assert after_row["mase_scale"] == pytest.approx(before_row["mase_scale"], abs=1e-12)
    assert after_row["actual"] != before_row["actual"]


def test_smearing_factor_uses_only_training_residuals(canonical):
    forecasts, _ = build(canonical)
    row = _row(forecasts, 2, "2019Q4")
    training_n = int(row.n_train)
    training_residuals = np.linspace(-0.04, 0.04, training_n, dtype=float)
    expected = float(np.exp(training_residuals).mean())
    assert row.smearing_factor == pytest.approx(expected, abs=1e-14)
    assert row.forecast == pytest.approx(np.exp(row.forecast_log) * row.smearing_factor)


def test_mase_scales_and_aggregate_formula_match_stage3_exactly(canonical):
    forecasts, _ = build(canonical)
    stage3 = evaluation.build_rolling_forecasts(canonical)
    scales = stage3.loc[stage3["model"] == "Naive", ["horizon", "target", "mase_scale"]]
    for horizon in evaluation.HORIZONS:
        ets = forecasts.loc[forecasts["horizon"] == horizon].sort_values("target", kind="stable")
        reference = scales.loc[scales["horizon"] == horizon].sort_values("target", kind="stable")
        assert np.array_equal(ets["target"].to_numpy(), reference["target"].to_numpy())
        assert np.array_equal(ets["mase_scale"].to_numpy(), reference["mase_scale"].to_numpy())
        expected_scaled = ets["absolute_error"].to_numpy() / reference["mase_scale"].to_numpy()
        assert np.allclose(ets["scaled_absolute_error"].to_numpy(), expected_scaled, rtol=0, atol=1e-14)
    expected_metrics = evaluation.compute_metrics(forecasts, models=(backtest_ets.MODEL_NAME,))
    observed_metrics = backtest_ets.compute_ets_metrics(forecasts)
    pd.testing.assert_frame_equal(observed_metrics, expected_metrics, check_exact=True)


def test_shock_sensitivity_only_masks_scores_and_retains_raw_shock_and_later_rows(canonical, monkeypatch):
    ets_forecasts, _ = build(canonical)
    sarima = pd.read_csv(backtest_ets.SARIMA_FORECAST_PATH, dtype={"origin": str, "target": str})
    cny = pd.read_csv(backtest_ets.CNY_FORECAST_PATH, dtype={"origin": str, "target": str})
    frames = {"SARIMA": sarima, "SARIMA+CNY": cny, "ETS": ets_forecasts}
    copies = {name: frame.copy(deep=True) for name, frame in frames.items()}
    monkeypatch.setattr(backtest_ets, "fit_fixed_ets", lambda *_: pytest.fail("sensitivity must not refit"))
    sensitivity = backtest_ets.compute_shock_sensitivity(frames)
    cny_sensitivity = backtest_ets.build_cny_shock_sensitivity(sensitivity)

    for model, frame in frames.items():
        h2 = frame.loc[frame["horizon"] == 2]
        assert set(backtest_ets.SHOCK_TARGETS).issubset(set(h2["target"]))
        pd.testing.assert_frame_equal(frame, copies[model])
        post_shock = h2.loc[h2["target"] > "2022Q2"]
        copy_post = copies[model].loc[(copies[model]["horizon"] == 2) & (copies[model]["target"] > "2022Q2")]
        pd.testing.assert_frame_equal(post_shock.reset_index(drop=True), copy_post.reset_index(drop=True))
        all_rows = sensitivity.loc[(sensitivity["model"] == model) & (sensitivity["scope"] == "all")].set_index("sample")
        q1_rows = sensitivity.loc[(sensitivity["model"] == model) & (sensitivity["scope"] == "q1")].set_index("sample")
        assert all_rows.loc["full", "n_forecasts"] == 86
        assert all_rows.loc["exclude_shocks", "n_forecasts"] == 84
        assert q1_rows.loc["full", "n_forecasts"] == 22
        assert q1_rows.loc["exclude_shocks", "n_forecasts"] == 21
    assert len(cny_sensitivity) == 4
    assert cny_sensitivity["n_sarima"].eq(cny_sensitivity["n_cny"]).all()


def test_full_sample_diagnostic_fit_does_not_create_a_forecast(canonical):
    result = FakeResult(canonical["log_retail"])

    def fit_only(training):
        return backtest_ets.ETSFit(result=result, converged=True)

    diagnostics = backtest_ets.full_sample_diagnostics(canonical, fit_function=fit_only).iloc[0]
    assert diagnostics.n_obs == 130
    assert diagnostics.descriptive_only
    assert not diagnostics.used_for_rolling_forecasts
    assert not diagnostics.forecast_created
    assert result.forecast_calls == 0
    assert diagnostics.ljung_box_lag == 8


def test_production_artifacts_are_deterministic_and_frozen_inputs_unchanged(tmp_path, canonical):
    protected = [
        backtest_ets.DATA_PATH,
        backtest_ets.SARIMA_FORECAST_PATH,
        backtest_ets.CNY_FORECAST_PATH,
    ]
    hashes_before = {path: _sha256(path) for path in protected}
    first = backtest_ets.run_backtest(output_root=tmp_path, fit_function=fake_fit)
    output_hashes_first = {
        relative: _sha256(tmp_path / relative) for relative in first["generated_artifacts"]
    }
    second = backtest_ets.run_backtest(output_root=tmp_path, fit_function=fake_fit)
    output_hashes_second = {
        relative: _sha256(tmp_path / relative) for relative in second["generated_artifacts"]
    }

    assert output_hashes_first == output_hashes_second
    assert {path: _sha256(path) for path in protected} == hashes_before
    assert first["rolling_fit_diagnostics"]["fit_success_count"] == 87
    assert second["rolling_fit_diagnostics"]["fit_failure_count"] == 0
    assert first["generated_artifacts"] == second["generated_artifacts"]
