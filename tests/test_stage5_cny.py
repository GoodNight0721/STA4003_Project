"""Stage 5 fixed-order, training-centered Lunar New Year value checks."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.modeling import backtest_cny, backtest_sarima


@pytest.fixture(scope="module")
def canonical():
    return backtest_cny.load_canonical()


@pytest.fixture(scope="module")
def stage4_forecasts():
    return backtest_cny.load_stage4_forecasts()


@pytest.fixture(scope="module")
def locked_orders():
    return backtest_cny.load_locked_orders()[0]


class FakePrediction:
    def __init__(self, mean, variance):
        self.predicted_mean = mean
        self.var_pred_mean = variance


class FakeResult:
    def __init__(self, training, exog, beta):
        self.beta = beta
        self.params = pd.Series({backtest_cny.CNY_EXOG_NAME: beta, "sigma2": 0.04})
        self.bse = pd.Series({backtest_cny.CNY_EXOG_NAME: 0.25, "sigma2": 0.01})
        self.pvalues = pd.Series({backtest_cny.CNY_EXOG_NAME: 0.32, "sigma2": 0.01})
        self.mle_retvals = {"converged": True}
        self.llf = -10.0
        self.aic = 24.0
        self.resid = np.sin(np.arange(len(training), dtype=float) * 1.37)
        self.model = SimpleNamespace(loglikelihood_burn=1)
        self.last_training_value = float(training.iloc[-1])
        self.training_length = len(training)

    def get_forecast(self, steps, exog):
        assert len(exog) == steps
        x = exog[backtest_cny.CNY_EXOG_NAME].to_numpy(dtype=float)
        mean = self.last_training_value + 0.01 * np.arange(1, steps + 1) + self.beta * x
        variance = np.full(steps, 0.04)
        return FakePrediction(mean, variance)


def fake_fit(training, exog, candidate):
    assert training.index.equals(exog.index)
    values = exog[backtest_cny.CNY_EXOG_NAME].to_numpy(dtype=float)
    centered_response = training.to_numpy(dtype=float) - float(training.mean())
    denominator = float(values @ values)
    beta = float(values @ centered_response / denominator * 0.001) if denominator else 0.0
    return backtest_cny.CnyFit(
        candidate=candidate,
        result=FakeResult(training, exog, beta),
        beta=beta,
        beta_se=0.25,
        beta_pvalue=0.32,
        converged=True,
    )


def _build(canonical, stage4_forecasts, locked_orders, fit_function=fake_fit):
    return backtest_cny.build_rolling_forecasts(
        canonical, locked_orders, stage4_forecasts, fit_function=fit_function
    )


def test_locked_orders_exactly_match_stage4_origin_and_forecast_orders(locked_orders, stage4_forecasts):
    selection, _ = backtest_cny.load_locked_orders()
    assert len(locked_orders) == len(selection) == 87
    assert set(locked_orders) == set(backtest_sarima.origin_horizons())
    for row in stage4_forecasts.itertuples(index=False):
        order = locked_orders[pd.Period(row.origin, freq="Q-DEC")]
        assert (row.p, row.d, row.q) == order.order
        assert (row.P, row.D, row.Q, row.seasonal_period) == order.seasonal_order


def test_stage5_does_not_run_order_search(monkeypatch, canonical, stage4_forecasts, locked_orders):
    def forbidden(*args, **kwargs):
        raise AssertionError("Stage 5 must not search model orders")

    monkeypatch.setattr(backtest_sarima, "generate_candidate_grid", forbidden)
    forecasts, _ = _build(canonical, stage4_forecasts, locked_orders)
    assert len(forecasts) == 172


def test_non_q1_regressor_is_exactly_zero_and_q1_is_centered(canonical):
    center = 0.37
    exog = backtest_cny.centered_cny_exog(canonical[["is_q1", "cny_position_fraction"]], center)
    non_q1 = canonical["is_q1"].eq(0)
    q1 = canonical["is_q1"].eq(1)
    assert (exog.loc[non_q1, backtest_cny.CNY_EXOG_NAME] == 0.0).all()
    expected = canonical.loc[q1, "cny_position_fraction"] - center
    assert np.allclose(exog.loc[q1, backtest_cny.CNY_EXOG_NAME], expected)


def test_training_center_uses_only_q1_calendar_rows_through_origin(canonical):
    origin = pd.Period("2010Q4", freq="Q-DEC")
    center, n_q1 = backtest_cny.training_center(canonical.loc[:origin])
    expected = canonical.loc[:origin].loc[lambda x: x["is_q1"].eq(1), "cny_position_fraction"].mean()
    full = canonical.loc[canonical["is_q1"].eq(1), "cny_position_fraction"].mean()
    assert center == pytest.approx(expected)
    assert n_q1 == 17
    assert center != pytest.approx(full)


def test_future_calendar_exog_reuses_training_center(canonical):
    origin = pd.Period("2020Q2", freq="Q-DEC")
    center, _, training_exog, future_exog = backtest_cny.origin_exog(canonical, origin, 3)
    assert training_exog.index[-1] == origin
    q1_target = pd.Period("2021Q1", freq="Q-DEC")
    expected = canonical.loc[q1_target, "cny_position_fraction"] - center
    assert future_exog.loc[q1_target, backtest_cny.CNY_EXOG_NAME] == pytest.approx(expected)
    assert future_exog.loc[pd.Period("2020Q3", freq="Q-DEC"), backtest_cny.CNY_EXOG_NAME] == 0.0


def test_future_retail_perturbation_cannot_change_early_beta_order_or_forecast(
    canonical, stage4_forecasts, locked_orders
):
    target = pd.Period("2019Q4", freq="Q-DEC")
    origin = target - 2
    original = backtest_cny.fit_origin(canonical, origin, 2, locked_orders[origin], fake_fit)
    changed = canonical.copy()
    after_origin = changed.index > origin
    changed.loc[after_origin, "retail_bn"] *= 1.4
    changed.loc[after_origin, "log_retail"] += 0.2
    perturbed = backtest_cny.fit_origin(changed, origin, 2, locked_orders[origin], fake_fit)
    assert perturbed.fitted.beta == pytest.approx(original.fitted.beta)
    assert np.allclose(perturbed.log_means, original.log_means)
    assert perturbed.fitted.candidate == original.fitted.candidate == locked_orders[origin]


def test_rolling_center_does_not_use_full_sample_cny_mean(canonical, stage4_forecasts, locked_orders):
    forecasts, _ = _build(canonical, stage4_forecasts, locked_orders)
    early = forecasts.loc[(forecasts["horizon"] == 2) & (forecasts["target"] == "2005Q1")].iloc[0]
    full_mean = canonical.loc[canonical["is_q1"].eq(1), "cny_position_fraction"].mean()
    expected = canonical.loc[:pd.Period(early.origin, freq="Q-DEC")]
    expected = expected.loc[expected["is_q1"].eq(1), "cny_position_fraction"].mean()
    assert early.cny_training_center == pytest.approx(expected)
    assert early.cny_training_center != pytest.approx(full_mean)


def test_same_origin_horizons_share_one_cny_fit(canonical, stage4_forecasts, locked_orders):
    calls = []

    def counted_fit(training, exog, candidate):
        calls.append(training.index[-1])
        return fake_fit(training, exog, candidate)

    forecasts, stability = _build(canonical, stage4_forecasts, locked_orders, counted_fit)
    shared_origin = pd.Period("2019Q2", freq="Q-DEC")
    assert len(calls) == 87
    assert calls.count(shared_origin) == 1
    assert len(stability.loc[stability["origin"] == str(shared_origin)]) == 1
    rows = forecasts.loc[forecasts["origin"] == str(shared_origin)]
    assert set(rows["horizon"]) == {1, 2}
    assert rows["cny_beta"].nunique() == 1
    assert rows["cny_training_center"].nunique() == 1


def test_targets_counts_orders_and_mase_scales_match_stage4(canonical, stage4_forecasts, locked_orders):
    forecasts, _ = _build(canonical, stage4_forecasts, locked_orders)
    targets = [str(value) for value in pd.period_range("2005Q1", "2026Q2", freq="Q-DEC")]
    for horizon in (1, 2):
        cny = forecasts.loc[forecasts["horizon"] == horizon]
        base = stage4_forecasts.loc[stage4_forecasts["horizon"] == horizon].sort_values("target")
        assert cny["target"].tolist() == targets == base["target"].tolist()
        assert len(cny) == len(base) == 86
        assert np.allclose(cny["mase_scale"], base["mase_scale"], rtol=0, atol=1e-10)
        assert np.allclose(cny["scaled_absolute_error"], cny["absolute_error"] / base["mase_scale"])
        for row in cny.itertuples(index=False):
            order = locked_orders[pd.Period(row.origin, freq="Q-DEC")]
            assert (row.p, row.d, row.q) == order.order
            assert (row.P, row.D, row.Q) == order.seasonal_order[:3]


def test_lognormal_retransformation_uses_conditional_mean_for_scoring(canonical, stage4_forecasts, locked_orders):
    forecasts, _ = _build(canonical, stage4_forecasts, locked_orders)
    row = forecasts.iloc[0]
    assert row.forecast_level_median == pytest.approx(np.exp(row.forecast_log_mean))
    assert row.forecast == pytest.approx(np.exp(row.forecast_log_mean + 0.5 * row.forecast_log_variance))
    assert row.forecast > row.forecast_level_median


def test_paired_error_differences_and_win_counts_are_correct(canonical, stage4_forecasts, locked_orders):
    forecasts, _ = _build(canonical, stage4_forecasts, locked_orders)
    paired = backtest_cny.paired_errors(forecasts, stage4_forecasts)
    assert len(paired) == 172
    assert np.allclose(paired["absolute_loss_difference"], paired["cny_absolute_error"] - paired["sarima_absolute_error"])
    assert np.allclose(paired["squared_loss_difference"], paired["cny_squared_error"] - paired["sarima_squared_error"])
    assert (paired["cny_absolute_error_win"] == paired["absolute_loss_difference"].lt(0)).all()
    summary = backtest_cny.paired_win_summary(paired)
    assert [(row["horizon"], row["scope"], row["n"]) for row in summary] == [
        (1, "all", 86), (1, "q1", 22), (2, "all", 86), (2, "q1", 22)
    ]


def test_metric_deltas_use_cny_minus_baseline_and_signed_percent(canonical, stage4_forecasts, locked_orders):
    forecasts, _ = _build(canonical, stage4_forecasts, locked_orders)
    cny_metrics = backtest_cny.compute_cny_metrics(forecasts)
    baseline_metrics = pd.read_csv(backtest_cny.STAGE4_METRICS_PATH)
    compared = backtest_cny.compare_metrics(cny_metrics, baseline_metrics)
    for metric in ("MAE", "RMSE", "MASE"):
        assert np.allclose(compared[f"delta_{metric}"], compared[f"cny_{metric}"] - compared[f"baseline_{metric}"])
        expected = 100 * compared[f"delta_{metric}"] / compared[f"baseline_{metric}"]
        assert np.allclose(compared[f"percent_change_{metric}"], expected)


def test_fit_failure_is_reported_without_baseline_fallback(canonical, stage4_forecasts, locked_orders):
    calls = []

    def failed_fit(training, exog, candidate):
        calls.append(training.index[-1])
        return None

    with pytest.raises(RuntimeError, match="no baseline fallback"):
        _build(canonical, stage4_forecasts, locked_orders, failed_fit)
    assert len(calls) == 1


def test_nonconvergence_retry_reuses_stage4_fit_settings_and_terminal_parameters(monkeypatch, canonical, locked_orders):
    origin = pd.Period("2004Q4", freq="Q-DEC")
    training = canonical.loc[:origin, "log_retail"].astype(float)
    _, _, exog, _ = backtest_cny.origin_exog(canonical, origin, 1)
    calls = []

    class FakeModel:
        def __init__(self, endog, exog, **kwargs):
            assert endog.index.equals(exog.index)
            assert kwargs["order"] == locked_orders[origin].order
            assert kwargs["seasonal_order"] == locked_orders[origin].seasonal_order
            assert kwargs["trend"] == "n"
            assert kwargs["simple_differencing"] is False
            assert kwargs["enforce_stationarity"] is True
            assert kwargs["enforce_invertibility"] is True

        def fit(self, **kwargs):
            calls.append(kwargs.copy())
            result = FakeResult(training, exog, -0.2)
            result.mle_retvals = {"converged": len(calls) > 1}
            return result

    monkeypatch.setattr(backtest_cny, "SARIMAX", FakeModel)
    fitted = backtest_cny.fit_cny_sarimax(training, exog, locked_orders[origin])
    assert fitted is not None and fitted.converged
    assert fitted.attempt_count == 2
    assert fitted.retry_count == 1
    assert calls[0] == {"method": "lbfgs", "disp": False, "maxiter": backtest_cny.MAX_ITERATIONS}
    assert calls[1]["method"] == "lbfgs"
    assert calls[1]["disp"] is False
    assert calls[1]["maxiter"] == backtest_cny.MAX_ITERATIONS
    pd.testing.assert_series_equal(calls[1]["start_params"], fitted.result.params)


def test_production_artifacts_are_deterministic(tmp_path):
    first = backtest_cny.run_backtest(output_root=tmp_path, fit_function=fake_fit)
    artifact_paths = [tmp_path / path for path in first["generated_artifacts"]]
    hashes_first = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact_paths}
    second = backtest_cny.run_backtest(output_root=tmp_path, fit_function=fake_fit)
    hashes_second = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact_paths}
    assert hashes_first == hashes_second
    assert "runtime_seconds" not in __import__("json").loads((tmp_path / "outputs/diagnostics/stage5_summary.json").read_text(encoding="utf-8"))
    assert second["rolling_fit_count"]["origins_attempted"] == 87
    assert second["rolling_fit_count"]["successful"] == 87
    assert second["rolling_fit_count"]["failed"] == 0


def test_canonical_dataset_is_not_modified():
    path = backtest_cny.DATA_PATH
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    backtest_cny.load_canonical(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
