"""Leakage and fixed-protocol checks for the Stage 6A PMI extension."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.modeling import backtest_pmi, backtest_sarima


@pytest.fixture(scope="module")
def canonical():
    return backtest_pmi.load_canonical()


@pytest.fixture(scope="module")
def locked_orders():
    return backtest_pmi.load_locked_orders()[0]


class FakePrediction:
    def __init__(self, mean, variance):
        self.predicted_mean = mean
        self.var_pred_mean = variance


class FakeResult:
    def __init__(self, training, model_name):
        beta_map = {
            "SARIMA-common": {},
            "SARIMA+PMI": {backtest_pmi.PMI_EXOG_NAME: 0.012},
            "SARIMA+CNY+PMI": {backtest_pmi.PMI_EXOG_NAME: 0.018, backtest_pmi.CNY_EXOG_NAME: -0.04},
        }
        self.params = pd.Series({**beta_map[model_name], "sigma2": 0.04})
        self.bse = pd.Series({**{key: 0.2 for key in beta_map[model_name]}, "sigma2": 0.01})
        self.pvalues = pd.Series({**{key: 0.3 for key in beta_map[model_name]}, "sigma2": 0.01})
        self.mle_retvals = {"converged": True}
        self.llf = -10.0
        self.aic = 24.0
        self.model = SimpleNamespace(loglikelihood_burn=1)
        self.last_training_value = float(training.iloc[-1])
        self.model_name = model_name

    def get_forecast(self, steps, exog):
        means = self.last_training_value + 0.003 * np.arange(1, steps + 1)
        if exog is not None:
            x = exog.to_numpy(dtype=float)
            if backtest_pmi.PMI_EXOG_NAME in exog.columns:
                means += 0.002 * x[:, exog.columns.get_loc(backtest_pmi.PMI_EXOG_NAME)]
            if backtest_pmi.CNY_EXOG_NAME in exog.columns:
                means += 0.01 * x[:, exog.columns.get_loc(backtest_pmi.CNY_EXOG_NAME)]
        return FakePrediction(means, np.full(steps, 0.04))


def fake_fit(training, exog, candidate, model_name, reference_fit=None):
    if exog is not None:
        assert training.index.equals(exog.index)
    result = FakeResult(training, model_name)
    return backtest_pmi.ModelFit(
        candidate=candidate,
        result=result,
        converged=True,
        attempt_count=1,
        retry_count=0,
        pmi_beta=result.params.get(backtest_pmi.PMI_EXOG_NAME),
        pmi_beta_se=0.2 if model_name != "SARIMA-common" else None,
        pmi_beta_pvalue=0.3 if model_name != "SARIMA-common" else None,
        cny_beta=-0.04 if model_name == "SARIMA+CNY+PMI" else None,
        cny_beta_se=0.2 if model_name == "SARIMA+CNY+PMI" else None,
        cny_beta_pvalue=0.3 if model_name == "SARIMA+CNY+PMI" else None,
    )


def build(canonical, locked_orders, fit_function=fake_fit):
    return backtest_pmi.build_rolling_forecasts(canonical, locked_orders, fit_function)


def test_secondary_window_horizons_training_start_and_first_sample_size(canonical):
    origins = backtest_pmi.target_origin_horizons()
    targets = pd.period_range(backtest_pmi.TARGET_START, backtest_pmi.TARGET_END, freq="Q-DEC")
    assert str(targets[0]) == "2010Q4"
    assert str(targets[-1]) == "2026Q2"
    assert len(targets) == 63
    assert len(origins) == 64
    assert origins[pd.Period("2010Q2", freq="Q-DEC")] == (2,)
    first_training = canonical.loc[backtest_pmi.TRAINING_START:pd.Period("2010Q2", freq="Q-DEC")]
    assert str(first_training.index[0]) == "2005Q3"
    assert len(first_training) == 20
    assert first_training["pmi_lag2"].notna().sum() == 20


def test_horizons_have_identical_target_window(canonical, locked_orders):
    forecasts, _ = build(canonical, locked_orders)
    expected = [str(value) for value in pd.period_range("2010Q4", "2026Q2", freq="Q-DEC")]
    for model in backtest_pmi.MODEL_NAMES:
        for horizon in (1, 2):
            subset = forecasts.loc[(forecasts.model == model) & (forecasts.horizon == horizon), "target"].tolist()
            assert subset == expected


def test_target_lag_is_canonical_pmi_target_minus_two_and_known_at_origin(canonical, locked_orders):
    forecasts, _ = build(canonical, locked_orders)
    for row in forecasts.itertuples(index=False):
        target = pd.Period(row.target, freq="Q-DEC")
        origin = pd.Period(row.origin, freq="Q-DEC")
        source = target - 2
        assert row.target_pmi_lag2 == pytest.approx(canonical.loc[target, "pmi_lag2"])
        assert row.target_pmi_lag2 == pytest.approx(canonical.loc[source, "pmi"])
        assert row.target_pmi_exog == pytest.approx(row.target_pmi_lag2 - 50)
        assert source <= origin


def test_future_raw_pmi_after_origin_cannot_change_training_or_forecast(canonical, locked_orders):
    origin = pd.Period("2019Q2", freq="Q-DEC")
    original = backtest_pmi.fit_origin_model(canonical, origin, 2, locked_orders[origin], "SARIMA+PMI", fake_fit)
    changed = canonical.copy()
    future = changed.index > origin
    changed.loc[future, "pmi"] *= 1.7
    changed["pmi_lag2"] = changed["pmi"].shift(2)
    perturbed = backtest_pmi.fit_origin_model(changed, origin, 2, locked_orders[origin], "SARIMA+PMI", fake_fit)
    assert original.fitted.candidate == perturbed.fitted.candidate == locked_orders[origin]
    assert original.fitted.pmi_beta == perturbed.fitted.pmi_beta
    assert np.allclose(original.log_means, perturbed.log_means)
    assert np.allclose(original.log_variances, perturbed.log_variances)


def test_stage4_orders_are_frozen_and_no_search_is_called(monkeypatch, canonical, locked_orders):
    def forbidden(*args, **kwargs):
        raise AssertionError("Stage 6A must not search SARIMA orders")

    monkeypatch.setattr(backtest_sarima, "generate_candidate_grid", forbidden)
    forecasts, _ = build(canonical, locked_orders)
    for row in forecasts.itertuples(index=False):
        candidate = locked_orders[pd.Period(row.origin, freq="Q-DEC")]
        assert (row.p, row.d, row.q) == candidate.order
        assert (row.P, row.D, row.Q, row.seasonal_period) == candidate.seasonal_order


def test_all_models_use_same_restricted_training_sample_and_reuse_each_origin_fit(canonical, locked_orders):
    calls = []

    def counted_fit(training, exog, candidate, model_name, reference_fit=None):
        calls.append((training.index[0], training.index[-1], len(training), candidate, model_name))
        return fake_fit(training, exog, candidate, model_name, reference_fit)

    forecasts, stability = build(canonical, locked_orders, counted_fit)
    assert len(calls) == 64 * 3
    assert len({(str(call[1]), call[4]) for call in calls}) == 64 * 3
    assert all(str(start) == "2005Q3" for start, _, _, _, _ in calls)
    for origin, group in forecasts.groupby("origin"):
        assert group["n_train"].nunique() == 1
        assert group["training_start"].eq("2005Q3").all()
        for _, subset in group.groupby("model"):
            assert subset["pmi_beta"].nunique(dropna=False) == 1
            assert subset["cny_beta"].nunique(dropna=False) == 1
    assert len(stability.loc[stability.row_type == "origin"]) == 64 * 3


def test_training_only_restricted_mase_scale_is_exact(canonical, locked_orders):
    forecasts, _ = build(canonical, locked_orders)
    row = forecasts.loc[(forecasts.model == "SARIMA+PMI") & (forecasts.horizon == 2)].iloc[0]
    origin = pd.Period(row.origin, freq="Q-DEC")
    training = canonical.loc[backtest_pmi.TRAINING_START:origin, "retail_bn"].to_numpy(dtype=float)
    expected = np.abs(training[4:] - training[:-4]).mean()
    assert row.mase_scale == pytest.approx(expected)
    assert row.scaled_absolute_error == pytest.approx(row.absolute_error / expected)


def test_combined_cny_center_uses_restricted_training_q1_rows(canonical, locked_orders):
    origin = pd.Period("2015Q2", freq="Q-DEC")
    _, stability = build(canonical, locked_orders)
    row = stability.loc[(stability.row_type == "origin") & (stability.model == "SARIMA+CNY+PMI") & (stability.origin == str(origin))].iloc[0]
    training = canonical.loc[backtest_pmi.TRAINING_START:origin]
    expected = training.loc[training.is_q1 == 1, "cny_position_fraction"].mean()
    assert row.cny_training_center == pytest.approx(expected)
    assert pd.notna(row.cny_beta)


def test_comparison_delta_sign_and_percent_change(canonical, locked_orders):
    forecasts, _ = build(canonical, locked_orders)
    metrics = backtest_pmi.compute_pmi_metrics(forecasts)
    comparisons = backtest_pmi.compare_metrics(metrics)
    assert len(comparisons) == 8
    for metric in ("MAE", "RMSE", "MASE"):
        assert np.allclose(comparisons[f"delta_{metric}"], comparisons[f"extended_{metric}"] - comparisons[f"reference_{metric}"])
        assert np.allclose(comparisons[f"percent_change_{metric}"], 100 * comparisons[f"delta_{metric}"] / comparisons[f"reference_{metric}"])
    assert set(comparisons.loc[comparisons.horizon == 2, "reference_model"]) == {"SARIMA-common", "SARIMA+PMI"}


def test_coefficient_stability_includes_rolling_values_and_model_summaries(canonical, locked_orders):
    _, stability = build(canonical, locked_orders)
    pmi = stability.loc[(stability.row_type == "summary") & (stability.model == "SARIMA+PMI")].iloc[0]
    combined = stability.loc[(stability.row_type == "summary") & (stability.model == "SARIMA+CNY+PMI")].iloc[0]
    assert pmi.n_origins == 64
    assert pmi.pmi_beta_positive_count == 64
    assert pmi.pmi_beta_negative_count == 0
    assert pmi.pmi_beta_sign_changes == 0
    assert combined.cny_beta_mean == pytest.approx(-0.04)
    assert len(stability.loc[stability.row_type == "origin"]) == 192


def test_optimizer_keeps_frozen_order_and_settings_through_deterministic_retry(monkeypatch, canonical, locked_orders):
    origin = pd.Period("2010Q2", freq="Q-DEC")
    candidate = locked_orders[origin]
    training_frame = canonical.loc[backtest_pmi.TRAINING_START:origin]
    training = training_frame["log_retail"]
    exog = backtest_pmi._pmi_exog(training_frame)
    calls = []

    class FakeFitResult:
        def __init__(self, converged):
            self.params = pd.Series(
                {"pmi_exog": 0.02, "ar.S.L4": 0.4, "ma.L1": -0.2, "sigma2": 0.03}
            )
            self.bse = pd.Series({"pmi_exog": 0.1, "ar.S.L4": 0.1, "ma.L1": 0.1, "sigma2": 0.01})
            self.pvalues = pd.Series({"pmi_exog": 0.2, "ar.S.L4": 0.2, "ma.L1": 0.2, "sigma2": 0.01})
            self.mle_retvals = {"converged": converged}
            self.llf = -10.0
            self.aic = 24.0
            self.model = SimpleNamespace(param_names=["pmi_exog", "ar.S.L4", "ma.L1", "sigma2"])

    class FakeModel:
        def __init__(self, endog, exog, **kwargs):
            assert endog.index.equals(exog.index)
            assert kwargs["order"] == candidate.order
            assert kwargs["seasonal_order"] == candidate.seasonal_order
            assert kwargs["trend"] == "n"
            assert kwargs["simple_differencing"] is False
            assert kwargs["enforce_stationarity"] is True
            assert kwargs["enforce_invertibility"] is True
            assert kwargs["concentrate_scale"] is False
            self.param_names = ["pmi_exog", "ar.S.L4", "ma.L1", "sigma2"]
            self.start_params = np.array([0.9, 0.0, -0.2, 0.03])

        def fit(self, **kwargs):
            calls.append(kwargs.copy())
            return FakeFitResult(converged=len(calls) == 3)

    monkeypatch.setattr(backtest_pmi, "SARIMAX", FakeModel)
    result = backtest_pmi.fit_sarimax(training, exog, candidate, "SARIMA+PMI")
    assert result.converged
    assert result.attempt_count == 3
    assert result.retry_count == 2
    assert calls[0] == {"method": "lbfgs", "disp": False, "maxiter": backtest_pmi.MAX_ITERATIONS}
    assert calls[1]["method"] == calls[2]["method"] == "lbfgs"
    assert calls[1]["disp"] is calls[2]["disp"] is False
    assert calls[1]["maxiter"] == calls[2]["maxiter"] == backtest_pmi.MAX_ITERATIONS
    pd.testing.assert_series_equal(calls[1]["start_params"], FakeFitResult(False).params)
    np.testing.assert_allclose(calls[2]["start_params"], np.array([0.0, 0.5, -0.1, 0.03]))


def test_production_artifacts_are_deterministic(tmp_path):
    first = backtest_pmi.run_backtest(output_root=tmp_path, fit_function=fake_fit)
    paths = [tmp_path / path for path in first["generated_artifacts"]]
    hash_first = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    second = backtest_pmi.run_backtest(output_root=tmp_path, fit_function=fake_fit)
    hash_second = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert hash_first == hash_second
    written = json.loads((tmp_path / "outputs/diagnostics/stage6a_summary.json").read_text(encoding="utf-8"))
    assert "runtime_seconds" not in written
    assert second["fit_count"]["attempted"] == 192
    assert second["fit_count"]["successful"] == 192
    assert second["fit_count"]["failed"] == 0


def test_canonical_dataset_is_not_modified():
    path = backtest_pmi.DATA_PATH
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    backtest_pmi.load_canonical(path)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after
