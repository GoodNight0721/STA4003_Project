"""Stage 4 SARIMA grid, training-only selection, scoring, and leakage tests."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.modeling import backtest_benchmarks, backtest_sarima


@pytest.fixture(scope="module")
def canonical():
    return backtest_sarima.load_analysis_data()


class _FakePrediction:
    def __init__(self, mean, variance, steps):
        self.predicted_mean = np.full(steps, mean, dtype=float)
        self.var_pred_mean = np.full(steps, variance, dtype=float)


class _FakeStateSpaceResult:
    def __init__(self, training):
        values = training.to_numpy(dtype=float)
        self.mean = float(values[-1])
        self.variance = 0.02
        self.resid = values - values.mean()
        self.model = SimpleNamespace(loglikelihood_burn=0, param_names=["sigma2"])
        self.params = np.asarray([0.02])

    def get_forecast(self, steps):
        return _FakePrediction(self.mean, self.variance, steps)


def _fake_selector(calls):
    def select(training):
        calls.append(training.index[-1])
        p = int(float(training.iloc[-1]) > 9.0)
        candidate = backtest_sarima.SarimaCandidate(p, 1, 0, 0, 0, 0)
        fitted = backtest_sarima.CandidateFit(
            candidate=candidate,
            result=_FakeStateSpaceResult(training),
            aic=1.0,
            aicc=2.0,
            llf=-0.5,
            nobs=len(training),
            parameter_count=1,
            converged=True,
        )
        return backtest_sarima.ModelSelection(
            selected_fit=fitted,
            candidate_count=46,
            successful_fit_count=43,
            failed_fit_count=3,
        )

    return select


def test_candidate_grid_is_exact_and_respects_all_constraints():
    candidates = backtest_sarima.generate_candidate_grid()
    assert len(candidates) == 46
    assert len(set(candidates)) == 46
    assert {candidate.p for candidate in candidates} == {0, 1, 2}
    assert {candidate.q for candidate in candidates} == {0, 1, 2}
    assert {candidate.P for candidate in candidates} == {0, 1}
    assert {candidate.Q for candidate in candidates} == {0, 1}
    assert {candidate.D for candidate in candidates} == {0, 1}
    assert {candidate.d for candidate in candidates} == {1}
    assert all(candidate.complexity <= 3 for candidate in candidates)
    assert all(candidate.seasonal_order[3] == 4 for candidate in candidates)


def test_aicc_uses_actual_parameter_count_and_invalidates_small_denominator():
    aic, k, n = 120.0, 4, 50
    expected = aic + 2 * k * (k + 1) / (n - k - 1)
    assert backtest_sarima.calculate_aicc(aic, k, n) == pytest.approx(expected)
    assert backtest_sarima.calculate_aicc(aic, 49, 50) is None
    assert backtest_sarima.calculate_aicc(np.inf, k, n) is None


def test_lognormal_level_forecast_returns_median_and_bias_corrected_mean():
    log_mean, log_variance = 8.5, 0.16
    median, mean = backtest_sarima.lognormal_level_forecasts(log_mean, log_variance)
    assert median == pytest.approx(np.exp(log_mean))
    assert mean == pytest.approx(np.exp(log_mean + 0.5 * log_variance))
    assert mean > median


def test_selector_uses_training_aicc_and_discards_failed_fits():
    candidates = backtest_sarima.generate_candidate_grid()[:3]
    training = pd.Series(np.arange(43.0), index=pd.period_range("1994Q1", periods=43, freq="Q-DEC"))
    supplied = []
    values = {candidates[0]: 5.0, candidates[1]: 2.0}

    def fake_fit(received_training, candidate):
        assert received_training.index[-1] == training.index[-1]
        assert len(received_training) == len(training)
        supplied.append(candidate)
        if candidate not in values:
            return None
        return backtest_sarima.CandidateFit(
            candidate=candidate,
            result=None,
            aic=values[candidate],
            aicc=values[candidate],
            llf=-1.0,
            nobs=len(training),
            parameter_count=1,
            converged=True,
        )

    selection = backtest_sarima.select_model(training, candidates, fake_fit)
    assert supplied == list(candidates)
    assert selection.selected_fit.candidate == candidates[1]
    assert selection.candidate_count == 3
    assert selection.successful_fit_count == 2
    assert selection.failed_fit_count == 1


def test_unique_origins_cover_both_horizons_once():
    origin_map = backtest_sarima.origin_horizons()
    assert len(origin_map) == 87
    assert origin_map[pd.Period("2019Q2", freq="Q-DEC")] == (1, 2)
    assert origin_map[pd.Period("2004Q3", freq="Q-DEC")] == (2,)
    assert origin_map[pd.Period("2026Q1", freq="Q-DEC")] == (1,)


def test_rolling_outputs_match_stage3_targets_and_counts(canonical):
    calls = []
    forecasts, selections, benchmarks = backtest_sarima.build_rolling_forecasts(
        canonical, selection_function=_fake_selector(calls)
    )
    assert len(forecasts) == 2 * 86
    assert len(selections) == len(backtest_sarima.origin_horizons()) == 87
    assert len(calls) == 87
    assert len(set(calls)) == 87
    targets = pd.period_range("2005Q1", "2026Q2", freq="Q-DEC")
    for horizon in backtest_sarima.HORIZONS:
        sarima = forecasts.loc[forecasts["horizon"] == horizon]
        baseline = benchmarks.loc[
            (benchmarks["model"] == "Naive") & (benchmarks["horizon"] == horizon)
        ]
        assert pd.PeriodIndex(sarima["target"], freq="Q-DEC").equals(targets)
        assert sarima["target"].tolist() == baseline["target"].tolist()
        assert len(sarima) == 86


def test_same_origin_selection_is_reused_for_h1_and_h2(canonical):
    calls = []
    forecasts, selections, _ = backtest_sarima.build_rolling_forecasts(
        canonical, selection_function=_fake_selector(calls)
    )
    origin = "2019Q2"
    assert calls.count(pd.Period(origin, freq="Q-DEC")) == 1
    selected = selections.loc[selections["origin"] == origin]
    assert len(selected) == 1
    records = forecasts.loc[forecasts["origin"] == origin].set_index("horizon")
    assert records.loc[1, "target"] == "2019Q3"
    assert records.loc[2, "target"] == "2019Q4"
    assert (records["p"] == int(selected.iloc[0]["selected_order"].strip("()").split(",")[0])).all()
    assert (records["aicc"] == selected.iloc[0]["aicc"]).all()


def test_selector_sees_training_data_only_and_future_changes_do_not_leak(canonical):
    target = pd.Period("2019Q4", freq="Q-DEC")
    origin = target - 2
    post_origin = origin + 1
    future = pd.Period("2020Q1", freq="Q-DEC")
    original_calls = []
    original, original_selections, _ = backtest_sarima.build_rolling_forecasts(
        canonical, selection_function=_fake_selector(original_calls)
    )
    changed = canonical.copy()
    changed.loc[target, "retail_bn"] *= 1.7
    changed.loc[target, "log_retail"] += 0.4
    changed.loc[post_origin, "retail_bn"] *= 1.9
    changed.loc[post_origin, "log_retail"] -= 0.2
    changed.loc[future, "retail_bn"] *= 2.2
    changed.loc[future, "log_retail"] += 0.3
    changed_calls = []
    mutated, mutated_selections, _ = backtest_sarima.build_rolling_forecasts(
        changed, selection_function=_fake_selector(changed_calls)
    )

    expected_origins = set(backtest_sarima.origin_horizons())
    assert set(original_calls) == expected_origins
    assert set(changed_calls) == expected_origins
    assert original_calls.count(origin) == 1
    assert changed_calls.count(origin) == 1
    original_record = original.loc[
        (original["horizon"] == 2) & (original["target"] == str(target))
    ].iloc[0]
    mutated_record = mutated.loc[
        (mutated["horizon"] == 2) & (mutated["target"] == str(target))
    ].iloc[0]
    assert mutated_record["forecast"] == pytest.approx(original_record["forecast"])
    assert mutated_record["p"] == original_record["p"]
    assert mutated_record["aicc"] == original_record["aicc"]
    assert mutated_record["mase_scale"] == pytest.approx(original_record["mase_scale"])
    original_selection = original_selections.loc[original_selections["origin"] == str(origin)].iloc[0]
    mutated_selection = mutated_selections.loc[mutated_selections["origin"] == str(origin)].iloc[0]
    assert mutated_selection["selected_order"] == original_selection["selected_order"]


def test_stage4_mase_reuses_stage3_origin_specific_scale_and_metrics(canonical):
    forecasts, _, benchmarks = backtest_sarima.build_rolling_forecasts(
        canonical, selection_function=_fake_selector([])
    )
    for horizon in backtest_sarima.HORIZONS:
        sarima = forecasts.loc[forecasts["horizon"] == horizon]
        baseline = benchmarks.loc[
            (benchmarks["model"] == "Naive") & (benchmarks["horizon"] == horizon)
        ]
        assert np.allclose(sarima["mase_scale"], baseline["mase_scale"])
    actual = backtest_sarima.compute_sarima_metrics(forecasts)
    expected = backtest_benchmarks.compute_metrics(forecasts, models=("SARIMA",))
    pd.testing.assert_frame_equal(actual, expected)
    assert (actual.loc[actual["scope"] == "all", "n_forecasts"] == 86).all()
    assert (actual.loc[actual["scope"] == "q1", "n_forecasts"] == 22).all()


def test_rolling_artifact_records_are_deterministic(canonical):
    first, first_selections, _ = backtest_sarima.build_rolling_forecasts(
        canonical, selection_function=_fake_selector([])
    )
    second, second_selections, _ = backtest_sarima.build_rolling_forecasts(
        canonical, selection_function=_fake_selector([])
    )
    first_bytes = first.to_csv(index=False, float_format="%.15g", lineterminator="\n")
    second_bytes = second.to_csv(index=False, float_format="%.15g", lineterminator="\n")
    assert first_bytes == second_bytes
    assert first_selections.to_csv(index=False) == second_selections.to_csv(index=False)
    assert first.columns.tolist() == backtest_sarima.FORECAST_COLUMNS
    assert len(first) == 172


def test_canonical_dataset_is_not_modified(canonical):
    del canonical
    path = backtest_sarima.DATA_PATH
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    backtest_sarima.load_analysis_data(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
