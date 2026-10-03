"""Fixed-order rolling test of the incremental Lunar New Year timing regressor."""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from scripts.modeling import backtest_benchmarks as evaluation
from scripts.modeling import backtest_sarima as sarima


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_quarterly.csv"
STAGE4_FORECAST_PATH = PROJECT_ROOT / "outputs" / "forecasts" / "sarima_rolling_forecasts.csv"
STAGE4_SELECTION_PATH = PROJECT_ROOT / "outputs" / "tables" / "sarima_origin_selection.csv"
STAGE4_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "diagnostics" / "stage4_summary.json"
STAGE4_METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "sarima_metrics.csv"
MODEL_NAME = "SARIMA+CNY"
CNY_EXOG_NAME = "cny_regressor"
MAX_ITERATIONS = sarima.MAX_ITERATIONS
LJUNG_BOX_LAG = sarima.LJUNG_BOX_LAG

FORECAST_COLUMNS = [
    "model", "horizon", "origin", "target", "p", "d", "q", "P", "D", "Q",
    "seasonal_period", "training_start", "training_end", "n_train",
    "cny_training_center", "target_is_q1", "target_cny_position_fraction",
    "target_cny_regressor", "cny_beta", "cny_beta_se", "cny_beta_pvalue",
    "converged", "fit_attempt_count", "same_protocol_retry_count", "actual", "forecast_log_mean", "forecast_log_variance",
    "forecast_level_median", "forecast", "error", "absolute_error", "squared_error",
    "mase_scale", "scaled_absolute_error",
]
STABILITY_COLUMNS = [
    "origin", "n_train", "n_training_q1", "cny_training_center", "p", "d", "q",
    "P", "D", "Q", "seasonal_period", "selected_order", "selected_seasonal_order",
    "cny_beta", "cny_beta_se", "cny_beta_pvalue", "converged", "fit_attempt_count", "same_protocol_retry_count",
]
METRIC_COLUMNS = ["model", "horizon", "scope", "n", "MAE", "RMSE", "MASE"]
COMPARISON_COLUMNS = [
    "horizon", "scope", "n",
    "baseline_MAE", "baseline_RMSE", "baseline_MASE",
    "cny_MAE", "cny_RMSE", "cny_MASE",
    "delta_MAE", "delta_RMSE", "delta_MASE",
    "percent_change_MAE", "percent_change_RMSE", "percent_change_MASE",
]
PAIRED_COLUMNS = [
    "horizon", "target", "is_q1", "sarima_absolute_error", "cny_absolute_error",
    "absolute_loss_difference", "sarima_squared_error", "cny_squared_error",
    "squared_loss_difference", "cny_absolute_error_win",
]


@dataclass(frozen=True)
class LockedOrder:
    p: int
    d: int
    q: int
    P: int
    D: int
    Q: int

    @property
    def order(self) -> tuple[int, int, int]:
        return (self.p, self.d, self.q)

    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        return (self.P, self.D, self.Q, evaluation.SEASONAL_PERIOD)

    @property
    def order_label(self) -> str:
        return f"({self.p},{self.d},{self.q})"

    @property
    def seasonal_order_label(self) -> str:
        return f"({self.P},{self.D},{self.Q},{evaluation.SEASONAL_PERIOD})"


@dataclass(frozen=True)
class CnyFit:
    candidate: LockedOrder
    result: object
    beta: float | None
    beta_se: float | None
    beta_pvalue: float | None
    converged: bool
    attempt_count: int = 1
    retry_count: int = 0
    failure_reason: str | None = None


@dataclass(frozen=True)
class OriginForecast:
    fitted: CnyFit
    center: float
    q1_count: int
    log_means: np.ndarray
    log_variances: np.ndarray


FitFunction = Callable[[pd.Series, pd.DataFrame, LockedOrder], CnyFit | None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _period_index(values: pd.Series, label: str) -> pd.PeriodIndex:
    result = pd.PeriodIndex(values.astype(str), freq="Q-DEC")
    _require(result.is_unique and result.is_monotonic_increasing, f"{label} quarters must be unique and ordered")
    return result


def load_canonical(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the canonical response and deterministic CNY calendar fields."""
    columns = ["quarter", "retail_bn", "log_retail", "is_q1", "cny_position_fraction"]
    source = pd.read_csv(path, usecols=columns, keep_default_na=False)
    index = _period_index(source["quarter"], "Canonical")
    _require(index.equals(evaluation.expected_quarters()), "Canonical quarters differ from Stage 3/4")
    result = pd.DataFrame(index=index)
    for column in columns[1:]:
        result[column] = pd.to_numeric(source[column], errors="raise").to_numpy(dtype=float)
    _require(np.isfinite(result["retail_bn"]).all() and (result["retail_bn"] > 0).all(), "Retail levels must be finite and positive")
    _require(np.isfinite(result[["log_retail", "is_q1", "cny_position_fraction"]]).all().all(), "Canonical model/calendar fields must be finite")
    expected_q1 = (result.index.quarter == 1).astype(float)
    _require(np.array_equal(result["is_q1"].to_numpy(), expected_q1), "Canonical is_q1 disagrees with quarter index")
    _require(((result["cny_position_fraction"] >= 0) & (result["cny_position_fraction"] < 1)).all(), "CNY fractions are out of range")
    _require(np.allclose(result["log_retail"], np.log(result["retail_bn"]), rtol=0, atol=5.1e-7), "Canonical log response does not match retail levels")
    return result


def parse_order(value: str, seasonal_value: str) -> LockedOrder:
    """Parse the already-selected Stage 4 tuple columns without searching orders."""
    def parse_tuple(raw: str, expected_size: int) -> tuple[int, ...]:
        text = str(raw).strip().strip("()")
        values = tuple(int(part.strip()) for part in text.split(","))
        _require(len(values) == expected_size, f"Invalid frozen Stage 4 order tuple: {raw}")
        return values

    p, d, q = parse_tuple(value, 3)
    P, D, Q, period = parse_tuple(seasonal_value, 4)
    _require(period == evaluation.SEASONAL_PERIOD, "Stage 4 seasonal period changed")
    _require(d == 1 and P in (0, 1) and D in (0, 1) and Q in (0, 1), "Unexpected Stage 4 order values")
    return LockedOrder(p=p, d=d, q=q, P=P, D=D, Q=Q)


def load_locked_orders(path: Path = STAGE4_SELECTION_PATH) -> tuple[dict[pd.Period, LockedOrder], pd.DataFrame]:
    """Read and validate the exact selected order at each Stage 4 rolling origin."""
    table = pd.read_csv(path, dtype={"origin": str, "selected_order": str, "selected_seasonal_order": str})
    expected = list(sarima.origin_horizons())
    origins = _period_index(table["origin"], "Stage 4 selection")
    _require(len(table) == 87 and list(origins) == expected, "Stage 4 must provide exactly 87 ordered rolling origins")
    _require(not table["origin"].duplicated().any(), "Stage 4 selection contains duplicate origins")
    _require(table["training_end"].astype(str).tolist() == table["origin"].astype(str).tolist(), "Stage 4 training end differs from origin")
    orders = {
        pd.Period(row.origin, freq="Q-DEC"): parse_order(row.selected_order, row.selected_seasonal_order)
        for row in table.itertuples(index=False)
    }
    return orders, table


def load_stage4_forecasts(path: Path = STAGE4_FORECAST_PATH) -> pd.DataFrame:
    """Load the scored Stage 4 records, including exact targets and MASE scales."""
    forecasts = pd.read_csv(path, dtype={"origin": str, "target": str})
    _require(forecasts["model"].eq("SARIMA").all(), "Stage 4 forecast file contains an unexpected model")
    _require(not forecasts.duplicated(["horizon", "target"]).any(), "Stage 4 baseline targets are duplicated")
    expected_targets = [str(value) for value in pd.period_range(evaluation.TARGET_START, evaluation.TARGET_END, freq="Q-DEC")]
    for horizon in evaluation.HORIZONS:
        subset = forecasts.loc[forecasts["horizon"] == horizon].sort_values("target", kind="stable")
        _require(subset["target"].tolist() == expected_targets and len(subset) == 86, f"Stage 4 h={horizon} targets are incomplete")
        target = pd.PeriodIndex(subset["target"], freq="Q-DEC")
        origin = pd.PeriodIndex(subset["origin"], freq="Q-DEC")
        _require((target - horizon).equals(origin), f"Stage 4 h={horizon} origins do not align")
    return forecasts


def training_center(training_calendar: pd.DataFrame) -> tuple[float, int]:
    """Calculate the origin-specific mean over Q1 calendar rows in training only."""
    q1 = training_calendar.loc[training_calendar["is_q1"].eq(1), "cny_position_fraction"]
    _require(len(q1) > 0, "Training sample has no Q1 calendar observations")
    center = float(q1.mean())
    _require(np.isfinite(center), "Training CNY center is non-finite")
    return center, int(len(q1))


def centered_cny_exog(calendar: pd.DataFrame, center: float) -> pd.DataFrame:
    """Build one centered CNY timing regressor, exactly zero outside Q1."""
    indicator = calendar["is_q1"].to_numpy(dtype=float)
    fraction = calendar["cny_position_fraction"].to_numpy(dtype=float)
    values = np.where(indicator == 1.0, fraction - center, 0.0)
    result = pd.DataFrame({CNY_EXOG_NAME: values}, index=calendar.index)
    _require((result.loc[indicator == 0.0, CNY_EXOG_NAME] == 0.0).all(), "Non-Q1 CNY regressor is not exactly zero")
    return result


def origin_exog(
    frame: pd.DataFrame, origin: pd.Period, steps: int
) -> tuple[float, int, pd.DataFrame, pd.DataFrame]:
    """Return training-only centered CNY exog and future deterministic calendar exog."""
    training_calendar = frame.loc[:origin, ["is_q1", "cny_position_fraction"]]
    _require(not training_calendar.empty and training_calendar.index[-1] == origin, f"Training calendar does not end at {origin}")
    center, q1_count = training_center(training_calendar)
    training_exog = centered_cny_exog(training_calendar, center)
    future_index = pd.period_range(origin + 1, periods=steps, freq="Q-DEC")
    _require(future_index.isin(frame.index).all(), "Future calendar metadata is unavailable")
    future_calendar = frame.loc[future_index, ["is_q1", "cny_position_fraction"]]
    future_exog = centered_cny_exog(future_calendar, center)
    return center, q1_count, training_exog, future_exog


def _named_parameter(result: object, name: str) -> float | None:
    params = result.params
    if hasattr(params, "index"):
        if name not in params.index:
            return None
        value = float(params[name])
    else:
        return None
    return value if np.isfinite(value) else None


def fit_cny_sarimax(
    training: pd.Series, training_exog: pd.DataFrame, candidate: LockedOrder
) -> CnyFit | None:
    """Fit one locked Stage 4 error structure with one CNY exogenous variable."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                training,
                exog=training_exog,
                order=candidate.order,
                seasonal_order=candidate.seasonal_order,
                trend="n",
                simple_differencing=False,
                enforce_stationarity=True,
                enforce_invertibility=True,
                concentrate_scale=False,
            )
            result = model.fit(method="lbfgs", disp=False, maxiter=MAX_ITERATIONS)
            attempt_count = 1
            retry_count = 0
            if not bool(result.mle_retvals.get("converged", False)):
                # Continue from the optimizer's own terminal parameters, with the
                # identical Stage 4 method and settings. Never switch model/order.
                attempt_count += 1
                retry_count += 1
                result = model.fit(
                    start_params=result.params,
                    method="lbfgs",
                    disp=False,
                    maxiter=MAX_ITERATIONS,
                )
        converged = bool(result.mle_retvals.get("converged", False))
        beta = _named_parameter(result, CNY_EXOG_NAME)
        valid = beta is not None and np.isfinite([result.llf, result.aic]).all()
        if not converged or not valid:
            return CnyFit(
                candidate=candidate,
                result=result,
                beta=beta,
                beta_se=_named_parameter_from_mapping(result, "bse", CNY_EXOG_NAME),
                beta_pvalue=_named_parameter_from_mapping(result, "pvalues", CNY_EXOG_NAME),
                converged=False,
                attempt_count=attempt_count,
                retry_count=retry_count,
                failure_reason=f"converged={converged}, valid_parameters={valid}, optimizer={result.mle_retvals}",
            )
        return CnyFit(
            candidate=candidate,
            result=result,
            beta=beta,
            beta_se=_named_parameter_from_mapping(result, "bse", CNY_EXOG_NAME),
            beta_pvalue=_named_parameter_from_mapping(result, "pvalues", CNY_EXOG_NAME),
            converged=converged,
            attempt_count=attempt_count,
            retry_count=retry_count,
        )
    except Exception as exc:
        raise RuntimeError(f"Stage 4 lbfgs fit protocol raised {type(exc).__name__}: {exc}") from exc


def _named_parameter_from_mapping(result: object, attribute: str, name: str) -> float | None:
    values = getattr(result, attribute, None)
    if values is None or not hasattr(values, "index") or name not in values.index:
        return None
    value = float(values[name])
    return value if np.isfinite(value) else None


def candidate_for_full_sample(stage4_summary_path: Path = STAGE4_SUMMARY_PATH) -> tuple[LockedOrder, dict[str, float]]:
    """Read the descriptive full-sample order and baseline diagnostics frozen by Stage 4."""
    summary = json.loads(stage4_summary_path.read_text(encoding="utf-8"))
    selected = summary["full_sample_descriptive_selection"]
    candidate = parse_order(selected["selected_order"], selected["selected_seasonal_order"])
    expected = LockedOrder(0, 1, 1, 1, 0, 1)
    _require(candidate == expected, "Stage 4 full-sample descriptive order differs from the requested frozen order")
    return candidate, {"aic": float(selected["aic"]), "aicc": float(selected["aicc"])}


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _fit_values(fitted: CnyFit) -> tuple[float | None, float | None]:
    return fitted.beta_se, fitted.beta_pvalue


def fit_origin(
    frame: pd.DataFrame,
    origin: pd.Period,
    steps: int,
    candidate: LockedOrder,
    fit_function: FitFunction = fit_cny_sarimax,
) -> OriginForecast:
    """Fit and forecast one origin from training data plus deterministic future calendar exog."""
    training = frame.loc[:origin, "log_retail"].astype(float)
    center, q1_count, training_exog, future_exog = origin_exog(frame, origin, steps)
    _require(training_exog.index.equals(training.index), f"Training CNY exog index differs at {origin}")
    try:
        fitted = fit_function(training, training_exog, candidate)
    except Exception as exc:
        raise RuntimeError(
            f"SARIMA+CNY fit raised an exception at origin {origin} with frozen Stage 4 order "
            f"{candidate.order_label} x {candidate.seasonal_order_label}; no baseline fallback was used: {exc}"
        ) from exc
    if fitted is None or not fitted.converged:
        reason = fitted.failure_reason if fitted is not None else "fit function returned no model"
        raise RuntimeError(
            f"SARIMA+CNY did not converge at origin {origin}; fixed Stage 4 order "
            f"{candidate.order_label} x {candidate.seasonal_order_label}; same-protocol attempts "
            f"{getattr(fitted, 'attempt_count', 0)}; {reason}; no baseline fallback was used."
        )
    _require(fitted.candidate == candidate, f"Fit function changed the frozen Stage 4 order at {origin}")
    _require(fitted.beta is not None and np.isfinite(fitted.beta), f"CNY coefficient is non-finite at {origin}")
    prediction = fitted.result.get_forecast(steps=steps, exog=future_exog)
    log_means = np.asarray(prediction.predicted_mean, dtype=float).reshape(-1)
    log_variances = np.asarray(prediction.var_pred_mean, dtype=float).reshape(-1)
    _require(len(log_means) == steps and len(log_variances) == steps, f"Unexpected forecast shape at {origin}")
    _require(np.isfinite(log_means).all() and np.isfinite(log_variances).all(), f"Non-finite forecast moments at {origin}")
    return OriginForecast(fitted, center, q1_count, log_means, log_variances)


def build_rolling_forecasts(
    frame: pd.DataFrame,
    locked_orders: dict[pd.Period, LockedOrder],
    stage4_forecasts: pd.DataFrame,
    fit_function: FitFunction = fit_cny_sarimax,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit once per Stage 4 origin, reusing each CNY fit across its h=1/h=2 targets."""
    _require(frame.index.equals(evaluation.expected_quarters()), "Input quarters do not match Stage 3 protocol")
    _require({"retail_bn", "log_retail", "is_q1", "cny_position_fraction"}.issubset(frame.columns), "Input fields are incomplete")
    origins = sarima.origin_horizons()
    _require(set(locked_orders) == set(origins), "Frozen Stage 4 orders do not cover exactly the rolling origins")
    base_lookup = stage4_forecasts.set_index(["horizon", "target"], verify_integrity=True)
    forecast_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []

    for position, (origin, horizons) in enumerate(origins.items(), start=1):
        training = frame.loc[:origin, "log_retail"].astype(float)
        candidate = locked_orders[origin]
        origin_forecast = fit_origin(frame, origin, max(horizons), candidate, fit_function)
        fitted = origin_forecast.fitted
        center = origin_forecast.center
        q1_count = origin_forecast.q1_count
        log_means = origin_forecast.log_means
        log_variances = origin_forecast.log_variances
        beta_se, beta_pvalue = _fit_values(fitted)
        stability_rows.append(
            {
                "origin": str(origin),
                "n_train": len(training),
                "n_training_q1": q1_count,
                "cny_training_center": center,
                "p": candidate.p, "d": candidate.d, "q": candidate.q,
                "P": candidate.P, "D": candidate.D, "Q": candidate.Q,
                "seasonal_period": evaluation.SEASONAL_PERIOD,
                "selected_order": candidate.order_label,
                "selected_seasonal_order": candidate.seasonal_order_label,
                "cny_beta": fitted.beta,
                "cny_beta_se": beta_se,
                "cny_beta_pvalue": beta_pvalue,
                "converged": True,
                "fit_attempt_count": fitted.attempt_count,
                "same_protocol_retry_count": fitted.retry_count,
            }
        )
        for horizon in horizons:
            target = origin + horizon
            calendar = frame.loc[target]
            target_is_q1 = int(calendar["is_q1"])
            target_fraction = float(calendar["cny_position_fraction"])
            target_regressor = target_is_q1 * (target_fraction - center) if target_is_q1 else 0.0
            _require(target_regressor == 0.0 if not target_is_q1 else np.isfinite(target_regressor), "Invalid target CNY regressor")
            median, level_mean = sarima.lognormal_level_forecasts(
                float(log_means[horizon - 1]), float(log_variances[horizon - 1])
            )
            base = base_lookup.loc[(horizon, str(target))]
            actual = float(frame.loc[target, "retail_bn"])
            _require(np.isclose(actual, float(base["actual"]), rtol=0, atol=1e-9), f"Actual target differs from Stage 4 at {target}")
            error = actual - level_mean
            forecast_rows.append(
                {
                    "model": MODEL_NAME,
                    "horizon": horizon,
                    "origin": str(origin),
                    "target": str(target),
                    "p": candidate.p, "d": candidate.d, "q": candidate.q,
                    "P": candidate.P, "D": candidate.D, "Q": candidate.Q,
                    "seasonal_period": evaluation.SEASONAL_PERIOD,
                    "training_start": str(training.index[0]),
                    "training_end": str(training.index[-1]),
                    "n_train": len(training),
                    "cny_training_center": center,
                    "target_is_q1": target_is_q1,
                    "target_cny_position_fraction": target_fraction,
                    "target_cny_regressor": target_regressor,
                    "cny_beta": fitted.beta,
                    "cny_beta_se": beta_se,
                    "cny_beta_pvalue": beta_pvalue,
                    "converged": True,
                    "fit_attempt_count": fitted.attempt_count,
                    "same_protocol_retry_count": fitted.retry_count,
                    "actual": actual,
                    "forecast_log_mean": float(log_means[horizon - 1]),
                    "forecast_log_variance": float(log_variances[horizon - 1]),
                    "forecast_level_median": median,
                    "forecast": level_mean,
                    "error": error,
                    "absolute_error": abs(error),
                    "squared_error": error**2,
                    "mase_scale": float(base["mase_scale"]),
                    "scaled_absolute_error": abs(error) / float(base["mase_scale"]),
                }
            )
        if progress_callback is not None:
            progress_callback(position, len(origins))

    forecasts = pd.DataFrame(forecast_rows, columns=FORECAST_COLUMNS)
    stability = pd.DataFrame(stability_rows, columns=STABILITY_COLUMNS)
    forecasts = forecasts.sort_values(["horizon", "target"], kind="stable").reset_index(drop=True)
    stability = stability.sort_values("origin", kind="stable").reset_index(drop=True)
    validate_rolling_outputs(forecasts, stability, locked_orders, stage4_forecasts)
    return forecasts, stability


def validate_rolling_outputs(
    forecasts: pd.DataFrame,
    stability: pd.DataFrame,
    locked_orders: dict[pd.Period, LockedOrder],
    stage4_forecasts: pd.DataFrame,
) -> None:
    expected_targets = [str(value) for value in pd.period_range(evaluation.TARGET_START, evaluation.TARGET_END, freq="Q-DEC")]
    _require(forecasts.columns.tolist() == FORECAST_COLUMNS, "SARIMA+CNY forecast schema changed")
    _require(not forecasts.duplicated(["horizon", "target"]).any(), "Duplicate SARIMA+CNY target forecast")
    _require(stability.columns.tolist() == STABILITY_COLUMNS and len(stability) == 87, "CNY coefficient table must have one row per origin")
    _require(stability["origin"].is_unique, "Duplicate CNY coefficient origin")
    _require(stability["converged"].astype(bool).all(), "A CNY rolling model did not converge")
    for horizon in evaluation.HORIZONS:
        subset = forecasts.loc[forecasts["horizon"] == horizon]
        base = stage4_forecasts.loc[stage4_forecasts["horizon"] == horizon].sort_values("target", kind="stable")
        _require(subset["target"].tolist() == expected_targets and len(subset) == 86, f"CNY h={horizon} targets differ from Stage 3/4")
        _require(subset["target"].tolist() == base["target"].tolist(), f"CNY h={horizon} targets differ from SARIMA")
        _require(np.allclose(subset["actual"], base["actual"], rtol=0, atol=1e-9), f"CNY h={horizon} actuals differ from SARIMA")
        _require((subset["training_end"] == subset["origin"]).all(), "CNY training extends past the forecast origin")
        target_periods = pd.PeriodIndex(subset["target"], freq="Q-DEC")
        origin_periods = pd.PeriodIndex(subset["origin"], freq="Q-DEC")
        _require((target_periods - horizon).equals(origin_periods), f"CNY target/origin alignment failed for h={horizon}")
        _require(subset["training_start"].eq(str(evaluation.SAMPLE_START)).all(), "CNY training start changed")
        _require((subset["converged"].astype(bool)).all(), "CNY forecast has non-converged fit")
        _require(np.isfinite(subset[["forecast", "mase_scale", "scaled_absolute_error", "cny_beta"]]).all().all(), "CNY forecast contains non-finite values")
        non_q1 = subset["target_is_q1"].eq(0)
        _require((subset.loc[non_q1, "target_cny_regressor"] == 0.0).all(), "Non-Q1 target regressor is not zero")
        expected_regressor = np.where(
            subset["target_is_q1"].to_numpy(dtype=int) == 1,
            subset["target_cny_position_fraction"].to_numpy(dtype=float)
            - subset["cny_training_center"].to_numpy(dtype=float),
            0.0,
        )
        _require(np.array_equal(subset["target_cny_regressor"].to_numpy(dtype=float), expected_regressor), "Target CNY regressor formula changed")
        expected_mean = np.exp(subset["forecast_log_mean"].to_numpy(dtype=float) + 0.5 * subset["forecast_log_variance"].to_numpy(dtype=float))
        expected_median = np.exp(subset["forecast_log_mean"].to_numpy(dtype=float))
        _require(np.allclose(subset["forecast"], expected_mean, rtol=1e-13, atol=1e-10), "Level forecast does not follow Stage 4 lognormal convention")
        _require(np.allclose(subset["forecast_level_median"], expected_median, rtol=1e-13, atol=1e-10), "Saved level median does not follow Stage 4 convention")
        _require(np.allclose(subset["error"], subset["actual"] - subset["forecast"], rtol=0, atol=1e-9), "Forecast error sign changed")
        _require(np.allclose(subset["mase_scale"], base["mase_scale"], rtol=0, atol=1e-10), "Stage 3/4 MASE scale changed")
        expected_order_rows = subset.apply(
            lambda row: locked_orders[pd.Period(row["origin"], freq="Q-DEC")], axis=1
        )
        for index, row in subset.iterrows():
            order = expected_order_rows.loc[index]
            _require(tuple(int(row[key]) for key in ("p", "d", "q")) == order.order, "Rolling order differs from frozen Stage 4 order")
            _require(tuple(int(row[key]) for key in ("P", "D", "Q")) == order.seasonal_order[:3], "Rolling seasonal order differs from frozen Stage 4 order")
    for origin, group in forecasts.groupby("origin"):
        _require(group["cny_beta"].nunique() == 1, f"h=1/h=2 do not share one fit at {origin}")
        _require(group["cny_training_center"].nunique() == 1, f"h=1/h=2 use different centers at {origin}")


def compute_cny_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    result = evaluation.compute_metrics(forecasts, models=(MODEL_NAME,)).rename(columns={"n_forecasts": "n"})
    return result[METRIC_COLUMNS]


def compare_metrics(cny_metrics: pd.DataFrame, baseline_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_metrics.loc[baseline_metrics["model"] == "SARIMA"].copy()
    baseline = baseline.rename(columns={"n_forecasts": "n", "MAE": "baseline_MAE", "RMSE": "baseline_RMSE", "MASE": "baseline_MASE"})
    cny = cny_metrics.rename(columns={"model": "cny_model", "MAE": "cny_MAE", "RMSE": "cny_RMSE", "MASE": "cny_MASE"})
    merged = baseline[["horizon", "scope", "n", "baseline_MAE", "baseline_RMSE", "baseline_MASE"]].merge(
        cny[["horizon", "scope", "n", "cny_MAE", "cny_RMSE", "cny_MASE"]],
        on=["horizon", "scope"], how="inner", suffixes=("_baseline", "_cny"), validate="one_to_one"
    )
    _require((merged["n_baseline"] == merged["n_cny"]).all(), "Paired metric sample sizes differ")
    merged["n"] = merged["n_baseline"]
    merged["delta_MAE"] = merged["cny_MAE"] - merged["baseline_MAE"]
    merged["delta_RMSE"] = merged["cny_RMSE"] - merged["baseline_RMSE"]
    merged["delta_MASE"] = merged["cny_MASE"] - merged["baseline_MASE"]
    for metric in ("MAE", "RMSE", "MASE"):
        merged[f"percent_change_{metric}"] = 100.0 * merged[f"delta_{metric}"] / merged[f"baseline_{metric}"]
    return merged[COMPARISON_COLUMNS].sort_values(["horizon", "scope"], kind="stable").reset_index(drop=True)


def paired_errors(cny_forecasts: pd.DataFrame, baseline_forecasts: pd.DataFrame) -> pd.DataFrame:
    cny = cny_forecasts[["horizon", "target", "actual", "absolute_error", "squared_error"]].rename(
        columns={"absolute_error": "cny_absolute_error", "squared_error": "cny_squared_error"}
    )
    baseline = baseline_forecasts[["horizon", "target", "actual", "absolute_error", "squared_error"]].rename(
        columns={"absolute_error": "sarima_absolute_error", "squared_error": "sarima_squared_error"}
    )
    paired = baseline.merge(cny, on=["horizon", "target"], how="inner", suffixes=("_sarima", "_cny"), validate="one_to_one")
    _require(len(paired) == 172, "Paired CNY/baseline target records are incomplete")
    _require(np.allclose(paired["actual_sarima"], paired["actual_cny"], rtol=0, atol=1e-9), "Paired models have different actuals")
    paired["is_q1"] = pd.PeriodIndex(paired["target"], freq="Q-DEC").quarter == 1
    paired["absolute_loss_difference"] = paired["cny_absolute_error"] - paired["sarima_absolute_error"]
    paired["squared_loss_difference"] = paired["cny_squared_error"] - paired["sarima_squared_error"]
    paired["cny_absolute_error_win"] = paired["absolute_loss_difference"] < 0
    return paired[PAIRED_COLUMNS].sort_values(["horizon", "target"], kind="stable").reset_index(drop=True)


def paired_win_summary(paired: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for horizon in evaluation.HORIZONS:
        for scope in ("all", "q1"):
            subset = paired.loc[paired["horizon"].eq(horizon)]
            if scope == "q1":
                subset = subset.loc[subset["is_q1"]]
            wins = int(subset["cny_absolute_error_win"].sum())
            rows.append(
                {
                    "horizon": horizon,
                    "scope": scope,
                    "n": int(len(subset)),
                    "absolute_error_win_count": wins,
                    "absolute_error_win_rate": wins / len(subset),
                    "ties": int((subset["absolute_loss_difference"] == 0).sum()),
                }
            )
    return rows


def coefficient_summary(stability: pd.DataFrame) -> dict[str, object]:
    beta = stability["cny_beta"].to_numpy(dtype=float)
    signs = np.sign(beta)
    positive = int((signs > 0).sum())
    negative = int((signs < 0).sum())
    zero = int((signs == 0).sum())
    changes = int(np.count_nonzero(signs[1:] != signs[:-1]))
    return {
        "n_origins": int(len(beta)),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "sign_changes_in_origin_order": changes,
        "one_nonzero_sign_only": bool(positive == 0 or negative == 0),
        "mean": float(np.mean(beta)),
        "median": float(np.median(beta)),
        "minimum": float(np.min(beta)),
        "maximum": float(np.max(beta)),
        "range": float(np.max(beta) - np.min(beta)),
    }


def full_sample_diagnostics(
    frame: pd.DataFrame,
    fit_function: FitFunction = fit_cny_sarimax,
    stage4_summary_path: Path = STAGE4_SUMMARY_PATH,
) -> pd.DataFrame:
    """Fit only the Stage 4 full-sample order for descriptive CNY diagnostics."""
    candidate, baseline = candidate_for_full_sample(stage4_summary_path)
    training = frame["log_retail"].astype(float)
    center, q1_count = training_center(frame[["is_q1", "cny_position_fraction"]])
    exog = centered_cny_exog(frame[["is_q1", "cny_position_fraction"]], center)
    fitted = fit_function(training, exog, candidate)
    if fitted is None or not fitted.converged:
        raise RuntimeError("Full-sample descriptive SARIMA+CNY fit did not converge; no order search or fallback was used.")
    beta_se, beta_pvalue = _fit_values(fitted)
    result = fitted.result
    aic = float(result.aic)
    parameter_count = int(np.asarray(result.params).size)
    aicc = sarima.calculate_aicc(aic, parameter_count, len(training))
    _require(aicc is not None, "Full-sample SARIMA+CNY AICc is invalid")
    residuals = np.asarray(result.resid, dtype=float).reshape(-1)
    burn = int(getattr(result.model, "loglikelihood_burn", 0) or 0)
    residuals = residuals[burn:]
    residuals = residuals[np.isfinite(residuals)]
    diagnostic = acorr_ljungbox(residuals, lags=[LJUNG_BOX_LAG], model_df=candidate.p + candidate.q + candidate.P + candidate.Q, return_df=True).iloc[0]
    return pd.DataFrame(
        [
            {
                "sample_start": str(training.index[0]),
                "sample_end": str(training.index[-1]),
                "n_train": len(training),
                "n_training_q1": q1_count,
                "cny_training_center": center,
                "selected_order": candidate.order_label,
                "selected_seasonal_order": candidate.seasonal_order_label,
                "p": candidate.p, "d": candidate.d, "q": candidate.q,
                "P": candidate.P, "D": candidate.D, "Q": candidate.Q,
                "cny_beta": fitted.beta,
                "cny_beta_se": beta_se,
                "cny_beta_pvalue": beta_pvalue,
                "aic": aic,
                "aicc": aicc,
                "baseline_stage4_aic": baseline["aic"],
                "baseline_stage4_aicc": baseline["aicc"],
                "delta_aicc": aicc - baseline["aicc"],
                "ljung_box_lag": LJUNG_BOX_LAG,
                "ljung_box_model_df": candidate.p + candidate.q + candidate.P + candidate.Q,
                "ljung_box_statistic": float(diagnostic["lb_stat"]),
                "ljung_box_pvalue": float(diagnostic["lb_pvalue"]),
                "ljung_box_n_residuals": int(len(residuals)),
                "residual_mean": float(np.mean(residuals)),
                "residual_std": float(np.std(residuals, ddof=1)),
                "residual_min": float(np.min(residuals)),
                "residual_max": float(np.max(residuals)),
                "converged": True,
                "descriptive_only": True,
                "oos_evidence": False,
            }
        ]
    )


def _save_figure(path: Path) -> None:
    plt.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "STA4003 Stage 5"})
    plt.close()


def plot_h2_q1_forecasts(forecasts: pd.DataFrame, path: Path) -> None:
    subset = forecasts.loc[(forecasts["horizon"] == 2) & (forecasts["target_is_q1"] == 1)]
    dates = pd.PeriodIndex(subset["target"], freq="Q-DEC").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(dates, subset["actual"], color="#111827", linewidth=1.8, marker="o", markersize=3, label="Actual")
    ax.plot(dates, subset["forecast_baseline"], color="#2563eb", linewidth=1.3, marker=".", label="SARIMA")
    ax.plot(dates, subset["forecast_cny"], color="#dc2626", linewidth=1.3, marker=".", label="SARIMA+CNY")
    ax.set_title("Q1 retail forecasts at h = 2")
    ax.set_ylabel("Retail sales (RMB 100 million)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    _save_figure(path)


def plot_h2_paired_difference(paired: pd.DataFrame, path: Path) -> None:
    subset = paired.loc[paired["horizon"] == 2]
    dates = pd.PeriodIndex(subset["target"], freq="Q-DEC").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(11, 4.7))
    ax.axhline(0, color="#111827", linewidth=0.9)
    q1 = subset["is_q1"].astype(bool).to_numpy()
    ax.scatter(dates[~q1], subset.loc[~q1, "absolute_loss_difference"], c="#64748b", s=25, alpha=0.85, label="Non-Q1")
    ax.scatter(dates[q1], subset.loc[q1, "absolute_loss_difference"], c="#dc2626", s=25, alpha=0.85, label="Q1")
    ax.set_title("h = 2 paired absolute-error difference (CNY − SARIMA)")
    ax.set_ylabel("Difference (RMB 100 million; negative = CNY better)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save_figure(path)


def plot_rolling_beta(stability: pd.DataFrame, path: Path) -> None:
    dates = pd.PeriodIndex(stability["origin"], freq="Q-DEC").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(11, 4.7))
    ax.plot(dates, stability["cny_beta"], color="#7c3aed", linewidth=1.35)
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_title("Rolling SARIMA+CNY coefficient by origin")
    ax.set_ylabel("CNY timing coefficient in log-retail model")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(path)


def _summary_metrics(metrics: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "model": str(row.model), "horizon": int(row.horizon), "scope": str(row.scope),
            "n": int(row.n), "MAE": float(row.MAE), "RMSE": float(row.RMSE), "MASE": float(row.MASE),
        }
        for row in metrics.itertuples(index=False)
    ]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.15g", lineterminator="\n")


def run_backtest(
    output_root: Path = PROJECT_ROOT,
    data_path: Path = DATA_PATH,
    stage4_forecast_path: Path = STAGE4_FORECAST_PATH,
    stage4_selection_path: Path = STAGE4_SELECTION_PATH,
    stage4_summary_path: Path = STAGE4_SUMMARY_PATH,
    stage4_metrics_path: Path = STAGE4_METRICS_PATH,
    fit_function: FitFunction = fit_cny_sarimax,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Generate Stage 5 forecasts, comparisons, diagnostics, figures, and summary."""
    started = time.perf_counter()
    frame = load_canonical(data_path)
    locked_orders, _ = load_locked_orders(stage4_selection_path)
    baseline_forecasts = load_stage4_forecasts(stage4_forecast_path)
    forecasts, stability = build_rolling_forecasts(
        frame, locked_orders, baseline_forecasts, fit_function, progress_callback
    )
    baseline_metrics = pd.read_csv(stage4_metrics_path)
    cny_metrics = compute_cny_metrics(forecasts)
    metrics = pd.concat(
        [
            baseline_metrics.loc[baseline_metrics["model"] == "SARIMA", ["model", "horizon", "scope", "n_forecasts", "MAE", "RMSE", "MASE"]]
            .rename(columns={"n_forecasts": "n"}),
            cny_metrics,
        ],
        ignore_index=True,
    )[METRIC_COLUMNS]
    comparisons = compare_metrics(cny_metrics, baseline_metrics)
    paired = paired_errors(forecasts, baseline_forecasts)
    win_summary = paired_win_summary(paired)
    full_sample = full_sample_diagnostics(frame, fit_function, stage4_summary_path)

    out = Path(output_root) / "outputs"
    paths = {
        "forecast": out / "forecasts" / "sarima_cny_rolling_forecasts.csv",
        "metrics": out / "tables" / "cny_metrics.csv",
        "comparison": out / "tables" / "cny_incremental_comparison.csv",
        "paired": out / "tables" / "cny_paired_errors.csv",
        "stability": out / "tables" / "cny_coefficient_stability.csv",
        "full_sample": out / "tables" / "cny_full_sample_diagnostics.csv",
        "summary": out / "diagnostics" / "stage5_summary.json",
        "figure_q1": out / "figures" / "cny_h2_q1_actual_vs_forecasts.png",
        "figure_paired": out / "figures" / "cny_h2_paired_absolute_error_difference.png",
        "figure_beta": out / "figures" / "cny_rolling_beta.png",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(forecasts, paths["forecast"])
    _write_csv(metrics, paths["metrics"])
    _write_csv(comparisons, paths["comparison"])
    _write_csv(paired, paths["paired"])
    _write_csv(stability, paths["stability"])
    _write_csv(full_sample, paths["full_sample"])
    plot_frame = forecasts.merge(
        baseline_forecasts[["horizon", "target", "forecast"]].rename(columns={"forecast": "forecast_baseline"}),
        on=["horizon", "target"], validate="one_to_one"
    ).rename(columns={"forecast": "forecast_cny"})
    plot_h2_q1_forecasts(plot_frame, paths["figure_q1"])
    plot_h2_paired_difference(paired, paths["figure_paired"])
    plot_rolling_beta(stability, paths["figure_beta"])

    coefficient = coefficient_summary(stability)
    full = full_sample.iloc[0]
    comparisons_json = [
        {key: (int(value) if key in {"horizon", "n"} else float(value) if isinstance(value, (float, np.floating)) else str(value))
         for key, value in row.items()}
        for row in comparisons.to_dict(orient="records")
    ]
    summary: dict[str, object] = {
        "stage": 5,
        "response": "log_retail",
        "evaluation_window": {"start": str(evaluation.TARGET_START), "end": str(evaluation.TARGET_END)},
        "horizons": list(evaluation.HORIZONS),
        "primary_result": {"horizon": 2, "scope": "all"},
        "prespecified_subgroup": {"horizon": 2, "scope": "q1"},
        "rolling_fit_count": {
            "origins_attempted": len(stability),
            "successful": int(stability["converged"].sum()),
            "failed": int((~stability["converged"].astype(bool)).sum()),
            "optimizer_attempts": int(stability["fit_attempt_count"].sum()),
            "same_protocol_retries_after_initial_nonconvergence": int(stability["same_protocol_retry_count"].sum()),
        },
        "locked_order_source": "outputs/tables/sarima_origin_selection.csv; one Stage 4 order read per origin; no order search.",
        "cny_regressor": "is_q1 * (cny_position_fraction - mean training Q1 cny_position_fraction through origin); same training center used for future exog.",
        "future_information": "Only deterministic, ex-ante-known CNY calendar fields are used for future exog; no future retail or PMI observation enters fitting or prediction.",
        "forecast_convention": {
            "response": "log_retail",
            "level_median": "exp(forecast_log_mean)",
            "primary_level_forecast": "exp(forecast_log_mean + 0.5 * forecast_log_variance)",
            "scoring": "All MAE/RMSE/MASE use bias-corrected conditional-mean level forecast.",
            "mase": "Stage 4 saved origin-specific training-only seasonal-period-4 scale reused by exact horizon/target.",
        },
        "rolling_coefficient_summary": coefficient,
        "metrics": _summary_metrics(metrics),
        "incremental_comparison": comparisons_json,
        "paired_absolute_error_wins": win_summary,
        "full_sample_descriptive_fit": {
            "order": str(full["selected_order"]), "seasonal_order": str(full["selected_seasonal_order"]),
            "beta": float(full["cny_beta"]), "beta_se": _optional_float(full["cny_beta_se"]),
            "beta_pvalue": _optional_float(full["cny_beta_pvalue"]), "aic": float(full["aic"]),
            "aicc": float(full["aicc"]), "baseline_stage4_aicc": float(full["baseline_stage4_aicc"]),
            "delta_aicc": float(full["delta_aicc"]), "ljung_box_lag8_pvalue": float(full["ljung_box_pvalue"]),
            "converged": bool(full["converged"]), "descriptive_only": True, "oos_evidence": False,
        },
        "interpretation": {
            "primary_endpoint": "h=2 all remains primary; h=2 Q1 is a prespecified subgroup.",
            "coefficient": "Conditional log-response association for Q1 timing after the fixed SARIMA dynamics; not causal evidence.",
            "p_values": "Coefficient p-values and full-sample AICc are descriptive and are not the forecasting-value criterion.",
        },
        "scope": {"pmi": False, "order_search": False, "oos_based_cny_redesign": False, "final_future_forecast": False},
        "generated_artifacts": [str(path.relative_to(Path(output_root))) for path in paths.values()],
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8", newline="",
    )
    summary["runtime_seconds"] = round(time.perf_counter() - started, 3)
    return summary


def main() -> None:
    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"Stage 5 CNY fits: {done}/{total}", flush=True)

    started = time.perf_counter()
    summary = run_backtest(progress_callback=progress)
    print(f"Stage 5 complete: {summary['rolling_fit_count']}; elapsed {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
