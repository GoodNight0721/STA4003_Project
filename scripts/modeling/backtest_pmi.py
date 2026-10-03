"""Leak-free PMI extension using the frozen Stage 4 SARIMA order at each origin."""

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

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from scripts.modeling import backtest_benchmarks as evaluation
from scripts.modeling import backtest_cny as cny
from scripts.modeling import backtest_sarima as sarima


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_quarterly.csv"
STAGE4_SELECTION_PATH = PROJECT_ROOT / "outputs" / "tables" / "sarima_origin_selection.csv"
TARGET_START = pd.Period("2010Q4", freq="Q-DEC")
TARGET_END = pd.Period("2026Q2", freq="Q-DEC")
TRAINING_START = pd.Period("2005Q3", freq="Q-DEC")
PMI_CENTER = 50.0
PMI_EXOG_NAME = "pmi_exog"
CNY_EXOG_NAME = cny.CNY_EXOG_NAME
MAX_ITERATIONS = sarima.MAX_ITERATIONS
HORIZONS = (1, 2)
MODEL_NAMES = ("SARIMA-common", "SARIMA+PMI", "SARIMA+CNY+PMI")

FORECAST_COLUMNS = [
    "model", "horizon", "origin", "target", "p", "d", "q", "P", "D", "Q",
    "seasonal_period", "training_start", "training_end", "n_train",
    "target_pmi_lag2", "target_pmi_exog", "pmi_beta", "pmi_beta_se", "pmi_beta_pvalue",
    "cny_training_center", "target_cny_regressor", "cny_beta", "cny_beta_se", "cny_beta_pvalue",
    "converged", "fit_attempt_count", "same_protocol_retry_count",
    "actual", "forecast_log_mean", "forecast_log_variance", "forecast_level_median", "forecast",
    "error", "absolute_error", "squared_error", "mase_scale", "scaled_absolute_error",
]
METRIC_COLUMNS = ["model", "horizon", "scope", "n", "MAE", "RMSE", "MASE"]
COMPARISON_COLUMNS = [
    "horizon", "scope", "n", "reference_model", "extended_model",
    "reference_MAE", "reference_RMSE", "reference_MASE",
    "extended_MAE", "extended_RMSE", "extended_MASE",
    "delta_MAE", "delta_RMSE", "delta_MASE",
    "percent_change_MAE", "percent_change_RMSE", "percent_change_MASE",
]
STABILITY_COLUMNS = [
    "row_type", "model", "origin", "n_origins", "n_train", "selected_order", "selected_seasonal_order",
    "cny_training_center", "pmi_beta", "pmi_beta_se", "pmi_beta_pvalue", "cny_beta", "cny_beta_se", "cny_beta_pvalue",
    "pmi_beta_mean", "pmi_beta_median", "pmi_beta_min", "pmi_beta_max",
    "pmi_beta_positive_count", "pmi_beta_negative_count", "pmi_beta_zero_count", "pmi_beta_sign_changes",
    "cny_beta_mean", "cny_beta_median", "cny_beta_min", "cny_beta_max",
    "cny_beta_positive_count", "cny_beta_negative_count", "cny_beta_zero_count", "cny_beta_sign_changes",
]


@dataclass(frozen=True)
class ModelFit:
    candidate: cny.LockedOrder
    result: object
    converged: bool
    attempt_count: int
    retry_count: int
    pmi_beta: float | None
    pmi_beta_se: float | None
    pmi_beta_pvalue: float | None
    cny_beta: float | None
    cny_beta_se: float | None
    cny_beta_pvalue: float | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class OriginModelForecast:
    fitted: ModelFit
    cny_center: float | None
    log_means: np.ndarray
    log_variances: np.ndarray


FitFunction = Callable[[pd.Series, pd.DataFrame | None, cny.LockedOrder, str, ModelFit | None], ModelFit]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_canonical(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load only the response, deterministic CNY calendar, and the canonical PMI lag."""
    columns = ["quarter", "retail_bn", "log_retail", "is_q1", "cny_position_fraction", "pmi", "pmi_lag2"]
    source = pd.read_csv(path, usecols=columns, keep_default_na=False)
    index = pd.PeriodIndex(source["quarter"].astype(str), freq="Q-DEC", name="quarter")
    _require(index.is_unique and index.is_monotonic_increasing, "Canonical quarters must be unique and ordered")
    _require(index.equals(evaluation.expected_quarters()), "Canonical quarters differ from Stage 3/4")
    frame = pd.DataFrame(index=index)
    for column in columns[1:]:
        frame[column] = pd.to_numeric(source[column], errors="coerce").to_numpy(dtype=float)
    _require(np.isfinite(frame[["retail_bn", "log_retail", "is_q1", "cny_position_fraction"]]).all().all(), "Canonical response/calendar fields must be finite")
    _require((frame["retail_bn"] > 0).all(), "Retail levels must be positive")
    _require(np.allclose(frame["log_retail"], np.log(frame["retail_bn"]), rtol=0, atol=5.1e-7), "Log response does not match retail levels")
    _require(np.array_equal(frame["is_q1"].to_numpy(), (index.quarter == 1).astype(float)), "is_q1 disagrees with the quarter index")
    expected_lag = frame["pmi"].shift(2)
    observed = frame["pmi_lag2"]
    _require(np.array_equal(observed.isna().to_numpy(), expected_lag.isna().to_numpy()), "Canonical PMI lag-2 missingness changed")
    _require(np.allclose(observed.dropna(), expected_lag.dropna(), rtol=0, atol=1e-12), "Canonical pmi_lag2 is not PMI[t-2]")
    expected_first = pd.Period("2005Q3", freq="Q-DEC")
    valid = observed.dropna()
    _require(not valid.empty and valid.index[0] == expected_first, "PMI lag-2 must first be available at 2005Q3")
    _require(valid.index[-1] == TARGET_END, "PMI lag-2 coverage must reach 2026Q2")
    return frame


def target_origin_horizons() -> dict[pd.Period, tuple[int, ...]]:
    """Return the fixed secondary target set and its unique sorted rolling origins."""
    targets = pd.period_range(TARGET_START, TARGET_END, freq="Q-DEC")
    _require(len(targets) == 63, "The fixed PMI secondary target window must contain 63 quarters")
    values: dict[pd.Period, list[int]] = {}
    for horizon in HORIZONS:
        for target in targets:
            origin = target - horizon
            values.setdefault(origin, []).append(horizon)
    return {origin: tuple(sorted(horizons)) for origin, horizons in sorted(values.items())}


def parse_order(value: str, seasonal_value: str) -> cny.LockedOrder:
    """Use the Stage 5 tuple parser; do not generate or select candidate orders."""
    return cny.parse_order(value, seasonal_value)


def load_locked_orders(path: Path = STAGE4_SELECTION_PATH) -> tuple[dict[pd.Period, cny.LockedOrder], pd.DataFrame]:
    return cny.load_locked_orders(path)


def mase_scale_for_training(training_levels: pd.Series) -> float:
    """Compute period-4 seasonal MASE scale using only the restricted training sample."""
    values = training_levels.to_numpy(dtype=float)
    _require(len(values) > evaluation.SEASONAL_PERIOD, "Too few restricted training observations for seasonal MASE")
    scale = float(np.abs(values[evaluation.SEASONAL_PERIOD:] - values[:-evaluation.SEASONAL_PERIOD]).mean())
    _require(np.isfinite(scale) and scale > 0, "Invalid restricted training MASE scale")
    return scale


def _named_stat(result: object, attribute: str, name: str) -> float | None:
    values = getattr(result, attribute, None)
    if values is None or not hasattr(values, "index") or name not in values.index:
        return None
    value = float(values[name])
    return value if np.isfinite(value) else None


def deterministic_order_start(model: object, exog_columns: tuple[str, ...]) -> np.ndarray:
    """Construct a stable, order-based L-BFGS start without using forecast outcomes."""
    values = np.asarray(model.start_params, dtype=float).copy()
    for position, name in enumerate(model.param_names):
        if name in exog_columns:
            values[position] = 0.0
        elif name.startswith("ar.S."):
            values[position] = 0.5
        elif name.startswith("ma.S."):
            values[position] = -0.5
        elif name.startswith("ar."):
            values[position] = 0.1
        elif name.startswith("ma."):
            values[position] = -0.1
    _require(np.isfinite(values).all(), "Deterministic optimizer start is non-finite")
    return values


def fit_sarimax(
    training: pd.Series,
    training_exog: pd.DataFrame | None,
    candidate: cny.LockedOrder,
    model_name: str,
    reference_fit: ModelFit | None = None,
) -> ModelFit:
    """Fit one frozen Stage 4 order using the shared Stage 4 L-BFGS configuration."""
    expected_columns = {
        "SARIMA-common": (),
        "SARIMA+PMI": (PMI_EXOG_NAME,),
        "SARIMA+CNY+PMI": (CNY_EXOG_NAME, PMI_EXOG_NAME),
    }
    _require(model_name in expected_columns, f"Unknown model: {model_name}")
    columns = expected_columns[model_name]
    _require((training_exog is None and not columns) or (training_exog is not None and tuple(training_exog.columns) == columns), f"Unexpected exogenous columns for {model_name}")
    if training_exog is not None:
        _require(training.index.equals(training_exog.index), "Endogenous and exogenous training indexes differ")
        _require(np.isfinite(training_exog.to_numpy(dtype=float)).all(), "Training exog contains missing or non-finite values")
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
            start_params = None
            if reference_fit is not None:
                _require(reference_fit.converged and reference_fit.candidate == candidate, "Warm-start SARIMA fit does not match the frozen order")
                reference_names = list(reference_fit.result.model.param_names)
                reference_values = np.asarray(reference_fit.result.params, dtype=float)
                start_params = np.asarray(model.start_params, dtype=float).copy()
                for position, name in enumerate(model.param_names):
                    if name in reference_names:
                        start_params[position] = reference_values[reference_names.index(name)]
                    elif name in columns:
                        start_params[position] = 0.0
            fit_options = {"method": "lbfgs", "disp": False, "maxiter": MAX_ITERATIONS}
            result = model.fit(start_params=start_params, **fit_options) if start_params is not None else model.fit(**fit_options)
            attempts = 1
            if not bool(result.mle_retvals.get("converged", False)):
                attempts += 1
                result = model.fit(start_params=result.params, **fit_options)
            if not bool(result.mle_retvals.get("converged", False)):
                attempts += 1
                result = model.fit(start_params=deterministic_order_start(model, columns), **fit_options)
        converged = bool(result.mle_retvals.get("converged", False))
        pmi_beta = _named_stat(result, "params", PMI_EXOG_NAME) if PMI_EXOG_NAME in columns else None
        cny_beta = _named_stat(result, "params", CNY_EXOG_NAME) if CNY_EXOG_NAME in columns else None
        required = [pmi_beta] if PMI_EXOG_NAME in columns else []
        if CNY_EXOG_NAME in columns:
            required.append(cny_beta)
        valid = all(value is not None for value in required) and np.isfinite([result.llf, result.aic]).all()
        reason = None if converged and valid else f"converged={converged}, valid_parameters={valid}, optimizer={result.mle_retvals}"
        return ModelFit(
            candidate=candidate, result=result, converged=converged and valid,
            attempt_count=attempts, retry_count=attempts - 1,
            pmi_beta=pmi_beta,
            pmi_beta_se=_named_stat(result, "bse", PMI_EXOG_NAME) if PMI_EXOG_NAME in columns else None,
            pmi_beta_pvalue=_named_stat(result, "pvalues", PMI_EXOG_NAME) if PMI_EXOG_NAME in columns else None,
            cny_beta=cny_beta,
            cny_beta_se=_named_stat(result, "bse", CNY_EXOG_NAME) if CNY_EXOG_NAME in columns else None,
            cny_beta_pvalue=_named_stat(result, "pvalues", CNY_EXOG_NAME) if CNY_EXOG_NAME in columns else None,
            failure_reason=reason,
        )
    except Exception as exc:
        raise RuntimeError(f"{model_name} Stage 4-configured SARIMAX fit raised {type(exc).__name__}: {exc}") from exc


def _pmi_exog(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame["pmi_lag2"].to_numpy(dtype=float) - PMI_CENTER
    _require(np.isfinite(values).all(), "PMI-lag2 exog is incomplete in this information set")
    return pd.DataFrame({PMI_EXOG_NAME: values}, index=frame.index)


def _training_exog(
    frame: pd.DataFrame, model_name: str
) -> tuple[pd.DataFrame | None, float | None, int]:
    if model_name == "SARIMA-common":
        return None, None, 0
    pmi = _pmi_exog(frame)
    if model_name == "SARIMA+PMI":
        return pmi, None, 0
    _require(model_name == "SARIMA+CNY+PMI", f"Unknown model: {model_name}")
    center, q1_count = cny.training_center(frame[["is_q1", "cny_position_fraction"]])
    cny_exog = cny.centered_cny_exog(frame[["is_q1", "cny_position_fraction"]], center)
    combined = pd.concat([cny_exog, pmi], axis=1)
    return combined, center, q1_count


def fit_origin_model(
    frame: pd.DataFrame,
    origin: pd.Period,
    steps: int,
    candidate: cny.LockedOrder,
    model_name: str,
    fit_function: FitFunction = fit_sarimax,
    reference_fit: ModelFit | None = None,
) -> OriginModelForecast:
    """Fit once through an origin and forecast using only observed PMI-lag2 exog."""
    training_frame = frame.loc[TRAINING_START:origin]
    _require(not training_frame.empty and training_frame.index[-1] == origin, f"No restricted training data through {origin}")
    _require(training_frame.index[0] == TRAINING_START, "Restricted training start is not 2005Q3")
    _require(training_frame["pmi_lag2"].notna().all(), f"PMI-lag2 training data are incomplete through {origin}")
    training = training_frame["log_retail"].astype(float)
    train_exog, center, _ = _training_exog(training_frame, model_name)
    if train_exog is not None:
        _require(training.index.equals(train_exog.index), f"{model_name} exog index differs at {origin}")
    try:
        fitted = fit_function(training, train_exog, candidate, model_name, reference_fit)
    except Exception as exc:
        raise RuntimeError(
            f"{model_name} fit failed at origin {origin}, frozen order {candidate.order_label} x "
            f"{candidate.seasonal_order_label}: {exc}"
        ) from exc
    if fitted is None or not fitted.converged:
        reason = fitted.failure_reason if fitted is not None else "fit function returned no model"
        raise RuntimeError(
            f"{model_name} did not converge at origin {origin}, frozen order {candidate.order_label} x "
            f"{candidate.seasonal_order_label}; attempts={getattr(fitted, 'attempt_count', 0)}; {reason}"
        )
    _require(fitted.candidate == candidate, f"{model_name} changed frozen order at {origin}")
    future_index = pd.period_range(origin + 1, periods=steps, freq="Q-DEC")
    _require(future_index.isin(frame.index).all(), f"Forecast calendar ends before {origin + steps}")
    future_frame = frame.loc[future_index]
    future_exog: pd.DataFrame | None = None
    if model_name == "SARIMA+PMI":
        future_exog = _pmi_exog(future_frame)
    elif model_name == "SARIMA+CNY+PMI":
        _require(center is not None, f"Combined model has no training Q1 center at {origin}")
        future_pmi = _pmi_exog(future_frame)
        future_cny = cny.centered_cny_exog(future_frame[["is_q1", "cny_position_fraction"]], center)
        future_exog = pd.concat([future_cny, future_pmi], axis=1)
    prediction = fitted.result.get_forecast(steps=steps, exog=future_exog)
    means = np.asarray(prediction.predicted_mean, dtype=float).reshape(-1)
    variances = np.asarray(prediction.var_pred_mean, dtype=float).reshape(-1)
    _require(len(means) == steps and len(variances) == steps, f"Unexpected forecast shape at {origin}")
    _require(np.isfinite(means).all() and np.isfinite(variances).all() and (variances >= 0).all(), f"Invalid log forecast moments at {origin}")
    return OriginModelForecast(fitted=fitted, cny_center=center, log_means=means, log_variances=variances)


def build_rolling_forecasts(
    frame: pd.DataFrame,
    locked_orders: dict[pd.Period, cny.LockedOrder],
    fit_function: FitFunction = fit_sarimax,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each model once per unique origin and reuse the fit across horizons."""
    _require(frame.index.equals(evaluation.expected_quarters()), "Input quarters differ from Stage 3/4")
    _require({"retail_bn", "log_retail", "is_q1", "cny_position_fraction", "pmi", "pmi_lag2"}.issubset(frame.columns), "Canonical PMI fields are incomplete")
    origins = target_origin_horizons()
    _require(set(locked_orders) == set(sarima.origin_horizons()), "Stage 4 frozen-order table does not cover the locked 87 origins")
    _require(set(origins).issubset(locked_orders), "PMI study origins are missing a Stage 4 frozen order")
    forecast_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    previous_fit: dict[str, ModelFit] = {}

    for position, (origin, horizons) in enumerate(origins.items(), start=1):
        candidate = locked_orders[origin]
        steps = max(horizons)
        training_frame = frame.loc[TRAINING_START:origin]
        training_levels = training_frame["retail_bn"].astype(float)
        training_n = len(training_frame)
        _require(training_n >= 20, f"PMI study has fewer than 20 training rows at {origin}")
        _require(training_frame.index[0] == TRAINING_START and training_frame.index[-1] == origin, "Restricted training sample bounds changed")
        scale = mase_scale_for_training(training_levels)
        origin_fits: dict[str, OriginModelForecast] = {}
        for model_name in MODEL_NAMES:
            reference_model = {
                "SARIMA-common": None,
                "SARIMA+PMI": "SARIMA-common",
                "SARIMA+CNY+PMI": "SARIMA+PMI",
            }[model_name]
            prior_fit = previous_fit.get(model_name)
            if prior_fit is not None and prior_fit.candidate == candidate:
                reference_fit = prior_fit
            else:
                reference_fit = origin_fits[reference_model].fitted if reference_model is not None else None
            origin_fit = fit_origin_model(frame, origin, steps, candidate, model_name, fit_function, reference_fit)
            origin_fits[model_name] = origin_fit
            previous_fit[model_name] = origin_fit.fitted
            fitted = origin_fit.fitted
            stability_rows.append(
                {
                    "row_type": "origin", "model": model_name, "origin": str(origin), "n_origins": None,
                    "n_train": training_n, "selected_order": candidate.order_label,
                    "selected_seasonal_order": candidate.seasonal_order_label,
                    "cny_training_center": origin_fit.cny_center,
                    "pmi_beta": fitted.pmi_beta, "pmi_beta_se": fitted.pmi_beta_se,
                    "pmi_beta_pvalue": fitted.pmi_beta_pvalue, "cny_beta": fitted.cny_beta,
                    "cny_beta_se": fitted.cny_beta_se, "cny_beta_pvalue": fitted.cny_beta_pvalue,
                }
            )
            for horizon in horizons:
                target = origin + horizon
                target_lag2 = float(frame.loc[target, "pmi_lag2"])
                pmi_source_quarter = target - 2
                _require(pmi_source_quarter <= origin, f"PMI information set leaks at {origin} for target {target}")
                _require(np.isclose(target_lag2, float(frame.loc[pmi_source_quarter, "pmi"]), rtol=0, atol=1e-12), f"Target pmi_lag2 is not PMI[{pmi_source_quarter}]")
                target_pmi_exog = target_lag2 - PMI_CENTER
                median, level_mean = sarima.lognormal_level_forecasts(
                    float(origin_fit.log_means[horizon - 1]), float(origin_fit.log_variances[horizon - 1])
                )
                actual = float(frame.loc[target, "retail_bn"])
                error = actual - level_mean
                target_cny_regressor: float | None = None
                if model_name == "SARIMA+CNY+PMI":
                    target_calendar = frame.loc[[target], ["is_q1", "cny_position_fraction"]]
                    target_cny_regressor = float(cny.centered_cny_exog(target_calendar, float(origin_fit.cny_center)).iloc[0, 0])
                forecast_rows.append(
                    {
                        "model": model_name, "horizon": horizon, "origin": str(origin), "target": str(target),
                        "p": candidate.p, "d": candidate.d, "q": candidate.q,
                        "P": candidate.P, "D": candidate.D, "Q": candidate.Q,
                        "seasonal_period": evaluation.SEASONAL_PERIOD,
                        "training_start": str(training_frame.index[0]), "training_end": str(training_frame.index[-1]),
                        "n_train": training_n,
                        "target_pmi_lag2": target_lag2, "target_pmi_exog": target_pmi_exog,
                        "pmi_beta": fitted.pmi_beta, "pmi_beta_se": fitted.pmi_beta_se,
                        "pmi_beta_pvalue": fitted.pmi_beta_pvalue,
                        "cny_training_center": origin_fit.cny_center, "target_cny_regressor": target_cny_regressor,
                        "cny_beta": fitted.cny_beta, "cny_beta_se": fitted.cny_beta_se,
                        "cny_beta_pvalue": fitted.cny_beta_pvalue,
                        "converged": True, "fit_attempt_count": fitted.attempt_count,
                        "same_protocol_retry_count": fitted.retry_count,
                        "actual": actual, "forecast_log_mean": float(origin_fit.log_means[horizon - 1]),
                        "forecast_log_variance": float(origin_fit.log_variances[horizon - 1]),
                        "forecast_level_median": median, "forecast": level_mean,
                        "error": error, "absolute_error": abs(error), "squared_error": error**2,
                        "mase_scale": scale, "scaled_absolute_error": abs(error) / scale,
                    }
                )
        if progress_callback is not None:
            progress_callback(position, len(origins))

    forecasts = pd.DataFrame(forecast_rows, columns=FORECAST_COLUMNS)
    stability = make_coefficient_stability(pd.DataFrame(stability_rows))
    forecasts = forecasts.sort_values(["model", "horizon", "target"], kind="stable").reset_index(drop=True)
    validate_rolling_outputs(forecasts, stability, locked_orders, frame)
    return forecasts, stability


def _series_summary(values: pd.Series) -> dict[str, float | int | None]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return {"mean": None, "median": None, "min": None, "max": None, "positive_count": 0, "negative_count": 0, "zero_count": 0, "sign_changes": 0}
    signs = np.sign(clean.to_numpy())
    return {
        "mean": float(clean.mean()), "median": float(clean.median()), "min": float(clean.min()), "max": float(clean.max()),
        "positive_count": int((clean > 0).sum()), "negative_count": int((clean < 0).sum()),
        "zero_count": int((clean == 0).sum()), "sign_changes": int(np.sum(signs[1:] != signs[:-1])),
    }


def make_coefficient_stability(origin_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep each rolling coefficient and append one descriptive summary row per model."""
    rows = origin_rows.to_dict(orient="records")
    for model in MODEL_NAMES:
        subset = origin_rows.loc[origin_rows["model"].eq(model)].sort_values("origin", kind="stable")
        pmi = _series_summary(subset["pmi_beta"])
        cny_beta = _series_summary(subset["cny_beta"])
        summary: dict[str, object] = {column: None for column in STABILITY_COLUMNS}
        summary.update({"row_type": "summary", "model": model, "n_origins": int(subset["origin"].nunique())})
        for prefix, values in (("pmi_beta", pmi), ("cny_beta", cny_beta)):
            for statistic, value in values.items():
                summary[f"{prefix}_{statistic}"] = value
        rows.append(summary)
    return pd.DataFrame(rows, columns=STABILITY_COLUMNS)


def validate_rolling_outputs(
    forecasts: pd.DataFrame,
    stability: pd.DataFrame,
    locked_orders: dict[pd.Period, cny.LockedOrder],
    frame: pd.DataFrame,
) -> None:
    targets = [str(value) for value in pd.period_range(TARGET_START, TARGET_END, freq="Q-DEC")]
    _require(forecasts.columns.tolist() == FORECAST_COLUMNS, "PMI forecast schema changed")
    _require(len(forecasts) == len(MODEL_NAMES) * len(HORIZONS) * len(targets), "PMI forecast row count is incorrect")
    _require(not forecasts.duplicated(["model", "horizon", "target"]).any(), "Duplicate model/horizon/target forecast")
    _require(stability.columns.tolist() == STABILITY_COLUMNS, "PMI coefficient stability schema changed")
    origins = target_origin_horizons()
    for model in MODEL_NAMES:
        for horizon in HORIZONS:
            subset = forecasts.loc[(forecasts["model"] == model) & (forecasts["horizon"] == horizon)].sort_values("target", kind="stable")
            _require(subset["target"].tolist() == targets and len(subset) == 63, f"{model} h={horizon} target window changed")
            target_periods = pd.PeriodIndex(subset["target"], freq="Q-DEC")
            origin_periods = pd.PeriodIndex(subset["origin"], freq="Q-DEC")
            _require((target_periods - horizon).equals(origin_periods), f"{model} h={horizon} origin alignment failed")
            _require(subset["training_start"].eq(str(TRAINING_START)).all(), "Restricted training start changed")
            _require(subset["training_end"].eq(subset["origin"]).all(), "Training extends beyond the origin")
            expected_n = (origin_periods.astype("int64") - TRAINING_START.ordinal + 1).to_numpy()
            _require(np.array_equal(subset["n_train"].to_numpy(dtype=int), expected_n), "Restricted training sizes changed")
            _require(subset["converged"].astype(bool).all(), f"{model} has a failed fit")
            _require(np.isfinite(subset[["forecast", "mase_scale", "scaled_absolute_error", "target_pmi_lag2", "target_pmi_exog"]]).all().all(), "Forecast contains invalid values")
            _require(np.allclose(subset["target_pmi_exog"], subset["target_pmi_lag2"] - PMI_CENTER, rtol=0, atol=1e-12), "PMI exog is not pmi_lag2 minus 50")
            for row in subset.itertuples(index=False):
                origin = pd.Period(row.origin, freq="Q-DEC")
                target = pd.Period(row.target, freq="Q-DEC")
                source_quarter = target - 2
                _require(source_quarter <= origin, "PMI source quarter is later than the forecast origin")
                _require(np.isclose(row.target_pmi_lag2, frame.loc[source_quarter, "pmi"], rtol=0, atol=1e-12), "Target PMI lag does not equal PMI[target-2]")
                candidate = locked_orders[origin]
                _require((row.p, row.d, row.q) == candidate.order, "Forecast order differs from frozen Stage 4 order")
                _require((row.P, row.D, row.Q, row.seasonal_period) == candidate.seasonal_order, "Forecast seasonal order differs from frozen Stage 4 order")
                training = frame.loc[TRAINING_START:origin, "retail_bn"].astype(float)
                _require(np.isclose(row.mase_scale, mase_scale_for_training(training), rtol=0, atol=1e-12), "MASE scale is not the restricted origin training scale")
                _require(np.isclose(row.scaled_absolute_error, row.absolute_error / row.mase_scale, rtol=1e-13, atol=1e-12), "Scaled absolute error changed")
                _require(np.isclose(row.forecast, np.exp(row.forecast_log_mean + 0.5 * row.forecast_log_variance), rtol=1e-13, atol=1e-10), "Level forecast is not lognormal conditional mean")
                _require(np.isclose(row.error, row.actual - row.forecast, rtol=0, atol=1e-9), "Forecast error sign changed")
    for (origin, model), group in forecasts.groupby(["origin", "model"], sort=False):
        _require(group["n_train"].nunique() == group["training_start"].nunique() == group["training_end"].nunique() == 1, "Models/horizons do not share a training sample")
        _require(group["pmi_beta"].nunique(dropna=False) == 1 and group["cny_beta"].nunique(dropna=False) == 1, f"Horizons do not reuse the same {model} fit at {origin}")
    for model in MODEL_NAMES:
        rows = stability.loc[(stability["model"] == model) & (stability["row_type"] == "origin")]
        _require(rows["origin"].nunique() == 64, f"{model} stability table must have 64 rolling fits")
        _require(set(rows["origin"]) == {str(value) for value in origins}, f"{model} fit origins differ from target-origin map")


def compute_pmi_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    metrics = evaluation.compute_metrics(forecasts, models=MODEL_NAMES).rename(columns={"n_forecasts": "n"})
    return metrics[METRIC_COLUMNS]


def compare_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = (("SARIMA-common", "SARIMA+PMI"), ("SARIMA+PMI", "SARIMA+CNY+PMI"))
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for scope in ("all", "q1"):
            for reference, extended in comparisons:
                ref = metrics.loc[(metrics["model"] == reference) & (metrics["horizon"] == horizon) & (metrics["scope"] == scope)].iloc[0]
                ext = metrics.loc[(metrics["model"] == extended) & (metrics["horizon"] == horizon) & (metrics["scope"] == scope)].iloc[0]
                row: dict[str, object] = {
                    "horizon": horizon, "scope": scope, "n": int(ref["n"]),
                    "reference_model": reference, "extended_model": extended,
                }
                for metric in ("MAE", "RMSE", "MASE"):
                    row[f"reference_{metric}"] = float(ref[metric])
                    row[f"extended_{metric}"] = float(ext[metric])
                    row[f"delta_{metric}"] = float(ext[metric] - ref[metric])
                    row[f"percent_change_{metric}"] = float(100 * (ext[metric] - ref[metric]) / ref[metric])
                rows.append(row)
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _summary_metrics(metrics: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {"model": str(row.model), "horizon": int(row.horizon), "scope": str(row.scope), "n": int(row.n),
         "MAE": float(row.MAE), "RMSE": float(row.RMSE), "MASE": float(row.MASE)}
        for row in metrics.itertuples(index=False)
    ]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.15g", lineterminator="\n")


def run_backtest(
    output_root: Path = PROJECT_ROOT,
    data_path: Path = DATA_PATH,
    stage4_selection_path: Path = STAGE4_SELECTION_PATH,
    fit_function: FitFunction = fit_sarimax,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Generate the Stage 6A secondary rolling study and deterministic artifacts."""
    started = time.perf_counter()
    frame = load_canonical(data_path)
    locked_orders, _ = load_locked_orders(stage4_selection_path)
    forecasts, stability = build_rolling_forecasts(frame, locked_orders, fit_function, progress_callback)
    metrics = compute_pmi_metrics(forecasts)
    comparisons = compare_metrics(metrics)
    fit_records = forecasts[["model", "origin", "fit_attempt_count"]].drop_duplicates()
    out = Path(output_root) / "outputs"
    paths = {
        "forecasts": out / "forecasts" / "pmi_rolling_forecasts.csv",
        "metrics": out / "tables" / "pmi_metrics.csv",
        "comparison": out / "tables" / "pmi_incremental_comparison.csv",
        "stability": out / "tables" / "pmi_coefficient_stability.csv",
        "summary": out / "diagnostics" / "stage6a_summary.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(forecasts, paths["forecasts"])
    _write_csv(metrics, paths["metrics"])
    _write_csv(comparisons, paths["comparison"])
    _write_csv(stability, paths["stability"])
    summaries = stability.loc[stability["row_type"] == "summary"].set_index("model")
    summary: dict[str, object] = {
        "stage": "6A",
        "secondary_evaluation_window": {"start": str(TARGET_START), "end": str(TARGET_END), "targets_per_horizon": 63},
        "horizons": list(HORIZONS), "primary_secondary_endpoint": {"horizon": 2, "scope": "all"},
        "unique_origins": len(target_origin_horizons()),
        "training_window": {"start": str(TRAINING_START), "end": "origin inclusive", "first_h2_origin": "2010Q2", "first_h2_n_train": 20},
        "fit_count": {
            "models": list(MODEL_NAMES), "attempted": int(len(stability.loc[stability["row_type"] == "origin"])),
            "successful": int(fit_records.shape[0]), "failed": 0,
            "optimizer_attempts": int(fit_records["fit_attempt_count"].sum()),
            "same_protocol_retries": int(fit_records["fit_attempt_count"].sum() - fit_records.shape[0]),
            "deterministic_start_fallbacks": int(fit_records["fit_attempt_count"].eq(3).sum()),
        },
        "optimizer_policy": {
            "optimizer": "L-BFGS",
            "settings": {"maxiter": MAX_ITERATIONS, "disp": False},
            "initialization": "Use the prior-origin fit when its frozen order matches; otherwise seed an extended model from its same-origin nested SARIMA model, with added exogenous coefficients initialized at zero.",
            "nonconvergence": "Continue once from terminal parameters; if still non-converged, try one deterministic order-based parameter start with the same optimizer and settings. No order, sample, or regressor changes.",
        },
        "locked_order_source": "outputs/tables/sarima_origin_selection.csv; exact Stage 4 order reused by origin; no order search.",
        "information_set": {
            "training_and_forecast_regressor": "pmi_lag2 - 50, where pmi_lag2[t] = PMI[t-2].",
            "future_pmi_used": "Only PMI at or before the forecast origin; for targets T+1 and T+2, target PMI source quarters are T-1 and T.",
            "contemporaneous_future_pmi_prohibited": True,
        },
        "models": {
            "SARIMA-common": "Frozen Stage 4 SARIMA on the shared 2005Q3-to-origin sample, no exog.",
            "SARIMA+PMI": "Same order and restricted sample with pmi_lag2 - 50.",
            "SARIMA+CNY+PMI": "Same order and sample with Stage 5 training-centered Q1 CNY regressor plus pmi_lag2 - 50.",
        },
        "forecast_convention": "exp(log forecast mean + 0.5 * log forecast variance); Stage 4 state-space SARIMAX settings.",
        "mase": "Origin-specific period-4 mean absolute seasonal difference calculated from retail levels in the restricted 2005Q3-to-origin training sample.",
        "metrics": _summary_metrics(metrics),
        "incremental_comparison": comparisons.to_dict(orient="records"),
        "coefficient_stability": {
            model: {key: (None if pd.isna(value) else int(value) if key.endswith("count") or key.endswith("changes") else float(value))
                    for key, value in row.items() if key not in {"row_type", "model", "origin", "n_origins", "n_train", "selected_order", "selected_seasonal_order", "cny_training_center"}}
            for model, row in summaries.iterrows()
        },
        "limitations": [
            "Secondary PMI-available target window is shorter than the Stage 3/5 primary window.",
            "The fixed lag-2 encoding is evaluated as specified; no other lag or full-sample correlation search is performed.",
            "Rolling coefficient and p-value summaries are descriptive and do not establish forecast value or causality.",
            "The combined CNY+PMI model is supplementary; it does not replace the Stage 5 primary conclusion.",
            "No ETS/Theta, shock robustness, full future forecast, or Stage 6B work is included.",
        ],
        "scope": {"pmi_only_extension": True, "order_search": False, "primary_window_changed": False, "stage6b_started": False},
        "generated_artifacts": [str(path.relative_to(Path(output_root))) for path in paths.values()],
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="")
    summary["runtime_seconds"] = round(time.perf_counter() - started, 3)
    return summary


def main() -> None:
    def progress(done: int, total: int) -> None:
        if done % 8 == 0 or done == total:
            print(f"Stage 6A PMI origins: {done}/{total}", flush=True)
    summary = run_backtest(progress_callback=progress)
    print(
        f"Stage 6A PMI complete: {summary['secondary_evaluation_window']['start']} to "
        f"{summary['secondary_evaluation_window']['end']}; targets per horizon 63; "
        f"fits {summary['fit_count']['successful']}/{summary['fit_count']['attempted']}; "
        f"runtime {summary['runtime_seconds']:.3f}s."
    )


if __name__ == "__main__":
    main()
