"""Fixed-specification ETS robustness and target-only shock sensitivity."""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

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
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from scripts.modeling import backtest_benchmarks as evaluation
from scripts.modeling import backtest_sarima


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_quarterly.csv"
BENCHMARK_METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "benchmark_metrics.csv"
SARIMA_FORECAST_PATH = PROJECT_ROOT / "outputs" / "forecasts" / "sarima_rolling_forecasts.csv"
SARIMA_METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "sarima_metrics.csv"
CNY_FORECAST_PATH = PROJECT_ROOT / "outputs" / "forecasts" / "sarima_cny_rolling_forecasts.csv"
CNY_METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "cny_metrics.csv"

MODEL_NAME = "ETS"
FIXED_ETS_SPEC = {
    "error": "add",
    "trend": "add",
    "damped_trend": True,
    "seasonal": "add",
    "seasonal_periods": evaluation.SEASONAL_PERIOD,
}
SHOCK_TARGETS = ("2020Q1", "2022Q2")
COMPARISON_MODELS = (
    "Historical Mean",
    "Naive",
    "Seasonal Naive",
    "SARIMA",
    "SARIMA+CNY",
    MODEL_NAME,
)
SENSITIVITY_MODELS = ("SARIMA", "SARIMA+CNY", MODEL_NAME)
FORECAST_COLUMNS = [
    "model",
    "horizon",
    "origin",
    "target",
    "training_start",
    "training_end",
    "n_train",
    "fit_success",
    "converged",
    "fit_failure_reason",
    "numerical_issues",
    "n_training_residuals",
    "actual",
    "forecast_log",
    "smearing_factor",
    "forecast",
    "error",
    "absolute_error",
    "squared_error",
    "mase_scale",
    "scaled_absolute_error",
]
METRIC_COLUMNS = ["model", "horizon", "scope", "n_forecasts", "MAE", "RMSE", "MASE"]
SHOCK_METRIC_COLUMNS = [
    "model", "horizon", "scope", "sample", "excluded_targets", "n_forecasts", "MAE", "RMSE", "MASE"
]
CNY_SHOCK_COLUMNS = [
    "horizon", "scope", "sample", "n_sarima", "n_cny",
    "delta_MAE", "delta_RMSE", "delta_MASE",
    "percent_change_MAE", "percent_change_RMSE", "percent_change_MASE",
]


@dataclass(frozen=True)
class ETSFit:
    result: object | None
    converged: bool
    warnings: tuple[str, ...] = ()
    failure_reason: str | None = None


FitFunction = Callable[[pd.Series], ETSFit]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_analysis_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the canonical quarterly retail series."""
    _require(Path(path).is_file(), f"Missing canonical input: {path}")
    source = pd.read_csv(path, usecols=["quarter", "retail_bn", "log_retail"], keep_default_na=False)
    quarters = pd.PeriodIndex(source["quarter"].astype(str), freq="Q-DEC", name="quarter")
    _require(quarters.equals(evaluation.expected_quarters()), "Canonical quarter sequence differs from Stage 3")
    retail = pd.to_numeric(source["retail_bn"], errors="raise").to_numpy(dtype=float)
    log_retail = pd.to_numeric(source["log_retail"], errors="raise").to_numpy(dtype=float)
    _require(np.isfinite(retail).all() and (retail > 0).all(), "Retail levels must be finite and positive")
    _require(np.isfinite(log_retail).all(), "log_retail must be finite")
    return pd.DataFrame({"retail_bn": retail, "log_retail": log_retail}, index=quarters)


def _validate_frame(frame: pd.DataFrame) -> None:
    _require(frame.index.equals(evaluation.expected_quarters()), "Input quarters do not match Stage 3")
    _require({"retail_bn", "log_retail"}.issubset(frame.columns), "Input must contain retail_bn and log_retail")
    levels = pd.to_numeric(frame["retail_bn"], errors="raise").to_numpy(dtype=float)
    logs = pd.to_numeric(frame["log_retail"], errors="raise").to_numpy(dtype=float)
    _require(np.isfinite(levels).all() and (levels > 0).all(), "Retail levels must be finite and positive")
    _require(np.isfinite(logs).all(), "log_retail must be finite")


def origin_horizons() -> dict[pd.Period, tuple[int, ...]]:
    """Map each Stage 3 origin to the horizons it serves."""
    requested: dict[pd.Period, set[int]] = {}
    targets = pd.period_range(evaluation.TARGET_START, evaluation.TARGET_END, freq="Q-DEC")
    for horizon in evaluation.HORIZONS:
        for target in targets:
            requested.setdefault(target - horizon, set()).add(horizon)
    return {origin: tuple(sorted(horizons)) for origin, horizons in sorted(requested.items())}


def fit_fixed_ets(training: pd.Series) -> ETSFit:
    """Fit the one pre-specified additive, damped-trend, quarterly ETS model."""
    captured: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = ETSModel(training, **FIXED_ETS_SPEC)
            result = model.fit(disp=False)
        retvals = getattr(result, "mle_retvals", {}) or {}
        converged = bool(retvals.get("converged", False))
        messages = tuple(f"{item.category.__name__}: {item.message}" for item in captured)
        return ETSFit(result=result, converged=converged, warnings=messages)
    except Exception as exc:  # a failed origin is retained in the raw record and diagnostics
        messages = tuple(f"{item.category.__name__}: {item.message}" for item in captured)
        return ETSFit(result=None, converged=False, warnings=messages, failure_reason=f"{type(exc).__name__}: {exc}")


def _forecast_for_origin(
    training: pd.Series, horizons: tuple[int, ...], fit_function: FitFunction
) -> tuple[ETSFit, dict[int, float], float, int, tuple[str, ...]]:
    """Fit once, then derive each requested horizon and training-only smear factor."""
    fit = fit_function(training)
    issues = list(fit.warnings)
    if fit.result is None:
        if fit.failure_reason:
            issues.append(fit.failure_reason)
        return fit, {}, float("nan"), 0, tuple(issues)

    residuals = np.asarray(getattr(fit.result, "resid", []), dtype=float).reshape(-1)
    finite_residuals = residuals[np.isfinite(residuals)]
    if len(finite_residuals) == 0:
        issues.append("No finite in-sample training residuals were available for smearing.")
        return fit, {}, float("nan"), 0, tuple(issues)
    if len(finite_residuals) != len(residuals):
        issues.append(f"Ignored {len(residuals) - len(finite_residuals)} non-finite training residual(s).")

    try:
        with np.errstate(over="raise", invalid="raise"):
            factor = float(np.exp(finite_residuals).mean())
        if not np.isfinite(factor) or factor <= 0:
            raise FloatingPointError("Smearing factor is non-finite or non-positive.")
    except FloatingPointError as exc:
        issues.append(str(exc))
        return fit, {}, float("nan"), len(finite_residuals), tuple(issues)

    try:
        forecast_warnings: list[warnings.WarningMessage] = []
        with warnings.catch_warnings(record=True) as forecast_warnings:
            warnings.simplefilter("always")
            values = np.asarray(fit.result.forecast(steps=max(horizons)), dtype=float).reshape(-1)
        issues.extend(f"{item.category.__name__}: {item.message}" for item in forecast_warnings)
        requested_values = values[np.asarray(horizons, dtype=int) - 1] if len(values) >= max(horizons) else np.array([])
        if len(values) < max(horizons) or not np.isfinite(requested_values).all():
            raise FloatingPointError("ETS forecast contains missing or non-finite requested steps.")
        log_forecasts = {horizon: float(values[horizon - 1]) for horizon in horizons}
    except Exception as exc:
        issues.append(f"{type(exc).__name__}: {exc}")
        return fit, {}, factor, len(finite_residuals), tuple(issues)

    if not fit.converged:
        issues.append("Optimizer returned a result without reporting convergence.")
    return fit, log_forecasts, factor, len(finite_residuals), tuple(issues)


def build_rolling_forecasts(
    frame: pd.DataFrame,
    fit_function: FitFunction = fit_fixed_ets,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one fixed-specification ETS fit per Stage 3 rolling origin."""
    _validate_frame(frame)
    mase_scales, _ = backtest_sarima.mase_scale_lookup(frame)
    forecast_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    requested = origin_horizons()

    for position, (origin, horizons) in enumerate(requested.items(), start=1):
        training = frame.loc[:origin, "log_retail"].astype(float)
        _require(not training.empty and training.index[-1] == origin, f"Training sample must end at {origin}")
        fit, log_forecasts, factor, n_residuals, issues = _forecast_for_origin(training, horizons, fit_function)
        fit_success = fit.result is not None and len(log_forecasts) == len(horizons) and np.isfinite(factor)
        diag_row = {
            "origin": str(origin),
            "n_train": int(len(training)),
            "fit_success": bool(fit_success),
            "converged": bool(fit.converged),
            "smearing_factor": float(factor) if np.isfinite(factor) else np.nan,
            "n_training_residuals": int(n_residuals),
            "numerical_issues": " | ".join(issues),
            "fit_failure_reason": fit.failure_reason or "",
        }
        diagnostic_rows.append(diag_row)

        for horizon in horizons:
            target = origin + horizon
            log_forecast = log_forecasts.get(horizon, float("nan"))
            with np.errstate(over="ignore", invalid="ignore"):
                level_forecast = float(np.exp(log_forecast) * factor) if fit_success else float("nan")
            actual = float(frame.loc[target, "retail_bn"])
            error = actual - level_forecast if np.isfinite(level_forecast) else np.nan
            scale = mase_scales[(horizon, str(target))]
            forecast_rows.append(
                {
                    "model": MODEL_NAME,
                    "horizon": int(horizon),
                    "origin": str(origin),
                    "target": str(target),
                    "training_start": str(training.index[0]),
                    "training_end": str(training.index[-1]),
                    "n_train": int(len(training)),
                    "fit_success": bool(fit_success),
                    "converged": bool(fit.converged),
                    "fit_failure_reason": fit.failure_reason or "",
                    "numerical_issues": " | ".join(issues),
                    "n_training_residuals": int(n_residuals),
                    "actual": actual,
                    "forecast_log": log_forecast,
                    "smearing_factor": factor,
                    "forecast": level_forecast,
                    "error": error,
                    "absolute_error": abs(error) if np.isfinite(error) else np.nan,
                    "squared_error": error**2 if np.isfinite(error) else np.nan,
                    "mase_scale": scale,
                    "scaled_absolute_error": abs(error) / scale if np.isfinite(error) else np.nan,
                }
            )
        if progress_callback:
            progress_callback(position, len(requested))

    forecasts = pd.DataFrame(forecast_rows, columns=FORECAST_COLUMNS)
    diagnostics = pd.DataFrame(diagnostic_rows)
    _validate_rolling_forecasts(forecasts)
    return forecasts, diagnostics


def _validate_rolling_forecasts(forecasts: pd.DataFrame) -> None:
    targets = [str(value) for value in pd.period_range(evaluation.TARGET_START, evaluation.TARGET_END, freq="Q-DEC")]
    _require(forecasts.columns.tolist() == FORECAST_COLUMNS, "ETS forecast schema changed unexpectedly")
    _require(not forecasts.duplicated(["horizon", "target"]).any(), "ETS forecast contains duplicate targets")
    for horizon in evaluation.HORIZONS:
        subset = forecasts.loc[forecasts["horizon"] == horizon].sort_values("target", kind="stable")
        _require(subset["target"].tolist() == targets and len(subset) == 86, f"ETS h={horizon} target window differs from Stage 3")
        target_index = pd.PeriodIndex(subset["target"], freq="Q-DEC")
        origins = pd.PeriodIndex(subset["origin"], freq="Q-DEC")
        _require((target_index - horizon).equals(origins), f"ETS h={horizon} origins do not equal target minus horizon")
        training_ends = pd.PeriodIndex(subset["training_end"], freq="Q-DEC")
        _require(training_ends.equals(origins), f"ETS h={horizon} training does not end at each origin")
        _require(subset["training_start"].eq(str(evaluation.SAMPLE_START)).all(), "ETS training start differs from Stage 3")
        expected_n = origins.astype("int64") - evaluation.SAMPLE_START.ordinal + 1
        _require(np.array_equal(subset["n_train"].to_numpy(dtype=int), np.asarray(expected_n)), "ETS expanding training sizes are incorrect")


def compute_ets_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Use the Stage 3 metric implementation over finite ETS forecast records."""
    valid = forecasts.loc[
        np.isfinite(pd.to_numeric(forecasts["forecast"], errors="coerce"))
        & np.isfinite(pd.to_numeric(forecasts["actual"], errors="coerce"))
        & np.isfinite(pd.to_numeric(forecasts["mase_scale"], errors="coerce"))
    ].copy()
    if valid.empty:
        return pd.DataFrame(
            [
                {"model": MODEL_NAME, "horizon": horizon, "scope": scope, "n_forecasts": 0,
                 "MAE": np.nan, "RMSE": np.nan, "MASE": np.nan}
                for horizon in evaluation.HORIZONS for scope in ("all", "q1")
            ],
            columns=METRIC_COLUMNS,
        )
    metrics = evaluation.compute_metrics(valid, models=(MODEL_NAME,))
    return metrics[METRIC_COLUMNS]


def _read_model_metrics(path: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "n_forecasts" not in frame.columns and "n" in frame.columns:
        frame = frame.rename(columns={"n": "n_forecasts"})
    _require({"model", "horizon", "scope", "n_forecasts", "MAE", "RMSE", "MASE"}.issubset(frame.columns), f"Unexpected metric schema in {path}")
    result = frame.loc[frame["model"] == model, METRIC_COLUMNS].copy()
    _require(len(result) == len(evaluation.HORIZONS) * 2, f"Expected four Stage 3 metric rows for {model} in {path}")
    _require(not result.duplicated(["horizon", "scope"]).any(), f"Duplicate metric rows for {model} in {path}")
    return result


def build_comparison_table(
    ets_metrics: pd.DataFrame,
    benchmark_metrics_path: Path = BENCHMARK_METRICS_PATH,
    sarima_metrics_path: Path = SARIMA_METRICS_PATH,
    cny_metrics_path: Path = CNY_METRICS_PATH,
) -> pd.DataFrame:
    """Combine frozen Stage 3–5 metrics with ETS without selecting or retuning."""
    reference = [
        _read_model_metrics(benchmark_metrics_path, "Historical Mean"),
        _read_model_metrics(benchmark_metrics_path, "Naive"),
        _read_model_metrics(benchmark_metrics_path, "Seasonal Naive"),
        _read_model_metrics(sarima_metrics_path, "SARIMA"),
        _read_model_metrics(cny_metrics_path, "SARIMA+CNY"),
        ets_metrics[METRIC_COLUMNS].copy(),
    ]
    comparison = pd.concat(reference, ignore_index=True)[METRIC_COLUMNS]
    _require(set(comparison["model"]) == set(COMPARISON_MODELS), "Stage 6B comparison model set is incomplete")
    _require(len(comparison) == len(COMPARISON_MODELS) * len(evaluation.HORIZONS) * 2, "Stage 6B comparison row count is incorrect")
    order = {name: i for i, name in enumerate(COMPARISON_MODELS)}
    scope_order = {"all": 0, "q1": 1}
    comparison["_model_order"] = comparison["model"].map(order)
    comparison["_scope_order"] = comparison["scope"].map(scope_order)
    return comparison.sort_values(["horizon", "_scope_order", "_model_order"], kind="stable").drop(
        columns=["_model_order", "_scope_order"]
    ).reset_index(drop=True)


def _score_subset(subset: pd.DataFrame) -> tuple[int, float | None, float | None, float | None]:
    valid = subset.loc[
        np.isfinite(pd.to_numeric(subset["forecast"], errors="coerce"))
        & np.isfinite(pd.to_numeric(subset["actual"], errors="coerce"))
        & np.isfinite(pd.to_numeric(subset["mase_scale"], errors="coerce"))
    ]
    if valid.empty:
        return 0, None, None, None
    errors = valid["actual"].to_numpy(dtype=float) - valid["forecast"].to_numpy(dtype=float)
    absolute_errors = np.abs(errors)
    scales = valid["mase_scale"].to_numpy(dtype=float)
    return len(valid), float(absolute_errors.mean()), float(np.sqrt(np.square(errors).mean())), float((absolute_errors / scales).mean())


def compute_shock_sensitivity(forecasts_by_model: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Re-score existing h=2 records after filtering only the two fixed target rows."""
    rows: list[dict[str, object]] = []
    expected_targets = [
        str(value) for value in pd.period_range(evaluation.TARGET_START, evaluation.TARGET_END, freq="Q-DEC")
    ]
    reference_actuals: np.ndarray | None = None
    reference_scales: np.ndarray | None = None
    for model in SENSITIVITY_MODELS:
        _require(model in forecasts_by_model, f"Missing original forecast records for {model}")
        forecasts = forecasts_by_model[model]
        _require(forecasts["model"].eq(model).all(), f"Unexpected model name in {model} forecast records")
        h2 = forecasts.loc[forecasts["horizon"] == evaluation.PRIMARY_HORIZON].sort_values("target", kind="stable").copy()
        _require(h2["target"].tolist() == expected_targets, f"Original {model} h=2 records do not match the complete Stage 3 window")
        actuals = h2["actual"].to_numpy(dtype=float)
        scales = h2["mase_scale"].to_numpy(dtype=float)
        if reference_actuals is None:
            reference_actuals, reference_scales = actuals, scales
        else:
            _require(np.allclose(actuals, reference_actuals, rtol=0, atol=1e-8), f"Original {model} actuals differ across models")
            _require(np.allclose(scales, reference_scales, rtol=0, atol=1e-10), f"Original {model} MASE scales differ across models")
        _require(set(SHOCK_TARGETS).issubset(set(h2["target"])), f"Original {model} records must retain both shock targets")
        for scope in ("all", "q1"):
            scoped = h2
            if scope == "q1":
                scoped = scoped.loc[pd.PeriodIndex(scoped["target"], freq="Q-DEC").quarter == 1]
            for sample, sample_rows in (
                ("full", scoped),
                ("exclude_shocks", scoped.loc[~scoped["target"].isin(SHOCK_TARGETS)]),
            ):
                n, mae, rmse, mase = _score_subset(sample_rows)
                rows.append(
                    {
                        "model": model,
                        "horizon": evaluation.PRIMARY_HORIZON,
                        "scope": scope,
                        "sample": sample,
                        "excluded_targets": "|".join(SHOCK_TARGETS) if sample == "exclude_shocks" else "",
                        "n_forecasts": n,
                        "MAE": mae,
                        "RMSE": rmse,
                        "MASE": mase,
                    }
                )
    return pd.DataFrame(rows, columns=SHOCK_METRIC_COLUMNS)


def build_cny_shock_sensitivity(shock_metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate SARIMA+CNY minus SARIMA deltas in each target aggregation."""
    baseline = shock_metrics.loc[shock_metrics["model"] == "SARIMA"].rename(
        columns={"n_forecasts": "n_sarima", "MAE": "sarima_MAE", "RMSE": "sarima_RMSE", "MASE": "sarima_MASE"}
    )
    cny = shock_metrics.loc[shock_metrics["model"] == "SARIMA+CNY"].rename(
        columns={"n_forecasts": "n_cny", "MAE": "cny_MAE", "RMSE": "cny_RMSE", "MASE": "cny_MASE"}
    )
    keys = ["horizon", "scope", "sample"]
    paired = baseline[keys + ["n_sarima", "sarima_MAE", "sarima_RMSE", "sarima_MASE"]].merge(
        cny[keys + ["n_cny", "cny_MAE", "cny_RMSE", "cny_MASE"]],
        on=keys,
        validate="one_to_one",
    )
    _require((paired["n_sarima"] == paired["n_cny"]).all(), "SARIMA and SARIMA+CNY sensitivity sample sizes differ")
    for metric in ("MAE", "RMSE", "MASE"):
        paired[f"delta_{metric}"] = paired[f"cny_{metric}"] - paired[f"sarima_{metric}"]
        paired[f"percent_change_{metric}"] = 100.0 * paired[f"delta_{metric}"] / paired[f"sarima_{metric}"]
    return paired[CNY_SHOCK_COLUMNS].sort_values(["horizon", "scope", "sample"], kind="stable").reset_index(drop=True)


def full_sample_diagnostics(frame: pd.DataFrame, fit_function: FitFunction = fit_fixed_ets) -> pd.DataFrame:
    """Fit ETS on all observed data for description only; do not forecast."""
    training = frame["log_retail"].astype(float)
    fit = fit_function(training)
    result = fit.result
    residuals = np.asarray(getattr(result, "resid", []), dtype=float).reshape(-1) if result is not None else np.array([])
    finite_residuals = residuals[np.isfinite(residuals)]
    lb_stat = lb_pvalue = None
    if len(finite_residuals) > 8:
        try:
            lb = acorr_ljungbox(finite_residuals, lags=[8], return_df=True)
            lb_stat = float(lb["lb_stat"].iloc[0])
            lb_pvalue = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            lb_stat = lb_pvalue = None

    def finite_attr(name: str) -> float | None:
        if result is None:
            return None
        try:
            value = float(getattr(result, name))
        except (AttributeError, TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    llf, aic, sse = finite_attr("llf"), finite_attr("aic"), finite_attr("sse")
    fit_success = result is not None and len(finite_residuals) == len(training)
    aic_reliable = bool(fit_success and fit.converged and llf is not None and aic is not None)
    return pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "response": "log_retail",
                "error": FIXED_ETS_SPEC["error"],
                "trend": FIXED_ETS_SPEC["trend"],
                "damped_trend": FIXED_ETS_SPEC["damped_trend"],
                "seasonal": FIXED_ETS_SPEC["seasonal"],
                "seasonal_periods": FIXED_ETS_SPEC["seasonal_periods"],
                "n_obs": len(training),
                "fit_success": bool(fit_success),
                "converged": bool(fit.converged),
                "failure_reason": fit.failure_reason or "",
                "sse": sse,
                "llf": llf,
                "aic": aic if aic_reliable else None,
                "aic_reliable": aic_reliable,
                "residual_mean": float(finite_residuals.mean()) if len(finite_residuals) else None,
                "residual_std": float(finite_residuals.std(ddof=1)) if len(finite_residuals) > 1 else None,
                "n_finite_residuals": len(finite_residuals),
                "ljung_box_lag": 8,
                "ljung_box_model_df": 0,
                "ljung_box_statistic": lb_stat,
                "ljung_box_pvalue": lb_pvalue,
                "warnings": " | ".join(fit.warnings),
                "descriptive_only": True,
                "used_for_rolling_forecasts": False,
                "forecast_created": False,
            }
        ]
    )


def plot_h2_actual_and_ets(forecasts: pd.DataFrame, path: Path) -> None:
    h2 = forecasts.loc[forecasts["horizon"] == evaluation.PRIMARY_HORIZON].sort_values("target", kind="stable")
    dates = pd.PeriodIndex(h2["target"], freq="Q-DEC").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, h2["actual"], color="#1f2937", linewidth=2.0, label="Actual")
    ax.plot(dates, h2["forecast"], color="#7c3aed", linewidth=1.25, label="ETS(A,Ad,A)")
    ax.set_title("Actual retail and fixed ETS forecasts (h = 2)")
    ax.set_ylabel("Retail sales (RMB 100 million)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "STA4003 Stage 6B"})
    plt.close(fig)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.15g", lineterminator="\n")


def _metric_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            {
                "model": str(row["model"]),
                "horizon": int(row["horizon"]),
                "scope": str(row["scope"]),
                "n_forecasts": int(row["n_forecasts"]),
                "MAE": float(row["MAE"]) if pd.notna(row["MAE"]) else None,
                "RMSE": float(row["RMSE"]) if pd.notna(row["RMSE"]) else None,
                "MASE": float(row["MASE"]) if pd.notna(row["MASE"]) else None,
            }
        )
    return records


def _json_table_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        clean: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, (bool, np.bool_)):
                clean[key] = bool(value)
            elif isinstance(value, (np.integer, int)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean[key] = float(value) if np.isfinite(value) else None
            elif pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records


def run_backtest(
    output_root: Path = PROJECT_ROOT,
    data_path: Path = DATA_PATH,
    benchmark_metrics_path: Path = BENCHMARK_METRICS_PATH,
    sarima_forecast_path: Path = SARIMA_FORECAST_PATH,
    sarima_metrics_path: Path = SARIMA_METRICS_PATH,
    cny_forecast_path: Path = CNY_FORECAST_PATH,
    cny_metrics_path: Path = CNY_METRICS_PATH,
    fit_function: FitFunction = fit_fixed_ets,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run ETS rolling forecasts, comparison tables, shock scoring, and diagnostics."""
    started = time.perf_counter()
    frame = load_analysis_data(data_path)
    forecasts, rolling_diagnostics = build_rolling_forecasts(frame, fit_function, progress_callback)
    ets_metrics = compute_ets_metrics(forecasts)
    comparison = build_comparison_table(
        ets_metrics,
        benchmark_metrics_path=benchmark_metrics_path,
        sarima_metrics_path=sarima_metrics_path,
        cny_metrics_path=cny_metrics_path,
    )

    sarima_forecasts = pd.read_csv(sarima_forecast_path, dtype={"origin": str, "target": str})
    cny_forecasts = pd.read_csv(cny_forecast_path, dtype={"origin": str, "target": str})
    shock_frames = {"SARIMA": sarima_forecasts, "SARIMA+CNY": cny_forecasts, MODEL_NAME: forecasts}
    shock_metrics = compute_shock_sensitivity(shock_frames)
    cny_shock_metrics = build_cny_shock_sensitivity(shock_metrics)
    full_diagnostics = full_sample_diagnostics(frame, fit_function)

    out = Path(output_root) / "outputs"
    paths = {
        "forecast": out / "forecasts" / "ets_rolling_forecasts.csv",
        "metrics": out / "tables" / "ets_metrics.csv",
        "comparison": out / "tables" / "model_comparison_stage6b.csv",
        "shock_sensitivity": out / "tables" / "shock_sensitivity_metrics.csv",
        "cny_shock_sensitivity": out / "tables" / "cny_shock_sensitivity.csv",
        "full_sample_diagnostics": out / "tables" / "ets_full_sample_diagnostics.csv",
        "summary": out / "diagnostics" / "stage6b_summary.json",
        "figure": out / "figures" / "ets_h2_actual_vs_forecast.png",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(forecasts, paths["forecast"])
    _write_csv(ets_metrics, paths["metrics"])
    _write_csv(comparison, paths["comparison"])
    _write_csv(shock_metrics, paths["shock_sensitivity"])
    _write_csv(cny_shock_metrics, paths["cny_shock_sensitivity"])
    _write_csv(full_diagnostics, paths["full_sample_diagnostics"])
    plot_h2_actual_and_ets(forecasts, paths["figure"])

    successful = int(rolling_diagnostics["fit_success"].sum())
    converged = int(rolling_diagnostics["converged"].sum())
    factors = rolling_diagnostics.loc[rolling_diagnostics["fit_success"], "smearing_factor"].astype(float)
    issue_records = rolling_diagnostics.loc[rolling_diagnostics["numerical_issues"].ne("")]
    shock_records = {
        model: {
            target: int(
                ((frame_model["horizon"] == evaluation.PRIMARY_HORIZON) & (frame_model["target"] == target)).sum()
            )
            for target in SHOCK_TARGETS
        }
        for model, frame_model in shock_frames.items()
    }
    summary: dict[str, object] = {
        "stage": "6B",
        "response": "log_retail",
        "fixed_ets_specification": FIXED_ETS_SPEC,
        "model_search": False,
        "rolling_protocol": {
            "target_start": str(evaluation.TARGET_START),
            "target_end": str(evaluation.TARGET_END),
            "target_count_per_horizon": 86,
            "horizons": list(evaluation.HORIZONS),
            "primary_endpoint": {"horizon": evaluation.PRIMARY_HORIZON, "scope": "all"},
            "training_start": str(evaluation.SAMPLE_START),
            "training_end_rule": "training ends at target minus horizon, inclusive",
            "unique_origins": len(rolling_diagnostics),
            "fit_once_per_origin": True,
        },
        "rolling_fit_diagnostics": {
            "fit_success_count": successful,
            "fit_failure_count": int(len(rolling_diagnostics) - successful),
            "converged_count": converged,
            "nonconverged_count": int(len(rolling_diagnostics) - converged),
            "numerical_issue_origin_count": int(len(issue_records)),
            "numerical_issue_records": _json_table_records(issue_records),
            "smearing_factor_min": float(factors.min()) if len(factors) else None,
            "smearing_factor_max": float(factors.max()) if len(factors) else None,
            "smearing_factor_count": int(len(factors)),
            "smearing_rule": "mean(exp(training ETS one-step residuals)) at each origin; finite residuals only",
            "fit_diagnostics": _json_table_records(rolling_diagnostics),
        },
        "forecast_convention": {
            "log_forecast": "ETS conditional forecast on log_retail scale",
            "level_forecast": "exp(log forecast) * origin-specific training-only residual smearing factor",
            "mase": "Stage 3 build_rolling_forecasts origin-specific training-only seasonal-period-4 scale",
            "score": "actual retail level minus level forecast; MAE, RMSE, MASE",
        },
        "metrics": _metric_records(ets_metrics),
        "model_comparison": _json_table_records(comparison.loc[comparison["horizon"] == evaluation.PRIMARY_HORIZON]),
        "shock_sensitivity": {
            "preset_targets": list(SHOCK_TARGETS),
            "aggregation_only": True,
            "refit_or_data_deletion": False,
            "original_h2_shock_records_retained": shock_records,
            "metrics": _json_table_records(shock_metrics),
            "cny_minus_sarima": _json_table_records(cny_shock_metrics),
        },
        "full_sample_descriptive_fit": _json_table_records(full_diagnostics)[0],
        "interpretation_guardrails": {
            "primary_endpoint_unchanged": True,
            "shock_sensitivity_replaces_primary": False,
            "cny_or_ets_conclusion_is_causal": False,
            "full_sample_fit_used_for_rolling_forecasts": False,
        },
        "generated_artifacts": [str(path.relative_to(Path(output_root))) for path in paths.values()],
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    summary["runtime_seconds"] = round(time.perf_counter() - started, 3)
    return summary


def main() -> None:
    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"Stage 6B ETS origins: {done}/{total}", flush=True)

    started = time.perf_counter()
    summary = run_backtest(progress_callback=progress)
    fit = summary["rolling_fit_diagnostics"]
    print(
        f"Stage 6B ETS complete: {fit['fit_success_count']} successful, "
        f"{fit['fit_failure_count']} failed, {fit['converged_count']} converged; "
        f"elapsed {time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
