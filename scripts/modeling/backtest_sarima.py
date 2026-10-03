"""Leak-free per-origin SARIMA selection and rolling evaluation for Stage 4."""

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
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from scripts.modeling import backtest_benchmarks as evaluation


DATA_PATH = evaluation.DATA_PATH
TARGET_START = evaluation.TARGET_START
TARGET_END = evaluation.TARGET_END
SAMPLE_START = evaluation.SAMPLE_START
SAMPLE_END = evaluation.SAMPLE_END
HORIZONS = evaluation.HORIZONS
PRIMARY_HORIZON = evaluation.PRIMARY_HORIZON
SEASONAL_PERIOD = evaluation.SEASONAL_PERIOD
LJUNG_BOX_LAG = 8
MAX_ITERATIONS = 100
MODEL_NAME = "SARIMA"

FORECAST_COLUMNS = [
    "model",
    "horizon",
    "origin",
    "target",
    "training_start",
    "training_end",
    "n_train",
    "p",
    "d",
    "q",
    "P",
    "D",
    "Q",
    "seasonal_period",
    "aic",
    "aicc",
    "converged",
    "actual",
    "forecast_log_mean",
    "forecast_log_variance",
    "forecast_level_median",
    "forecast",
    "error",
    "absolute_error",
    "squared_error",
    "mase_scale",
    "scaled_absolute_error",
]
SELECTION_COLUMNS = [
    "origin",
    "n_train",
    "training_start",
    "training_end",
    "selected_order",
    "selected_seasonal_order",
    "selected_D",
    "aic",
    "aicc",
    "candidate_count",
    "successful_fit_count",
    "failed_fit_count",
    "ljung_box_lag",
    "ljung_box_model_df",
    "ljung_box_statistic",
    "ljung_box_pvalue",
    "ljung_box_n_residuals",
]
METRIC_COLUMNS = evaluation.METRIC_COLUMNS
FULL_DIAGNOSTIC_COLUMNS = [
    "sample_start",
    "sample_end",
    "n_train",
    "selected_order",
    "selected_seasonal_order",
    "selected_D",
    "aic",
    "aicc",
    "llf",
    "parameter_count",
    "parameter_estimates",
    "converged",
    "candidate_count",
    "successful_fit_count",
    "failed_fit_count",
    "ljung_box_lag",
    "ljung_box_model_df",
    "ljung_box_statistic",
    "ljung_box_pvalue",
    "ljung_box_n_residuals",
    "residual_mean",
    "residual_std",
    "deterministic_trend",
    "simple_differencing",
]


@dataclass(frozen=True)
class SarimaCandidate:
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
        return (self.P, self.D, self.Q, SEASONAL_PERIOD)

    @property
    def complexity(self) -> int:
        return self.p + self.q + self.P + self.Q

    @property
    def order_label(self) -> str:
        return f"({self.p},{self.d},{self.q})"

    @property
    def seasonal_order_label(self) -> str:
        return f"({self.P},{self.D},{self.Q},{SEASONAL_PERIOD})"


@dataclass(frozen=True)
class CandidateFit:
    candidate: SarimaCandidate
    result: object
    aic: float
    aicc: float
    llf: float
    nobs: int
    parameter_count: int
    converged: bool


@dataclass(frozen=True)
class ModelSelection:
    selected_fit: CandidateFit
    candidate_count: int
    successful_fit_count: int
    failed_fit_count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def generate_candidate_grid() -> tuple[SarimaCandidate, ...]:
    """Return exactly the pre-specified SARIMA family."""
    candidates = tuple(
        SarimaCandidate(p=p, d=1, q=q, P=P, D=D, Q=Q)
        for D in (0, 1)
        for p in range(3)
        for q in range(3)
        for P in range(2)
        for Q in range(2)
        if p + q + P + Q <= 3
    )
    _require(len(candidates) == 46, f"Candidate grid changed: expected 46, got {len(candidates)}")
    return candidates


def calculate_aicc(aic: float, parameter_count: int, nobs: int) -> float | None:
    """Calculate AICc using k estimated parameters and n training observations."""
    if parameter_count < 0 or nobs - parameter_count - 1 <= 0 or not np.isfinite(aic):
        return None
    value = float(aic) + (
        2.0 * parameter_count * (parameter_count + 1) / (nobs - parameter_count - 1)
    )
    return float(value) if np.isfinite(value) else None


def lognormal_level_forecasts(log_mean: float, log_variance: float) -> tuple[float, float]:
    """Return lognormal median and conditional-mean level forecasts."""
    _require(np.isfinite([log_mean, log_variance]).all(), "Log forecast moments must be finite")
    _require(log_variance >= 0, "Log forecast variance must be non-negative")
    with np.errstate(over="ignore", invalid="ignore"):
        median = float(np.exp(log_mean))
        mean = float(np.exp(log_mean + 0.5 * log_variance))
    _require(np.isfinite([median, mean]).all(), "Retransformed level forecast is non-finite")
    return median, mean


def load_analysis_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the canonical level and log response and verify their alignment."""
    level_frame = evaluation.load_canonical(path)
    log_frame = pd.read_csv(path, usecols=["quarter", "log_retail"], keep_default_na=False)
    log_index = pd.PeriodIndex(log_frame["quarter"].astype(str), freq="Q-DEC", name="quarter")
    _require(log_index.equals(level_frame.index), "Log and level quarter indexes do not match")
    log_retail = pd.to_numeric(log_frame["log_retail"], errors="raise").to_numpy(dtype=float)
    _require(np.isfinite(log_retail).all(), "Canonical log_retail must be finite")
    _require(
        np.allclose(log_retail, np.log(level_frame["retail_bn"]), rtol=0.0, atol=5.1e-7),
        "Canonical log_retail does not match retail_bn",
    )
    level_frame["log_retail"] = log_retail
    return level_frame


def fit_candidate(training: pd.Series, candidate: SarimaCandidate) -> CandidateFit | None:
    """Fit one state-space candidate; reject failed, non-converged, or invalid fits."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                training,
                order=candidate.order,
                seasonal_order=candidate.seasonal_order,
                trend="n",
                simple_differencing=False,
                enforce_stationarity=True,
                enforce_invertibility=True,
                concentrate_scale=False,
            )
            result = model.fit(method="lbfgs", disp=False, maxiter=MAX_ITERATIONS)
        converged = bool(result.mle_retvals.get("converged", False))
        llf = float(result.llf)
        aic = float(result.aic)
        nobs = int(round(float(result.nobs)))
        parameter_count = int(np.asarray(result.params).size)
        aicc = calculate_aicc(aic, parameter_count, nobs)
        if (
            not converged
            or aicc is None
            or not np.isfinite([llf, aic, aicc]).all()
            or nobs != len(training)
        ):
            return None
        return CandidateFit(
            candidate=candidate,
            result=result,
            aic=aic,
            aicc=aicc,
            llf=llf,
            nobs=nobs,
            parameter_count=parameter_count,
            converged=converged,
        )
    except Exception:
        return None


def select_model(
    training: pd.Series,
    candidates: tuple[SarimaCandidate, ...] | None = None,
    fit_function: Callable[[pd.Series, SarimaCandidate], CandidateFit | None] = fit_candidate,
) -> ModelSelection:
    """Select the training-sample AICc minimum; no evaluation target is accepted."""
    candidate_grid = candidates if candidates is not None else generate_candidate_grid()
    successful: list[CandidateFit] = []
    for candidate in candidate_grid:
        try:
            fitted = fit_function(training, candidate)
        except Exception:
            fitted = None
        if fitted is not None and fitted.converged and np.isfinite(fitted.aicc):
            successful.append(fitted)
    if not successful:
        raise RuntimeError(f"No valid converged SARIMA candidate for training end {training.index[-1]}")
    selected = min(
        successful,
        key=lambda fitted: (
            fitted.aicc,
            fitted.candidate.order,
            fitted.candidate.seasonal_order,
        ),
    )
    return ModelSelection(
        selected_fit=selected,
        candidate_count=len(candidate_grid),
        successful_fit_count=len(successful),
        failed_fit_count=len(candidate_grid) - len(successful),
    )


def _diagnostic_residuals(fitted: CandidateFit) -> np.ndarray:
    residuals = np.asarray(fitted.result.resid, dtype=float).reshape(-1)
    burn = int(getattr(fitted.result.model, "loglikelihood_burn", 0) or 0)
    residuals = residuals[burn:]
    residuals = residuals[np.isfinite(residuals)]
    _require(len(residuals) > LJUNG_BOX_LAG, "Too few finite residuals for lag-8 Ljung-Box")
    return residuals


def ljung_box_diagnostic(fitted: CandidateFit) -> dict[str, float | int]:
    """Calculate a lag-8 Ljung-Box diagnostic, adjusting for AR/MA terms."""
    residuals = _diagnostic_residuals(fitted)
    model_df = fitted.candidate.complexity
    diagnostic = acorr_ljungbox(
        residuals,
        lags=[LJUNG_BOX_LAG],
        model_df=model_df,
        return_df=True,
    ).iloc[0]
    return {
        "ljung_box_lag": LJUNG_BOX_LAG,
        "ljung_box_model_df": int(model_df),
        "ljung_box_statistic": float(diagnostic["lb_stat"]),
        "ljung_box_pvalue": float(diagnostic["lb_pvalue"]),
        "ljung_box_n_residuals": int(len(residuals)),
    }


def _selection_row(origin: pd.Period, training: pd.Series, selection: ModelSelection) -> dict[str, object]:
    fitted = selection.selected_fit
    candidate = fitted.candidate
    return {
        "origin": str(origin),
        "n_train": len(training),
        "training_start": str(training.index[0]),
        "training_end": str(training.index[-1]),
        "selected_order": candidate.order_label,
        "selected_seasonal_order": candidate.seasonal_order_label,
        "selected_D": candidate.D,
        "aic": fitted.aic,
        "aicc": fitted.aicc,
        "candidate_count": selection.candidate_count,
        "successful_fit_count": selection.successful_fit_count,
        "failed_fit_count": selection.failed_fit_count,
        **ljung_box_diagnostic(fitted),
    }


def origin_horizons() -> dict[pd.Period, tuple[int, ...]]:
    """Map each unique rolling origin to the horizons it serves."""
    requested: dict[pd.Period, set[int]] = {}
    targets = pd.period_range(TARGET_START, TARGET_END, freq="Q-DEC")
    for horizon in HORIZONS:
        for target in targets:
            origin = target - horizon
            requested.setdefault(origin, set()).add(horizon)
    return {origin: tuple(sorted(values)) for origin, values in sorted(requested.items())}


def mase_scale_lookup(frame: pd.DataFrame) -> tuple[dict[tuple[int, str], float], pd.DataFrame]:
    """Take MASE denominators directly from the Stage 3 implementation."""
    reference = evaluation.build_rolling_forecasts(frame)
    scales = reference.loc[reference["model"] == "Naive", ["horizon", "target", "mase_scale"]]
    lookup = {
        (int(row.horizon), str(row.target)): float(row.mase_scale)
        for row in scales.itertuples(index=False)
    }
    _require(len(lookup) == len(HORIZONS) * 86, "Stage 3 MASE scale lookup is incomplete")
    return lookup, reference


def build_rolling_forecasts(
    frame: pd.DataFrame,
    selection_function: Callable[[pd.Series], ModelSelection] = select_model,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select once per unique origin and forecast the requested one/two-step targets."""
    _require(frame.index.equals(evaluation.expected_quarters()), "Input quarters do not match Stage 3 protocol")
    _require({"retail_bn", "log_retail"}.issubset(frame.columns), "Input must contain retail_bn and log_retail")
    mase_scales, benchmark_forecasts = mase_scale_lookup(frame)
    requested = origin_horizons()
    forecast_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for position, (origin, horizons) in enumerate(requested.items(), start=1):
        training = frame.loc[:origin, "log_retail"].astype(float)
        _require(training.index[-1] == origin, f"Training data does not end at {origin}")
        selection = selection_function(training)
        fitted = selection.selected_fit
        candidate = fitted.candidate
        prediction = fitted.result.get_forecast(steps=max(horizons))
        log_means = np.asarray(prediction.predicted_mean, dtype=float).reshape(-1)
        log_variances = np.asarray(prediction.var_pred_mean, dtype=float).reshape(-1)
        selection_rows.append(_selection_row(origin, training, selection))

        for horizon in horizons:
            target = origin + horizon
            _require(TARGET_START <= target <= TARGET_END, "Origin map created an out-of-window target")
            log_mean = float(log_means[horizon - 1])
            log_variance = float(log_variances[horizon - 1])
            median, level_mean = lognormal_level_forecasts(log_mean, log_variance)
            actual = float(frame.loc[target, "retail_bn"])
            error = actual - level_mean
            scale = mase_scales[(horizon, str(target))]
            forecast_rows.append(
                {
                    "model": MODEL_NAME,
                    "horizon": horizon,
                    "origin": str(origin),
                    "target": str(target),
                    "training_start": str(training.index[0]),
                    "training_end": str(training.index[-1]),
                    "n_train": len(training),
                    "p": candidate.p,
                    "d": candidate.d,
                    "q": candidate.q,
                    "P": candidate.P,
                    "D": candidate.D,
                    "Q": candidate.Q,
                    "seasonal_period": SEASONAL_PERIOD,
                    "aic": fitted.aic,
                    "aicc": fitted.aicc,
                    "converged": fitted.converged,
                    "actual": actual,
                    "forecast_log_mean": log_mean,
                    "forecast_log_variance": log_variance,
                    "forecast_level_median": median,
                    "forecast": level_mean,
                    "error": error,
                    "absolute_error": abs(error),
                    "squared_error": error**2,
                    "mase_scale": scale,
                    "scaled_absolute_error": abs(error) / scale,
                }
            )
        if progress_callback is not None:
            progress_callback(position, len(requested))

    forecasts = pd.DataFrame(forecast_rows, columns=FORECAST_COLUMNS)
    selection_table = pd.DataFrame(selection_rows, columns=SELECTION_COLUMNS)
    forecasts = forecasts.sort_values(["horizon", "target"], kind="stable").reset_index(drop=True)
    selection_table = selection_table.sort_values("origin", kind="stable").reset_index(drop=True)
    _validate_rolling_outputs(forecasts, selection_table, benchmark_forecasts)
    return forecasts, selection_table, benchmark_forecasts


def _validate_rolling_outputs(
    forecasts: pd.DataFrame, selections: pd.DataFrame, benchmark_forecasts: pd.DataFrame
) -> None:
    targets = [str(value) for value in pd.period_range(TARGET_START, TARGET_END, freq="Q-DEC")]
    _require(forecasts.columns.tolist() == FORECAST_COLUMNS, "Unexpected SARIMA forecast columns")
    _require(not forecasts.duplicated(["model", "horizon", "target"]).any(), "Duplicate SARIMA target record")
    _require(forecasts["model"].eq(MODEL_NAME).all(), "Unexpected model label in SARIMA records")
    for horizon in HORIZONS:
        subset = forecasts.loc[forecasts["horizon"] == horizon]
        _require(subset["target"].tolist() == targets, f"SARIMA target sequence changed at h={horizon}")
        origins = pd.PeriodIndex(subset["origin"], freq="Q-DEC")
        target_periods = pd.PeriodIndex(subset["target"], freq="Q-DEC")
        _require((target_periods - horizon).equals(origins), f"SARIMA target/origin alignment failed at h={horizon}")
        _require(pd.PeriodIndex(subset["training_end"], freq="Q-DEC").equals(origins), "Training end does not equal origin")
        _require(subset["training_start"].eq(str(SAMPLE_START)).all(), "SARIMA training start changed")
        expected_n = (origins.astype("int64") - SAMPLE_START.ordinal + 1).to_numpy()
        _require(np.array_equal(subset["n_train"].to_numpy(dtype=int), expected_n), "SARIMA training sizes are incorrect")
        _require(len(subset) == 86, f"SARIMA forecast count is not 86 at h={horizon}")
        _require(subset["converged"].astype(bool).all(), "A selected rolling model did not converge")
        _require(np.isfinite(subset[["actual", "forecast", "mase_scale", "scaled_absolute_error"]]).all().all(), "Non-finite rolling record")
    for horizon in HORIZONS:
        baseline_targets = benchmark_forecasts.loc[
            (benchmark_forecasts["model"] == "Naive") & (benchmark_forecasts["horizon"] == horizon), "target"
        ].tolist()
        sarima_targets = forecasts.loc[forecasts["horizon"] == horizon, "target"].tolist()
        _require(sarima_targets == baseline_targets, f"SARIMA and Stage 3 targets differ at h={horizon}")
    expected_origins = [str(value) for value in origin_horizons()]
    _require(selections["origin"].tolist() == expected_origins, "Selection diagnostics do not have one row per unique origin")
    _require(not selections["origin"].duplicated().any(), "Duplicate SARIMA selection origin")
    _require((selections["candidate_count"] == len(generate_candidate_grid())).all(), "Candidate count changed")
    _require((selections["successful_fit_count"] + selections["failed_fit_count"] == selections["candidate_count"]).all(), "Fit counts do not reconcile")
    _require(selections["training_end"].tolist() == selections["origin"].tolist(), "Selection training end is not its origin")


def compute_sarima_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Score SARIMA through the shared Stage 3 metric implementation."""
    return evaluation.compute_metrics(forecasts, models=(MODEL_NAME,))


def compute_order_frequency(selections: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected order frequencies without using OOS performance."""
    total_origins = len(selections)
    order_groups = (
        selections.groupby(
            ["selected_order", "selected_seasonal_order", "selected_D"],
            as_index=False,
            sort=True,
        )
        .size()
        .rename(columns={"size": "count"})
        .sort_values(
            ["count", "selected_order", "selected_seasonal_order"],
            ascending=[False, True, True],
            kind="stable",
        )
    )
    rows: list[dict[str, object]] = []
    for row in order_groups.itertuples(index=False):
        rows.append(
            {
                "selection_type": "order",
                "selected_order": row.selected_order,
                "selected_seasonal_order": row.selected_seasonal_order,
                "selected_D": int(row.selected_D),
                "count": int(row.count),
                "share": float(row.count / total_origins),
            }
        )
    for seasonal_difference in (0, 1):
        count = int((selections["selected_D"] == seasonal_difference).sum())
        rows.append(
            {
                "selection_type": "D",
                "selected_order": "",
                "selected_seasonal_order": f"D={seasonal_difference}",
                "selected_D": seasonal_difference,
                "count": count,
                "share": float(count / total_origins),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "selection_type",
            "selected_order",
            "selected_seasonal_order",
            "selected_D",
            "count",
            "share",
        ],
    )


def full_sample_diagnostics(
    log_retail: pd.Series, selection: ModelSelection
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Create descriptive full-sample fit diagnostics; do not make forecasts."""
    fitted = selection.selected_fit
    candidate = fitted.candidate
    residuals = _diagnostic_residuals(fitted)
    ljung_box = ljung_box_diagnostic(fitted)
    parameter_values = {
        str(name): float(value)
        for name, value in zip(fitted.result.model.param_names, np.asarray(fitted.result.params, dtype=float))
    }
    row = {
        "sample_start": str(log_retail.index[0]),
        "sample_end": str(log_retail.index[-1]),
        "n_train": len(log_retail),
        "selected_order": candidate.order_label,
        "selected_seasonal_order": candidate.seasonal_order_label,
        "selected_D": candidate.D,
        "aic": fitted.aic,
        "aicc": fitted.aicc,
        "llf": fitted.llf,
        "parameter_count": fitted.parameter_count,
        "parameter_estimates": json.dumps(parameter_values, sort_keys=True, separators=(",", ":"), allow_nan=False),
        "converged": fitted.converged,
        "candidate_count": selection.candidate_count,
        "successful_fit_count": selection.successful_fit_count,
        "failed_fit_count": selection.failed_fit_count,
        **ljung_box,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)),
        "deterministic_trend": "n",
        "simple_differencing": False,
    }
    return pd.DataFrame([row], columns=FULL_DIAGNOSTIC_COLUMNS), residuals, int(
        getattr(fitted.result.model, "loglikelihood_burn", 0) or 0
    )


def _save_figure(path: Path) -> None:
    plt.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "STA4003 Stage 4"})
    plt.close()


def plot_h2_actual_and_forecasts(forecasts: pd.DataFrame, path: Path) -> None:
    subset = forecasts.loc[forecasts["horizon"] == PRIMARY_HORIZON]
    dates = pd.PeriodIndex(subset["target"], freq="Q-DEC").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, subset["actual"], color="#1f2937", linewidth=2.0, label="Actual")
    ax.plot(dates, subset["forecast"], color="#7c3aed", linewidth=1.4, label="SARIMA conditional mean")
    ax.set_title("Actual retail and SARIMA forecasts (h = 2)")
    ax.set_ylabel("Retail sales (RMB 100 million)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(path)


def plot_h2_errors(forecasts: pd.DataFrame, path: Path) -> None:
    subset = forecasts.loc[forecasts["horizon"] == PRIMARY_HORIZON]
    dates = pd.PeriodIndex(subset["target"], freq="Q-DEC").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, subset["error"], color="#7c3aed", linewidth=1.2)
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_title("SARIMA forecast errors (actual − bias-corrected mean, h = 2)")
    ax.set_ylabel("Error (RMB 100 million)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(path)


def plot_full_sample_residuals(
    log_retail: pd.Series, diagnostic_row: pd.Series, residuals: np.ndarray, burn: int, path: Path
) -> None:
    residual_index = log_retail.index[burn : burn + len(residuals)]
    candidate_label = f"SARIMA{diagnostic_row['selected_order']}{diagnostic_row['selected_seasonal_order']}"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    dates = residual_index.to_timestamp(how="end")
    axes[0].plot(dates, residuals, color="#2563eb", linewidth=1.0)
    axes[0].axhline(0, color="#111827", linewidth=0.8)
    axes[0].set_title("Selected-model residuals")
    axes[0].xaxis.set_major_locator(mdates.YearLocator(8))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].grid(axis="y", alpha=0.2)

    plot_acf(residuals, lags=20, zero=False, ax=axes[1], title="Residual ACF (20 lags)")
    axes[1].grid(axis="y", alpha=0.2)

    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Residual normal Q-Q plot")
    axes[2].grid(alpha=0.2)
    fig.suptitle(f"Full-sample descriptive diagnostics: {candidate_label}", y=1.02)
    fig.tight_layout()
    _save_figure(path)


def _native_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for source in frame.to_dict(orient="records"):
        row: dict[str, object] = {}
        for key, value in source.items():
            if isinstance(value, (bool, np.bool_)):
                row[key] = bool(value)
            elif isinstance(value, (int, np.integer)):
                row[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                row[key] = float(value)
            elif pd.isna(value):
                row[key] = None
            else:
                row[key] = str(value)
        records.append(row)
    return records


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(
        frame.to_csv(index=False, float_format="%.15g", lineterminator="\n"),
        encoding="utf-8",
        newline="",
    )


def run_backtest(
    output_root: Path = PROJECT_ROOT,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run rolling-origin selection, full-sample diagnostics, scores, and outputs."""
    frame = load_analysis_data()
    forecasts, selections, benchmark_forecasts = build_rolling_forecasts(
        frame, progress_callback=progress_callback
    )

    full_selection = select_model(frame["log_retail"])
    full_diagnostic, full_residuals, residual_burn = full_sample_diagnostics(
        frame["log_retail"], full_selection
    )
    sarima_metrics = compute_sarima_metrics(forecasts)
    benchmark_metrics = evaluation.compute_metrics(benchmark_forecasts)
    model_comparison = pd.concat([benchmark_metrics, sarima_metrics], ignore_index=True)
    order_frequency = compute_order_frequency(selections)

    output_root = Path(output_root)
    forecast_path = output_root / "outputs" / "forecasts" / "sarima_rolling_forecasts.csv"
    selection_path = output_root / "outputs" / "tables" / "sarima_origin_selection.csv"
    frequency_path = output_root / "outputs" / "tables" / "sarima_order_frequency.csv"
    full_diagnostic_path = output_root / "outputs" / "tables" / "sarima_full_sample_diagnostics.csv"
    metrics_path = output_root / "outputs" / "tables" / "sarima_metrics.csv"
    comparison_path = output_root / "outputs" / "tables" / "model_comparison_stage4.csv"
    figure_dir = output_root / "outputs" / "figures"
    summary_path = output_root / "outputs" / "diagnostics" / "stage4_summary.json"
    for directory in (
        forecast_path.parent,
        selection_path.parent,
        figure_dir,
        summary_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    actual_figure = figure_dir / "sarima_h2_actual_vs_forecast.png"
    error_figure = figure_dir / "sarima_h2_errors.png"
    residual_figure = figure_dir / "sarima_full_sample_residual_diagnostics.png"
    plot_h2_actual_and_forecasts(forecasts, actual_figure)
    plot_h2_errors(forecasts, error_figure)
    plot_full_sample_residuals(
        frame["log_retail"],
        full_diagnostic.iloc[0],
        full_residuals,
        residual_burn,
        residual_figure,
    )

    for table, path in (
        (forecasts, forecast_path),
        (selections, selection_path),
        (order_frequency, frequency_path),
        (full_diagnostic, full_diagnostic_path),
        (sarima_metrics, metrics_path),
        (model_comparison, comparison_path),
    ):
        _write_csv(table, path)

    d_counts = {
        str(value): int((selections["selected_D"] == value).sum())
        for value in (0, 1)
    }
    order_rows = order_frequency.loc[order_frequency["selection_type"] == "order"]
    most_common_orders = _native_records(order_rows.head(10))
    full_row = full_diagnostic.iloc[0]
    generated_artifacts = [
        "outputs/forecasts/sarima_rolling_forecasts.csv",
        "outputs/tables/sarima_origin_selection.csv",
        "outputs/tables/sarima_order_frequency.csv",
        "outputs/tables/sarima_full_sample_diagnostics.csv",
        "outputs/tables/sarima_metrics.csv",
        "outputs/tables/model_comparison_stage4.csv",
        "outputs/figures/sarima_h2_actual_vs_forecast.png",
        "outputs/figures/sarima_h2_errors.png",
        "outputs/figures/sarima_full_sample_residual_diagnostics.png",
        "outputs/diagnostics/stage4_summary.json",
    ]
    candidate_grid = generate_candidate_grid()
    summary: dict[str, object] = {
        "stage": 4,
        "response": "log_retail",
        "forecast_scoring_scale": "Original retail level in RMB 100 million (亿元).",
        "evaluation_window": {"start": str(TARGET_START), "end": str(TARGET_END)},
        "horizons": list(HORIZONS),
        "primary_horizon": PRIMARY_HORIZON,
        "secondary_horizon": 1,
        "primary_scope": "all",
        "seasonal_period": SEASONAL_PERIOD,
        "candidate_family": {
            "model": "statsmodels state-space SARIMAX",
            "nonseasonal_p_values": [0, 1, 2],
            "d": 1,
            "nonseasonal_q_values": [0, 1, 2],
            "seasonal_P_values": [0, 1],
            "seasonal_D_values": [0, 1],
            "seasonal_Q_values": [0, 1],
            "complexity_cap": "p + q + P + Q <= 3",
            "candidate_count": len(candidate_grid),
            "trend": "n",
            "simple_differencing": False,
            "exogenous_variables": False,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
        },
        "selection_rule": "For each unique rolling origin, fit all valid candidates on 1994Q1 through that origin and select the lowest training-sample AICc.",
        "aicc_convention": {
            "formula": "AICc = AIC + 2*k*(k+1)/(n-k-1)",
            "k": "Number of actually estimated parameters, including the innovation variance.",
            "n": "State-space training nobs, required to equal the origin training-sample length.",
            "invalid_if": "n-k-1 <= 0, non-finite likelihood/AIC/AICc, fit exception, or non-convergence.",
        },
        "rolling_selection": {
            "unique_origins": int(len(selections)),
            "candidate_fits_attempted": int(selections["candidate_count"].sum()),
            "successful_fit_count": int(selections["successful_fit_count"].sum()),
            "failed_fit_count": int(selections["failed_fit_count"].sum()),
            "selected_D_frequency": d_counts,
            "selected_order_frequency": most_common_orders,
        },
        "full_sample_descriptive_selection": {
            "selected_order": str(full_row["selected_order"]),
            "selected_seasonal_order": str(full_row["selected_seasonal_order"]),
            "selected_D": int(full_row["selected_D"]),
            "aic": float(full_row["aic"]),
            "aicc": float(full_row["aicc"]),
            "candidate_count": int(full_row["candidate_count"]),
            "successful_fit_count": int(full_row["successful_fit_count"]),
            "failed_fit_count": int(full_row["failed_fit_count"]),
            "ljung_box_lag": int(full_row["ljung_box_lag"]),
            "ljung_box_model_df": int(full_row["ljung_box_model_df"]),
            "ljung_box_statistic": float(full_row["ljung_box_statistic"]),
            "ljung_box_pvalue": float(full_row["ljung_box_pvalue"]),
            "residual_mean": float(full_row["residual_mean"]),
            "residual_std": float(full_row["residual_std"]),
            "parameter_estimates": json.loads(str(full_row["parameter_estimates"])),
            "descriptive_only": True,
            "used_for_rolling_forecasts": False,
        },
        "retransformation": {
            "primary_forecast": "exp(forecast_log_mean + 0.5 * forecast_log_variance)",
            "saved_median": "exp(forecast_log_mean)",
            "scoring_uses": "Bias-corrected conditional-mean level forecast.",
        },
        "mase_convention": {
            "reused_from": "Stage 3 build_rolling_forecasts and compute_metrics",
            "seasonal_period": SEASONAL_PERIOD,
            "origin_specific_training_only_scale": True,
        },
        "metrics": _native_records(sarima_metrics),
        "benchmark_comparison": _native_records(model_comparison),
        "leakage_safeguards": [
            "A selector receives only the log_retail Series ending at its origin.",
            "The candidate is selected by training-sample AICc, never by target actual or OOS error.",
            "Selection runs once per unique origin; its fitted result supplies all horizons needed at that origin.",
            "The full-sample descriptive fit is separate and is never used for rolling forecasts.",
            "The Stage 3 target window, origins, and MASE scale implementation are reused.",
        ],
        "scope": {
            "cny_regressors": False,
            "pmi_regressors": False,
            "arimax": False,
            "ets_or_theta": False,
            "oos_order_tuning": False,
            "final_future_forecast": False,
        },
        "generated_artifacts": generated_artifacts,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    return summary


def main() -> None:
    started = time.perf_counter()

    def report_progress(position: int, total: int) -> None:
        if position % 5 == 0 or position == total:
            print(f"Selected {position}/{total} rolling origins.", flush=True)

    summary = run_backtest(progress_callback=report_progress)
    print("Selecting the separate full-sample descriptive model is complete.", flush=True)
    elapsed = time.perf_counter() - started
    rolling = summary["rolling_selection"]
    print(
        f"Stage 4 complete in {elapsed:.1f}s: {rolling['unique_origins']} origins, "
        f"{rolling['successful_fit_count']} successful / {rolling['failed_fit_count']} failed rolling fits."
    )


if __name__ == "__main__":
    main()
