"""Build the frozen Stage 7 forecast without repeating any model search."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

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
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.modeling import backtest_benchmarks as evaluation
from scripts.modeling import backtest_cny as cny
from scripts.modeling import backtest_ets as ets
from scripts.modeling import backtest_sarima as sarima


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_quarterly.csv"
STAGE4_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "diagnostics" / "stage4_summary.json"
PROTECTED_MANIFEST_PATH = PROJECT_ROOT / "docs" / "stage7_protected_artifact_sha256.json"
FORECAST_PATH = PROJECT_ROOT / "outputs" / "forecasts" / "final_forecast.csv"
DIAGNOSTICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "final_model_diagnostics.csv"
SUPPLEMENTARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "final_forecast_supplementary.csv"
RESULTS_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "final_results_summary.csv"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "final_forecast.png"
STAGE7_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "diagnostics" / "stage7_summary.json"
PDF_PATH = PROJECT_ROOT / "outputs" / "reports" / "STA4003_Final_Report.pdf"

MODEL_NAME = "SARIMA"
SEASONAL_PERIOD = 4
FORECAST_ORIGIN = pd.Period("2026Q2", freq="Q-DEC")
FORECAST_TARGETS = (pd.Period("2026Q3", freq="Q-DEC"), pd.Period("2026Q4", freq="Q-DEC"))
PRIMARY_EXOG_COLUMNS: tuple[str, ...] = ()
FORECAST_COLUMNS = [
    "origin",
    "target",
    "horizon",
    "model",
    "forecast_log_mean",
    "forecast_log_variance",
    "forecast_level_median",
    "forecast_level_mean",
    "lower_80",
    "upper_80",
    "lower_95",
    "upper_95",
]
SUPPLEMENTARY_COLUMNS = [
    "model",
    "target",
    "horizon",
    "forecast_level",
    "forecast_convention",
    "exogenous_regressors",
    "target_cny_regressor",
    "converged",
]
RESULTS_SUMMARY_COLUMNS = [
    "study/sample",
    "model",
    "horizon",
    "scope",
    "n",
    "MAE",
    "RMSE",
    "MASE",
    "interpretation",
    "source_file",
]


def json_ready(value: Any) -> Any:
    """Convert pandas/numpy values and CSV blanks to strict JSON types."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def word_converter_path() -> Path | None:
    """Return an installed local Microsoft Word executable, if present."""
    candidates = (
        Path(r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"),
        Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE"),
    )
    return next((path for path in candidates if path.is_file()), None)


def frozen_order() -> cny.LockedOrder:
    """Load Stage 4's full-sample descriptive order, never a new search result."""
    selected = read_json(STAGE4_SUMMARY_PATH)["full_sample_descriptive_selection"]
    candidate, _ = cny.candidate_for_full_sample(STAGE4_SUMMARY_PATH)
    _require(
        selected["selected_order"] == candidate.order_label
        and selected["selected_seasonal_order"] == candidate.seasonal_order_label
        and int(selected["selected_D"]) == candidate.D,
        "Stage 4 frozen full-sample order labels do not match the parsed order",
    )
    return candidate


def verify_protected_artifacts() -> dict[str, str]:
    """Check every Stage 0-6B tracked input/output against its pre-Stage 7 hash."""
    manifest = read_json(PROTECTED_MANIFEST_PATH)["files"]
    _require(bool(manifest), "Protected-artifact manifest is empty")
    actual: dict[str, str] = {}
    for relative_path, expected_hash in manifest.items():
        path = PROJECT_ROOT / Path(relative_path)
        _require(path.is_file(), f"Protected artifact is missing: {relative_path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(digest == expected_hash, f"Protected historical artifact changed: {relative_path}")
        actual[relative_path] = digest
    return actual


def load_stage7_data() -> pd.DataFrame:
    """Load Stage 4's validated response and align only observed calendar metadata."""
    frame = sarima.load_analysis_data(DATA_PATH)
    calendar = pd.read_csv(
        DATA_PATH,
        usecols=["quarter", "is_q1", "cny_position_fraction"],
        keep_default_na=False,
    )
    calendar.index = pd.PeriodIndex(calendar.pop("quarter").astype(str), freq="Q-DEC", name="quarter")
    _require(calendar.index.equals(frame.index), "Calendar fields do not match the validated retail index")
    frame = frame.join(calendar)
    _require(frame[["is_q1", "cny_position_fraction"]].notna().all().all(), "Historical calendar metadata is incomplete")
    return frame


def forecast_row(target: pd.Period, horizon: int, log_mean: float, log_variance: float) -> dict[str, Any]:
    """Transform one predictive log-normal distribution to level forecasts."""
    values = np.asarray([log_mean, log_variance], dtype=float)
    _require(np.isfinite(values).all(), "Log forecast moments must be finite")
    _require(log_variance >= 0.0, "Log forecast variance must be non-negative")
    standard_error = float(np.sqrt(log_variance))
    median = float(np.exp(log_mean))
    conditional_mean = float(np.exp(log_mean + 0.5 * log_variance))
    lower_80 = float(np.exp(log_mean - norm.ppf(0.90) * standard_error))
    upper_80 = float(np.exp(log_mean + norm.ppf(0.90) * standard_error))
    lower_95 = float(np.exp(log_mean - norm.ppf(0.975) * standard_error))
    upper_95 = float(np.exp(log_mean + norm.ppf(0.975) * standard_error))
    _require(
        np.isfinite([median, conditional_mean, lower_80, upper_80, lower_95, upper_95]).all()
        and conditional_mean > 0.0,
        "Retransformed forecast is not finite and positive",
    )
    _require(lower_95 < lower_80 < median < upper_80 < upper_95, "Forecast intervals are not properly nested")
    return {
        "origin": str(FORECAST_ORIGIN),
        "target": str(target),
        "horizon": int(horizon),
        "model": MODEL_NAME,
        "forecast_log_mean": float(log_mean),
        "forecast_log_variance": float(log_variance),
        "forecast_level_median": median,
        "forecast_level_mean": conditional_mean,
        "lower_80": lower_80,
        "upper_80": upper_80,
        "lower_95": lower_95,
        "upper_95": upper_95,
    }


def future_cny_exog(targets: tuple[pd.Period, ...], center: float) -> pd.DataFrame:
    """Build deterministic calendar-only exog; Q3/Q4 are exactly outside CNY Q1."""
    _require(all(target.quarter != 1 for target in targets), "Future CNY helper is scoped to non-Q1 targets")
    calendar = pd.DataFrame(
        {
            "is_q1": [0] * len(targets),
            # The fraction is immaterial when is_q1 is zero; no future economic
            # observations or future CNY timing values enter these forecasts.
            "cny_position_fraction": [0.0] * len(targets),
        },
        index=pd.PeriodIndex(targets, freq="Q-DEC"),
    )
    result = cny.centered_cny_exog(calendar, center)
    _require(result[cny.CNY_EXOG_NAME].eq(0.0).all(), "Non-Q1 future CNY regressors must equal zero")
    return result


def metric_record(path: Path, model: str, horizon: int = 2, scope: str = "all") -> dict[str, Any]:
    frame = pd.read_csv(path)
    rows = frame.loc[
        frame["model"].eq(model) & frame["horizon"].eq(horizon) & frame["scope"].eq(scope)
    ]
    _require(len(rows) == 1, f"Expected one metric row for {model}, h={horizon}, {scope} in {path}")
    row = rows.iloc[0]
    n_field = "n" if "n" in rows.columns else "n_forecasts"
    return {
        "model": model,
        "horizon": int(horizon),
        "scope": scope,
        "n": int(row[n_field]),
        "MAE": float(row["MAE"]),
        "RMSE": float(row["RMSE"]),
        "MASE": float(row["MASE"]),
    }


def build_results_summary() -> pd.DataFrame:
    """Assemble the primary common window and the separately labeled PMI window."""
    primary_window = "Primary 2005Q1-2026Q2"
    pmi_window = "Stage 6A secondary 2010Q4-2026Q2"
    specifications = [
        ("Historical Mean", "outputs/tables/benchmark_metrics.csv", "outputs/tables/benchmark_metrics.csv", "Historical level benchmark"),
        ("Naive", "outputs/tables/benchmark_metrics.csv", "outputs/tables/benchmark_metrics.csv", "Last observed level benchmark"),
        ("Seasonal Naive", "outputs/tables/benchmark_metrics.csv", "outputs/tables/benchmark_metrics.csv", "Same-quarter prior-year benchmark"),
        ("SARIMA", "outputs/tables/sarima_metrics.csv", "outputs/tables/sarima_metrics.csv", "Primary fixed-protocol seasonal ARIMA baseline"),
        ("SARIMA+CNY", "outputs/tables/cny_metrics.csv", "outputs/tables/cny_metrics.csv", "Supplementary; incremental changes are small and mixed"),
        ("ETS", "outputs/tables/ets_metrics.csv", "outputs/tables/ets_metrics.csv", "Fixed supplementary ETS; below SARIMA on the primary endpoint"),
        ("SARIMA+PMI", "outputs/tables/pmi_metrics.csv", "outputs/tables/pmi_metrics.csv", "Secondary window; does not improve on SARIMA-common"),
    ]
    rows: list[dict[str, Any]] = []
    for model, source_name, _, interpretation in specifications:
        source_path = PROJECT_ROOT / source_name
        record = metric_record(source_path, model)
        rows.append(
            {
                "study/sample": pmi_window if model == "SARIMA+PMI" else primary_window,
                **record,
                "interpretation": interpretation,
                "source_file": source_name,
            }
        )
    return pd.DataFrame(rows, columns=RESULTS_SUMMARY_COLUMNS)


def _series_forecast_rows(result: Any, exog: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = result.get_forecast(steps=len(FORECAST_TARGETS), exog=exog)
    means = np.asarray(prediction.predicted_mean, dtype=float).reshape(-1)
    variances = np.asarray(prediction.var_pred_mean, dtype=float).reshape(-1)
    _require(len(means) == len(FORECAST_TARGETS) and len(variances) == len(FORECAST_TARGETS), "Model returned the wrong forecast length")
    _require(np.isfinite(means).all() and np.isfinite(variances).all(), "Model forecast moments must be finite")
    return means, variances


def _forecast_supplementary(
    frame: pd.DataFrame,
    candidate: cny.LockedOrder,
    primary_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    training = frame.loc[:FORECAST_ORIGIN, "log_retail"].astype(float)
    _require(training.index[0] == pd.Period("1994Q1", freq="Q-DEC"), "Supplementary training start changed")
    _require(training.index[-1] == FORECAST_ORIGIN and len(training) == 130, "Supplementary training sample changed")
    order = sarima.SarimaCandidate(
        p=candidate.p,
        d=candidate.d,
        q=candidate.q,
        P=candidate.P,
        D=candidate.D,
        Q=candidate.Q,
    )

    # The CNY extension uses Stage 5's training-only Q1 center and one future
    # regressor. Its Q3/Q4 values are identically zero by the frozen definition.
    training_calendar = frame.loc[:FORECAST_ORIGIN, ["is_q1", "cny_position_fraction"]]
    center, _ = cny.training_center(training_calendar)
    training_exog = cny.centered_cny_exog(training_calendar, center)
    future_exog = future_cny_exog(FORECAST_TARGETS, center)
    cny_fit = cny.fit_cny_sarimax(training, training_exog, candidate)
    _require(cny_fit is not None and cny_fit.converged, "Full-sample supplementary SARIMA+CNY did not converge")
    cny_means, cny_variances = _series_forecast_rows(cny_fit.result, future_exog)

    # Stage 6B's fixed ETS(A,Ad,A) specification and training-residual smearing.
    ets_fit = ets.fit_fixed_ets(training)
    _require(ets_fit.result is not None and ets_fit.converged, "Full-sample supplementary ETS did not converge")
    ets_log_forecast = np.asarray(ets_fit.result.forecast(steps=len(FORECAST_TARGETS)), dtype=float).reshape(-1)
    ets_residuals = np.asarray(ets_fit.result.resid, dtype=float).reshape(-1)
    ets_residuals = ets_residuals[np.isfinite(ets_residuals)]
    _require(len(ets_residuals) == len(training), "ETS residual smearing sample must match training observations")
    with np.errstate(over="ignore", invalid="ignore"):
        smearing_factor = float(np.mean(np.exp(ets_residuals)))
        ets_level_forecast = np.exp(ets_log_forecast) * smearing_factor
    _require(np.isfinite(ets_level_forecast).all() and (ets_level_forecast > 0).all(), "ETS level forecast must be finite and positive")

    rows: list[dict[str, Any]] = []
    for i, target in enumerate(FORECAST_TARGETS):
        primary_row = primary_forecasts.iloc[i]
        rows.append(
            {
                "model": "SARIMA",
                "target": str(target),
                "horizon": i + 1,
                "forecast_level": float(primary_row["forecast_level_mean"]),
                "forecast_convention": "exp(log mean + 0.5 * log forecast variance)",
                "exogenous_regressors": "none",
                "target_cny_regressor": np.nan,
                "converged": True,
            }
        )
        rows.append(
            {
                "model": "SARIMA+CNY",
                "target": str(target),
                "horizon": i + 1,
                "forecast_level": float(np.exp(cny_means[i] + 0.5 * cny_variances[i])),
                "forecast_convention": "exp(log mean + 0.5 * log forecast variance)",
                "exogenous_regressors": "cny_regressor",
                "target_cny_regressor": float(future_exog.iloc[i][cny.CNY_EXOG_NAME]),
                "converged": bool(cny_fit.converged),
            }
        )
        rows.append(
            {
                "model": "ETS",
                "target": str(target),
                "horizon": i + 1,
                "forecast_level": float(ets_level_forecast[i]),
                "forecast_convention": "exp(log forecast) * mean(exp(training residuals))",
                "exogenous_regressors": "none",
                "target_cny_regressor": np.nan,
                "converged": bool(ets_fit.converged),
            }
        )
    return pd.DataFrame(rows, columns=SUPPLEMENTARY_COLUMNS)


def _quarter_timestamp(period: pd.Period) -> pd.Timestamp:
    return period.to_timestamp(how="end").normalize()


def save_forecast_figure(frame: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    recent_start = FORECAST_ORIGIN - 37
    history = frame.loc[recent_start:FORECAST_ORIGIN, "retail_bn"].astype(float)
    targets = pd.PeriodIndex(forecasts["target"], freq="Q-DEC")
    forecast_dates = [_quarter_timestamp(period) for period in targets]
    history_dates = [_quarter_timestamp(period) for period in history.index]
    point_values = forecasts["forecast_level_mean"].to_numpy(dtype=float)
    last_date = history_dates[-1]
    last_value = float(history.iloc[-1])

    fig, ax = plt.subplots(figsize=(10.5, 5.7), constrained_layout=True)
    observed_line, = ax.plot(history_dates, history.to_numpy(), color="#24476B", linewidth=2.1, label="Observed retail")
    lower_95 = forecasts["lower_95"].to_numpy(dtype=float)
    upper_95 = forecasts["upper_95"].to_numpy(dtype=float)
    lower_80 = forecasts["lower_80"].to_numpy(dtype=float)
    upper_80 = forecasts["upper_80"].to_numpy(dtype=float)
    band95 = ax.fill_between(forecast_dates, lower_95, upper_95, color="#9DB7CE", alpha=0.33, label="95% prediction interval")
    band80 = ax.fill_between(forecast_dates, lower_80, upper_80, color="#5D8DB5", alpha=0.30, label="80% prediction interval")
    forecast_line, = ax.plot(
        [last_date, *forecast_dates],
        [last_value, *point_values],
        color="#C56A2D",
        linewidth=2.0,
        linestyle=(0, (4, 3)),
        marker="D",
        markersize=5.5,
        markevery=[1, 2],
        label="SARIMA conditional mean forecast",
    )
    boundary = last_date + (forecast_dates[0] - last_date) / 2
    ax.axvline(boundary, color="#697987", linewidth=1.0, linestyle=(0, (3, 3)))
    ax.text(boundary, 0.98, "Forecast origin 2026Q2", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=8.5, color="#45515B")
    ax.set_title("Quarterly retail sales and final SARIMA forecast", fontsize=14, color="#111111", pad=14)
    cjk_font = Path("C:/Windows/Fonts/msyh.ttc")
    if cjk_font.is_file():
        ax.set_ylabel("Retail sales (RMB 100 million, 亿元)", fontproperties=FontProperties(fname=str(cjk_font)))
    else:
        ax.set_ylabel("Retail sales (RMB 100 million)")
    ax.set_xlabel("Quarter")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color="#D9E0E5", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[observed_line, forecast_line, band80, band95],
        loc="upper left",
        frameon=False,
        ncol=2,
        fontsize=8.5,
    )
    ax.set_xlim(history_dates[0], forecast_dates[-1] + pd.Timedelta(days=150))
    fig.savefig(FIGURE_PATH, dpi=180, facecolor="white", metadata={"Software": "matplotlib"})
    plt.close(fig)


def _row_for_metric(summary: pd.DataFrame, model: str) -> dict[str, Any]:
    rows = summary.loc[summary["model"].eq(model)]
    _require(len(rows) == 1, f"Missing final historical metric row for {model}")
    row = rows.iloc[0]
    return {key: (int(value) if key == "n" else float(value) if key in ("MAE", "RMSE", "MASE") else value) for key, value in row.items()}


def create_diagnostics_row(fitted: sarima.CandidateFit, training: pd.Series) -> dict[str, Any]:
    result = fitted.result
    residuals = sarima._diagnostic_residuals(fitted)
    lb = sarima.ljung_box_diagnostic(fitted)
    parameters = {name: float(value) for name, value in zip(result.param_names, np.asarray(result.params, dtype=float))}
    return {
        "training_start": str(training.index[0]),
        "training_end": str(training.index[-1]),
        "n": len(training),
        "model": MODEL_NAME,
        "order": fitted.candidate.order_label,
        "seasonal_order": fitted.candidate.seasonal_order_label,
        "model_spec": f"SARIMA{fitted.candidate.order_label}({fitted.candidate.P},{fitted.candidate.D},{fitted.candidate.Q})[{SEASONAL_PERIOD}]",
        "seasonal_period": SEASONAL_PERIOD,
        "AIC": float(fitted.aic),
        "AICc": float(fitted.aicc),
        **parameters,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)),
        "ljung_box_lag": int(lb["ljung_box_lag"]),
        "ljung_box_model_df": int(lb["ljung_box_model_df"]),
        "ljung_box_statistic": float(lb["ljung_box_statistic"]),
        "ljung_box_pvalue": float(lb["ljung_box_pvalue"]),
        "ljung_box_n_residuals": int(lb["ljung_box_n_residuals"]),
        "converged": bool(fitted.converged),
        "primary_exog_variables": "none",
    }


def _cny_shock_comparison() -> dict[str, dict[str, float]]:
    source = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "cny_shock_sensitivity.csv")
    result: dict[str, dict[str, float]] = {}
    for sample in ("full", "exclude_shocks"):
        rows = source.loc[(source["horizon"] == 2) & (source["scope"] == "all") & (source["sample"] == sample)]
        _require(len(rows) == 1, f"Missing CNY sensitivity row for {sample}")
        row = rows.iloc[0]
        result[sample] = {metric: float(row[f"delta_{metric}"]) for metric in ("MAE", "RMSE", "MASE")}
    return result


def run_final_forecast() -> dict[str, Any]:
    protected_hashes = verify_protected_artifacts()
    previous_pdf_status: dict[str, Any] | None = None
    if STAGE7_SUMMARY_PATH.is_file():
        previous_pdf_status = read_json(STAGE7_SUMMARY_PATH).get("pdf_status")
    pdf_status_was_checked = bool(previous_pdf_status and previous_pdf_status.get("runtime_tested"))
    frame = load_stage7_data()
    _require(len(frame) == 130, "Canonical final training data must contain 130 quarters")
    _require(frame.index[0] == pd.Period("1994Q1", freq="Q-DEC"), "Final training start changed")
    _require(frame.index[-1] == FORECAST_ORIGIN, "Final training must end exactly at 2026Q2")

    selected = read_json(STAGE4_SUMMARY_PATH)["full_sample_descriptive_selection"]
    candidate = frozen_order()
    training = frame.loc[:FORECAST_ORIGIN, "log_retail"].astype(float)
    _require(training.index[-1] == FORECAST_ORIGIN and len(training) == 130, "Primary model training window changed")
    frozen_candidate = sarima.SarimaCandidate(
        p=candidate.p,
        d=candidate.d,
        q=candidate.q,
        P=candidate.P,
        D=candidate.D,
        Q=candidate.Q,
    )
    fitted = sarima.fit_candidate(training, frozen_candidate)
    _require(fitted is not None and fitted.converged, "Frozen full-sample SARIMA fit failed or did not converge")
    primary_log_mean, primary_log_variance = _series_forecast_rows(fitted.result)

    target_periods = pd.PeriodIndex(FORECAST_TARGETS, freq="Q-DEC")
    _require(target_periods.tolist() == [pd.Period("2026Q3", freq="Q-DEC"), pd.Period("2026Q4", freq="Q-DEC")], "Final forecast target set changed")
    _require([int(x) for x in target_periods.year] == [2026, 2026], "Unexpected forecast year")
    forecasts = pd.DataFrame(
        [forecast_row(target, horizon, primary_log_mean[i], primary_log_variance[i]) for i, (target, horizon) in enumerate(zip(FORECAST_TARGETS, (1, 2)))],
        columns=FORECAST_COLUMNS,
    )
    _require(forecasts["forecast_level_mean"].gt(0).all() and np.isfinite(forecasts["forecast_level_mean"]).all(), "Primary point forecasts must be finite and positive")

    diagnostics = pd.DataFrame([create_diagnostics_row(fitted, training)])
    supplementary = _forecast_supplementary(frame, candidate, forecasts)
    results_summary = build_results_summary()
    save_forecast_figure(frame, forecasts)

    forecast_path = FORECAST_PATH
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path = DIAGNOSTICS_PATH
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    supplementary_path = SUPPLEMENTARY_PATH
    supplementary_path.parent.mkdir(parents=True, exist_ok=True)
    results_summary_path = RESULTS_SUMMARY_PATH
    results_summary_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(forecast_path, index=False, float_format="%.17g", lineterminator="\n")
    diagnostics.to_csv(diagnostics_path, index=False, float_format="%.17g", lineterminator="\n")
    supplementary.to_csv(supplementary_path, index=False, float_format="%.17g", lineterminator="\n")
    results_summary.to_csv(results_summary_path, index=False, float_format="%.17g", lineterminator="\n")

    metric_rows = {model: _row_for_metric(results_summary, model) for model in ("Historical Mean", "Naive", "Seasonal Naive", "SARIMA", "SARIMA+CNY", "ETS", "SARIMA+PMI")}
    cny_shock_deltas = _cny_shock_comparison()
    pmi_common = metric_record(PROJECT_ROOT / "outputs" / "tables" / "pmi_metrics.csv", "SARIMA-common")
    pmi_with_predictor = metric_rows["SARIMA+PMI"]
    forecast_records = forecasts.to_dict(orient="records")
    supplementary_records = supplementary.to_dict(orient="records")
    diag_record = diagnostics.iloc[0].to_dict()

    generated_artifacts = [
        "outputs/forecasts/final_forecast.csv",
        "outputs/tables/final_model_diagnostics.csv",
        "outputs/tables/final_forecast_supplementary.csv",
        "outputs/tables/final_results_summary.csv",
        "outputs/figures/final_forecast.png",
        "docs/final_report.md",
        "outputs/reports/STA4003_Final_Report.docx",
    ]
    local_word = word_converter_path()
    if pdf_status_was_checked:
        pdf_status = dict(previous_pdf_status or {})
        pdf_generated = bool(pdf_status.get("generated")) and PDF_PATH.is_file()
    else:
        pdf_generated = PDF_PATH.is_file()
        pdf_status = {
            "converter_installed": local_word is not None,
            "converter_available": None,
            "runtime_tested": False,
            "generated": pdf_generated,
            "output": "outputs/reports/STA4003_Final_Report.pdf" if pdf_generated else None,
            "reason": "Report builder will test the installed local converter." if local_word else "No local converter executable was found.",
        }
    if pdf_generated:
        generated_artifacts.append("outputs/reports/STA4003_Final_Report.pdf")

    stage7_summary = {
        "stage": 7,
        "stage_name": "Final forecast and report",
        "primary_model": MODEL_NAME,
        "primary_model_reason": [
            "SARIMA materially outperforms Historical Mean, Naive, and Seasonal Naive at h=2 across all primary-window targets.",
            "Fixed ETS improves on the simple benchmarks but has higher h=2 all-target errors than SARIMA.",
            "CNY changes are very small and mixed across metrics; after preset-shock target exclusion, the MAE and MASE reductions disappear.",
            "The fixed PMI lag-2 extension does not improve the separate secondary-window h=2 all-target metrics.",
            "The parsimonious Stage 4 full-sample SARIMA order is retained without a post-hoc order search.",
        ],
        "candidate_search_performed": False,
        "primary_exog_variables": list(PRIMARY_EXOG_COLUMNS),
        "response": "log_retail",
        "fit_settings": {
            "trend": "n",
            "simple_differencing": False,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
            "optimizer": "lbfgs",
            "maxiter": sarima.MAX_ITERATIONS,
        },
        "final_order": {
            "order": list(candidate.order),
            "seasonal_order": list(candidate.seasonal_order),
            "label": f"SARIMA{candidate.order_label}({candidate.P},{candidate.D},{candidate.Q})[{SEASONAL_PERIOD}]",
            "source": "outputs/diagnostics/stage4_summary.json:full_sample_descriptive_selection",
            "stage4_aicc": float(selected["aicc"]),
        },
        "training": {
            "start": str(training.index[0]),
            "end": str(training.index[-1]),
            "n": len(training),
            "future_economic_observations_used": False,
        },
        "forecast_origin": str(FORECAST_ORIGIN),
        "targets": [str(target) for target in FORECAST_TARGETS],
        "forecast_convention": {
            "median": "exp(log forecast mean)",
            "primary_point_forecast": "exp(log forecast mean + 0.5 * log forecast variance)",
            "intervals": "exp(log forecast mean +/- normal quantile * sqrt(log forecast variance)); no 0.5 variance shift",
            "parameter_uncertainty_included": False,
        },
        "forecasts": forecast_records,
        "final_model_diagnostics": diag_record,
        "historical_oos_metrics": {
            "primary_window": "2005Q1-2026Q2",
            "primary_horizon": 2,
            "primary_scope": "all",
            "models": [metric_rows[name] for name in ("Historical Mean", "Naive", "Seasonal Naive", "SARIMA", "SARIMA+CNY", "ETS")],
            "source_files": [
                "outputs/tables/benchmark_metrics.csv",
                "outputs/tables/sarima_metrics.csv",
                "outputs/tables/cny_metrics.csv",
                "outputs/tables/ets_metrics.csv",
            ],
        },
        "cny_conclusion": {
            "primary_window_h2_all_delta_cny_minus_sarima": {
                metric: float(metric_rows["SARIMA+CNY"][metric] - metric_rows["SARIMA"][metric])
                for metric in ("MAE", "RMSE", "MASE")
            },
            "shock_sensitivity_delta_cny_minus_sarima": cny_shock_deltas,
            "conclusion": "No material or consistently positive incremental out-of-sample forecasting improvement under the prespecified protocol; this is not a claim that CNY has no effect.",
            "full_sample_beta_used_as_oos_evidence": False,
            "source_files": ["outputs/tables/cny_metrics.csv", "outputs/tables/cny_shock_sensitivity.csv"],
        },
        "pmi_conclusion": {
            "window": "2010Q4-2026Q2",
            "n": int(pmi_with_predictor["n"]),
            "horizon": 2,
            "scope": "all",
            "common_sarima": pmi_common,
            "sarima_plus_pmi": pmi_with_predictor,
            "conclusion": "Lag-2 PMI does not improve the secondary-window h=2 all-target MAE, RMSE, or MASE.",
            "source_file": "outputs/tables/pmi_metrics.csv",
        },
        "ets_robustness_conclusion": {
            "specification": "ETS(A,Ad,A), seasonal period 4",
            "h2_all": metric_rows["ETS"],
            "conclusion": "The fixed ETS comparator improves on simple benchmarks but has higher h=2 all-target errors than SARIMA.",
            "shock_targets": ["2020Q1", "2022Q2"],
            "source_file": "outputs/tables/ets_metrics.csv",
        },
        "supplementary_forecasts": supplementary_records,
        "protected_artifacts": {
            "manifest": "docs/stage7_protected_artifact_sha256.json",
            "checked_count": len(protected_hashes),
            "unchanged": True,
            "sha256": protected_hashes,
        },
        "generated_artifacts": generated_artifacts,
        "pdf_status": pdf_status,
    }
    STAGE7_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGE7_SUMMARY_PATH.write_text(
        json.dumps(json_ready(stage7_summary), indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stage7_summary


def main() -> None:
    started = time.perf_counter()
    summary = run_final_forecast()
    elapsed = time.perf_counter() - started
    for row in summary["forecasts"]:
        print(
            f"{row['target']} mean={row['forecast_level_mean']:.2f}; "
            f"80%=[{row['lower_80']:.2f}, {row['upper_80']:.2f}]; "
            f"95%=[{row['lower_95']:.2f}, {row['upper_95']:.2f}] 亿元"
        )
    print(f"Stage 7 forecasts and diagnostics written; runtime {elapsed:.2f}s")


if __name__ == "__main__":
    main()
