"""Leak-free expanding-window benchmark forecasts for Stage 3.

Only three pre-specified benchmarks are evaluated, on original retail levels.
Every forecast and seasonal MASE scale uses information available by its origin.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_quarterly.csv"
TARGET_START = pd.Period("2005Q1", freq="Q-DEC")
TARGET_END = pd.Period("2026Q2", freq="Q-DEC")
SAMPLE_START = pd.Period("1994Q1", freq="Q-DEC")
SAMPLE_END = pd.Period("2026Q2", freq="Q-DEC")
HORIZONS = (1, 2)
PRIMARY_HORIZON = 2
SEASONAL_PERIOD = 4
BENCHMARKS = ("Historical Mean", "Naive", "Seasonal Naive")
FORECAST_COLUMNS = [
    "model",
    "horizon",
    "origin",
    "target",
    "training_start",
    "training_end",
    "n_train",
    "actual",
    "forecast",
    "error",
    "absolute_error",
    "squared_error",
    "mase_scale",
    "scaled_absolute_error",
]
METRIC_COLUMNS = ["model", "horizon", "scope", "n_forecasts", "MAE", "RMSE", "MASE"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_quarters(
    start: pd.Period = SAMPLE_START, end: pd.Period = SAMPLE_END
) -> pd.PeriodIndex:
    return pd.period_range(start=start, end=end, freq="Q-DEC", name="quarter")


def load_canonical(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the immutable canonical retail series."""
    _require(path.is_file(), f"Missing canonical input: {path}")
    frame = pd.read_csv(path, usecols=["quarter", "retail_bn"], keep_default_na=False)
    quarters = pd.PeriodIndex(frame["quarter"].astype(str), freq="Q-DEC", name="quarter")
    _require(quarters.equals(expected_quarters()), "Canonical quarter sequence is not 1994Q1–2026Q2")
    retail = pd.to_numeric(frame["retail_bn"], errors="raise").to_numpy(dtype=float)
    _require(np.isfinite(retail).all() and (retail > 0).all(), "Retail levels must be finite and positive")
    return pd.DataFrame({"retail_bn": retail}, index=quarters)


def _validate_frame(frame: pd.DataFrame) -> pd.Series:
    _require("retail_bn" in frame.columns, "Input frame must contain retail_bn")
    _require(isinstance(frame.index, pd.PeriodIndex), "Input index must be quarterly PeriodIndex")
    _require(frame.index.freqstr == "Q-DEC", "Input PeriodIndex frequency must be Q-DEC")
    _require(frame.index.equals(expected_quarters()), "Input must cover 1994Q1–2026Q2 without gaps")
    retail = pd.to_numeric(frame["retail_bn"], errors="raise").astype(float)
    values = retail.to_numpy()
    _require(np.isfinite(values).all() and (values > 0).all(), "Retail levels must be finite and positive")
    return retail


def build_rolling_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Create all benchmark forecasts using expanding training samples."""
    retail = _validate_frame(frame)
    targets = pd.period_range(TARGET_START, TARGET_END, freq="Q-DEC")
    rows: list[dict[str, object]] = []

    for horizon in HORIZONS:
        for target in targets:
            origin = target - horizon
            training = retail.loc[:origin]
            _require(not training.empty, f"No training observations available at {origin}")
            _require(training.index[-1] == origin, f"Training sample does not end at origin {origin}")
            _require(len(training) > SEASONAL_PERIOD, f"Insufficient history at origin {origin}")

            training_values = training.to_numpy(dtype=float)
            seasonal_differences = np.abs(
                training_values[SEASONAL_PERIOD:] - training_values[:-SEASONAL_PERIOD]
            )
            mase_scale = float(seasonal_differences.mean())
            _require(np.isfinite(mase_scale) and mase_scale > 0, f"Invalid MASE scale at origin {origin}")
            actual = float(retail.loc[target])
            forecasts = {
                "Historical Mean": float(training_values.mean()),
                "Naive": float(training_values[-1]),
                "Seasonal Naive": float(retail.loc[target - SEASONAL_PERIOD]),
            }

            for model in BENCHMARKS:
                forecast = forecasts[model]
                error = actual - forecast
                rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "origin": str(origin),
                        "target": str(target),
                        "training_start": str(training.index[0]),
                        "training_end": str(training.index[-1]),
                        "n_train": int(len(training)),
                        "actual": actual,
                        "forecast": forecast,
                        "error": error,
                        "absolute_error": abs(error),
                        "squared_error": error**2,
                        "mase_scale": mase_scale,
                        "scaled_absolute_error": abs(error) / mase_scale,
                    }
                )

    forecasts = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
    _validate_forecasts(forecasts)
    return forecasts


def _validate_forecasts(forecasts: pd.DataFrame) -> None:
    targets = [str(period) for period in pd.period_range(TARGET_START, TARGET_END, freq="Q-DEC")]
    expected_count = len(targets)
    _require(forecasts.columns.tolist() == FORECAST_COLUMNS, "Forecast record schema changed unexpectedly")
    _require(not forecasts.duplicated(["model", "horizon", "target"]).any(), "Duplicate model/horizon/target record")

    for horizon in HORIZONS:
        for model in BENCHMARKS:
            subset = forecasts.loc[(forecasts["horizon"] == horizon) & (forecasts["model"] == model)]
            _require(subset["target"].tolist() == targets, f"Unexpected target sequence for {model}, h={horizon}")
            target_periods = pd.PeriodIndex(subset["target"], freq="Q-DEC")
            origins = pd.PeriodIndex(subset["origin"], freq="Q-DEC")
            _require((target_periods - horizon).equals(origins), f"Origin-target alignment failed for {model}, h={horizon}")
            training_ends = pd.PeriodIndex(subset["training_end"], freq="Q-DEC")
            _require(training_ends.equals(origins), f"Training extends beyond or stops before origin for {model}, h={horizon}")
            _require((subset["training_start"] == str(SAMPLE_START)).all(), "Training start changed unexpectedly")
            expected_train_n = (origins.astype("int64") - SAMPLE_START.ordinal + 1).to_numpy()
            _require(np.array_equal(subset["n_train"].to_numpy(dtype=int), expected_train_n), "Expanding training size is incorrect")
            _require(len(subset) == expected_count, f"Incorrect forecast count for {model}, h={horizon}")

    for horizon in HORIZONS:
        model_targets = [
            forecasts.loc[(forecasts["horizon"] == horizon) & (forecasts["model"] == model), "target"].tolist()
            for model in BENCHMARKS
        ]
        _require(all(items == model_targets[0] for items in model_targets[1:]), f"Models do not share targets at h={horizon}")


def compute_metrics(
    forecasts: pd.DataFrame, models: tuple[str, ...] = BENCHMARKS
) -> pd.DataFrame:
    """Report MAE, RMSE, and origin-scaled seasonal MASE for requested models."""
    rows: list[dict[str, object]] = []
    for model in models:
        for horizon in HORIZONS:
            model_horizon = forecasts.loc[
                (forecasts["model"] == model) & (forecasts["horizon"] == horizon)
            ]
            scopes = (
                ("all", model_horizon),
                ("q1", model_horizon.loc[pd.PeriodIndex(model_horizon["target"], freq="Q-DEC").quarter == 1]),
            )
            for scope, subset in scopes:
                absolute_errors = subset["absolute_error"].to_numpy(dtype=float)
                squared_errors = subset["squared_error"].to_numpy(dtype=float)
                scaled_absolute_errors = subset["scaled_absolute_error"].to_numpy(dtype=float)
                rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "scope": scope,
                        "n_forecasts": int(len(subset)),
                        "MAE": float(absolute_errors.mean()),
                        "RMSE": float(np.sqrt(squared_errors.mean())),
                        "MASE": float(scaled_absolute_errors.mean()),
                    }
                )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _save_figure(fpath: Path) -> None:
    plt.savefig(fpath, dpi=180, bbox_inches="tight", metadata={"Software": "STA4003 Stage 3"})
    plt.close()


def plot_h2_actual_and_forecasts(forecasts: pd.DataFrame, path: Path) -> None:
    h2 = forecasts.loc[forecasts["horizon"] == PRIMARY_HORIZON]
    actual = h2.loc[h2["model"] == BENCHMARKS[0]]
    dates = pd.PeriodIndex(actual["target"], freq="Q-DEC").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, actual["actual"], color="#1f2937", linewidth=2.0, label="Actual")
    colors = {"Historical Mean": "#2563eb", "Naive": "#dc2626", "Seasonal Naive": "#059669"}
    for model in BENCHMARKS:
        subset = h2.loc[h2["model"] == model]
        ax.plot(dates, subset["forecast"], color=colors[model], linewidth=1.25, label=model)
    ax.set_title("Actual retail and benchmark forecasts (h = 2)")
    ax.set_ylabel("Retail sales (RMB 100 million)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    _save_figure(path)


def plot_h2_errors(forecasts: pd.DataFrame, path: Path) -> None:
    h2 = forecasts.loc[forecasts["horizon"] == PRIMARY_HORIZON]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"Historical Mean": "#2563eb", "Naive": "#dc2626", "Seasonal Naive": "#059669"}
    for model in BENCHMARKS:
        subset = h2.loc[h2["model"] == model]
        dates = pd.PeriodIndex(subset["target"], freq="Q-DEC").to_timestamp(how="end")
        ax.plot(dates, subset["error"], color=colors[model], linewidth=1.15, label=model)
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_title("Benchmark forecast errors (actual − forecast, h = 2)")
    ax.set_ylabel("Error (RMB 100 million)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    _save_figure(path)


def _summary_metrics(metrics: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "model": str(row.model),
            "horizon": int(row.horizon),
            "scope": str(row.scope),
            "n_forecasts": int(row.n_forecasts),
            "MAE": float(row.MAE),
            "RMSE": float(row.RMSE),
            "MASE": float(row.MASE),
        }
        for row in metrics.itertuples(index=False)
    ]


def run_backtest(output_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Run the fixed benchmark protocol and write its reproducible artifacts."""
    frame = load_canonical()
    forecasts = build_rolling_forecasts(frame)
    metrics = compute_metrics(forecasts)

    output_root = Path(output_root)
    forecast_path = output_root / "outputs" / "forecasts" / "benchmark_rolling_forecasts.csv"
    metrics_path = output_root / "outputs" / "tables" / "benchmark_metrics.csv"
    figure_dir = output_root / "outputs" / "figures"
    summary_path = output_root / "outputs" / "diagnostics" / "stage3_summary.json"
    for directory in (forecast_path.parent, metrics_path.parent, figure_dir, summary_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    forecast_path.write_text(
        forecasts.to_csv(index=False, float_format="%.15g", lineterminator="\n"),
        encoding="utf-8",
        newline="",
    )
    metrics_path.write_text(
        metrics.to_csv(index=False, float_format="%.15g", lineterminator="\n"),
        encoding="utf-8",
        newline="",
    )
    actual_figure = figure_dir / "benchmark_h2_actual_vs_forecasts.png"
    error_figure = figure_dir / "benchmark_h2_errors.png"
    plot_h2_actual_and_forecasts(forecasts, actual_figure)
    plot_h2_errors(forecasts, error_figure)

    generated_artifacts = [
        "outputs/forecasts/benchmark_rolling_forecasts.csv",
        "outputs/tables/benchmark_metrics.csv",
        "outputs/figures/benchmark_h2_actual_vs_forecasts.png",
        "outputs/figures/benchmark_h2_errors.png",
        "outputs/diagnostics/stage3_summary.json",
    ]
    counts = {
        str(horizon): int(
            forecasts.loc[forecasts["horizon"] == horizon, ["model", "target"]]
            .drop_duplicates()["target"]
            .nunique()
        )
        for horizon in HORIZONS
    }
    summary: dict[str, object] = {
        "stage": 3,
        "evaluation_window": {"start": str(TARGET_START), "end": str(TARGET_END)},
        "horizons": list(HORIZONS),
        "primary_horizon": PRIMARY_HORIZON,
        "secondary_horizon": 1,
        "primary_scope": "all",
        "reporting_scopes": {
            "all": "All targets in the fixed window.",
            "q1": "Pre-specified Q1 target subgroup.",
        },
        "target_count_per_model_horizon": counts,
        "benchmark_definitions": [
            {
                "model": "Historical Mean",
                "forecast": "Arithmetic mean of retail_bn from 1994Q1 through the forecast origin.",
            },
            {
                "model": "Naive",
                "forecast": "retail_bn observed at the forecast origin.",
            },
            {
                "model": "Seasonal Naive",
                "forecast": "retail_bn observed four quarters before the target.",
            },
        ],
        "forecast_scale": "Original retail level in RMB 100 million (亿元).",
        "mase_convention": {
            "type": "seasonal MASE",
            "seasonal_period": SEASONAL_PERIOD,
            "origin_specific": True,
            "training_only": True,
            "scale": "Mean absolute difference retail_bn[t] - retail_bn[t-4] within the origin training sample.",
            "scaled_absolute_error": "absolute_error divided by the origin-specific training-only MASE scale.",
            "reported_mase": "Arithmetic mean of scaled_absolute_error over the requested model, horizon, and scope.",
        },
        "metrics": _summary_metrics(metrics),
        "leakage_safeguards": [
            "For target t and horizon h, origin is t-h.",
            "The expanding training sample starts at 1994Q1 and ends at the origin, inclusive.",
            "Historical Mean and Naive use only that training sample.",
            "Seasonal Naive uses t-4, which is at or before the origin for both evaluated horizons.",
            "Each target's seasonal MASE scale uses only seasonal differences in that origin's training sample.",
            "All three benchmarks use the same complete target quarters for each horizon.",
        ],
        "scope": {
            "sarima_or_other_fitted_model": False,
            "model_order_search": False,
            "cny_predictive_test": False,
            "pmi_predictive_test": False,
            "final_forecast": False,
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
    summary = run_backtest()
    window = summary["evaluation_window"]
    counts = summary["target_count_per_model_horizon"]
    print(
        f"Stage 3 benchmark backtest complete: window {window['start']} to {window['end']}, "
        f"targets per model/horizon {counts}."
    )


if __name__ == "__main__":
    main()
