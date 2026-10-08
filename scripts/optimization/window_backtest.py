"""Checkpointed fixed-window forecasts using archived origin-specific orders."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.modeling import backtest_cny as archived
from scripts.modeling import backtest_pmi, backtest_sarima

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

WINDOWS = (40, 60)
FORECAST_COLUMNS = [
    "model", "horizon", "origin", "target", "training_start", "training_end",
    "n_train", "p", "d", "q", "P", "D", "Q", "actual", "forecast_log_mean",
    "forecast_log_variance", "forecast_level_median", "forecast", "error",
    "absolute_error", "squared_error", "mase_scale", "scaled_absolute_error",
]
FIT_COLUMNS = [
    "model", "window", "origin", "training_start", "training_end", "n_train",
    "p", "d", "q", "P", "D", "Q", "status", "attempt_count", "attempts_json",
    "failure_reason", "checkpoint_reused",
]
ESTIMATOR = {
    "trend": "n", "simple_differencing": False, "enforce_stationarity": True,
    "enforce_invertibility": True, "concentrate_scale": False,
}
FIT_OPTIONS = {"method": "lbfgs", "disp": False, "maxiter": 100}


@dataclass
class WindowFit:
    result: object | None
    attempts: list[dict]
    failure_reason: str | None

    @property
    def converged(self) -> bool:
        return self.result is not None and self.failure_reason is None


def window_training(frame: pd.DataFrame, origin: pd.Period, window: int) -> pd.DataFrame:
    """Return the final min(window, available) contiguous quarters through origin."""
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("Training window must be a positive integer")
    if not isinstance(frame.index, pd.PeriodIndex) or not frame.index.is_unique:
        raise ValueError("Training frame must have a unique quarterly PeriodIndex")
    if not frame.index.is_monotonic_increasing or origin not in frame.index:
        raise ValueError("Training quarters must be ordered and contain the origin")
    training = frame.loc[:origin].tail(window).copy()
    expected = pd.period_range(training.index[0], origin, freq="Q-DEC")
    if not training.index.equals(expected):
        raise ValueError("Training quarters must be contiguous")
    if "log_retail" not in training or not np.isfinite(training.log_retail.to_numpy(dtype=float)).all():
        raise ValueError("Training log response must be finite")
    return training


def _finite_values(values) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None
            for value in np.asarray(values, dtype=float).reshape(-1)]


def fit_window(training: pd.DataFrame, candidate) -> WindowFit:
    """Fit one order with initial, terminal continuation and deterministic starts."""
    attempts = []
    result = None
    try:
        model = SARIMAX(training["log_retail"].astype(float), order=candidate.order,
                        seasonal_order=candidate.seasonal_order, **ESTIMATOR)
    except Exception as exc:
        reason = f"Model construction: {type(exc).__name__}: {exc}"
        return WindowFit(None, [{"attempt": 0, "error": reason, "converged": False}], reason)
    for strategy in ("initial", "terminal_continuation", "deterministic_order"):
        if strategy == "terminal_continuation" and result is None:
            continue
        record = {"attempt": len(attempts) + 1, "start": strategy, **FIT_OPTIONS}
        try:
            start = (result.params if strategy == "terminal_continuation" else
                     backtest_pmi.deterministic_order_start(model, ())
                     if strategy == "deterministic_order" else None)
            record["start_params"] = None if start is None else _finite_values(start)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(**FIT_OPTIONS) if start is None else model.fit(start_params=start, **FIT_OPTIONS)
            converged = bool(result.mle_retvals.get("converged", False))
            valid = bool(np.isfinite(np.asarray(result.params, dtype=float)).all()
                         and np.isfinite([result.llf, result.aic]).all())
            record.update(converged=converged, valid_parameters=valid,
                          terminal_params=_finite_values(result.params),
                          llf=_finite_values([result.llf])[0], aic=_finite_values([result.aic])[0],
                          iterations=int(result.mle_retvals.get("iterations", 0)),
                          warnings=[str(item.message) for item in caught])
            attempts.append(record)
            if converged:
                return WindowFit(result, attempts, None if valid else "Non-finite fitted parameters or likelihood")
        except Exception as exc:
            record.update(converged=False, error=f"{type(exc).__name__}: {exc}")
            attempts.append(record)
            result = None
    return WindowFit(result, attempts, "Frozen-order L-BFGS fit did not converge after permitted attempts")


def load_inputs() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Read frozen Stage 4 orders and scoring records without regenerating stages."""
    frame = backtest_sarima.load_analysis_data()
    orders, _ = archived.load_locked_orders()
    reference = archived.load_stage4_forecasts()
    for row in reference.itertuples(index=False):
        candidate = orders[pd.Period(row.origin, freq="Q-DEC")]
        if (row.p, row.d, row.q, row.P, row.D, row.Q) != (*candidate.order, *candidate.seasonal_order[:3]):
            raise ValueError(f"Archived forecast order differs from origin selection at {row.origin}")
        actual = float(frame.loc[pd.Period(row.target, freq="Q-DEC"), "retail_bn"])
        if not np.isclose(actual, row.actual, rtol=1e-12, atol=1e-8):
            raise ValueError(f"Archived actual differs from canonical target {row.target}")
    return frame, orders, reference


def _specification(training, candidate, scoring, window) -> dict:
    return {
        "schema_version": 1, "window": window,
        "training": [[str(quarter), float(value).hex()]
                     for quarter, value in training.log_retail.items()],
        "order": list(candidate.order), "seasonal_order": list(candidate.seasonal_order),
        "estimator": ESTIMATOR, "fit_options": FIT_OPTIONS,
        "retry_sequence": ["initial", "terminal_continuation", "deterministic_order"],
        "scoring": [{"horizon": int(row.horizon), "origin": str(row.origin), "target": str(row.target),
                     "actual": float(row.actual).hex(), "mase_scale": float(row.mase_scale).hex()}
                    for row in scoring.itertuples(index=False)],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__},
    }


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _base_row(training, candidate, model, origin) -> dict:
    return {"model": model, "origin": str(origin), "training_start": str(training.index[0]),
            "training_end": str(training.index[-1]), "n_train": len(training),
            **{name: int(getattr(candidate, name)) for name in ("p", "d", "q", "P", "D", "Q")}}


def _forecast_rows(fitted, base, scoring) -> list[dict]:
    if not fitted.converged:
        raise ValueError(fitted.failure_reason or "Fit failed")
    steps = int(scoring.horizon.max())
    prediction = fitted.result.get_forecast(steps=steps)
    means = np.asarray(prediction.predicted_mean, dtype=float).reshape(-1)
    variances = np.asarray(prediction.var_pred_mean, dtype=float).reshape(-1)
    if len(means) != steps or len(variances) != steps:
        raise ValueError("Unexpected forecast moment shape")
    rows = []
    for row in scoring.itertuples(index=False):
        mu, variance = float(means[row.horizon - 1]), float(variances[row.horizon - 1])
        median, forecast = backtest_sarima.lognormal_level_forecasts(mu, variance)
        actual, scale = float(row.actual), float(row.mase_scale)
        error = actual - forecast
        if not np.isfinite([actual, scale, error, error * error]).all() or scale <= 0:
            raise ValueError("Invalid scoring values")
        rows.append({**base, "horizon": int(row.horizon), "target": str(row.target),
                     "actual": actual, "forecast_log_mean": mu, "forecast_log_variance": variance,
                     "forecast_level_median": median, "forecast": forecast, "error": error,
                     "absolute_error": abs(error), "squared_error": error * error,
                     "mase_scale": scale, "scaled_absolute_error": abs(error) / scale})
    return rows


def _load_checkpoint(path, specification, base) -> dict | None:
    """Reject stale specifications and malformed saved successes, then resume."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["specification"] != specification or not isinstance(saved["attempts"], list):
            return None
        if saved["status"] == "failed":
            return saved if saved["failure_reason"] and saved["forecasts"] == [] else None
        if saved["status"] != "successful" or saved["failure_reason"] is not None:
            return None
        if len(saved["forecasts"]) != len(specification["scoring"]):
            return None
        for row, scoring in zip(saved["forecasts"], specification["scoring"]):
            if set(row) != set(FORECAST_COLUMNS) or any(row[key] != value for key, value in base.items()):
                return None
            if any(row[key] != scoring[key] for key in ("origin", "target", "horizon")):
                return None
            if row["actual"] != float.fromhex(scoring["actual"]) or row["mase_scale"] != float.fromhex(scoring["mase_scale"]):
                return None
            median, forecast = backtest_sarima.lognormal_level_forecasts(row["forecast_log_mean"], row["forecast_log_variance"])
            error, scale = row["actual"] - forecast, row["mase_scale"]
            expected = {"forecast_level_median": median, "forecast": forecast, "error": error,
                        "absolute_error": abs(error), "squared_error": error * error,
                        "scaled_absolute_error": abs(error) / scale}
            if any(row[key] != value or not np.isfinite(value) for key, value in expected.items()):
                return None
        return saved
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        return None


def _export(output_dir, forecast_rows, fit_rows, requested_count) -> dict:
    forecasts = pd.DataFrame(forecast_rows, columns=FORECAST_COLUMNS).sort_values(
        ["model", "horizon", "target"], kind="stable")
    fits = pd.DataFrame(fit_rows, columns=FIT_COLUMNS).sort_values(["model", "origin"], kind="stable")
    for name, frame in (("window_forecasts.csv", forecasts), ("window_fits.csv", fits)):
        _atomic_text(output_dir / name, frame.to_csv(index=False, float_format="%.17g", lineterminator="\n"))
    coverage = {}
    for window in WINDOWS:
        model = f"SARIMA-W{window}"
        subset = fits.loc[fits.model == model]
        successful = int(subset.status.eq("successful").sum())
        failed = int(subset.status.eq("failed").sum())
        coverage[model] = {"requested": requested_count, "successful": successful, "failed": failed,
                           "pending": requested_count - len(subset), "complete": successful == requested_count}
    failures = fits.loc[fits.status == "failed", ["model", "origin", "failure_reason"]].to_dict("records")
    summary = {"complete": all(item["complete"] for item in coverage.values()),
               "requested_fits": requested_count * len(WINDOWS),
               "successful_fits": int(fits.status.eq("successful").sum()),
               "failed_fits": len(failures), "forecast_count": len(forecasts),
               "model_coverage": coverage, "failures": failures,
               "generated_artifacts": ["window_forecasts.csv", "window_fits.csv", "window_summary.json"]}
    _atomic_text(output_dir / "window_summary.json", json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def run_windows(output_dir: Path, retry_failed: bool = False) -> dict:
    """Checkpoint every origin/window; export successes and explicit failures."""
    output_dir = Path(output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frame, orders, reference = load_inputs()
    if reference.duplicated(["origin", "horizon", "target"]).any():
        raise ValueError("Duplicate archived scoring rows")
    forecast_rows, fit_rows = [], []
    summary = _export(output_dir, forecast_rows, fit_rows, len(orders))
    for window in WINDOWS:
        model = f"SARIMA-W{window}"
        for position, (origin, candidate) in enumerate(sorted(orders.items()), start=1):
            training = window_training(frame, origin, window)
            scoring = reference.loc[reference.origin == str(origin)].sort_values("horizon", kind="stable")
            if scoring.empty or not scoring.horizon.isin([1, 2]).all():
                raise ValueError(f"Missing/invalid archived scoring metadata at {origin}")
            for row in scoring.itertuples(index=False):
                if pd.Period(row.target, freq="Q-DEC") != origin + int(row.horizon):
                    raise ValueError(f"Archived target/horizon alignment failed at {origin}")
            base = _base_row(training, candidate, model, origin)
            specification = _specification(training, candidate, scoring, window)
            path = checkpoint_dir / f"{model}_{origin}.json"
            saved = _load_checkpoint(path, specification, base)
            reused = saved is not None and not (retry_failed and saved["status"] == "failed")
            if not reused:
                attempts = []
                try:
                    fitted = fit_window(training, candidate)
                    attempts = fitted.attempts
                    rows = _forecast_rows(fitted, base, scoring)
                    status, failure = "successful", None
                except Exception as exc:
                    rows, status, failure = [], "failed", f"{type(exc).__name__}: {exc}"
                saved = {"specification": specification, "status": status, "attempts": attempts,
                         "failure_reason": failure, "forecasts": rows}
                _atomic_text(path, json.dumps(saved, indent=2, sort_keys=True, allow_nan=False) + "\n")
            forecast_rows.extend(saved["forecasts"])
            fit_rows.append({**base, "window": window, "status": saved["status"],
                             "attempt_count": len(saved["attempts"]),
                             "attempts_json": json.dumps(saved["attempts"], sort_keys=True, allow_nan=False),
                             "failure_reason": saved["failure_reason"], "checkpoint_reused": reused})
            summary = _export(output_dir, forecast_rows, fit_rows, len(orders))
            action = "resumed" if reused else "fitted"
            print(f"{model} {position}/{len(orders)} origin={origin} {saved['status']} ({action}, attempts={len(saved['attempts'])})", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/optimization/round1")
    parser.add_argument("--retry-failed", action="store_true")
    arguments = parser.parse_args()
    result = run_windows(arguments.output_dir, retry_failed=arguments.retry_failed)
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0 if result["complete"] else 1)
