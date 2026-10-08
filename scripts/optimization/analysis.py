"""Exploratory scoring of frozen forecasts; never fit or tune a model here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEYS = ["origin", "target", "horizon"]
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261008
BLOCK_LENGTHS = (4, 8)
COVERAGES = (0.8, 0.95)
CALIBRATION_WINDOW = 40
CALIBRATION_MINIMUM = 20
ORIGINAL_SHOCK_TARGETS = ("2020Q1", "2022Q2")
POINT_COLUMNS = [
    "model", "horizon", "scope", "point_convention", "n", "MAE", "RMSE", "MASE",
]
INTERVAL_COLUMNS = [
    "model", "horizon", "scope", "coverage", "method", "subset", "n",
    "empirical_coverage", "mean_width", "winkler_score",
]


# Validation, pairing and chronological block resampling.


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["model", *KEYS, "actual", "forecast", "mase_scale"]
    _require(set(required).issubset(frame), f"Missing forecast columns: {set(required) - set(frame)}")
    _require(not frame.empty, "Forecast records must not be empty")
    result = frame.copy()
    _require(result["model"].notna().all(), "Model labels must not be missing")
    horizons = pd.to_numeric(result["horizon"], errors="raise").to_numpy(dtype=float)
    _require(np.isfinite(horizons).all() and np.isin(horizons, [1, 2]).all(), "Horizons must be 1 or 2")
    result["horizon"] = horizons.astype(int)
    origins = pd.PeriodIndex(result["origin"].astype(str), freq="Q-DEC")
    targets = pd.PeriodIndex(result["target"].astype(str), freq="Q-DEC")
    _require(not origins.hasnans and not targets.hasnans, "Origin and target must be valid quarters")
    _require(np.array_equal(targets.asi8 - origins.asi8, horizons), "Origin-target horizon mismatch")
    result["origin"] = origins.astype(str)
    result["target"] = targets.astype(str)
    _require(not result.duplicated(["model", *KEYS]).any(), "Duplicate model/forecast key")
    for column in ("actual", "forecast", "mase_scale"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        _require(np.isfinite(result[column]).all(), f"{column} must be finite")
    _require((result["actual"] > 0).all(), "Actual levels must be positive")
    _require((result["mase_scale"] > 0).all(), "MASE scales must be positive")
    return result.sort_values(["model", "horizon", "target", "origin"]).reset_index(drop=True)


def paired_losses(reference: pd.DataFrame, extended: pd.DataFrame) -> pd.DataFrame:
    """Pair complete keys, preserving archived scales and signed loss deltas."""
    reference = _validate_forecasts(reference)
    extended = _validate_forecasts(extended)
    _require(reference["model"].nunique() == extended["model"].nunique() == 1, "Pairing needs one model per input")
    columns = [*KEYS, "model", "actual", "forecast", "mase_scale"]
    paired = reference[columns].merge(
        extended[columns], on=KEYS, how="outer", validate="one_to_one",
        suffixes=("_reference", "_extended"), indicator=True,
    )
    _require((paired["_merge"] == "both").all(), "Reference and extension target keys differ")
    for column in ("actual", "mase_scale"):
        _require(
            np.allclose(
                paired[f"{column}_reference"], paired[f"{column}_extended"],
                rtol=1e-12, atol=1e-10,
            ),
            f"Paired {column} values differ",
        )
    output = paired[KEYS].copy()
    output["reference_model"] = paired["model_reference"]
    output["extended_model"] = paired["model_extended"]
    output["actual"] = paired["actual_reference"]
    output["mase_scale"] = paired["mase_scale_reference"]
    for name in ("reference", "extended"):
        output[f"{name}_forecast"] = paired[f"forecast_{name}"]
        error = output["actual"] - output[f"{name}_forecast"]
        output[f"{name}_error"] = error
        output[f"{name}_absolute_error"] = error.abs()
        output[f"{name}_squared_error"] = error**2
        output[f"{name}_scaled_absolute_error"] = error.abs() / output["mase_scale"]
    for loss in ("absolute_error", "squared_error", "scaled_absolute_error"):
        output[f"{loss}_delta"] = output[f"extended_{loss}"] - output[f"reference_{loss}"]
    return output.sort_values(["target", "origin", "horizon"]).reset_index(drop=True)


def bootstrap_deltas(paired: pd.DataFrame, block_length: int, reps: int, seed: int) -> pd.DataFrame:
    """Circular blocks count retained chronological observations, including Q1."""
    _require(isinstance(block_length, (int, np.integer)) and block_length > 0, "Block length must be positive integer")
    _require(isinstance(reps, (int, np.integer)) and reps > 0, "Repetitions must be positive integer")
    _require(not paired.empty, "Paired losses must not be empty")
    _require(paired["horizon"].nunique() == 1, "Bootstrap must retain one horizon")
    _require(not paired.duplicated(KEYS).any(), "Duplicate paired key")
    paired = paired.sort_values(["target", "origin", "horizon"]).reset_index(drop=True)
    n = len(paired)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(reps, int(np.ceil(n / block_length))))
    indices = ((starts[:, :, None] + np.arange(block_length)) % n).reshape(reps, -1)[:, :n]
    rows = []
    for metric, loss in (
        ("MAE", "absolute_error"), ("RMSE", "squared_error"),
        ("MASE", "scaled_absolute_error"),
    ):
        reference = paired[f"reference_{loss}"].to_numpy(dtype=float)
        extended = paired[f"extended_{loss}"].to_numpy(dtype=float)
        _require(np.isfinite(reference).all() and np.isfinite(extended).all(), "Paired losses must be finite")
        transform = np.sqrt if metric == "RMSE" else lambda value: value
        delta = float(transform(extended.mean()) - transform(reference.mean()))
        samples = transform(extended[indices].mean(axis=1)) - transform(reference[indices].mean(axis=1))
        lower, upper = np.quantile(samples, [0.025, 0.975])
        rows.append({
            "metric": metric, "delta": delta, "lower_95": float(lower),
            "upper_95": float(upper), "n": n, "block_length": int(block_length),
            "reps": int(reps), "seed": int(seed),
        })
    return pd.DataFrame(rows)


# Point-functional sensitivity and fixed combination.


def _scopes(frame: pd.DataFrame):
    yield "all", frame
    q1 = frame.loc[pd.PeriodIndex(frame["target"], freq="Q-DEC").quarter == 1]
    if not q1.empty:
        yield "Q1", q1


def point_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Score saved means and medians separately, using unchanged MASE scales."""
    forecasts = _validate_forecasts(forecasts)
    rows = []
    for (model, horizon), group in forecasts.groupby(["model", "horizon"], sort=True):
        conventions = [("mean", "forecast")]
        if "forecast_level_median" in group and group["forecast_level_median"].notna().any():
            _require(np.isfinite(group["forecast_level_median"]).all() and (group["forecast_level_median"] > 0).all(), f"Invalid median for {model}")
            conventions.append(("median", "forecast_level_median"))
        for scope, subset in _scopes(group):
            for convention, column in conventions:
                error = (subset["actual"] - subset[column]).to_numpy(dtype=float)
                rows.append({
                    "model": model, "horizon": int(horizon), "scope": scope,
                    "point_convention": convention, "n": len(subset),
                    "MAE": float(np.abs(error).mean()),
                    "RMSE": float(np.sqrt(np.mean(error**2))),
                    "MASE": float((np.abs(error) / subset["mase_scale"].to_numpy(dtype=float)).mean()),
                })
    return pd.DataFrame(rows, columns=POINT_COLUMNS)


def combine_forecasts(reference: pd.DataFrame, ets: pd.DataFrame) -> pd.DataFrame:
    """Fixed half weights on level points; infer no median or distribution."""
    paired = paired_losses(reference, ets)
    result = paired[KEYS + ["actual", "mase_scale"]].copy()
    result["model"] = "SARIMA+ETS-50:50"
    result["forecast"] = 0.5 * paired["reference_forecast"] + 0.5 * paired["extended_forecast"]
    result["error"] = result["actual"] - result["forecast"]
    result["absolute_error"] = result["error"].abs()
    result["squared_error"] = result["error"]**2
    result["scaled_absolute_error"] = result["absolute_error"] / result["mase_scale"]
    return result


def shock_sensitivity(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Secondary mean scoring excludes only the original Stage 6B targets."""
    forecasts = _validate_forecasts(forecasts).drop(
        columns=["forecast_level_median"], errors="ignore",
    )
    tables = []
    for sample, subset in (
        ("all", forecasts),
        ("exclude_original_shocks", forecasts.loc[~forecasts["target"].isin(ORIGINAL_SHOCK_TARGETS)]),
    ):
        if subset.empty:
            continue
        table = point_metrics(subset)
        table["score_sample"] = sample
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


# Parametric and prior-error calibrated intervals.


def interval_records(forecasts: pd.DataFrame, calibrate: bool = False) -> pd.DataFrame:
    """Use log quantiles; calibration can observe only targets <= origin."""
    forecasts = _validate_forecasts(forecasts)
    for column in ("forecast_log_mean", "forecast_log_variance"):
        _require(column in forecasts, f"Missing {column}")
        forecasts[column] = pd.to_numeric(forecasts[column], errors="raise").astype(float)
        _require(np.isfinite(forecasts[column]).all(), "Log forecast moments must be finite")
    _require((forecasts["forecast_log_variance"] >= 0).all(), "Log variance must be nonnegative")
    rows = []
    for (model, horizon), group in forecasts.groupby(["model", "horizon"], sort=True):
        targets = pd.PeriodIndex(group["target"], freq="Q-DEC")
        with np.errstate(divide="ignore", invalid="ignore"):
            standardized = (
                np.abs(np.log(group["actual"].to_numpy()) - group["forecast_log_mean"].to_numpy())
                / np.sqrt(group["forecast_log_variance"].to_numpy())
            )
        for record in group.itertuples(index=False):
            eligible = (
                np.flatnonzero(targets <= pd.Period(record.origin, freq="Q-DEC"))[-CALIBRATION_WINDOW:]
                if calibrate else np.asarray([], dtype=int)
            )
            n_calibration = len(eligible)
            last_target = str(targets[eligible[-1]]) if n_calibration else ""
            fallback_reason = ""
            if calibrate:
                if n_calibration < CALIBRATION_MINIMUM:
                    fallback_reason = "insufficient_history"
                elif record.forecast_log_variance <= 0:
                    fallback_reason = "invalid_current_variance"
                elif not np.isfinite(standardized[eligible]).all():
                    fallback_reason = "invalid_calibration_errors"
            for coverage in COVERAGES:
                critical = float(norm.ppf((1.0 + coverage) / 2.0))
                if calibrate and not fallback_reason:
                    critical = float(np.quantile(standardized[eligible], coverage, method="higher"))
                radius = critical * np.sqrt(record.forecast_log_variance)
                with np.errstate(over="ignore", invalid="ignore"):
                    lower, upper = np.exp([record.forecast_log_mean - radius, record.forecast_log_mean + radius])
                _require(np.isfinite([lower, upper]).all() and lower > 0, "Interval endpoints must be finite and positive")
                width = float(upper - lower)
                penalty = (2.0 / (1.0 - coverage)) * max(
                    float(lower - record.actual), float(record.actual - upper), 0.0,
                )
                rows.append({
                    "model": model, "horizon": int(horizon), "origin": record.origin,
                    "target": record.target, "actual": float(record.actual),
                    "coverage": coverage, "method": "calibrated" if calibrate else "parametric",
                    "lower": float(lower), "upper": float(upper), "width": width,
                    "covered": bool(lower <= record.actual <= upper),
                    "winkler_score": width + penalty, "critical_value": critical,
                    "n_calibration": n_calibration, "fallback": bool(calibrate and fallback_reason),
                    "fallback_reason": fallback_reason, "calibration_last_target": last_target,
                })
    return pd.DataFrame(rows)


def _interval_metrics(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon, coverage, method), group in records.groupby(["model", "horizon", "coverage", "method"], sort=True):
        for scope, scoped in _scopes(group):
            subsets = [("all", scoped)]
            if method == "calibrated":
                subsets.append(("calibrated_only", scoped.loc[~scoped["fallback"]]))
            for subset_name, subset in subsets:
                rows.append({
                    "model": model, "horizon": int(horizon), "scope": scope,
                    "coverage": float(coverage), "method": method, "subset": subset_name,
                    "n": len(subset),
                    "empirical_coverage": float(subset["covered"].mean()) if len(subset) else np.nan,
                    "mean_width": float(subset["width"].mean()) if len(subset) else np.nan,
                    "winkler_score": float(subset["winkler_score"].mean()) if len(subset) else np.nan,
                })
    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS)


# Calendar aggregation and file-backed research scoring.


def calendar_diagnostic(calendar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition each inclusive calendar window into Q1, prior Q4 and other."""
    _require({"year", "cny_date"}.issubset(calendar), "Calendar needs year and cny_date")
    _require(not calendar.empty and not calendar["year"].duplicated().any(), "Calendar years must be nonempty and unique")
    rows = []
    for record in calendar.sort_values("year").itertuples(index=False):
        year = int(record.year)
        date = pd.Timestamp(record.cny_date).normalize()
        _require(not pd.isna(date) and date.year == year and date.month in (1, 2), "CNY date and year mismatch")
        for start_offset in (-14, -30):
            days = pd.date_range(date + pd.Timedelta(days=start_offset), date + pd.Timedelta(days=7), freq="D")
            q1_days = int(((days.year == year) & (days.quarter == 1)).sum())
            prior_days = int(((days.year == year - 1) & (days.quarter == 4)).sum())
            other_days = int(len(days) - q1_days - prior_days)
            _require(q1_days + prior_days + other_days == 8 - start_offset, "Calendar day partition failed")
            rows.append({
                "year": year, "cny_date": date.date().isoformat(),
                "window": f"[{start_offset},+7]", "start_offset": start_offset,
                "end_offset": 7, "total_days": len(days), "q1_days": q1_days,
                "previous_q4_days": prior_days, "other_days": other_days,
                "q1_fraction": q1_days / len(days), "previous_q4_fraction": prior_days / len(days),
            })
    allocations = pd.DataFrame(rows)
    variations = []
    for window, group in allocations.groupby("window", sort=True):
        row = {"window": window, "n_years": len(group)}
        for prefix in ("q1", "previous_q4"):
            fractions = group[f"{prefix}_fraction"]
            row.update({
                f"{prefix}_fraction_min": float(fractions.min()),
                f"{prefix}_fraction_max": float(fractions.max()),
                f"{prefix}_fraction_std": float(fractions.std(ddof=0)),
                f"{prefix}_n_unique": int(fractions.nunique()),
                f"{prefix}_constant_regressor_risk": bool(fractions.nunique() == 1),
            })
        variations.append(row)
    return allocations, pd.DataFrame(variations)


def _archive_agreement(metrics: pd.DataFrame) -> None:
    for name in ("sarima_metrics.csv", "cny_metrics.csv"):
        archived = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / name)
        for record in archived.itertuples(index=False):
            computed = metrics.loc[(metrics["model"] == record.model) & (metrics["horizon"] == record.horizon) & (metrics["scope"].str.lower() == record.scope.lower()) & (metrics["point_convention"] == "mean")]
            _require(len(computed) == 1, f"Missing archived metric key in {name}")
            _require(np.allclose(computed[["MAE", "RMSE", "MASE"]].to_numpy(dtype=float)[0], [record.MAE, record.RMSE, record.MASE], rtol=1e-12, atol=1e-8), f"Archived metrics differ in {name}")


def _period_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    years = pd.PeriodIndex(forecasts["target"], freq="Q-DEC").year
    tables = []
    for first, last in ((2005, 2011), (2012, 2019), (2020, 2026)):
        subset = forecasts.loc[(years >= first) & (years <= last)]
        if not subset.empty:
            table = point_metrics(subset)
            table.insert(0, "period", f"{first}-{last}")
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def run_analysis(output_dir: Path) -> dict:
    """Read archives and complete windows, writing exclusively to output_dir."""
    output_dir = Path(output_dir).resolve()
    archive_dir = PROJECT_ROOT / "outputs" / "forecasts"
    reference = _validate_forecasts(pd.read_csv(archive_dir / "sarima_rolling_forecasts.csv"))
    cny = _validate_forecasts(pd.read_csv(archive_dir / "sarima_cny_rolling_forecasts.csv"))
    ets = _validate_forecasts(pd.read_csv(archive_dir / "ets_rolling_forecasts.csv"))
    targets = pd.period_range("2005Q1", "2026Q2", freq="Q-DEC")
    expected_keys = {(str(target - horizon), str(target), horizon) for target in targets for horizon in (1, 2)}
    _require(set(map(tuple, reference[KEYS].to_numpy())) == expected_keys, "SARIMA archive target keys are incomplete")
    paired_losses(reference, cny)
    combination = combine_forecasts(reference, ets)
    all_forecasts = [reference, cny, ets, combination]
    moment_forecasts = [reference, cny]
    extensions = [cny, combination]
    windows_path = output_dir / "window_forecasts.csv"
    windows = pd.read_csv(windows_path) if windows_path.is_file() else pd.DataFrame()
    if not windows.empty:
        windows = _validate_forecasts(windows)
        _require(set(windows["model"]).issubset({"SARIMA-W40", "SARIMA-W60"}), "Unknown window model label")
    completeness = {}
    warnings = []
    for model in ("SARIMA-W40", "SARIMA-W60"):
        group = windows.loc[windows["model"] == model] if not windows.empty else pd.DataFrame()
        keys = set(map(tuple, group[KEYS].to_numpy())) if not group.empty else set()
        complete = len(group) == len(expected_keys) and keys == expected_keys
        completeness[model] = {"complete": bool(complete), "n_records": len(group), "n_expected": len(expected_keys), "n_missing_keys": len(expected_keys - keys), "n_unexpected_keys": len(keys - expected_keys)}
        if complete:
            paired_losses(reference, group)
            all_forecasts.append(group)
            moment_forecasts.append(group)
            extensions.append(group)
        else:
            warnings.append(f"{model} incomplete: {len(group)}/{len(expected_keys)} records; no aggregate scores.")
    forecasts = pd.concat(all_forecasts, ignore_index=True, sort=False)
    metrics = point_metrics(forecasts)
    _archive_agreement(metrics)
    pair_tables = []
    bootstrap_tables = []
    for extension in extensions:
        conventions = ["mean"]
        if "forecast_level_median" in extension:
            conventions.append("median")
        for convention in conventions:
            ref = reference.copy()
            ext = extension.copy()
            if convention == "median":
                ref["forecast"] = ref["forecast_level_median"]
                ext["forecast"] = ext["forecast_level_median"]
            paired = paired_losses(ref, ext)
            paired["point_convention"] = convention
            pair_tables.append(paired)
            for horizon, horizon_group in paired.groupby("horizon", sort=True):
                for scope, subset in _scopes(horizon_group):
                    for block_length in BLOCK_LENGTHS:
                        table = bootstrap_deltas(subset, block_length, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
                        for key, value in {"reference_model": str(paired.iloc[0]["reference_model"]), "extended_model": str(paired.iloc[0]["extended_model"]), "horizon": int(horizon), "scope": scope, "point_convention": convention}.items():
                            table[key] = value
                        bootstrap_tables.append(table)
    moments = pd.concat(moment_forecasts, ignore_index=True, sort=False)
    intervals = pd.concat([interval_records(moments), interval_records(moments, calibrate=True)], ignore_index=True)
    interval_metrics = _interval_metrics(intervals)
    allocations, variation = calendar_diagnostic(pd.read_csv(PROJECT_ROOT / "data" / "meta" / "lunar_new_year.csv"))
    periods = _period_metrics(forecasts)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in {
        "point_metrics.csv": metrics,
        "shock_sensitivity.csv": shock_sensitivity(forecasts),
        "paired_losses.csv": pd.concat(pair_tables, ignore_index=True),
        "paired_bootstrap.csv": pd.concat(bootstrap_tables, ignore_index=True),
        "interval_records.csv": intervals,
        "interval_metrics.csv": interval_metrics,
        "period_metrics.csv": periods,
        "calendar_allocations.csv": allocations,
        "calendar_variation.csv": variation,
    }.items():
        table.to_csv(output_dir / filename, index=False, float_format="%.15g")
    warnings.extend([
        "Historical experiment specifications were defined after benchmark inspection; results are exploratory.",
        "Percentile intervals are descriptive conditional on archived fits; no significance, zero-effect or multiplicity-adjusted claim.",
        "Q1 blocks count retained annual observations; block length 4 spans four Q1 years.",
        "Calibrated all-target intervals mix parametric fallback and calibrated intervals; calibrated_only excludes fallback.",
        "Period comparisons are supplementary descriptive analyses; no independent future evaluation.",
        "Target-only shock sensitivity excludes only the original Stage 6B targets 2020Q1 and 2022Q2; secondary scores do not replace the primary endpoint.",
    ])
    if variation["q1_constant_regressor_risk"].any():
        warnings.append("At least one holiday window has constant Q1 allocation; calendar diagnostic does not establish an economic effect.")
    summary = {
        "evidence_status": "exploratory",
        "archived_metric_agreement": True,
        "window_completeness": completeness,
        "primary_h2_mean_scores": metrics.loc[(metrics["horizon"] == 2) & (metrics["scope"] == "all") & (metrics["point_convention"] == "mean")].to_dict(orient="records"),
        "experiment_specs": {
            "target_window": "2005Q1-2026Q2", "primary_horizon": 2,
            "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "block_lengths": list(BLOCK_LENGTHS), "method": "paired_chronological_circular_blocks", "delta_direction": "extended_minus_reference", "percentile_interval": [0.025, 0.975]},
            "calibration": {"last_n": CALIBRATION_WINDOW, "minimum": CALIBRATION_MINIMUM, "eligible": "same model/horizon target <= origin", "statistic": "absolute standardized log error", "quantile_method": "higher", "coverages": list(COVERAGES)},
            "combination": {"model": "SARIMA+ETS-50:50", "sarima_weight": 0.5, "ets_weight": 0.5},
            "mase_scale": "unchanged archived Stage 4 origin-specific scale",
            "periods": ["2005-2011", "2012-2019", "2020-2026"],
            "shock_sensitivity": {"excluded_targets": list(ORIGINAL_SHOCK_TARGETS), "source": "original Stage 6B preset targets", "role": "secondary target-only scoring; no refits"},
            "calendar_windows": [[-14, 7], [-30, 7]],
        },
        "warnings": warnings,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary
