"""Execute and report the frozen exploratory research round without altering archives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_REF = "4b03bdc3b4602b4419478584b170da2e8b5dd927"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/optimization/round1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def validate_output_dir(path: Path) -> Path:
    """Prevent research output from overlapping original repository artifacts."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        if not resolved.is_relative_to(PROJECT_ROOT / "outputs/optimization"):
            raise ValueError("Repository research output must be under outputs/optimization")
    return resolved


def require_archived_bytes(path: Path, expected: bytes) -> None:
    if not path.is_file():
        raise ValueError(f"Original archive missing: {path}")
    if path.read_bytes() != expected:
        raise ValueError(f"Original archive changed: {path}")


def verify_baseline() -> dict:
    """Reuse the existing manifest, then check remaining original artifact bytes."""
    from scripts.modeling.final_forecast import verify_protected_artifacts

    protected = verify_protected_artifacts()
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE_REF, "data", "outputs", "proposal"],
        cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE,
    ).decode("utf-8").splitlines()
    count = 0
    for relative in names:
        if relative in protected:
            continue
        expected = subprocess.check_output(
            ["git", "show", f"{BASE_REF}:{relative}"], cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        require_archived_bytes(PROJECT_ROOT / relative, expected)
        count += 1
    return {"protected_count": len(protected), "other_archived_count": count, "unchanged": True}


def runtime_info() -> dict:
    import matplotlib
    import numpy
    import pandas
    import scipy
    import statsmodels

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {module.__name__: module.__version__ for module in (
            numpy, pandas, scipy, statsmodels, matplotlib
        )},
        "baseline_commit": BASE_REF,
        "blas_threads": 1,
        "data_cutoff": "2026Q2",
        "evidence_status": "exploratory_historical_comparison",
    }


def current_window_status(output_dir: Path) -> tuple[dict, object]:
    """Reconcile current origin fits and forecast keys with the frozen archives."""
    import pandas as pd
    from scripts.optimization.window_backtest import FIT_COLUMNS, FORECAST_COLUMNS, WINDOWS, load_inputs

    _, orders, reference = load_inputs()
    models = {f"SARIMA-W{window}": window for window in WINDOWS}
    expected_origins = {str(origin) for origin in orders}
    expected_keys = set(reference[["origin", "target", "horizon"]].itertuples(index=False, name=None))
    frames = []
    for name, columns, keys in (
        ("window_fits.csv", FIT_COLUMNS, ["model", "origin"]),
        ("window_forecasts.csv", FORECAST_COLUMNS, ["model", "origin", "target", "horizon"]),
    ):
        path = output_dir / name
        frame = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=columns)
        if not set(columns).issubset(frame):
            raise ValueError(f"Missing current {name} columns")
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate current {name} keys")
        if not frame.model.isin(models).all() or not frame.origin.isin(expected_origins).all():
            raise ValueError(f"Unknown current {name} model/origin key")
        for row in frame.itertuples(index=False):
            candidate = orders[pd.Period(row.origin, freq="Q-DEC")]
            if (row.p, row.d, row.q, row.P, row.D, row.Q) != (*candidate.order, *candidate.seasonal_order[:3]):
                raise ValueError(f"Current {name} order differs from archive at {row.origin}")
        frames.append(frame)
    fits, forecasts = frames
    if not fits.status.isin(["successful", "failed"]).all():
        raise ValueError("Unknown current fit status")
    if not fits.empty and not fits.window.eq(fits.model.map(models)).all():
        raise ValueError("Current fit window differs from model label")
    actual_keys = set(forecasts[["origin", "target", "horizon"]].itertuples(index=False, name=None))
    if not actual_keys.issubset(expected_keys):
        raise ValueError("Unknown current forecast target/horizon key")
    coverage, warnings = {}, []
    for model in models:
        subset = fits.loc[fits.model == model]
        successful_origins = set(subset.loc[subset.status == "successful", "origin"])
        group = forecasts.loc[forecasts.model == model]
        keys = set(group[["origin", "target", "horizon"]].itertuples(index=False, name=None))
        successful_keys = {key for key in expected_keys if key[0] in successful_origins}
        fit_complete = successful_origins == expected_origins
        forecast_complete = keys == expected_keys
        consistent = keys == successful_keys
        coverage[model] = {
            "requested": len(expected_origins), "successful": len(successful_origins),
            "failed": int(subset.status.eq("failed").sum()), "pending": len(expected_origins) - len(subset),
            "fit_complete": fit_complete, "forecast_complete": forecast_complete,
            "forecasts_match_successful_fits": consistent,
            "n_records": len(group), "n_expected": len(expected_keys),
            "n_missing_keys": len(expected_keys - keys), "n_unexpected_keys": len(keys - expected_keys),
            "complete": fit_complete and forecast_complete and consistent,
        }
        if not consistent:
            warnings.append(f"{model}: current forecast keys disagree with current successful fit origins.")
    failures = fits.loc[fits.status == "failed", ["model", "origin", "failure_reason"]].fillna("").to_dict("records")
    status = {
        "complete": all(item["complete"] for item in coverage.values()),
        "requested_fits": len(models) * len(expected_origins),
        "successful_fits": int(fits.status.eq("successful").sum()), "failed_fits": len(failures),
        "pending_fits": len(models) * len(expected_origins) - len(fits),
        "forecast_count": len(forecasts), "model_coverage": coverage, "failures": failures,
        "warnings": warnings,
    }
    return status, forecasts


def score_current_windows(output_dir: Path, status: dict, forecasts) -> dict:
    """Exclude models without complete current fit evidence from aggregate scoring."""
    from scripts.optimization.analysis import run_analysis

    eligible_models = {model for model, item in status["model_coverage"].items() if item["fit_complete"]}
    if len(eligible_models) == len(status["model_coverage"]):
        summary = run_analysis(output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="sta4003-analysis-") as directory:
            staging = Path(directory)
            forecasts.loc[forecasts.model.isin(eligible_models)].to_csv(staging / "window_forecasts.csv", index=False)
            summary = run_analysis(staging)
            for name in ("point_metrics.csv", "shock_sensitivity.csv", "paired_losses.csv", "paired_bootstrap.csv",
                         "interval_records.csv", "interval_metrics.csv", "period_metrics.csv",
                         "calendar_allocations.csv", "calendar_variation.csv"):
                shutil.copyfile(staging / name, output_dir / name)
    summary["warnings"] = [warning for warning in summary["warnings"] if not any(
        warning.startswith(f"{model} incomplete:") for model in status["model_coverage"]
    )]
    for model, coverage in status["model_coverage"].items():
        analysis_complete = bool(summary["window_completeness"][model]["complete"])
        if coverage["complete"] and not analysis_complete:
            status["warnings"].append(f"{model}: current forecast coverage disagrees with analysis completeness.")
        coverage["analysis_complete"] = analysis_complete
        coverage["complete"] = coverage["complete"] and analysis_complete
        summary["window_completeness"][model] = {
            key: coverage[key] for key in ("complete", "n_records", "n_expected", "n_missing_keys", "n_unexpected_keys")
        }
        if not coverage["complete"]:
            summary["warnings"].append(
                f"{model} incomplete: {coverage['successful']}/{coverage['requested']} current successful fits; "
                f"{coverage['n_records']}/{coverage['n_expected']} current forecast records; no aggregate scores."
            )
    status["complete"] = all(item["complete"] for item in status["model_coverage"].values())
    summary["warnings"].extend(status["warnings"])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def run_research(output_dir: Path = DEFAULT_OUTPUT, analysis_only: bool = False, retry_failed: bool = False) -> dict:
    from scripts.optimization.reporting import build_figures, build_report
    from scripts.optimization.window_backtest import run_windows

    output_dir = validate_output_dir(output_dir)
    before = verify_baseline()
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_path = output_dir / "window_execution.json"
    if analysis_only:
        if not execution_path.is_file():
            raise ValueError("Run the window experiment before --analysis-only")
        saved_status = json.loads(execution_path.read_text(encoding="utf-8"))
    else:
        saved_status = run_windows(output_dir, retry_failed=retry_failed)
        execution_path.write_text(json.dumps(saved_status, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    window_status, forecasts = current_window_status(output_dir)
    print("Scoring frozen research comparisons", flush=True)
    summary = score_current_windows(output_dir, window_status, forecasts)
    saved_consistent = all(saved_status.get(key) == window_status[key] for key in (
        "complete", "requested_fits", "successful_fits", "failed_fits", "forecast_count"
    )) and all(
        saved_status.get("model_coverage", {}).get(model, {}).get(key) == coverage[key]
        for model, coverage in window_status["model_coverage"].items()
        for key in ("requested", "successful", "failed", "pending", "complete")
    )
    window_status["saved_execution_consistent"] = saved_consistent
    if not saved_consistent:
        window_status["warnings"].append("Saved window_execution.json is stale or inconsistent with current fit/forecast/analysis outputs; retained for traceability only.")
    figures = build_figures(output_dir)
    execution = {
        "runtime": runtime_info(),
        "window_study": window_status,
        "saved_window_execution": saved_status,
        "baseline_before": before,
        "baseline_after": verify_baseline(),
        "figures": figures,
        "complete": bool(window_status["complete"]),
        "commands": [
            "python scripts/optimization/run_research.py",
            "python -m pytest -q",
            "python -m compileall scripts tests",
        ],
    }
    (output_dir / "execution.json").write_text(
        json.dumps(execution, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    build_report(output_dir, execution)
    with (output_dir / "research_report.md").open("a", encoding="utf-8") as report:
        report.write("\n## 当前窗口输出核对\n\n")
        report.write("当前拟合表、预测键与分析覆盖共同决定完成状态；缓存仅用于追溯。\n\n")
        for model, coverage in window_status["model_coverage"].items():
            report.write(f"- {model}：成功 {coverage['successful']}/{coverage['requested']}，"
                         f"失败 {coverage['failed']}，待完成 {coverage['pending']}；"
                         f"当前预测 {coverage['n_records']}/{coverage['n_expected']}，完整状态 {coverage['complete']}。\n")
        if not saved_consistent:
            report.write("\n缓存 window_execution.json 与当前输出不一致，旧完成状态不作为本次研究完成依据。\n")
        for warning in window_status["warnings"]:
            report.write(f"\n{warning}\n")
    print(json.dumps({"complete": execution["complete"], "output": str(output_dir),
                      "window_fits": window_status["successful_fits"],
                      "failed_fits": window_status["failed_fits"]}), flush=True)
    return {"execution": execution, "analysis": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis-only", action="store_true", help="Rebuild scoring/report from completed fitting artifacts")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly retry failed per-origin checkpoints")
    args = parser.parse_args()
    result = run_research(args.output_dir, args.analysis_only, args.retry_failed)
    if not result["execution"]["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
