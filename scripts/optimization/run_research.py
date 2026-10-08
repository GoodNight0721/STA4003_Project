"""Execute and report the frozen exploratory research round without altering archives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

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
        cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
    ).decode("utf-8").splitlines()
    count = 0
    for relative in names:
        if relative in protected:
            continue
        expected = subprocess.check_output(
            ["git", "show", f"{BASE_REF}:{relative}"], cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL
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


def run_research(output_dir: Path = DEFAULT_OUTPUT, analysis_only: bool = False, retry_failed: bool = False) -> dict:
    from scripts.optimization.analysis import run_analysis
    from scripts.optimization.reporting import build_figures, build_report
    from scripts.optimization.window_backtest import run_windows

    output_dir = validate_output_dir(output_dir)
    before = verify_baseline()
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_path = output_dir / "window_execution.json"
    if analysis_only:
        if not execution_path.is_file():
            raise ValueError("Run the window experiment before --analysis-only")
        window_status = json.loads(execution_path.read_text(encoding="utf-8"))
    else:
        window_status = run_windows(output_dir, retry_failed=retry_failed)
        execution_path.write_text(json.dumps(window_status, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Scoring frozen research comparisons", flush=True)
    summary = run_analysis(output_dir)
    figures = build_figures(output_dir)
    execution = {
        "runtime": runtime_info(),
        "window_study": window_status,
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
