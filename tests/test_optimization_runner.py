"""Research-output isolation and frozen archive verification."""

from pathlib import Path
import json
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from scripts.optimization import run_research as research
from scripts.optimization import analysis, reporting, window_backtest


@pytest.mark.parametrize("relative", ["data", "outputs", "proposal", "docs", "."])
def test_output_directory_cannot_overlap_original_artifacts(relative):
    with pytest.raises(ValueError, match="outputs/optimization"):
        research.validate_output_dir(research.PROJECT_ROOT / relative)


def test_output_directory_accepts_research_subdirectory_and_external_temp(tmp_path):
    assert research.validate_output_dir(research.PROJECT_ROOT / "outputs/optimization/round1") == (
        research.PROJECT_ROOT / "outputs/optimization/round1"
    )
    assert research.validate_output_dir(tmp_path) == tmp_path.resolve()


def test_frozen_baseline_verification_checks_git_bytes_and_existing_manifest():
    result = research.verify_baseline()
    assert result["protected_count"] == 67
    assert result["other_archived_count"] > 0
    assert result["unchanged"] is True


def test_baseline_git_failure_retains_stderr_evidence(monkeypatch):
    monkeypatch.setattr(research, "BASE_REF", "missing-optimization-test-baseline")
    with pytest.raises(subprocess.CalledProcessError) as error:
        research.verify_baseline()
    assert error.value.returncode != 0
    assert error.value.stderr and b"fatal" in error.value.stderr.lower()


def test_byte_comparison_detects_changed_and_missing_archives(tmp_path):
    path = tmp_path / "archive.csv"
    path.write_bytes(b"x\n1\n")
    research.require_archived_bytes(path, b"x\n1\n")
    with pytest.raises(ValueError, match="changed"):
        research.require_archived_bytes(path, b"x\n2\n")
    with pytest.raises(ValueError, match="missing"):
        research.require_archived_bytes(tmp_path / "missing.csv", b"x\n")


@pytest.fixture
def current_outputs(tmp_path, monkeypatch):
    """Reuse real completed forecasts/fit records, without fitting any model."""
    for name in ("window_forecasts.csv", "window_fits.csv", "window_execution.json"):
        shutil.copyfile(research.DEFAULT_OUTPUT / name, tmp_path / name)
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPS", 20)
    monkeypatch.setattr(reporting, "build_figures", lambda output_dir: [])

    def forbid_refits(*args, **kwargs):
        pytest.fail("Analysis-only must not refit window models")

    monkeypatch.setattr(window_backtest, "run_windows", forbid_refits)
    return tmp_path


def run_analysis_cli(output_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_research.py", "--analysis-only", "--output-dir", str(output_dir)])
    research.main()


def test_analysis_only_rejects_stale_complete_cache_after_partial_rerun(current_outputs, monkeypatch, capsys):
    fits = pd.read_csv(current_outputs / "window_fits.csv").iloc[:1]
    forecasts = pd.read_csv(current_outputs / "window_forecasts.csv")
    forecasts = forecasts.loc[(forecasts.model == "SARIMA-W40") & (forecasts.origin == "2004Q3")]
    assert len(forecasts) == 1
    fits.to_csv(current_outputs / "window_fits.csv", index=False)
    forecasts.to_csv(current_outputs / "window_forecasts.csv", index=False)

    with pytest.raises(SystemExit) as exit_info:
        run_analysis_cli(current_outputs, monkeypatch)
    assert exit_info.value.code == 2
    execution = json.loads((current_outputs / "execution.json").read_text(encoding="utf-8"))
    assert execution["complete"] is False
    assert execution["window_study"]["successful_fits"] == 1
    assert execution["window_study"]["pending_fits"] == 173
    assert execution["window_study"]["saved_execution_consistent"] is False
    assert execution["saved_window_execution"]["complete"] is True
    report = (current_outputs / "research_report.md").read_text(encoding="utf-8")
    assert "研究不完整" in report
    assert "缓存" in report
    assert "所有窗口模型已覆盖完整目标集" not in report
    assert '"complete": false' in capsys.readouterr().out
    metrics = pd.read_csv(current_outputs / "point_metrics.csv")
    assert not metrics.model.str.startswith("SARIMA-W").any()


def test_analysis_only_complete_current_outputs_preserve_point_results(current_outputs, monkeypatch):
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPS", 5000)
    run_analysis_cli(current_outputs, monkeypatch)
    execution = json.loads((current_outputs / "execution.json").read_text(encoding="utf-8"))
    status = execution["window_study"]
    assert execution["complete"] is True
    assert status["requested_fits"] == status["successful_fits"] == 174
    assert status["failed_fits"] == status["pending_fits"] == 0
    assert status["saved_execution_consistent"] is True
    for coverage in status["model_coverage"].values():
        assert coverage["requested"] == coverage["successful"] == 87
        assert coverage["n_records"] == coverage["n_expected"] == 172
        assert coverage["forecasts_match_successful_fits"] is True
        assert coverage["complete"] is True
    for name in ("point_metrics.csv", "shock_sensitivity.csv", "paired_losses.csv", "paired_bootstrap.csv",
                 "interval_records.csv", "interval_metrics.csv", "period_metrics.csv",
                 "calendar_allocations.csv", "calendar_variation.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(current_outputs / name),
                                      pd.read_csv(research.DEFAULT_OUTPUT / name), check_exact=True)


@pytest.mark.parametrize("fit_change,successful,failed,pending", [
    ("missing", 173, 0, 1), ("failed", 173, 1, 0),
])
def test_full_forecasts_cannot_override_failed_or_missing_current_fit(current_outputs, monkeypatch,
                                                                    fit_change, successful, failed, pending):
    fits = pd.read_csv(current_outputs / "window_fits.csv")
    if fit_change == "missing":
        fits = fits.iloc[1:]
    else:
        fits["failure_reason"] = fits["failure_reason"].astype(object)
        fits.loc[0, ["status", "failure_reason"]] = ["failed", "Interrupted retry failed"]
    fits.to_csv(current_outputs / "window_fits.csv", index=False)
    forecast_bytes = (current_outputs / "window_forecasts.csv").read_bytes()
    with pytest.raises(SystemExit) as exit_info:
        run_analysis_cli(current_outputs, monkeypatch)
    assert exit_info.value.code == 2
    execution = json.loads((current_outputs / "execution.json").read_text(encoding="utf-8"))
    status = execution["window_study"]
    assert (status["successful_fits"], status["failed_fits"], status["pending_fits"]) == (successful, failed, pending)
    coverage = status["model_coverage"]["SARIMA-W40"]
    assert coverage["n_records"] == 172
    assert coverage["forecasts_match_successful_fits"] is False
    assert coverage["complete"] is False
    assert status["model_coverage"]["SARIMA-W60"]["complete"] is True
    metrics = pd.read_csv(current_outputs / "point_metrics.csv")
    assert "SARIMA-W40" not in set(metrics.model)
    assert "SARIMA-W60" in set(metrics.model)
    assert (current_outputs / "window_forecasts.csv").read_bytes() == forecast_bytes
    assert not (current_outputs / "checkpoints").exists()


@pytest.mark.parametrize("table,change,message", [
    ("window_fits.csv", "duplicate", "Duplicate"),
    ("window_fits.csv", "unknown_model", "Unknown"),
    ("window_fits.csv", "unknown_origin", "Unknown"),
    ("window_fits.csv", "wrong_order", "order"),
    ("window_forecasts.csv", "duplicate", "Duplicate"),
    ("window_forecasts.csv", "unknown_model", "Unknown"),
    ("window_forecasts.csv", "unknown_origin", "Unknown"),
    ("window_forecasts.csv", "unknown_target", "Unknown"),
])
def test_analysis_only_rejects_invalid_current_keys(current_outputs, table, change, message):
    frame = pd.read_csv(current_outputs / table)
    if change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif change == "unknown_model":
        frame.loc[0, "model"] = "SARIMA-W80"
    elif change == "unknown_origin":
        frame.loc[0, "origin"] = "2000Q1"
    elif change == "unknown_target":
        frame.loc[0, "target"] = "2000Q2"
    else:
        frame.loc[0, "p"] = 99
    frame.to_csv(current_outputs / table, index=False)
    with pytest.raises(ValueError, match=message):
        research.run_research(current_outputs, analysis_only=True)


def test_analysis_completeness_disagreement_prevents_success(current_outputs, monkeypatch):
    real_analysis = analysis.run_analysis

    def incomplete_analysis(output_dir):
        summary = real_analysis(output_dir)
        summary["window_completeness"]["SARIMA-W40"]["complete"] = False
        return summary

    monkeypatch.setattr(analysis, "run_analysis", incomplete_analysis)
    with pytest.raises(SystemExit) as exit_info:
        run_analysis_cli(current_outputs, monkeypatch)
    assert exit_info.value.code == 2
    execution = json.loads((current_outputs / "execution.json").read_text(encoding="utf-8"))
    assert execution["complete"] is False
    assert execution["window_study"]["model_coverage"]["SARIMA-W40"]["complete"] is False
    assert any("analysis" in warning for warning in execution["window_study"]["warnings"])
