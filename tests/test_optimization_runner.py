"""Research-output isolation and frozen archive verification."""

from pathlib import Path

import pytest

from scripts.optimization import run_research as research


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


def test_byte_comparison_detects_changed_and_missing_archives(tmp_path):
    path = tmp_path / "archive.csv"
    path.write_bytes(b"x\n1\n")
    research.require_archived_bytes(path, b"x\n1\n")
    with pytest.raises(ValueError, match="changed"):
        research.require_archived_bytes(path, b"x\n2\n")
    with pytest.raises(ValueError, match="missing"):
        research.require_archived_bytes(tmp_path / "missing.csv", b"x\n")
