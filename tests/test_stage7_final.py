"""Checks for the frozen Stage 7 final forecast and report inputs."""

from __future__ import annotations

import hashlib
import json
import zipfile
from xml.etree import ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.modeling import backtest_sarima as sarima
from scripts.modeling import final_forecast as stage7


ROOT = Path(__file__).resolve().parents[1]
FORECAST_OUTPUTS = (
    ROOT / "outputs/forecasts/final_forecast.csv",
    ROOT / "outputs/tables/final_model_diagnostics.csv",
    ROOT / "outputs/tables/final_forecast_supplementary.csv",
    ROOT / "outputs/tables/final_results_summary.csv",
    ROOT / "outputs/figures/final_forecast.png",
    ROOT / "outputs/diagnostics/stage7_summary.json",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_final_order_matches_stage4_frozen_full_sample_order() -> None:
    frozen = _read_json(ROOT / "outputs/diagnostics/stage4_summary.json")["full_sample_descriptive_selection"]
    order = stage7.frozen_order()
    assert order.order == (0, 1, 1)
    assert order.seasonal_order == (1, 0, 1, 4)
    assert frozen["selected_order"] == "(0,1,1)"
    assert frozen["selected_seasonal_order"] == "(1,0,1,4)"


def test_stage7_does_not_call_candidate_search() -> None:
    source = stage7.__file__
    text = Path(source).read_text(encoding="utf-8")
    assert "select_model(" not in text
    assert "generate_candidate_grid(" not in text
    summary = _read_json(ROOT / "outputs/diagnostics/stage7_summary.json")
    assert summary["candidate_search_performed"] is False


def test_training_and_targets_stop_at_the_frozen_boundary() -> None:
    frame = stage7.load_stage7_data()
    training = frame.loc[:stage7.FORECAST_ORIGIN, "log_retail"]
    assert str(frame.index[0]) == "1994Q1"
    assert str(frame.index[-1]) == "2026Q2"
    assert str(training.index[-1]) == "2026Q2"
    assert len(training) == 130
    assert [str(value) for value in stage7.FORECAST_TARGETS] == ["2026Q3", "2026Q4"]
    assert not pd.PeriodIndex(stage7.FORECAST_TARGETS, freq="Q-DEC").isin(frame.index).any()


def test_future_economic_observations_and_primary_exog_are_absent() -> None:
    summary = _read_json(ROOT / "outputs/diagnostics/stage7_summary.json")
    diagnostics = pd.read_csv(ROOT / "outputs/tables/final_model_diagnostics.csv").iloc[0]
    assert summary["training"]["future_economic_observations_used"] is False
    assert summary["primary_exog_variables"] == []
    assert stage7.PRIMARY_EXOG_COLUMNS == ()
    assert diagnostics["primary_exog_variables"] == "none"

    frame = stage7.load_stage7_data()
    training = frame.loc[:stage7.FORECAST_ORIGIN, "log_retail"].astype(float)
    order = stage7.frozen_order()
    candidate = sarima.SarimaCandidate(p=order.p, d=order.d, q=order.q, P=order.P, D=order.D, Q=order.Q)
    fitted = sarima.fit_candidate(training, candidate)
    assert fitted is not None and fitted.converged
    assert fitted.result.model.exog is None


def test_point_forecast_and_prediction_interval_formulas() -> None:
    mean, variance = 11.5, 0.04
    row = stage7.forecast_row(pd.Period("2026Q3", freq="Q-DEC"), 1, mean, variance)
    se = np.sqrt(variance)
    assert np.isclose(row["forecast_level_median"], np.exp(mean), rtol=1e-14)
    assert np.isclose(row["forecast_level_mean"], np.exp(mean + 0.5 * variance), rtol=1e-14)
    assert np.isclose(row["lower_80"], np.exp(mean - norm.ppf(0.90) * se), rtol=1e-14)
    assert np.isclose(row["upper_80"], np.exp(mean + norm.ppf(0.90) * se), rtol=1e-14)
    assert np.isclose(row["lower_95"], np.exp(mean - norm.ppf(0.975) * se), rtol=1e-14)
    assert np.isclose(row["upper_95"], np.exp(mean + norm.ppf(0.975) * se), rtol=1e-14)
    assert row["lower_95"] < row["lower_80"] < row["forecast_level_median"] < row["upper_80"] < row["upper_95"]
    assert np.isfinite(row["forecast_level_mean"]) and row["forecast_level_mean"] > 0


def test_saved_forecasts_have_correct_targets_and_nested_intervals() -> None:
    frame = pd.read_csv(ROOT / "outputs/forecasts/final_forecast.csv")
    assert frame["target"].tolist() == ["2026Q3", "2026Q4"]
    assert frame["horizon"].tolist() == [1, 2]
    assert frame["origin"].eq("2026Q2").all()
    assert frame["model"].eq("SARIMA").all()
    assert (frame["lower_95"] < frame["lower_80"]).all()
    assert (frame["lower_80"] < frame["forecast_level_median"]).all()
    assert (frame["forecast_level_median"] < frame["upper_80"]).all()
    assert (frame["upper_80"] < frame["upper_95"]).all()
    assert np.isfinite(frame["forecast_level_mean"]).all()
    assert (frame["forecast_level_mean"] > 0).all()


def test_primary_summary_and_supplementary_cny_future_regressors() -> None:
    summary = _read_json(ROOT / "outputs/diagnostics/stage7_summary.json")
    supplementary = pd.read_csv(ROOT / "outputs/tables/final_forecast_supplementary.csv")
    assert summary["primary_model"] == "SARIMA"
    assert set(supplementary["model"]) == {"SARIMA", "SARIMA+CNY", "ETS"}
    cny_rows = supplementary.loc[supplementary["model"].eq("SARIMA+CNY")]
    assert cny_rows["target"].tolist() == ["2026Q3", "2026Q4"]
    assert cny_rows["target_cny_regressor"].eq(0.0).all()
    assert cny_rows["exogenous_regressors"].eq("cny_regressor").all()
    assert summary["primary_exog_variables"] == []
    assert all("PMI" not in str(value) for value in supplementary["exogenous_regressors"])

    centered = stage7.future_cny_exog(stage7.FORECAST_TARGETS, center=0.5)
    assert centered["cny_regressor"].tolist() == [0.0, 0.0]


def test_final_report_metrics_match_frozen_stage_outputs() -> None:
    results = pd.read_csv(ROOT / "outputs/tables/final_results_summary.csv")
    report = (ROOT / "docs/final_report.md").read_text(encoding="utf-8")
    stage_metrics = pd.read_csv(ROOT / "outputs/tables/sarima_metrics.csv")
    expected = stage_metrics.query("model == 'SARIMA' and horizon == 2 and scope == 'all'").iloc[0]
    final = results.query("model == 'SARIMA' and horizon == 2 and scope == 'all'").iloc[0]
    assert final["study/sample"] == "Primary 2005Q1-2026Q2"
    expected_n = expected["n"] if "n" in expected.index else expected["n_forecasts"]
    assert int(final["n"]) == int(expected_n)
    for metric in ("MAE", "RMSE", "MASE"):
        assert np.isclose(final[metric], expected[metric], rtol=0.0, atol=1e-10)
    assert f"{expected['MAE']:,.2f}" in report
    assert f"{expected['RMSE']:,.2f}" in report
    pmi = results.query("model == 'SARIMA+PMI'").iloc[0]
    assert pmi["study/sample"] == "Stage 6A secondary 2010Q4-2026Q2"
    assert int(pmi["n"]) == 63
    assert "CNY timing has no effect" not in report
    assert "share of combined January-February" in report


def test_local_pdf_conversion_matches_converter_availability() -> None:
    summary = _read_json(ROOT / "outputs/diagnostics/stage7_summary.json")
    pdf_path = ROOT / "outputs/reports/STA4003_Final_Report.pdf"
    assert summary["pdf_status"]["runtime_tested"] is True
    assert summary["pdf_status"]["reason"] is None or "AppData" not in summary["pdf_status"]["reason"]
    if summary["pdf_status"]["converter_available"]:
        assert pdf_path.is_file()
        assert pdf_path.read_bytes().startswith(b"%PDF-")
        assert summary["pdf_status"]["generated"] is True
        assert "outputs/reports/STA4003_Final_Report.pdf" in summary["generated_artifacts"]
    else:
        if summary["pdf_status"]["generated"]:
            assert pdf_path.is_file()
            assert pdf_path.read_bytes().startswith(b"%PDF-")
        else:
            assert not pdf_path.exists()


def test_final_report_docx_package_timestamps_are_canonical() -> None:
    report_path = ROOT / "outputs/reports/STA4003_Final_Report.docx"
    with zipfile.ZipFile(report_path) as package:
        assert package.namelist()
        assert all(member.date_time == (2020, 1, 1, 0, 0, 0) for member in package.infolist())


def test_final_report_figure_one_caption_and_table_three_pagination() -> None:
    expected_caption = (
        "Figure 1. Annual-mean-centered log retail by quarter, showing the "
        "quarterly seasonal pattern. Source: Stage 2 frozen output."
    )
    report_markdown = (ROOT / "docs/final_report.md").read_text(encoding="utf-8")
    assert f"*{expected_caption}*" in report_markdown
    assert "long-run scale retained for context" not in report_markdown

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    word = "{" + namespace["w"] + "}"
    report_path = ROOT / "outputs/reports/STA4003_Final_Report.docx"
    with zipfile.ZipFile(report_path) as package:
        document = ET.fromstring(package.read("word/document.xml"))
    body = document.find(".//w:body", namespace)
    assert body is not None
    blocks = list(body)

    def paragraph_text(block: ET.Element) -> str:
        return "".join(node.text or "" for node in block.findall(".//w:t", namespace))

    caption_index = next(
        index
        for index, block in enumerate(blocks)
        if block.tag == word + "p" and paragraph_text(block).startswith("Table 3.")
    )
    caption = blocks[caption_index]
    caption_keep = caption.find("./w:pPr/w:keepNext", namespace)
    assert caption_keep is not None and caption_keep.get(word + "val") not in {"0", "false", "off"}

    table = blocks[caption_index + 1]
    assert table.tag == word + "tbl"
    rows = table.findall("./w:tr", namespace)
    assert len(rows) == 3
    assert "SARIMA-common" in paragraph_text(rows[1])
    assert "SARIMA+PMI" in paragraph_text(rows[2])
    for row_index, row in enumerate(rows):
        assert row.find("./w:trPr/w:cantSplit", namespace) is not None
        for paragraph in row.findall(".//w:p", namespace):
            keep = paragraph.find("./w:pPr/w:keepNext", namespace)
            enabled = keep is not None and keep.get(word + "val") not in {"0", "false", "off"}
            assert enabled is (row_index < len(rows) - 1)


def test_stage0_to_6b_protected_files_match_frozen_manifest() -> None:
    actual = stage7.verify_protected_artifacts()
    assert len(actual) == 67
    manifest = _read_json(ROOT / "docs/stage7_protected_artifact_sha256.json")["files"]
    assert actual == manifest


def test_model_outputs_are_deterministic() -> None:
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in FORECAST_OUTPUTS}
    stage7.run_final_forecast()
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in FORECAST_OUTPUTS}
    assert before == after
