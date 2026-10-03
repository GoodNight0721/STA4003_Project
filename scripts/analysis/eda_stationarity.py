"""Stage 2 descriptive, transformation, stationarity, and dependence diagnostics.

All economic observations are read from the canonical Stage 1 analysis dataset.
No forecasting model is estimated by this script.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, adfuller, kpss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_quarterly.csv"
SEASONAL_PERIOD = 4
ADF_MAX_LAG = 8
ACF_MAX_LAG = 20
ROLLING_WINDOW = 20
REQUIRED_COLUMNS = [
    "quarter",
    "year",
    "quarter_num",
    "retail_bn",
    "log_retail",
    "is_q1",
    "cny_date",
    "cny_month",
    "cny_position_fraction",
    "pmi",
    "pmi_lag1",
    "pmi_lag2",
]
STATIONARITY_COLUMNS = [
    "transformation",
    "n_obs",
    "adf_n_obs",
    "adf_regression",
    "adf_maxlag",
    "adf_usedlag",
    "adf_statistic",
    "adf_pvalue",
    "kpss_regression",
    "kpss_nlags",
    "kpss_statistic",
    "kpss_pvalue",
    "kpss_pvalue_note",
    "interpretation",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_quarters() -> list[str]:
    return [
        f"{year}Q{quarter}"
        for year in range(1994, 2027)
        for quarter in range(1, 5)
        if (year, quarter) <= (2026, 2)
    ]


def load_canonical(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read and minimally validate the immutable canonical quarterly input."""
    _require(path.is_file(), f"Missing canonical input: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    _require(frame.columns.tolist() == REQUIRED_COLUMNS, "Canonical dataset columns do not match the Stage 1 schema")
    _require(frame["quarter"].astype(str).tolist() == expected_quarters(), "Canonical quarter sequence is not 1994Q1–2026Q2")

    for column in ("year", "quarter_num", "retail_bn", "log_retail", "is_q1", "cny_month", "cny_position_fraction"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("pmi", "pmi_lag1", "pmi_lag2"):
        frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="raise")

    _require(frame["retail_bn"].gt(0).all(), "Canonical retail levels must be positive")
    _require(np.isfinite(frame["log_retail"].to_numpy()).all(), "Canonical log retail must be finite")
    _require(
        np.allclose(frame["log_retail"], np.log(frame["retail_bn"]), rtol=0.0, atol=5.1e-7),
        "Canonical log retail does not match the natural log of retail",
    )
    frame.index = pd.PeriodIndex(frame["quarter"], freq="Q-DEC", name="quarter_period")
    return frame


def make_transformations(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the four requested transformations with their natural time indexes."""
    log_retail = frame["log_retail"].astype(float).copy()
    return {
        "log_retail": log_retail,
        "first_difference": log_retail.diff(1).dropna(),
        "seasonal_difference": log_retail.diff(SEASONAL_PERIOD).dropna(),
        "combined_difference": log_retail.diff(SEASONAL_PERIOD).diff(1).dropna(),
    }


def stationarity_interpretation(adf_pvalue: float, kpss_pvalue: float) -> str:
    """Summarize the opposing null hypotheses at the pre-set five-percent level."""
    adf_rejects_unit_root = adf_pvalue < 0.05
    kpss_rejects_stationarity = kpss_pvalue < 0.05
    if adf_rejects_unit_root and not kpss_rejects_stationarity:
        return "ADF rejects a unit root and KPSS does not reject stationarity; evidence is consistent with stationarity under the stated deterministic terms."
    if not adf_rejects_unit_root and kpss_rejects_stationarity:
        return "ADF does not reject a unit root and KPSS rejects stationarity; evidence is consistent with non-stationarity under the stated deterministic terms."
    if adf_rejects_unit_root and kpss_rejects_stationarity:
        return "The tests conflict: ADF rejects a unit root, while KPSS rejects stationarity."
    return "Neither null is rejected at five percent; the tests are inconclusive about stationarity."


def run_stationarity_tests(transformations: dict[str, pd.Series]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Run fixed-specification ADF and automatic-bandwidth KPSS tests."""
    deterministic_terms = {
        "log_retail": ("ct", "ct"),
        "first_difference": ("c", "c"),
        "seasonal_difference": ("c", "c"),
        "combined_difference": ("c", "c"),
    }
    rows = []
    warning_log: dict[str, list[str]] = {}
    for name, series in transformations.items():
        values = series.to_numpy(dtype=float)
        _require(len(values) > ADF_MAX_LAG + 12, f"Too few observations for {name} stationarity tests")
        _require(np.isfinite(values).all(), f"{name} contains non-finite values")
        adf_regression, kpss_regression = deterministic_terms[name]
        adf_result = adfuller(values, maxlag=ADF_MAX_LAG, regression=adf_regression, autolag=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kpss_result = kpss(values, regression=kpss_regression, nlags="auto")
        adf_stat, adf_pvalue, adf_usedlag, adf_n_obs = adf_result[:4]
        kpss_stat, kpss_pvalue, kpss_nlags = kpss_result[:3]
        adf_stat = float(adf_stat)
        adf_pvalue = float(adf_pvalue)
        kpss_stat = float(kpss_stat)
        kpss_pvalue = float(kpss_pvalue)
        _require(np.isfinite([adf_stat, adf_pvalue, kpss_stat, kpss_pvalue]).all(), f"Non-finite test result for {name}")
        _require(0 <= adf_pvalue <= 1 and 0 <= kpss_pvalue <= 1, f"Invalid p-value for {name}")
        warning_log[name] = [str(item.message) for item in caught]
        if any("actual p-value is smaller" in str(item.message) for item in caught):
            kpss_pvalue_note = "Reported at lower table bound; actual p-value is smaller."
        elif any("actual p-value is greater" in str(item.message) for item in caught):
            kpss_pvalue_note = "Reported at upper table bound; actual p-value is greater."
        else:
            kpss_pvalue_note = ""
        rows.append(
            {
                "transformation": name,
                "n_obs": int(len(values)),
                "adf_n_obs": int(adf_n_obs),
                "adf_regression": adf_regression,
                "adf_maxlag": ADF_MAX_LAG,
                "adf_usedlag": int(adf_usedlag),
                "adf_statistic": adf_stat,
                "adf_pvalue": adf_pvalue,
                "kpss_regression": kpss_regression,
                "kpss_nlags": int(kpss_nlags),
                "kpss_statistic": kpss_stat,
                "kpss_pvalue": kpss_pvalue,
                "kpss_pvalue_note": kpss_pvalue_note,
                "interpretation": stationarity_interpretation(adf_pvalue, kpss_pvalue),
            }
        )
    return pd.DataFrame(rows, columns=STATIONARITY_COLUMNS), warning_log


def _dates(index: pd.PeriodIndex) -> list:
    return index.to_timestamp(how="end").to_pydatetime().tolist()


def _style_time_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)


def _mark_reference_quarters(ax, frame: pd.DataFrame, values: pd.Series) -> None:
    event_styles = {
        "2020Q1": {"color": "#c43c39", "offset": (9, 11)},
        "2022Q2": {"color": "#6b4c9a", "offset": (9, -18)},
    }
    for quarter, style in event_styles.items():
        position = frame.index.get_loc(pd.Period(quarter, freq="Q-DEC"))
        x_value = _dates(frame.index[[position]])[0]
        y_value = float(values.iloc[position])
        ax.axvline(x_value, color=style["color"], linestyle="--", linewidth=1.0, alpha=0.65)
        ax.scatter([x_value], [y_value], color=style["color"], edgecolor="white", linewidth=0.7, zorder=5, s=55)
        ax.annotate(
            quarter,
            (x_value, y_value),
            xytext=style["offset"],
            textcoords="offset points",
            fontsize=9,
            color=style["color"],
            weight="bold",
        )


def plot_retail_series(frame: pd.DataFrame, figure_dir: Path) -> None:
    x_values = _dates(frame.index)
    plot_specs = [
        ("retail_bn", "Quarterly retail sales", "RMB 100 million", "retail_level.png"),
        ("log_retail", "Log quarterly retail sales", "Natural log of retail (RMB 100 million)", "log_retail.png"),
    ]
    for column, title, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        ax.plot(x_values, frame[column].to_numpy(dtype=float), color="#245b78", linewidth=1.6)
        _mark_reference_quarters(ax, frame, frame[column])
        ax.set_title(title)
        ax.set_xlabel("Quarter")
        ax.set_ylabel(ylabel)
        _style_time_axis(ax)
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=160, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        plt.close(fig)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.12g", lineterminator="\n")


def seasonal_tables_and_plots(frame: pd.DataFrame, table_dir: Path, figure_dir: Path) -> tuple[pd.DataFrame, dict]:
    level_summary = frame.groupby("quarter_num")["log_retail"].agg(
        n_obs="count", mean_log_retail="mean", median_log_retail="median", std_log_retail="std"
    )
    complete_years = frame.groupby("year").filter(lambda group: len(group) == SEASONAL_PERIOD).copy()
    complete_years["within_year_log_deviation"] = complete_years["log_retail"] - complete_years.groupby("year")["log_retail"].transform("mean")
    seasonal_summary = complete_years.groupby("quarter_num")["within_year_log_deviation"].agg(
        n_complete_years="count",
        mean_year_centered_log_deviation="mean",
        median_year_centered_log_deviation="median",
        std_year_centered_log_deviation="std",
    )
    positive_share = complete_years.assign(
        positive_deviation=complete_years["within_year_log_deviation"] > 0
    ).groupby("quarter_num")["positive_deviation"].mean()
    seasonal_summary["share_complete_years_positive"] = positive_share
    summary = level_summary.join(seasonal_summary).reset_index()
    summary.insert(1, "quarter_label", summary["quarter_num"].map(lambda value: f"Q{value}"))
    _write_csv(summary, table_dir / "seasonal_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for quarter_num, ax in enumerate(axes.flat, start=1):
        subset = complete_years.loc[complete_years["quarter_num"] == quarter_num]
        years = subset["year"].to_numpy(dtype=int)
        deviations = subset["within_year_log_deviation"].to_numpy(dtype=float)
        mean_deviation = float(np.mean(deviations))
        ax.plot(years, deviations, marker="o", markersize=3.2, linewidth=0.9, color="#2c718e")
        ax.axhline(mean_deviation, color="#c43c39", linestyle="--", linewidth=1.1, label="Quarter mean")
        ax.set_title(f"Q{quarter_num}: annual-mean centered log retail")
        ax.set_ylabel("Log-point deviation")
        ax.grid(True, alpha=0.25)
        if quarter_num == 1:
            ax.legend(frameon=False, loc="best")
    for ax in axes[-1, :]:
        ax.set_xlabel("Year (complete years 1994–2025)")
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    fig.suptitle("Quarterly seasonal subseries on the log scale", y=1.01)
    fig.tight_layout()
    fig.savefig(figure_dir / "seasonal_subseries_log.png", dpi=160, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    box_data = [frame.loc[frame["quarter_num"] == q, "log_retail"].to_numpy(dtype=float) for q in range(1, 5)]
    ax.boxplot(box_data, showmeans=True, showfliers=True)
    ax.set_xticks(range(1, 5), [f"Q{q}" for q in range(1, 5)])
    ax.set_title("Log retail distribution by quarter")
    ax.set_xlabel("Quarter of year")
    ax.set_ylabel("Natural log retail (RMB 100 million)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "seasonal_distribution_log.png", dpi=160, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)

    seasonal_effects = summary.set_index("quarter_label")["mean_year_centered_log_deviation"].to_dict()
    positive_shares = summary.set_index("quarter_label")["share_complete_years_positive"].to_dict()
    highest_quarter = max(seasonal_effects, key=seasonal_effects.get)
    lowest_quarter = min(seasonal_effects, key=seasonal_effects.get)
    return summary, {
        "complete_year_count": int(complete_years["year"].nunique()),
        "highest_mean_centered_quarter": highest_quarter,
        "lowest_mean_centered_quarter": lowest_quarter,
        "mean_year_centered_log_deviation_by_quarter": {key: float(value) for key, value in seasonal_effects.items()},
        "share_complete_years_positive_by_quarter": {key: float(value) for key, value in positive_shares.items()},
    }


def rolling_scale_diagnostics(frame: pd.DataFrame, table_dir: Path, figure_dir: Path) -> tuple[pd.DataFrame, dict]:
    rolling = pd.DataFrame(index=frame.index)
    rolling["level_rolling_sd"] = frame["retail_bn"].rolling(ROLLING_WINDOW).std(ddof=1)
    rolling["level_rolling_cv"] = rolling["level_rolling_sd"] / frame["retail_bn"].rolling(ROLLING_WINDOW).mean()
    rolling["log_rolling_sd"] = frame["log_retail"].rolling(ROLLING_WINDOW).std(ddof=1)
    rows = []
    for metric in rolling:
        values = rolling[metric].dropna()
        rows.append(
            {
                "metric": metric,
                "window_quarters": ROLLING_WINDOW,
                "n_windows": int(len(values)),
                "first_quarter": str(values.index[0]),
                "last_quarter": str(values.index[-1]),
                "minimum": float(values.min()),
                "median": float(values.median()),
                "maximum": float(values.max()),
                "first_value": float(values.iloc[0]),
                "last_value": float(values.iloc[-1]),
            }
        )
    summary = pd.DataFrame(rows)
    _write_csv(summary, table_dir / "transformation_summary.csv")

    dates = _dates(frame.index)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(dates, rolling["level_rolling_sd"], color="#bc5a31", linewidth=1.5)
    axes[0].set_ylabel("Retail rolling SD (RMB 100 million)")
    axes[0].set_title(f"{ROLLING_WINDOW}-quarter rolling scale")
    axes[1].plot(dates, rolling["level_rolling_cv"], color="#245b78", linewidth=1.5, label="Level coefficient of variation")
    axes[1].plot(dates, rolling["log_rolling_sd"], color="#c43c39", linewidth=1.4, label="Log-retail rolling SD")
    axes[1].set_ylabel("Dimensionless scale")
    axes[1].set_xlabel("Quarter")
    axes[1].legend(frameon=False, loc="best")
    for ax in axes:
        _style_time_axis(ax)
    fig.tight_layout()
    fig.savefig(figure_dir / "rolling_scale_comparison.png", dpi=160, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)

    diagnostics = {
        row["metric"]: {
            "window_quarters": ROLLING_WINDOW,
            "n_windows": row["n_windows"],
            "first_quarter": row["first_quarter"],
            "last_quarter": row["last_quarter"],
            "minimum": row["minimum"],
            "median": row["median"],
            "maximum": row["maximum"],
            "first_value": row["first_value"],
            "last_value": row["last_value"],
        }
        for row in rows
    }
    return summary, diagnostics


def acf_diagnostics(transformations: dict[str, pd.Series], figure_dir: Path) -> dict:
    selected_lags = [1, 4, 8, 12, 16, 20]
    specs = [
        ("log_retail", "Log retail", "acf_pacf_log_retail.png"),
        ("first_difference", "First difference of log retail", "acf_pacf_first_difference.png"),
        ("seasonal_difference", "Seasonal difference (lag 4) of log retail", "acf_pacf_seasonal_difference.png"),
        ("combined_difference", "Regular + seasonal difference of log retail", "acf_pacf_combined_difference.png"),
    ]
    summary = {}
    for key, title, filename in specs:
        series = transformations[key]
        values = series.to_numpy(dtype=float)
        acf_values = acf(values, nlags=ACF_MAX_LAG, fft=False, missing="raise")
        summary[key] = {
            "n_obs": int(len(values)),
            "acf_at_lags": {str(lag): float(acf_values[lag]) for lag in selected_lags},
        }
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        plot_acf(values, lags=ACF_MAX_LAG, alpha=0.05, ax=axes[0], zero=True, title="ACF")
        plot_pacf(values, lags=ACF_MAX_LAG, alpha=0.05, ax=axes[1], zero=True, method="ywm", title="PACF")
        axes[0].set_xlabel("Lag (quarters)")
        axes[1].set_xlabel("Lag (quarters)")
        fig.suptitle(f"{title}: dependence diagnostics only", y=1.02)
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=160, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        plt.close(fig)
    return summary


def cny_q1_descriptive(frame: pd.DataFrame, table_dir: Path, figure_dir: Path) -> dict:
    q1 = frame.loc[frame["quarter_num"] == 1, ["year", "cny_position_fraction", "log_retail"]].copy()
    q1["q1_yoy_log_growth"] = q1["log_retail"] - q1["log_retail"].shift(1)
    usable = q1.dropna(subset=["q1_yoy_log_growth"])
    x_values = usable["cny_position_fraction"].to_numpy(dtype=float)
    y_values = usable["q1_yoy_log_growth"].to_numpy(dtype=float)
    correlation = float(np.corrcoef(x_values, y_values)[0, 1])
    _require(np.isfinite(correlation), "CNY descriptive correlation is not finite")
    result = {
        "n_q1_year_over_year_pairs": int(len(usable)),
        "first_year": int(usable["year"].iloc[0]),
        "last_year": int(usable["year"].iloc[-1]),
        "pearson_correlation_cny_position_vs_q1_yoy_log_growth": correlation,
        "mean_q1_yoy_log_growth": float(y_values.mean()),
        "sd_q1_yoy_log_growth": float(y_values.std(ddof=1)),
        "interpretation": "Descriptive contemporaneous association only; no causal claim, significance test, or out-of-sample predictive evidence.",
    }
    _write_csv(pd.DataFrame([result]), table_dir / "cny_q1_descriptive.csv")

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.scatter(x_values, 100 * y_values, color="#2c718e", edgecolor="white", linewidth=0.6, alpha=0.85, s=46)
    ax.axhline(0, color="#666666", linewidth=0.9, linestyle="--")
    ax.set_title("Q1 CNY timing and Q1 year-over-year log growth\nDescriptive association only")
    ax.set_xlabel("CNY position fraction in Q1")
    ax.set_ylabel("Q1 year-over-year log growth (percent points)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "cny_q1_descriptive.png", dpi=160, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return result


def differencing_diagnostics(transformations: dict[str, pd.Series]) -> dict:
    combined_acf = acf(transformations["combined_difference"].to_numpy(dtype=float), nlags=4, fft=False)
    return {
        "regular_d1_n_obs": int(len(transformations["first_difference"])),
        "seasonal_D1_n_obs": int(len(transformations["seasonal_difference"])),
        "combined_d1_D1_n_obs": int(len(transformations["combined_difference"])),
        "combined_difference_acf_lag1": float(combined_acf[1]),
        "combined_difference_acf_lag4": float(combined_acf[4]),
    }


def _write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_analysis(data_path: Path = DATA_PATH, output_root: Path = PROJECT_ROOT) -> dict:
    """Generate Stage 2 artifacts under output_root without writing to data/."""
    figure_dir = output_root / "outputs" / "figures"
    table_dir = output_root / "outputs" / "tables"
    diagnostic_dir = output_root / "outputs" / "diagnostics"
    for directory in (figure_dir, table_dir, diagnostic_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame = load_canonical(data_path)
    transformations = make_transformations(frame)
    plot_retail_series(frame, figure_dir)
    _, seasonality = seasonal_tables_and_plots(frame, table_dir, figure_dir)
    _, rolling = rolling_scale_diagnostics(frame, table_dir, figure_dir)
    stationarity_table, test_warnings = run_stationarity_tests(transformations)
    _write_csv(stationarity_table, table_dir / "stationarity_tests.csv")
    acf_summary = acf_diagnostics(transformations, figure_dir)
    cny_summary = cny_q1_descriptive(frame, table_dir, figure_dir)
    differencing = differencing_diagnostics(transformations)

    generated_artifacts = [
        "outputs/figures/retail_level.png",
        "outputs/figures/log_retail.png",
        "outputs/figures/rolling_scale_comparison.png",
        "outputs/figures/seasonal_subseries_log.png",
        "outputs/figures/seasonal_distribution_log.png",
        "outputs/figures/acf_pacf_log_retail.png",
        "outputs/figures/acf_pacf_first_difference.png",
        "outputs/figures/acf_pacf_seasonal_difference.png",
        "outputs/figures/acf_pacf_combined_difference.png",
        "outputs/figures/cny_q1_descriptive.png",
        "outputs/tables/seasonal_summary.csv",
        "outputs/tables/transformation_summary.csv",
        "outputs/tables/stationarity_tests.csv",
        "outputs/tables/cny_q1_descriptive.csv",
        "outputs/diagnostics/stage2_summary.json",
    ]
    summary = {
        "sample_range": {"first_quarter": str(frame.index[0]), "last_quarter": str(frame.index[-1])},
        "n_obs": int(len(frame)),
        "seasonal_period": SEASONAL_PERIOD,
        "canonical_input": "data/processed/analysis_quarterly.csv",
        "chosen_transformation": {
            "series": "log_retail",
            "decision": "Use log_retail as the working response for later modeling, while retaining the original level for interpretation.",
            "rationale": "Level-scale volatility grows with the series; the log scale expresses changes proportionally and its rolling standard deviation tracks the level coefficient of variation. The log transform does not itself make the series stationary.",
        },
        "stationarity_test_settings": {
            "significance_level": 0.05,
            "adf_maxlag": ADF_MAX_LAG,
            "adf_autolag": None,
            "adf_regression": {"log_retail": "ct", "first_difference": "c", "seasonal_difference": "c", "combined_difference": "c"},
            "kpss_nlags": "auto",
            "kpss_regression": {"log_retail": "ct", "first_difference": "c", "seasonal_difference": "c", "combined_difference": "c"},
        },
        "stationarity_tests": stationarity_table.to_dict(orient="records"),
        "stationarity_test_warnings": test_warnings,
        "seasonality": {
            **seasonality,
            "summary_table": "outputs/tables/seasonal_summary.csv",
            "interpretation": "The report evaluates the complete-year centered log seasonal effects and their dispersion; these descriptive averages are not a forecasting model.",
        },
        "transformation_scale": {
            "summary_table": "outputs/tables/transformation_summary.csv",
            "rolling_window_quarters": ROLLING_WINDOW,
            "metrics": rolling,
        },
        "acf_pacf": {
            "maximum_lag_quarters": ACF_MAX_LAG,
            "seasonal_lags_examined": [4, 8, 12, 16, 20],
            "diagnostics": acf_summary,
            "interpretation": "ACF/PACF figures are structural diagnostics only; no ARMA or seasonal ARMA orders are selected.",
        },
        "differencing_diagnostics": differencing,
        "differencing_recommendation": {
            "regular_d": 1,
            "seasonal_D": None,
            "seasonal_D_status": "not established; retain as unresolved for later modeling diagnostics",
            "conclusion": "A regular difference is a cautious working recommendation given the trend, but its ADF/KPSS results are inconclusive. Seasonal differencing alone remains non-stationary by the selected tests. Combined differencing passes both tests, but its negative lag-4 ACF raises a seasonal over-differencing concern; do not commit to D=1 from Stage 2 alone.",
            "over_differencing_assessment": "The combined-difference ACF is -0.332 at lag 1 and -0.492 at lag 4. The lag-1 value is negative but not extreme; the seasonal-lag negative spike is a caution, not an automatic decision rule.",
        },
        "cny_q1_descriptive": cny_summary,
        "pmi_coverage": {
            "first_quarter": "2005Q1",
            "last_quarter": str(frame.loc[frame["pmi"].notna()].index[-1]),
            "n_obs": int(frame["pmi"].notna().sum()),
            "predictive_analysis_performed": False,
        },
        "generated_artifacts": generated_artifacts,
        "scope": {
            "models_fitted": False,
            "arma_orders_searched": False,
            "forecasts_or_evaluation_performed": False,
            "cny_predictive_value_tested": False,
            "pmi_predictive_value_tested": False,
        },
    }
    _write_json(summary, diagnostic_dir / "stage2_summary.json")
    return summary


def main() -> None:
    summary = run_analysis()
    print(
        f"Stage 2 diagnostics complete: {summary['n_obs']} quarters, "
        f"{len(summary['generated_artifacts'])} output artifacts, "
        f"{summary['sample_range']['first_quarter']} to {summary['sample_range']['last_quarter']}"
    )


if __name__ == "__main__":
    main()
