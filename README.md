# STA4003 Time Series Project

## Project title

Forecasting China's Quarterly Retail Sales With a Lunar New Year Adjustment

## Research question

This project studies quarterly Chinese retail-sales forecasting and asks whether information about Lunar New Year timing adds forecasting value beyond standard seasonal time-series structure. No improvement is assumed in advance.

## Current data

The repository contains NBS retail-sales and manufacturing-PMI source series, quarterly processed series, annual Lunar New Year calendar metadata, and a canonical quarterly analysis dataset. The Lunar New Year table is calendar metadata, not an NBS economic series.

## Repository structure

- `proposal/` — submitted proposal, its template, and the accompanying figure.
- `data/` — raw source series, reproducibly derived quarterly files, and metadata.
- `scripts/` — data acquisition/construction and reporting scripts.
- `notebooks/` — optional exploratory notebooks, if useful in a later stage.
- `outputs/` — generated figures, tables, diagnostics, forecasts, and reports.
- `docs/` — research plan, data dictionary, and decision log.
- `tests/` — automated checks for project code as they are added.

## Setup

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux), then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Reproduce Stage 1

From the repository root:

```bash
python scripts/data/validate_data.py
python scripts/data/build_analysis_dataset.py
python scripts/data/validate_data.py
pytest -q
```

## Reproduce Stage 2

From the repository root:

```bash
python scripts/analysis/eda_stationarity.py
pytest -q
python -m compileall scripts tests
```

Stage 2 outputs include descriptive figures and tables, stationarity tests, and ACF/PACF diagnostics. They do not include fitted forecasting models or model-order selection.

## Reproduce Stage 3

    python scripts/modeling/backtest_benchmarks.py
    pytest -q
    python -m compileall scripts tests

Stage 3 fixes an expanding-window evaluation protocol and evaluates only the three pre-specified level benchmarks. The common target window is 2005Q1–2026Q2; h = 2 is primary and h = 1 is secondary. See docs/evaluation_protocol.md and docs/stage3_evaluation_benchmarks.md.

## Reproduce Stage 4

    python scripts/modeling/backtest_sarima.py
    pytest -q
    python -m compileall scripts tests

Stage 4 applies training-only AICc selection to the fixed SARIMA candidate family at each Stage 3 origin. The full-sample fit is descriptive only.

## Reproduce Stage 5

    python scripts/modeling/backtest_cny.py
    pytest -q
    python -m compileall scripts tests

Stage 5 reads and freezes each Stage 4 rolling order, then adds one Q1-only, training-centered Lunar New Year timing regressor. Future exogenous values use only deterministic calendar metadata known at the origin. The primary comparison remains h = 2 across all targets; h = 2 Q1 is a prespecified subgroup. See [docs/stage5_cny_incremental_value.md](docs/stage5_cny_incremental_value.md).

## Project workflow

0. Repository bootstrap
1. Data audit and analysis dataset
2. EDA and stationarity analysis
3. Rolling-origin evaluation framework
4. Seasonal ARIMA baseline
5. Lunar New Year incremental-value analysis
6. Additional predictors and robustness checks
7. Final forecasting and reporting

## Current status

**Stage 5 — Lunar New Year incremental-value analysis complete.** No PMI regressor or final future forecast is included.
