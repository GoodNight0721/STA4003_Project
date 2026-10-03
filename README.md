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

**Stage 2 — EDA and stationarity diagnostics complete.** No forecasting model or predictive-value result is reported here.
