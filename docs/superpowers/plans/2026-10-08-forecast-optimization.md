# Forecast optimization Implementation Plan

> **For agentic workers:** Use the bounded task briefs below. Execute in this session with focused tests and independent review; the user has already authorized implementation and publishing the research branch.

**Goal:** Produce a reproducible first-round optimization study without changing the completed course-project baseline.

**Architecture:** Read frozen Stage 4–6B artifacts, fit only two fixed-window SARIMA variants, and derive paired-loss, interval, combination, median and calendar diagnostics. Separate fitting/checkpointing from statistical scoring and reporting.

**Tech Stack:** Existing Python, pandas, NumPy, SciPy, statsmodels, matplotlib, pytest.

## Global Constraints

- Follow `docs/optimization_protocol.md` verbatim for samples, estimators, retries, bootstrap and calibration parameters.
- Preserve existing data, proposal and Stage 1–7 generated outputs byte-for-byte.
- Resolve paths from `__file__`; generated research artifacts stay in `outputs/optimization/round1/`.
- All historical new-specification results remain exploratory; no primary-model adoption or merge to main.

## Task 1: Window fitting and checkpointed execution

**Files:** Create `scripts/optimization/__init__.py`, `scripts/optimization/window_backtest.py`, `tests/test_optimization_windows.py`.

**Interfaces:** `window_training(frame, origin, window)` returns the last min(window,n) observed rows; `fit_window(training,candidate)` returns forecast-capable fit plus attempt metadata; `run_windows(output_dir)` saves `window_forecasts.csv`, `window_fits.csv`, and per-origin checkpoint JSON.

- [x] Write tests for exact truncated sample, real fit future-perturbation invariance, original per-origin orders/MASE pairing, checkpoint reuse and invalidation, and failure persistence.
- [x] Run the focused file and observe a missing-module/function failure before implementing.
- [x] Implement per-origin fitting, h=1/h=2 reuse, deterministic retries, finite-moment validation, fail-visible resumability and saved prediction moments.
- [x] Run focused tests, then run the complete real window experiment. Preserve failed origins explicitly.
- [x] Review task correctness and evidence before publishing.

## Task 2: Research scoring and diagnostics

**Files:** Create `scripts/optimization/analysis.py`, `tests/test_optimization_analysis.py`.

**Interfaces:** `paired_losses(reference,extended)`, `bootstrap_deltas(paired,block_length,reps,seed)`, `interval_records(forecasts,calibrate=False)`, `point_metrics(forecasts)`, `combine_forecasts(reference,ets)`, `calendar_diagnostic(calendar)`; `run_analysis(output_dir)` writes all tables and `summary.json`.

- [x] Write hand-checked tests for signed paired metrics, target mismatch rejection, identical-input zero bootstrap intervals, interval/Winkler formulas, calibration eligibility at origin and future-perturbation invariance, exact equal-weight combination, and holiday allocation.
- [x] Run focused tests and observe the expected failure before implementation.
- [x] Implement complete-key matching, chronological block resampling, original-scale scoring, mean/median sensitivity, parametric/calibrated interval evaluation, descriptive period tables, and calendar allocation variation.
- [x] Run focused tests and real scoring. Verify archived SARIMA/CNY metrics reproduce stored tables.
- [x] Review task correctness and evidence before publishing.

## Task 3: Integration, report and GitHub branch

**Files:** Create `scripts/optimization/run_research.py`, `docs/optimization_research.md`; append research instructions to `README.md` and authorization/methods to `docs/decisions.md`.

- [x] Integrate experiment execution and scoring with explicit output-dir CLI, progress, checkpoint resume and original-artifact verification using existing repository manifest plus Git baseline comparison.
- [x] Generate figures and a Chinese report using actual result tables; retain uncertainty, negative results and incomplete-status limitations.
- [x] Save runtime versions and commands. Run `python -m pytest -q`, `python -m compileall scripts tests`, and `git diff --check`.
- [x] Request read-only review of code, tests, methodology, report and scope; address material findings.
- [x] Commit and push to `research/forecast-optimization-20261008`; verify the remote SHA. Keep main unchanged.
