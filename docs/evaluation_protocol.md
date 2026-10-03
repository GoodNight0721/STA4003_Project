# Rolling-origin evaluation protocol

## Purpose and comparison window

Use an expanding-window rolling-origin evaluation for every future forecasting model in this project. A random train/test split would break the chronological order of the observations and would not reproduce the information set available when each forecast was made.

The common target evaluation window is **2005Q1 through 2026Q2, inclusive**. This is 86 consecutive quarterly targets. The window is fixed independently of predictor availability: later studies must not shorten it because PMI or another predictor has a shorter history. If a future predictor is unavailable at an origin, that limitation must be handled within the model design and documented without changing this target window.

## Forecast horizons and origin

- **Primary result:** h = 2 quarters.
- **Secondary diagnostic:** h = 1 quarter.
- For each target quarter t, its forecast origin is T = t - h.
- The training sample for that forecast is every available canonical observation from 1994Q1 through T, inclusive.
- Training expands by one quarter as the target moves forward. No observation after T may be used in fitting, transformations estimated from data, forecast construction, or scale calculations.

Both horizons use the exact same 86 target quarters. Consequently, their origins differ, but the scored target set does not.

## Forecast records and scoring

Store one record for every model, horizon, and target, including the origin, target, training start/end and sample size, actual, forecast, error (actual - forecast), absolute error, squared error, MASE scale, and scaled absolute error.

For quarterly seasonal MASE, use period s = 4. At each origin T, calculate:

scale_T = mean(|y_t - y_(t-4)|) over seasonal differences whose two observations are both in the training sample ending at T.

For a forecast with actual y_t and forecast f_t, calculate scaled_absolute_error = |y_t - f_t| / scale_T. A model/horizon/scope MASE is the arithmetic mean of those per-origin scaled absolute errors. Never calculate one denominator from the full sample or from observations after an origin.

Report only MAE, RMSE, and MASE for each model and horizon, with:

- all: all 86 evaluation targets.
- q1: the 22 Q1 targets in the same window. Q1 is a pre-specified subgroup.

The primary comparison is h = 2, scope all; Q1 performance is a pre-specified subgroup result, and h = 1 remains secondary.

## Protocol for later stages

All subsequent models must reuse the target window, horizon definitions, origin convention, expanding training sample, score calculations, and all/q1 scopes defined here. The Stage 3 benchmarks establish references only. This protocol does not select a model structure or authorize changing the evaluation window based on observed results.
