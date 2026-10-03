# Stage 6B — ETS Forecasting Robustness and COVID Shock Sensitivity

## Scope and fixed model

Stage 6B adds one supplementary forecasting family and checks the sensitivity of the existing h=2 scores to two preset shock targets. The primary endpoint remains h=2 across all targets in the Stage 3 window. Neither the ETS fit nor the sensitivity analysis selects or retunes a model.

The sole ETS specification is statsmodels ETS(A,Ad,A): additive error, additive trend with damping, additive seasonal component, and seasonal period 4. It models `log_retail`. Each of the 87 unique Stage 3 origins receives one fit on observations from 1994Q1 through that origin; its one- and two-step forecasts supply the requested horizons. Each horizon contains the same 86 targets, 2005Q1–2026Q2. No ETS component search is performed.

For each origin, the level forecast is:

`exp(log forecast) * mean(exp(training one-step residuals))`

The smearing factor uses residuals from that origin's fitted training sample only. MASE scales are the exact origin-specific seasonal-period-4 scales from the Stage 3 benchmark implementation. Scores use the actual level minus the forecast level. All/q1 scopes and MAE, RMSE, and MASE match the earlier stages.

## Rolling fit diagnostics

All **87/87** origin fits returned finite forecasts and reported optimizer convergence; there were no failed fits or recorded numerical warnings. The origin-specific smearing factors ranged from **1.001777 to 1.004744**. This range is a retransformation diagnostic, not an estimate of forecast uncertainty.

## Forecast comparison

MAE and RMSE are in RMB 100 million (亿元); MASE is unit-free. The primary h=2 all-target result is:

| Model | n | MAE | RMSE | MASE |
|---|---:|---:|---:|---:|
| Historical Mean | 86 | 47,701.43 | 53,655.68 | 14.752580 |
| Naive | 86 | 6,914.05 | 9,376.89 | 2.203949 |
| Seasonal Naive | 86 | 5,991.50 | 7,023.89 | 2.246349 |
| SARIMA | 86 | 2,970.24 | 5,115.52 | 0.947893 |
| SARIMA+CNY | 86 | 2,965.73 | 5,021.59 | 0.946407 |
| ETS(A,Ad,A) | 86 | 3,242.32 | 5,211.84 | 1.059760 |

ETS improves on the three Stage 3 benchmarks on all three h=2 all-target metrics. It does not match either SARIMA forecast: compared with SARIMA, ETS has MAE higher by **272.08**, RMSE higher by **96.31**, and MASE higher by **0.111867**. Its Q1 h=2 MAE/RMSE/MASE are **4,146.72 / 7,194.91 / 1.229487**, each above SARIMA's **3,824.04 / 7,159.17 / 1.108583**. The fixed ETS specification therefore provides a useful family robustness check, while SARIMA retains the lower primary errors in this evaluation.

The complete six-model comparison for h=1/h=2 and all/Q1 is in [model_comparison_stage6b.csv](../outputs/tables/model_comparison_stage6b.csv). The h=1 results remain secondary.

## COVID shock-target sensitivity

The preset target rows are **2020Q1** and **2022Q2**. The sensitivity step filters these rows only when aggregating scores. It does not remove observations from the canonical data or any origin's training sample, refit a model, or rewrite the original forecast records. Both shock forecasts remain in the raw h=2 forecasts for SARIMA, SARIMA+CNY, and ETS; later-origin forecasts retain the same histories and fitted values.

| Model | h=2 all n | RMSE, full | RMSE, excluding targets | RMSE change |
|---|---:|---:|---:|---:|
| SARIMA | 86 → 84 | 5,115.52 | 4,218.66 | −896.87 (−17.53%) |
| SARIMA+CNY | 86 → 84 | 5,021.59 | 4,131.79 | −889.80 (−17.72%) |
| ETS | 86 → 84 | 5,211.84 | 4,415.40 | −796.44 (−15.28%) |

Removing the two scored targets lowers the all-target RMSE for each model. For the Q1 subgroup, 2020Q1 is the only excluded row, reducing n from 22 to 21. Q1 RMSE falls from **7,159.17 to 4,536.32** for SARIMA, from **7,044.62 to 4,439.32** for SARIMA+CNY, and from **7,194.91 to 4,886.20** for ETS. These are alternative aggregations of the same forecasts, not a new primary result.

### Does shock exclusion change the CNY reading?

For the primary h=2 all-target endpoint, the CNY-minus-SARIMA deltas are:

| Score sample | Δ MAE | Δ RMSE | Δ MASE |
|---|---:|---:|---:|
| Full evaluation | −4.51 (−0.15%) | −93.93 (−1.84%) | −0.001486 (−0.16%) |
| Exclude 2020Q1 and 2022Q2 | +2.05 (+0.08%) | −86.87 (−2.06%) | +0.000104 (+0.01%) |

After excluding the two targets, MAE and MASE move from tiny reductions to tiny increases, while the RMSE reduction remains. The interpretation does not change: CNY does not show a consistent incremental improvement across metrics, and the sizes remain small. In the Q1 subgroup, the MAE change remains positive and RMSE remains negative; MASE changes from almost zero negative to a small positive value. This does not replace the primary all-target result.

## Full-sample descriptive fit

A separate fit on all 130 quarters is descriptive only and is never used in rolling forecasts. It converged for the same fixed ETS specification. Its SSE is **0.256195**, log likelihood **220.445862**, and AIC **−418.891725** (reported because the fit converged and the likelihood/AIC were finite). Residual mean is **0.002575**, sample standard deviation **0.044490**. The unadjusted Ljung–Box statistic at lag 8 is **7.971675** with p-value **0.436242** (8 degrees of freedom). This single residual check does not prove independent errors, and the ETS AIC is not used to compare or select rolling models.

## Limitations

- ETS is represented by one fixed specification; these results do not establish how other ETS components or forecast methods would perform.
- The rolling sample contains 86 targets, and the Q1 subgroup contains 22. The shock check uses only two preset target quarters and is descriptive sensitivity analysis, not a general shock taxonomy.
- The smearing factor is based on in-sample training residuals and does not quantify parameter uncertainty or produce prediction intervals.
- The full-sample residual and likelihood diagnostics are descriptive. No causal claim, significance test, model search, or final future forecast is made.

## Reproduction and outputs

Run from the repository root:

```text
python scripts/modeling/backtest_ets.py
pytest -q
python -m compileall scripts tests
```

- Rolling forecasts: [ets_rolling_forecasts.csv](../outputs/forecasts/ets_rolling_forecasts.csv)
- ETS metrics: [ets_metrics.csv](../outputs/tables/ets_metrics.csv)
- Six-model comparison: [model_comparison_stage6b.csv](../outputs/tables/model_comparison_stage6b.csv)
- Full/excluded shock metrics: [shock_sensitivity_metrics.csv](../outputs/tables/shock_sensitivity_metrics.csv)
- CNY sensitivity deltas: [cny_shock_sensitivity.csv](../outputs/tables/cny_shock_sensitivity.csv)
- Full-sample fit diagnostics: [ets_full_sample_diagnostics.csv](../outputs/tables/ets_full_sample_diagnostics.csv)
- Machine-readable summary: [stage6b_summary.json](../outputs/diagnostics/stage6b_summary.json)
- h=2 actual and ETS forecasts: [ets_h2_actual_vs_forecast.png](../outputs/figures/ets_h2_actual_vs_forecast.png)
