"""Static research figures and a Chinese report backed by actual result tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_COLORS = {
    "SARIMA": "#34495e", "SARIMA+CNY": "#2c7fb8", "SARIMA-W40": "#b7791f",
    "SARIMA-W60": "#6a8055", "SARIMA+ETS-50:50": "#b45b80",
}


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame[columns].itertuples(index=False, name=None):
        cells = []
        for value in values:
            if pd.isna(value):
                cells.append("—")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:,.6f}" if abs(value) < 10 else f"{value:,.2f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def build_figures(output_dir: Path) -> list[str]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    created = []
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(figures / name, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        created.append(f"figures/{name}")

    metrics = pd.read_csv(output_dir / "point_metrics.csv")
    primary = metrics.loc[(metrics.horizon == 2) & (metrics.scope == "all") & (metrics.point_convention == "mean")]
    primary = primary.sort_values("MAE", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        ax.barh(primary.model, primary[metric], color=[MODEL_COLORS.get(m, "#87959c") for m in primary.model])
        ax.set_xlabel(f"{metric} (RMB 100 million)")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("Two-quarter forecasts: all 86 targets, 2005Q1–2026Q2")
    save(fig, "primary_errors.png")

    baseline = pd.read_csv(PROJECT_ROOT / "outputs/forecasts/sarima_rolling_forecasts.csv")
    cny = pd.read_csv(PROJECT_ROOT / "outputs/forecasts/sarima_cny_rolling_forecasts.csv")
    keys = ["origin", "target", "horizon"]
    paired = baseline.merge(cny, on=keys, suffixes=("_baseline", "_cny"), validate="one_to_one")
    paired = paired.loc[paired.horizon == 2].sort_values("target")
    dates = pd.PeriodIndex(paired.target, freq="Q-DEC").to_timestamp()
    delta = paired.absolute_error_cny - paired.absolute_error_baseline
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.plot(dates, delta.cumsum(), color=MODEL_COLORS["SARIMA+CNY"], linewidth=1.8)
    ax.axhline(0, color="#34495e", linestyle="--", linewidth=1)
    ax.set_xlim(dates[0], dates[-1])
    ticks = [dates[0], *[pd.Timestamp(f"{year}-01-01") for year in (2010, 2015, 2020)], dates[-1]]
    ax.set_xticks(ticks, ["2005Q1", "2010", "2015", "2020", "2026Q2"])
    ax.set_ylabel("Cumulative absolute-loss difference\n(RMB 100 million)")
    ax.set_title("CNY minus SARIMA: negative values favor the CNY extension")
    ax.grid(alpha=0.15)
    save(fig, "cny_cumulative_loss.png")

    intervals = pd.read_csv(output_dir / "interval_metrics.csv")
    selected = intervals.loc[(intervals.horizon == 2) & (intervals.scope == "all") & (intervals.subset == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, nominal in zip(axes, (0.8, 0.95)):
        rows = selected.loc[np.isclose(selected.coverage, nominal)].sort_values(["model", "method"])
        labels = [f"{row.model} [{row.method}]" for row in rows.itertuples()]
        colors = [MODEL_COLORS.get(m, "#87959c") for m in rows.model]
        bars = ax.barh(labels, rows.empirical_coverage * 100, color=colors)
        for bar, method in zip(bars, rows.method):
            if method == "calibrated":
                bar.set_hatch("//")
                bar.set_edgecolor("#34495e")
        ax.axvline(nominal * 100, color="#34495e", linestyle="--", linewidth=1)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Observed coverage (%)")
        ax.set_title(f"Nominal {nominal:.0%} interval; h=2, all targets")
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("Calibrated series includes flagged parametric warm-up intervals")
    save(fig, "interval_coverage.png")

    boot = pd.read_csv(output_dir / "paired_bootstrap.csv")
    rows = boot.loc[(boot.horizon == 2) & (boot.scope == "all") & (boot.block_length == 4)
                    & (boot.metric == "MAE") & (boot.point_convention == "mean")]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for index, row in enumerate(rows.itertuples()):
        color = MODEL_COLORS.get(row.extended_model, "#87959c")
        ax.hlines(index, row.lower_95, row.upper_95, color=color, linewidth=2)
        ax.scatter(row.delta, index, color=color, marker="o", s=40)
    ax.set_yticks(range(len(rows)), rows.extended_model)
    ax.axvline(0, color="#34495e", linestyle="--", linewidth=1)
    ax.set_xlabel("Extended minus SARIMA MAE (RMB 100 million)")
    ax.set_title("Historical paired-block bootstrap: 95% percentile intervals, block=4")
    ax.grid(axis="x", alpha=0.15)
    save(fig, "paired_mae_uncertainty.png")
    return created


def build_report(output_dir: Path, execution: dict) -> None:
    metrics = pd.read_csv(output_dir / "point_metrics.csv")
    boot = pd.read_csv(output_dir / "paired_bootstrap.csv")
    intervals = pd.read_csv(output_dir / "interval_metrics.csv")
    periods = pd.read_csv(output_dir / "period_metrics.csv")
    allocations = pd.read_csv(output_dir / "calendar_allocations.csv")
    primary = metrics.loc[(metrics.horizon == 2) & (metrics.scope == "all") & (metrics.point_convention == "mean")]
    baseline = primary.loc[primary.model == "SARIMA"].iloc[0]
    candidates = primary.loc[~primary.model.isin(["SARIMA", "SARIMA+CNY", "ETS"])]
    improved = candidates.loc[(candidates.MAE < baseline.MAE) & (candidates.RMSE < baseline.RMSE) & (candidates.MASE < baseline.MASE)]
    names = "、".join(improved.model) if len(improved) else "无"
    window = execution["window_study"]
    bootstrap_primary = boot.loc[(boot.horizon == 2) & (boot.scope == "all") & (boot.block_length == 4)
                                 & (boot.point_convention == "mean")]
    interval_primary = intervals.loc[(intervals.horizon == 2) & (intervals.scope == "all")]
    medians = metrics.loc[(metrics.horizon == 2) & (metrics.scope == "all") & (metrics.point_convention == "median")]
    recent = periods.loc[(periods.horizon == 2) & (periods.scope == "all") & (periods.point_convention == "mean")]
    sections = [
        "# STA4003 优化研究：第一轮实验报告",
        "研究日期：2026-10-08；数据截至 2026Q2；原始 Stage 1–7 的预测、数据和报告保持不变。",
        "## 已验证的研究状态",
        f"窗口拟合成功 {window['successful_fits']}/{window['requested_fits']}，失败 {window['failed_fits']}。"
        + ("所有窗口模型已覆盖完整目标集。" if window["complete"] else "研究不完整；未完整覆盖的模型不进入总体比较。"),
        f"在已完成的新增规格中，同时低于 SARIMA 的 h=2 全目标 MAE、RMSE、MASE 的规格：**{names}**。"
        "这只是固定历史窗口的点估计比较，不能证明未来泛化；所有新增规格均保持探索性，原主模型不自动替换。",
        "## 1. 点预测比较",
        "主要评价为提前两个季度、2005Q1–2026Q2 的 86 个目标；MAE/RMSE 单位为亿元，MASE 无单位。"
        "所有模型使用同一个已归档的逐起点 MASE 分母。均值预测与中位数预测单独列示。",
        markdown_table(primary, ["model", "n", "MAE", "RMSE", "MASE"]),
        "![主要误差比较](figures/primary_errors.png)",
        "40/60 季度实验保留每个起点原有的 Stage 4 阶数，仅改变训练长度；不足窗口长度时使用当时全部可用历史。"
        "因此检验的是固定已知阶数下的窗口效应，不是重新搜索模型阶数。等权组合为原尺度 SARIMA 与 ETS 的算术平均，不调权重。",
        "## 2. 配对损失与历史不确定性",
        "下表差值均为扩展模型减 SARIMA，负数有利于扩展。使用配对循环区块 bootstrap，"
        "5,000 次、种子 20261008、主要区块长度 4；长度 8 的结果保存在 paired_bootstrap.csv。"
        "区块数的是保留后的观测：Q1 子样本的长度 4 对应四年。",
        markdown_table(bootstrap_primary, ["extended_model", "metric", "delta", "lower_95", "upper_95"]),
        "![MAE 差异的不确定性](figures/paired_mae_uncertainty.png)",
        "![春节变量累计绝对损失差](figures/cny_cumulative_loss.png)",
        "这些区间只描述已归档预测流程下的历史损失差，不重新拟合模型，不包含规格搜索的不确定性，"
        "也不是多重比较校正后的显著性结论。区间跨零既不证明有效，也不证明零效应。",
        "## 3. 滚动预测区间与补充校准",
        "参数区间从各起点保存的对数均值与方差计算；上下界不加半方差的均值修正。"
        "覆盖率、宽度和 Winkler 分数共同评价，Winkler 越低越好。",
        markdown_table(interval_primary, ["model", "coverage", "method", "subset", "n", "empirical_coverage", "mean_width", "winkler_score"]),
        "![预测区间历史覆盖率](figures/interval_coverage.png)",
        "补充校准只使用该模型、该预测步长在当前起点之前已经兑现的误差（目标季度 ≤ 当前起点），"
        "最多最近 40 个，至少 20 个。使用绝对标准化对数误差的经验分位数（higher）。"
        "不足 20 个时回退到参数区间，并逐行标记；all 是含启动期回退的全目标结果，"
        "calibrated_only 仅统计实际校准行，二者样本不同，不能据此直接归因。此方法不保证未来覆盖率。",
        "## 4. 中位数敏感性",
        markdown_table(medians, ["model", "n", "MAE", "RMSE", "MASE"]),
        "中位数 exp(mu) 与条件均值 exp(mu+v/2) 是不同预测目标；"
        "MAE 下的中位数表现用于补充解释，不重写原条件均值结果。组合未提供中位数或区间。",
        "## 5. 分时期稳定性",
        markdown_table(recent, ["model", "period", "n", "MAE", "RMSE", "MASE"]),
        "时期为 2005–2011、2012–2019、2020–2026；分期在看过基线后定义，只是描述性补充，"
        "不用于选择训练窗口或替代全目标主要评价。",
        "## 6. 春节窗口与季度聚合",
    ]
    for label, group in allocations.groupby("window", sort=False):
        q1_unique = group.q1_fraction.nunique()
        q4_unique = group.previous_q4_fraction.nunique()
        sections.append(
            f"窗口 **{label}**：Q1 分配比例范围 {group.q1_fraction.min():.6f}–{group.q1_fraction.max():.6f}，"
            f"有 {q1_unique} 个不同值；前一年 Q4 有 {q4_unique} 个不同值。"
            + ("Q1 变量没有年际变化，中心化后不能识别额外的日期效应。" if q1_unique == 1 else "存在跨季度分配变化，但本轮没有拟合或验证其预测价值。")
        )
    sections.extend([
        "这是对预设日历窗口的机械分配检查；不能据此证明实际消费影响持续多少天，"
        "也不能证明春节没有经济效应。没有填补或拆分 2012 年后的一二月联合数据。",
        "## 7. 数值与证据限制",
        "所有历史结果都使用当前归档的数据版本，未重建历史发布版本与发布滞后，属于伪实时比较。"
        "新规格是在观察既有成绩后提出的；本轮未使用一个新的、未查看的外部测试集。"
        "需要后续未查看季度或另行设计的独立验证才能支持主模型替换。",
        "原 Stage 7 跨环境重拟合产生细小数值差异；因此确定性测试现在在临时目录比较两次本机运行，"
        "并检查原归档未变。明确的逐文件换行规则保留已有归档清单记录的混合 LF/CRLF。",
        "拟合失败不会被其他模型预测替代；详见 window_fits.csv、window_execution.json 和 checkpoints/。"
        "本轮没有改变原订单选择、春节定义、PMI 规格或最终预测。",
        "## 8. 复现与来源",
        "完整协议见 [optimization_protocol.md](../../../docs/optimization_protocol.md)。"
        "原始数值来自 outputs/forecasts 下的 SARIMA、SARIMA+CNY、ETS 归档，"
        "新窗口结果来自本目录 window_forecasts.csv；其余表格由 analysis.py 推导。",
        "```powershell\npython scripts/optimization/run_research.py\n"
        "python scripts/optimization/run_research.py --analysis-only\n"
        "python -m pytest -q\npython -m compileall scripts tests\n```",
        "已有成功 checkpoint 可直接续用；失败 checkpoint 仅在显式 --retry-failed 时重试。"
        "运行版本和基线保留结果见 execution.json。",
        "```json\n" + json.dumps(execution["runtime"], ensure_ascii=False, indent=2) + "\n```",
    ])
    if "SARIMA-W60" in set(primary.model):
        candidate = primary.loc[primary.model == "SARIMA-W60"].iloc[0]
        mae_change = 100 * (candidate.MAE / baseline.MAE - 1)
        rmse_change = 100 * (candidate.RMSE / baseline.RMSE - 1)
        loss_interval = bootstrap_primary.loc[(bootstrap_primary.extended_model == "SARIMA-W60")
                                               & (bootstrap_primary.metric == "MAE")].iloc[0]
        sections.insert(sections.index("## 2. 配对损失与历史不确定性"),
                        f"60 季度窗口的主要 MAE 相对变化为 {mae_change:+.2f}%，RMSE 为 {rmse_change:+.2f}%。"
                        f"但 MAE 差的主要历史 bootstrap 区间为 [{loss_interval.lower_95:.2f}, {loss_interval.upper_95:.2f}] 亿元；"
                        "应作为后续验证候选，不应直接宣称获得稳定提升。分期表显示改善集中在哪些年份，而非假设短窗口普遍更好。")
    parameter95 = interval_primary.loc[(interval_primary.model == "SARIMA") & (interval_primary.coverage == 0.95)
                                      & (interval_primary.method == "parametric") & (interval_primary.subset == "all")].iloc[0]
    calibrated95 = interval_primary.loc[(interval_primary.model == "SARIMA") & (interval_primary.coverage == 0.95)
                                       & (interval_primary.method == "calibrated") & (interval_primary.subset == "all")].iloc[0]
    sections.insert(sections.index("## 4. 中位数敏感性"),
                    f"原 SARIMA 的名义 95% 区间实际覆盖率为 {parameter95.empirical_coverage:.2%}，"
                    f"补充校准全目标覆盖率为 {calibrated95.empirical_coverage:.2%}；"
                    f"Winkler 分数分别为 {parameter95.winkler_score:,.2f} 和 {calibrated95.winkler_score:,.2f}。"
                    "覆盖率与区间代价必须一起解读，不能仅凭区间变宽就认定校准改善。")
    shock_path = output_dir / "shock_sensitivity.csv"
    if shock_path.is_file():
        shocks = pd.read_csv(shock_path)
        excluded = shocks.loc[(shocks.horizon == 2) & (shocks.scope == "all")
                              & (shocks.point_convention == "mean") & (shocks.score_sample == "exclude_original_shocks")]
        index = sections.index("## 6. 春节窗口与季度聚合")
        sections[index:index] = [
            "### 原有两个冲击季度的补充评分敏感性",
            "仅从评分目标中剔除原 Stage 6B 已预设的 2020Q1 和 2022Q2；不删除训练数据、不重新拟合，"
            "也不替换 86 个目标的主要评价。这项补充评分在观察首轮结果后加入，未选择新的冲击日期。",
            markdown_table(excluded, ["model", "n", "MAE", "RMSE", "MASE"]),
        ]
    (output_dir / "research_report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
