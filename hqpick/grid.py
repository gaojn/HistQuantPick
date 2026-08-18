"""参数网格扫描：同一份选股、同一份行情，只变执行口径。

行情只加载一次，网格内所有配置复用，避免重复 IO。
可选同时跑一遍随机基线——比较策略优劣时，两边的口径必须完全一致。
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from hqpick.analysis.metrics import calc_metrics, equal_weight_benchmark
from hqpick.data.panel import WideFrames, load_wide_frames
from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester
from hqpick.picks import normalize_frame

logger = logging.getLogger(__name__)

# 汇总表里跨函数读写的列名，集中定义以避免 _summarize 改字段名时
# _summarize_baseline / run_grid 只能在运行时才炸出 KeyError。
# 其余仅在 format_grid 内部使用、不跨函数传递的列名不在此列。
COL_STRATEGY = "策略"
COL_ANN_RETURN = "年化"
COL_UTILIZATION = "资金利用率"
COL_NO_FREE_SLOT_DAYS = "无空槽跳过天数"


@dataclass(frozen=True)
class GridSpec:
    """要扫的参数轴。每个轴给一组取值，笛卡尔积展开。"""

    hold_days: Sequence[int] = (1,)
    n_buckets: Sequence[int | None] = (None,)
    entry_price: Sequence[str] = ("open",)
    exit_price: Sequence[str] = ("close",)
    capital_mode: Sequence[str] = ("slots",)
    writeoff_stuck_days: Sequence[int] = (0,)
    fixed: dict = field(default_factory=dict)      # 其余 ExecConfig 参数

    def configs(self) -> list[ExecConfig]:
        axes = itertools.product(
            self.hold_days, self.n_buckets, self.entry_price,
            self.exit_price, self.capital_mode, self.writeoff_stuck_days,
        )
        return [
            ExecConfig(
                hold_days=h, n_buckets=n, entry_price=ep, exit_price=xp,
                capital_mode=cm, writeoff_stuck_days=wd, **self.fixed,
            )
            for h, n, ep, xp, cm, wd in axes
        ]


def _summarize(config: ExecConfig, result, metrics: dict, label: str) -> dict:
    st = result.exec_stats
    turnover = result.turnover
    return {
        COL_STRATEGY: label,
        "config": config.label,
        "hold_days": config.hold_days,
        "N": config.buckets,
        "entry": config.entry_price,
        "exit": config.exit_price,
        "mode": config.capital_mode,
        COL_UTILIZATION: 1.0 - st["avg_cash_pct"],
        COL_NO_FREE_SLOT_DAYS: st["no_free_slot_days"],
        "平均持仓数": st["avg_holding_count"],
        COL_ANN_RETURN: metrics.get("ann_return", 0.0),
        "年化波动": metrics.get("ann_vol", 0.0),
        "最大回撤": metrics.get("max_drawdown", 0.0),
        "Sharpe": metrics.get("sharpe", 0.0),
        "超额IR": metrics.get("information_ratio", 0.0),
        "日胜率": metrics.get("win_rate_daily", 0.0),
        "日均换手": float(turnover.mean()) if len(turnover) else 0.0,
        "买入失败_涨停": st["buy_fail_breakdown"]["limit_up"],
        "冻结市值均值": st["avg_frozen_pct"],
    }


def _summarize_baseline(
    config: ExecConfig,
    path_rows: list[dict],
    strategy_row: dict,
) -> dict:
    """汇总多条随机路径，并把可比较的分布统计挂到策略行。"""
    paths = pd.DataFrame(path_rows)
    annual = paths[COL_ANN_RETURN]
    n_paths = len(paths)
    strategy_ann_return = strategy_row[COL_ANN_RETURN]
    baseline_stats = {
        "基线样本数": n_paths,
        "基线年化均值": float(annual.mean()),
        "基线年化P05": float(annual.quantile(0.05)),
        "基线年化P95": float(annual.quantile(0.95)),
        # 加一校正避免有限模拟下出现 p=0；值越小，策略优于随机路径的证据越强。
        "年化经验p": float(((annual >= strategy_ann_return).sum() + 1) / (n_paths + 1)),
        "策略年化分位": float((annual <= strategy_ann_return).mean()),
    }
    strategy_row.update(baseline_stats)

    summary = path_rows[0].copy()
    for column in paths.columns:
        if pd.api.types.is_numeric_dtype(paths[column]):
            summary[column] = float(paths[column].mean())
    summary.update(baseline_stats)
    summary[COL_STRATEGY] = "随机基线"
    summary["基线样本数"] = n_paths
    return summary


def _signal_schedule(picks: pd.DataFrame) -> pd.Series:
    """标准化后的信号日 → 当日选股数，用于基线可比性校验。"""
    return normalize_frame(picks).groupby("date").size()


def run_grid(
    picks: pd.DataFrame,
    start: date,
    end: date,
    spec: GridSpec | None = None,
    baseline_picks: pd.DataFrame | Sequence[pd.DataFrame] | None = None,
    wide: WideFrames | None = None,
    cache_dir: Path | str | None = None,
    return_baseline_paths: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """跑完整网格，返回汇总表（每行一个配置）。

    Parameters
    ----------
    picks          : 选股长表 [date, code]
    baseline_picks : 一条或多条对照选股路径。多路径时返回其分布汇总；每条路径
                     必须与策略拥有相同的评价日历。
    wide           : 已加载的行情；不给则按 [start, end] 加载一次
    """
    spec = spec or GridSpec()
    configs = spec.configs()
    if wide is None:
        logger.info("加载行情 %s ~ %s ...", start, end)
        wide = load_wide_frames(start, end, cache_dir=cache_dir)

    if baseline_picks is None:
        baseline_paths: list[pd.DataFrame] = []
    elif isinstance(baseline_picks, pd.DataFrame):
        baseline_paths = [baseline_picks]
    else:
        baseline_paths = list(baseline_picks)
    if not baseline_paths and baseline_picks is not None:
        raise ValueError("至少需要一条随机基线路径")
    reference_schedule = _signal_schedule(picks)
    for path_idx, baseline in enumerate(baseline_paths, start=1):
        if not _signal_schedule(baseline).equals(reference_schedule):
            raise ValueError(
                f"随机基线第 {path_idx} 条路径的信号日期或每日选股数与策略不一致；"
                "请使用 build_matched_random_baselines 生成基线。"
            )

    rows: list[dict] = []
    path_rows_all: list[dict] = []
    total = len(configs) * (1 + len(baseline_paths))
    done = 0

    for config in configs:
        strategy_result = PickBacktester(config).run(picks, wide)
        bm_ret = equal_weight_benchmark(wide.adj["close"], strategy_result.nav.index)
        strategy_metrics = calc_metrics(
            strategy_result.daily_ret, bm_ret, config.risk_free
        )
        strategy_row = _summarize(config, strategy_result, strategy_metrics, "策略")
        rows.append(strategy_row)
        done += 1
        logger.info(
            "[%d/%d] 策略 %s → 年化 %.2f%% 利用率 %.1f%% 跳过 %d",
            done, total, config.label,
            strategy_row[COL_ANN_RETURN] * 100, strategy_row[COL_UTILIZATION] * 100,
            strategy_row[COL_NO_FREE_SLOT_DAYS],
        )

        baseline_rows: list[dict] = []
        for path_idx, baseline in enumerate(baseline_paths, start=1):
            result = PickBacktester(config).run(baseline, wide)
            if not result.nav.index.equals(strategy_result.nav.index):
                raise ValueError(
                    "随机基线与策略的评价日历不一致；"
                    "请使用 build_matched_random_baselines 生成基线。"
                )
            metrics = calc_metrics(result.daily_ret, bm_ret, config.risk_free)
            path_row = _summarize(config, result, metrics, "随机基线")
            baseline_rows.append(path_row)
            path_rows_all.append({"基线路径": path_idx, **path_row})
            done += 1
            logger.info(
                "[%d/%d] %s %s → 年化 %.2f%% 利用率 %.1f%% 跳过 %d",
                done, total, "随机基线", config.label,
                path_row[COL_ANN_RETURN] * 100, path_row[COL_UTILIZATION] * 100,
                path_row[COL_NO_FREE_SLOT_DAYS],
            )

        if baseline_rows:
            rows.append(_summarize_baseline(config, baseline_rows, strategy_row))

    summary = pd.DataFrame(rows)
    path_frame = pd.DataFrame(path_rows_all)
    return (summary, path_frame) if return_baseline_paths else summary


def format_grid(frame: pd.DataFrame) -> str:
    """把汇总表格式化成终端可读的字符串。"""
    if frame.empty:
        return "（网格为空）"
    view = frame.copy()
    for col in ("资金利用率", "年化", "年化波动", "最大回撤", "日胜率", "冻结市值均值",
                "基线年化均值", "基线年化P05", "基线年化P95", "年化经验p", "策略年化分位"):
        if col not in view:
            continue
        view[col] = view[col].map(lambda v: f"{v:.2%}")
    for col in ("Sharpe", "超额IR", "日均换手"):
        view[col] = view[col].map(lambda v: f"{v:.2f}")
    view["平均持仓数"] = view["平均持仓数"].map(lambda v: f"{v:.0f}")
    cols = [
        "策略", "hold_days", "N", "entry", "exit", "mode",
        "资金利用率", "无空槽跳过天数", "年化", "最大回撤", "Sharpe", "超额IR",
    ]
    cols += [
        col for col in ("基线样本数", "基线年化P05", "基线年化P95", "年化经验p")
        if col in view
    ]
    return view[cols].to_string(index=False)
