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

logger = logging.getLogger(__name__)


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
        "策略": label,
        "config": config.label,
        "hold_days": config.hold_days,
        "N": config.buckets,
        "entry": config.entry_price,
        "exit": config.exit_price,
        "mode": config.capital_mode,
        "资金利用率": 1.0 - st["avg_cash_pct"],
        "无空槽跳过天数": st["no_free_slot_days"],
        "平均持仓数": st["avg_holding_count"],
        "年化": metrics.get("ann_return", 0.0),
        "年化波动": metrics.get("ann_vol", 0.0),
        "最大回撤": metrics.get("max_drawdown", 0.0),
        "Sharpe": metrics.get("sharpe", 0.0),
        "超额IR": metrics.get("information_ratio", 0.0),
        "日胜率": metrics.get("win_rate_daily", 0.0),
        "日均换手": float(turnover.mean()) if len(turnover) else 0.0,
        "买入失败_涨停": st["buy_fail_breakdown"]["limit_up"],
        "冻结市值均值": st["avg_frozen_pct"],
    }


def run_grid(
    picks: pd.DataFrame,
    start: date,
    end: date,
    spec: GridSpec | None = None,
    baseline_picks: pd.DataFrame | None = None,
    wide: WideFrames | None = None,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """跑完整网格，返回汇总表（每行一个配置）。

    Parameters
    ----------
    picks          : 选股长表 [date, code]
    baseline_picks : 对照组选股（通常是随机基线）；给了就每个配置各跑两遍
    wide           : 已加载的行情；不给则按 [start, end] 加载一次
    """
    spec = spec or GridSpec()
    configs = spec.configs()
    if wide is None:
        logger.info("加载行情 %s ~ %s ...", start, end)
        wide = load_wide_frames(start, end, cache_dir=cache_dir)

    runs: list[tuple[str, pd.DataFrame]] = [("策略", picks)]
    if baseline_picks is not None:
        runs.append(("随机基线", baseline_picks))

    bm_cache: dict = {}
    rows: list[dict] = []
    total = len(configs) * len(runs)
    done = 0

    for config in configs:
        for label, frame in runs:
            result = PickBacktester(config).run(frame, wide)
            if "bm" not in bm_cache:
                bm_cache["bm"] = equal_weight_benchmark(
                    wide.adj["close"], result.nav.index
                )
            metrics = calc_metrics(result.daily_ret, bm_cache["bm"], config.risk_free)
            rows.append(_summarize(config, result, metrics, label))
            done += 1
            logger.info(
                "[%d/%d] %s %s → 年化 %.2f%% 利用率 %.1f%% 跳过 %d",
                done, total, label, config.label,
                rows[-1]["年化"] * 100, rows[-1]["资金利用率"] * 100,
                rows[-1]["无空槽跳过天数"],
            )

    return pd.DataFrame(rows)


def format_grid(frame: pd.DataFrame) -> str:
    """把汇总表格式化成终端可读的字符串。"""
    if frame.empty:
        return "（网格为空）"
    view = frame.copy()
    for col in ("资金利用率", "年化", "年化波动", "最大回撤", "日胜率", "冻结市值均值"):
        view[col] = view[col].map(lambda v: f"{v:.2%}")
    for col in ("Sharpe", "超额IR", "日均换手"):
        view[col] = view[col].map(lambda v: f"{v:.2f}")
    view["平均持仓数"] = view["平均持仓数"].map(lambda v: f"{v:.0f}")
    cols = [
        "策略", "hold_days", "N", "entry", "exit", "mode",
        "资金利用率", "无空槽跳过天数", "年化", "最大回撤", "Sharpe", "超额IR",
    ]
    return view[cols].to_string(index=False)
