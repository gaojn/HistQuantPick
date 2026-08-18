"""随机选股信号：管线冒烟与 alpha 显著性对照的 baseline。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from hqpick.constants import SUSPENDED
from hqpick.data.panel import load_panel
from hqpick.picks import normalize_frame

_COLUMNS = ("code", "date", "close", "trade_status", "is_st", "list_days")


def _eligible_pool(
    panel: pl.DataFrame,
    *,
    exclude_st: bool,
    min_list_days: int,
) -> pl.DataFrame:
    """筛出随机抽样可用的 T 日股票池。"""
    cond = pl.col("close").is_not_null() & (pl.col("trade_status") != SUSPENDED)
    if exclude_st:
        cond = cond & (pl.col("is_st") == 0)
    if min_list_days > 0:
        cond = cond & (pl.col("list_days") >= min_list_days)
    return panel.filter(cond).select(["date", "code"]).sort(["date", "code"])


def build_random_picks(
    t1: date,
    t2: date,
    n_per_day: int = 10,
    seed: int = 42,
    exclude_st: bool = True,
    min_list_days: int = 120,
    cache_dir: Path | str | None = None,
) -> pl.DataFrame:
    """每个交易日从可交易股票中随机抽 ``n_per_day`` 只，返回 [date, code]。

    随机基线的意义：同一套执行口径下跑一遍，任何策略的净值都必须显著
    跑赢它，否则「超额」只是执行规则或市场 beta 带来的假象。
    """
    if n_per_day < 1:
        raise ValueError(f"n_per_day 须 ≥ 1，当前为 {n_per_day}")

    panel = load_panel(t1, t2, columns=_COLUMNS, cache_dir=cache_dir)
    pool = _eligible_pool(
        panel, exclude_st=exclude_st, min_list_days=min_list_days,
    )

    rng = np.random.default_rng(seed)
    rows: list[pl.DataFrame] = []
    for (day,), group in pool.group_by(["date"], maintain_order=True):
        codes = group["code"].to_list()
        take = min(n_per_day, len(codes))
        if take == 0:
            continue
        chosen = rng.choice(codes, size=take, replace=False)
        rows.append(pl.DataFrame({"date": [day] * take, "code": sorted(chosen)}))

    if not rows:
        return pl.DataFrame({"date": [], "code": []})
    return pl.concat(rows).sort(["date", "code"])


def build_matched_random_baselines(
    reference_picks: pd.DataFrame,
    n_paths: int = 1,
    seed: int = 42,
    exclude_st: bool = True,
    min_list_days: int = 120,
    cache_dir: Path | str | None = None,
) -> list[pl.DataFrame]:
    """生成与策略日程严格匹配的多条随机基线。

    每条路径都复用 ``reference_picks`` 的每个信号日和当日选股数，只随机替换
    股票代码。因此策略与基线的建仓频率、评价起点和资金周转日程完全一致。
    若 T 日可抽样股票不足以匹配策略当日选股数，直接报错而不悄悄缩减样本。
    """
    if n_paths < 1:
        raise ValueError(f"n_paths 须 ≥ 1，当前为 {n_paths}")

    reference = normalize_frame(reference_picks)
    schedule = reference.groupby("date").size()
    if schedule.empty:
        raise ValueError("参考选股表为空，无法生成匹配日程的随机基线")

    panel = load_panel(
        schedule.index.min().date(), schedule.index.max().date(),
        columns=_COLUMNS, cache_dir=cache_dir,
    )
    pool = _eligible_pool(
        panel, exclude_st=exclude_st, min_list_days=min_list_days,
    )
    candidates_by_day = {
        pd.Timestamp(day): group["code"].to_list()
        for (day,), group in pool.group_by("date", maintain_order=True)
    }

    paths: list[pl.DataFrame] = []
    for path_idx in range(n_paths):
        rng = np.random.default_rng(seed + path_idx)
        rows: list[pl.DataFrame] = []
        for day, count in schedule.items():
            candidates = candidates_by_day.get(pd.Timestamp(day), [])
            if len(candidates) < count:
                raise ValueError(
                    f"{pd.Timestamp(day):%Y-%m-%d} 的随机股票池仅有 {len(candidates)} 只，"
                    f"不足以匹配策略当日 {count} 只选股；请放宽基线股票池或缩小策略日程。"
                )
            chosen = rng.choice(candidates, size=count, replace=False)
            rows.append(pl.DataFrame({"date": [day] * count, "code": sorted(chosen)}))
        paths.append(pl.concat(rows).sort(["date", "code"]))
    return paths
