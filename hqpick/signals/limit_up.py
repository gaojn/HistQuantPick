"""涨停信号：T 日收盘涨停的股票作为 T 日选股。

只用 T 日及之前的数据，成交在 T+entry_offset，符合前视红线。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from hqpick.constants import LIMIT_UP_TOL, SUSPENDED
from hqpick.data.panel import load_panel

_COLUMNS = (
    "code", "date", "close", "limit_up", "trade_status", "is_st", "list_days", "amount",
)


def build_limit_up_picks(
    t1: date,
    t2: date,
    exclude_st: bool = True,
    min_list_days: int = 120,
    min_amount: float = 0.0,
    max_per_day: int | None = None,
    cache_dir: Path | str | None = None,
) -> pl.DataFrame:
    """生成涨停选股长表 [date, code]。

    Parameters
    ----------
    exclude_st    : 剔除 ST/*ST（涨跌停幅度不同且流动性差）
    min_list_days : 最少上市天数，剔除次新股（默认 120 个自然日）
    min_amount    : 当日最低成交额（元），过滤过小的票；0 表示不过滤
    max_per_day   : 每日最多保留几只，按成交额降序取；None 表示全取
    """
    panel = load_panel(t1, t2, columns=_COLUMNS, cache_dir=cache_dir)

    cond = (
        pl.col("close").is_not_null()
        & pl.col("limit_up").is_not_null()
        & (pl.col("close") >= pl.col("limit_up") * LIMIT_UP_TOL)
        & (pl.col("trade_status") != SUSPENDED)
    )
    if exclude_st:
        cond = cond & (pl.col("is_st") == 0)
    if min_list_days > 0:
        cond = cond & (pl.col("list_days") >= min_list_days)
    if min_amount > 0:
        cond = cond & (pl.col("amount") >= min_amount)

    picks = panel.filter(cond).select(["date", "code", "amount"])
    if max_per_day is not None:
        picks = (
            picks.sort(["date", "amount"], descending=[False, True])
            .group_by("date", maintain_order=True)
            .head(max_per_day)
        )
    return picks.select(["date", "code"]).sort(["date", "code"])
