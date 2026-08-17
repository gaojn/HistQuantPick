"""涨停信号：T 日收盘涨停（可要求连续 N 日涨停）的股票作为 T 日选股。

``consecutive=2`` 即「连续两天涨停」，配默认口径就是：
第 1、2 天涨停 → 第 3 天（T+1）开盘买 → 第 4 天（T+2）收盘卖。

只用 T 日及之前的数据，成交在 T+entry_offset，符合前视红线。
连续判定的滚动窗口右对齐、含当日（当日涨停在收盘后即可知）。
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from hqpick.constants import LIMIT_UP_TOL, SUSPENDED
from hqpick.data.panel import ensure_years_available, load_panel

_COLUMNS = (
    "code", "date", "close", "limit_up", "trade_status", "is_st", "list_days", "amount",
)


def build_limit_up_picks(
    t1: date,
    t2: date,
    consecutive: int = 1,
    exclude_st: bool = True,
    min_list_days: int = 120,
    min_amount: float = 0.0,
    max_per_day: int | None = None,
    cache_dir: Path | str | None = None,
) -> pl.DataFrame:
    """生成涨停选股长表 [date, code]。

    Parameters
    ----------
    consecutive   : 要求连续几个交易日涨停（含当日），默认 1。
                    停牌日不算涨停，会打断连续。
    exclude_st    : 剔除 ST/*ST（涨跌停幅度不同且流动性差）
    min_list_days : 最少上市天数，剔除次新股（默认 120 个自然日）
    min_amount    : 当日最低成交额（**千元**，缓存单位），过滤过小的票；0 表示不过滤
    max_per_day   : 每日最多保留几只，按成交额降序取；None 表示全取
    """
    if consecutive < 1:
        raise ValueError(f"consecutive 须 ≥ 1，当前为 {consecutive}")

    ensure_years_available(t1, t2, cache_dir)
    # 连续判定需要 T 之前的交易日；预热段允许缺年份（起点靠近缓存最早年份时）
    warmup = timedelta(days=int((consecutive - 1) * 4 + 10)) if consecutive > 1 else timedelta(0)
    panel = load_panel(
        t1 - warmup, t2, columns=_COLUMNS, cache_dir=cache_dir,
        require_all_years=False,
    )

    is_limit_up = (
        pl.col("close").is_not_null()
        & pl.col("limit_up").is_not_null()
        & (pl.col("close") >= pl.col("limit_up") * LIMIT_UP_TOL)
        & (pl.col("trade_status") != SUSPENDED)
    )

    if consecutive > 1:
        panel = panel.sort(["code", "date"]).with_columns(
            is_limit_up.cast(pl.Int8).alias("_lu")
        ).with_columns(
            # 右对齐含当日：连续 consecutive 天都涨停才算命中
            pl.col("_lu")
            .rolling_sum(window_size=consecutive, min_samples=consecutive)
            .over("code")
            .alias("_lu_streak")
        )
        cond = pl.col("_lu_streak") == consecutive
    else:
        cond = is_limit_up

    cond = cond & (pl.col("date") >= t1)
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
