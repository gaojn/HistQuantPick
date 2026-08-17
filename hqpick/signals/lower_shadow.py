"""下影线放量信号：T 日收出长下影线且放量的股票。

下影线 = 实体下沿到最低价的那一段，反映当日盘中被砸下去又被买回来——
「下方有承接」。单看影线长度容易选到无人问津的票，所以叠加放量条件：
有量的下影线才是真买盘，缩量的下影线可能只是流动性差导致的插针。

口径
----
    实体下沿 = min(开盘价, 收盘价)
    下影线比率 = (实体下沿 − 最低价) / 昨收
    放量倍数   = 当日成交量 / 过去 volume_window 日均量（不含当日）

两项各取当日截面百分位，等权相加为总分，取分数最高的若干只。
全部只用 T 日及之前的数据；均量窗口右对齐到 T−1，不含当日，避免自相关。
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from hqpick.constants import LIMIT_UP_TOL, SUSPENDED
from hqpick.data.panel import ensure_years_available, load_panel

_COLUMNS = (
    "code", "date", "open", "high", "low", "close", "pre_close", "limit_up",
    "volume", "amount", "trade_status", "is_st", "list_days", "float_mv",
)


def build_lower_shadow_picks(
    t1: date,
    t2: date,
    max_per_day: int = 5,
    volume_window: int = 20,
    min_shadow: float = 0.02,
    min_volume_ratio: float = 1.0,
    exclude_limit_up: bool = True,
    exclude_st: bool = True,
    min_list_days: int = 120,
    min_amount: float = 5e4,
    min_float_mv: float = 0.0,
    cache_dir: Path | str | None = None,
) -> pl.DataFrame:
    """生成下影线放量选股长表 [date, code]。

    Parameters
    ----------
    max_per_day      : 每日最多选几只（按总分降序）
    volume_window    : 均量窗口（交易日），右对齐到 T−1、不含当日
    min_shadow       : 最小下影线比率，默认 2%
    min_volume_ratio : 最小放量倍数，默认 1.0（即不低于近期均量）
    exclude_limit_up : 剔除当日涨停的票（T+1 大概率买不进）
    min_amount       : 当日最低成交额（**千元**，缓存单位），默认 5e4 即 5000 万元
    min_float_mv     : 最低流通市值（**亿元**），0=不过滤。小盘股的下影线
                       更多是流动性插针而非真实承接，设 50 可显著改善
    """
    ensure_years_available(t1, t2, cache_dir)
    # 均量窗口需要 T 之前的数据，向前多取一段日历日（含非交易日与节假日冗余）。
    # 预热段允许缺年份：从缓存最早年份起算时不该因此失败，只是最初几天没信号。
    warmup = timedelta(days=int(volume_window * 2 + 30))
    panel = load_panel(
        t1 - warmup, t2, columns=_COLUMNS, cache_dir=cache_dir,
        require_all_years=False,
    )

    tradable = (
        pl.col("close").is_not_null()
        & pl.col("low").is_not_null()
        & (pl.col("trade_status") != SUSPENDED)
        & (pl.col("volume") > 0)
    )

    frame = panel.sort(["code", "date"]).with_columns(
        # 均量右对齐到 T−1：先 rolling 再 shift(1)，绝不含当日
        pl.col("volume")
        .rolling_mean(window_size=volume_window, min_samples=volume_window)
        .shift(1)
        .over("code")
        .alias("avg_volume"),
    )

    body_low = pl.min_horizontal("open", "close")
    frame = frame.with_columns(
        ((body_low - pl.col("low")) / pl.col("pre_close")).alias("shadow_ratio"),
        (pl.col("volume") / pl.col("avg_volume")).alias("volume_ratio"),
    )

    cond = (
        tradable
        & pl.col("shadow_ratio").is_not_null()
        & pl.col("volume_ratio").is_not_null()
        & (pl.col("shadow_ratio") >= min_shadow)
        & (pl.col("volume_ratio") >= min_volume_ratio)
        & (pl.col("date") >= t1)
    )
    if exclude_limit_up:
        cond = cond & (
            pl.col("limit_up").is_null()
            | (pl.col("close") < pl.col("limit_up") * LIMIT_UP_TOL)
        )
    if exclude_st:
        cond = cond & (pl.col("is_st") == 0)
    if min_list_days > 0:
        cond = cond & (pl.col("list_days") >= min_list_days)
    if min_amount > 0:
        cond = cond & (pl.col("amount") >= min_amount)
    if min_float_mv > 0:
        # 参数用亿元，缓存 float_mv 单位是万元
        cond = cond & (pl.col("float_mv") >= min_float_mv * 1e4)

    picked = frame.filter(cond)
    if picked.is_empty():
        return pl.DataFrame({"date": [], "code": []})

    # 两项各取当日截面百分位，等权相加
    picked = picked.with_columns(
        (
            pl.col("shadow_ratio").rank(method="average").over("date")
            / pl.len().over("date")
            + pl.col("volume_ratio").rank(method="average").over("date")
            / pl.len().over("date")
        ).alias("score")
    )

    return (
        picked.sort(["date", "score"], descending=[False, True])
        .group_by("date", maintain_order=True)
        .head(max_per_day)
        .select(["date", "code"])
        .sort(["date", "code"])
    )
