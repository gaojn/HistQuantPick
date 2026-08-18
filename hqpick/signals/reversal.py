"""短期反转族信号：截面上按「近期跌幅」打分选股，可叠加质量/流动性过滤。

经济逻辑
--------
A 股散户占比高、短期情绪波动大，个股短期超跌后存在均值回复。这是日频最有
实证基础的 alpha 之一，比形态/打板类稳健。但裸反转会选到两类噪音票：
流动性差的小票（价格发现失效，跌是真跌）和趋势性下跌的票（跌是基本面），
所以提供流动性、波动、市值、中期动量几个过滤轴。

前视
----
所有因子只用 T 日及之前的数据：
- ``ret_k`` = close / close.shift(k) − 1，含 T 日收盘，T 日收盘后可知；
- 滚动统计（均量、波动）右对齐，放量倍数的均量窗口 shift(1) 不含当日；
- 成交在 T+entry_offset（引擎强制 ≥1）。
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from hqpick.constants import LIMIT_UP_TOL, SUSPENDED
from hqpick.data.panel import ensure_years_available, load_panel

_COLUMNS = (
    "code", "date", "open", "high", "low", "close", "pre_close", "limit_up",
    "volume", "amount", "turnover", "trade_status", "is_st", "list_days", "float_mv",
)

# 滚动窗口用的固定长度
_MOM_WINDOW = 20
_VOL_WINDOW = 20


def load_reversal_factors(
    t1: date,
    t2: date,
    cache_dir: Path | str | None = None,
) -> pl.DataFrame:
    """加载行情并算好反转族所需的全部因子列（含 warmup，未裁剪到 t1）。

    抽出来单独提供，是为了让研究脚本一次算完、多个候选共用，避免重复 IO。
    """
    ensure_years_available(t1, t2, cache_dir)
    warmup = timedelta(days=int(max(_MOM_WINDOW, _VOL_WINDOW) * 3 + 40))
    panel = load_panel(
        t1 - warmup, t2, columns=_COLUMNS, cache_dir=cache_dir,
        require_all_years=False,
    )

    frame = panel.sort(["code", "date"]).with_columns(
        (pl.col("close") / pl.col("pre_close") - 1.0).alias("ret1d"),
    )
    return frame.with_columns(
        # 近期涨跌幅：含 T 日收盘
        (pl.col("close") / pl.col("close").shift(1).over("code") - 1.0).alias("ret1"),
        (pl.col("close") / pl.col("close").shift(5).over("code") - 1.0).alias("ret5"),
        (pl.col("close") / pl.col("close").shift(_MOM_WINDOW).over("code") - 1.0)
        .alias("ret20"),
        # 波动：过去 20 日日收益标准差（右对齐，含 T）
        pl.col("ret1d")
        .rolling_std(window_size=_VOL_WINDOW, min_samples=_VOL_WINDOW)
        .over("code")
        .alias("vol20"),
        # 流动性：20 日均成交额（右对齐，含 T）
        pl.col("amount")
        .rolling_mean(window_size=_VOL_WINDOW, min_samples=_VOL_WINDOW)
        .over("code")
        .alias("amount_ma20"),
        # 放量倍数：均量右对齐到 T−1，不含当日
        (
            pl.col("volume")
            / pl.col("volume")
            .rolling_mean(window_size=_VOL_WINDOW, min_samples=_VOL_WINDOW)
            .shift(1)
            .over("code")
        ).alias("volume_ratio"),
    )


def base_universe(
    exclude_st: bool = True,
    min_list_days: int = 120,
    min_amount: float = 5e4,
    exclude_limit_up: bool = True,
) -> pl.Expr:
    """可交易域：剔除停牌、ST、次新、成交额过小，以及当日涨停（T+1 买不进）。

    ``min_amount`` 单位是**千元**（缓存口径），默认 5e4 即 5000 万元。
    """
    cond = (
        pl.col("close").is_not_null()
        & (pl.col("trade_status") != SUSPENDED)
        & (pl.col("volume") > 0)
    )
    if exclude_st:
        cond = cond & (pl.col("is_st") == 0)
    if min_list_days > 0:
        cond = cond & (pl.col("list_days") >= min_list_days)
    if min_amount > 0:
        cond = cond & (pl.col("amount") >= min_amount)
    if exclude_limit_up:
        cond = cond & (
            pl.col("limit_up").is_null()
            | (pl.col("close") < pl.col("limit_up") * LIMIT_UP_TOL)
        )
    return cond


def select_top(
    factors: pl.DataFrame,
    t1: date,
    score: pl.Expr,
    extra_filter: pl.Expr | None = None,
    max_per_day: int = 20,
    universe: pl.Expr | None = None,
) -> pl.DataFrame:
    """按 ``score`` 降序在每日截面取前 N，返回 [date, code]。

    ``score`` 需自行处理方向（反转类传 ``-ret5``）。截面分位类过滤请放在
    ``extra_filter`` 里，用 ``over("date")`` 表达。
    """
    cond = universe if universe is not None else base_universe()
    cond = cond & (pl.col("date") >= t1) & score.is_not_null()
    if extra_filter is not None:
        cond = cond & extra_filter

    picked = factors.filter(cond).with_columns(score.alias("_score"))
    if picked.is_empty():
        return pl.DataFrame({"date": [], "code": []})

    return (
        picked.sort(["date", "_score"], descending=[False, True])
        .group_by("date", maintain_order=True)
        .head(max_per_day)
        .select(["date", "code"])
        .sort(["date", "code"])
    )


def build_reversal_picks(
    t1: date,
    t2: date,
    lookback: int = 5,
    max_per_day: int = 20,
    min_liquidity_pct: float = 0.0,
    max_vol_pct: float = 1.0,
    min_ret20: float | None = None,
    min_float_mv: float = 0.0,
    min_volume_ratio: float | None = None,
    cache_dir: Path | str | None = None,
) -> pl.DataFrame:
    """短期反转选股：取 ``lookback`` 日跌幅最大的股票。

    Parameters
    ----------
    lookback          : 反转窗口，1 或 5
    max_per_day       : 每日选股数
    min_liquidity_pct : 成交额截面分位下限（0.5=只要流动性前 50%）
    max_vol_pct       : 波动率截面分位上限（0.5=只要低波动那一半）
    min_ret20         : 20 日涨跌幅下限（0.0=只在中期上行的票里做回调买入）
    min_float_mv      : 流通市值下限（**亿元**）
    min_volume_ratio  : 当日放量倍数下限
    """
    if lookback not in (1, 5):
        raise ValueError(f"lookback 目前只支持 1 或 5，当前为 {lookback}")

    factors = load_reversal_factors(t1, t2, cache_dir=cache_dir)
    ret_col = "ret1" if lookback == 1 else "ret5"

    extra = None

    def _and(expr: pl.Expr) -> None:
        nonlocal extra
        extra = expr if extra is None else extra & expr

    if min_liquidity_pct > 0:
        _and(
            (pl.col("amount_ma20").rank(descending=False).over("date") / pl.len().over("date"))
            >= min_liquidity_pct
        )
    if max_vol_pct < 1.0:
        _and(
            (pl.col("vol20").rank(descending=False).over("date") / pl.len().over("date"))
            <= max_vol_pct
        )
    if min_ret20 is not None:
        _and(pl.col("ret20") >= min_ret20)
    if min_float_mv > 0:
        _and(pl.col("float_mv") >= min_float_mv * 1e4)
    if min_volume_ratio is not None:
        _and(pl.col("volume_ratio") >= min_volume_ratio)

    return select_top(
        factors, t1,
        score=-pl.col(ret_col),          # 跌得越多、分数越高
        extra_filter=extra,
        max_per_day=max_per_day,
    )
