"""分年、分月的收益拆解。

年度表回答「哪几年赚、哪几年亏」，月度矩阵回答「收益是均匀的还是靠个别月份」。
两者都基于净值层日收益，逐笔指标（笔数/胜率）按**买入年份**归属。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hqpick.analysis.metrics import max_drawdown
from hqpick.constants import TRADING_DAYS_PER_YEAR

MONTH_COLUMNS = [f"{m}月" for m in range(1, 13)]


def _compound(returns: pd.Series) -> float:
    """日收益复利成区间收益。"""
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def monthly_returns(daily_ret: pd.Series) -> pd.Series:
    """日收益 → 月收益（复利），index 为 Period[M]。"""
    ret = daily_ret.dropna()
    if ret.empty:
        return pd.Series(dtype=float)
    return ret.groupby(ret.index.to_period("M")).apply(_compound)


def monthly_matrix(
    daily_ret: pd.Series, bm_ret: pd.Series | None = None
) -> pd.DataFrame:
    """月度收益矩阵：行=年份，列=1~12月，外加「全年」与可选「全年超额」。"""
    monthly = monthly_returns(daily_ret)
    if monthly.empty:
        return pd.DataFrame()

    frame = pd.DataFrame({
        "year": [p.year for p in monthly.index],
        "month": [p.month for p in monthly.index],
        "ret": monthly.to_numpy(),
    })
    matrix = frame.pivot(index="year", columns="month", values="ret")
    matrix = matrix.reindex(columns=range(1, 13))
    matrix.columns = MONTH_COLUMNS

    ret = daily_ret.dropna()
    yearly = ret.groupby(ret.index.year).apply(_compound)
    matrix["全年"] = yearly

    if bm_ret is not None and not bm_ret.empty:
        bm = bm_ret.reindex(ret.index).fillna(0.0)
        bm_yearly = bm.groupby(bm.index.year).apply(_compound)
        # 超额按净值比口径：(1+策略)/(1+基准) - 1，与年化超额一致
        matrix["全年超额"] = (1.0 + yearly) / (1.0 + bm_yearly) - 1.0

    matrix.index.name = "年份"
    return matrix


def yearly_stats(
    daily_ret: pd.Series,
    bm_ret: pd.Series | None = None,
    round_trips: pd.DataFrame | None = None,
    risk_free: float = 0.02,
) -> pd.DataFrame:
    """分年表现：收益、基准、超额、波动、回撤、Sharpe，以及逐笔笔数与胜率。

    逐笔指标按**买入年份**归属——这一年选出来的票表现如何。
    不足整年的年份（首尾）指标照算，但波动与 Sharpe 已年化，样本少时会失真。
    """
    ret = daily_ret.dropna()
    if ret.empty:
        return pd.DataFrame()

    rows = []
    for year, group in ret.groupby(ret.index.year):
        nav = (1.0 + group).cumprod()
        vol = float(group.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(group) > 1 else 0.0
        total = _compound(group)
        row = {
            "年份": year,
            "交易日": int(len(group)),
            "收益": total,
            "波动": vol,
            "最大回撤": max_drawdown(nav),
            "Sharpe": (
                float((float(group.mean()) * TRADING_DAYS_PER_YEAR - risk_free) / vol)
                if vol > 1e-12 else 0.0
            ),
            "日胜率": float((group > 0).mean()),
        }
        if bm_ret is not None and not bm_ret.empty:
            bm = bm_ret.reindex(group.index).fillna(0.0)
            bm_total = _compound(bm)
            row["基准"] = bm_total
            row["超额"] = (1.0 + total) / (1.0 + bm_total) - 1.0
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("年份")

    if round_trips is not None and not round_trips.empty:
        rt = round_trips.copy()
        rt["year"] = pd.to_datetime(rt["buy_date"]).dt.year
        grouped = rt.groupby("year")
        frame["交易笔数"] = grouped.size()
        frame["逐笔胜率"] = grouped["net_ret"].apply(lambda s: float((s > 0).mean()))
        frame["平均单笔"] = grouped["net_ret"].mean()
        frame[["交易笔数", "逐笔胜率", "平均单笔"]] = frame[
            ["交易笔数", "逐笔胜率", "平均单笔"]
        ].fillna(0.0)

    return frame


def month_of_year_stats(daily_ret: pd.Series) -> pd.DataFrame:
    """跨年汇总的月份效应：每个自然月份（1~12）的平均收益与为正的比例。

    样本通常很少（每个月份只有 N 年），**不要据此下季节性结论**，
    它只用来发现「收益是不是集中在少数几个月」。
    """
    monthly = monthly_returns(daily_ret)
    if monthly.empty:
        return pd.DataFrame()

    frame = pd.DataFrame({
        "month": [p.month for p in monthly.index],
        "ret": monthly.to_numpy(),
    })
    grouped = frame.groupby("month")["ret"]
    out = pd.DataFrame({
        "样本年数": grouped.size(),
        "平均月收益": grouped.mean(),
        "中位月收益": grouped.median(),
        "为正比例": grouped.apply(lambda s: float((s > 0).mean())),
        "最好": grouped.max(),
        "最差": grouped.min(),
    })
    out.index = [f"{m}月" for m in out.index]
    out.index.name = "月份"
    return out
