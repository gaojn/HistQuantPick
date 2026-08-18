"""净值层绩效指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hqpick.constants import TRADING_DAYS_PER_YEAR


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤（正数表示回撤幅度）。"""
    if nav.empty:
        return 0.0
    running_max = nav.cummax()
    return float((1.0 - nav / running_max).max())


def calc_metrics(
    daily_ret: pd.Series,
    bm_ret: pd.Series | None = None,
    risk_free: float = 0.02,
) -> dict:
    """组合日收益 → 年化收益/波动/Sharpe/回撤/胜率，以及相对基准的超额指标。"""
    ret = daily_ret.dropna()
    n = len(ret)
    if n == 0:
        return {}

    nav = (1.0 + ret).cumprod()
    years = n / TRADING_DAYS_PER_YEAR
    total_return = float(nav.iloc[-1] - 1.0)
    ann_return = float(nav.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    ann_vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if n > 1 else 0.0
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 1e-12 else 0.0
    mdd = max_drawdown(nav)

    metrics = {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "risk_free": float(risk_free),
        "max_drawdown": mdd,
        "calmar": float(ann_return / mdd) if mdd > 1e-12 else 0.0,
        "win_rate_daily": float((ret > 0).mean()),
        "n_days": n,
    }

    if bm_ret is not None and not bm_ret.empty:
        aligned = bm_ret.reindex(ret.index).fillna(0.0)
        excess = ret - aligned
        bm_nav = (1.0 + aligned).cumprod()
        excess_nav = nav / bm_nav
        te = float(excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if n > 1 else 0.0
        ann_excess = (
            float(excess_nav.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
        )
        metrics.update(
            {
                "bm_ann_return": (
                    float(bm_nav.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
                ),
                "ann_excess_return": ann_excess,
                "tracking_error": te,
                "information_ratio": float(ann_excess / te) if te > 1e-12 else 0.0,
                "excess_max_drawdown": max_drawdown(excess_nav),
                "win_rate_excess": float((excess > 0).mean()),
            }
        )
    return metrics


def equal_weight_benchmark(adj_close: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """全市场等权日收益基准：每日对当日有效收益的股票取截面均值。"""
    prices = adj_close.reindex(index=dates)
    # 明确不填充缺失报价：不同 pandas 版本对 pct_change 的默认填充语义不同。
    rets = prices.pct_change(fill_method=None)
    bm = rets.mean(axis=1, skipna=True).fillna(0.0)
    bm.name = "benchmark"
    return bm
