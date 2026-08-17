"""逐笔归因：把成交流水配对成完整交易，算胜率/赔率/持有期分布。

一笔完整交易 = 同一 (signal_date, code) 的买入与卖出。桶内同一只票只买一次、
卖出时一次清空，因此这个键唯一确定一笔往返。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 卖出方式 → 可读标签
EXIT_KIND = {
    "sell": "正常到期",
    "sell_deferred": "顺延卖出",
    "sell_delist": "退市核销",
    "sell_writeoff": "卡仓核销",
}


ROUND_TRIP_COLUMNS = [
    "signal_date", "code", "buy_date", "sell_date", "buy_price", "sell_price",
    "shares", "notional", "exit_kind", "gross_ret", "net_ret", "pnl",
    "hold_days", "hold_bars",
]


def _empty_round_trips() -> pd.DataFrame:
    return pd.DataFrame(columns=ROUND_TRIP_COLUMNS)


def build_round_trips(
    trades: pd.DataFrame,
    cost_buy: float = 0.0,
    cost_sell: float = 0.0,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """成交流水 → 逐笔完整交易。

    Returns
    -------
    DataFrame，一行一笔：signal_date / code / buy_date / sell_date / buy_price /
    sell_price / shares / notional / exit_kind / gross_ret / net_ret / pnl /
    hold_days（自然日）/ hold_bars（交易日，需传 calendar）。
    未平仓的买入（回测区间截断）不出现在结果里，由 open_positions 单独统计。
    """
    if trades.empty:
        return _empty_round_trips()

    buys = trades[trades["side"] == "buy"].copy()
    sells = trades[trades["side"] != "buy"].copy()

    key = ["signal_date", "code"]
    merged = buys.merge(
        sells[[*key, "date", "price", "shares", "side"]],
        on=key, how="inner", suffixes=("_buy", "_sell"),
    )
    if merged.empty:
        return _empty_round_trips()

    out = pd.DataFrame({
        "signal_date": merged["signal_date"],
        "code": merged["code"],
        "buy_date": merged["date_buy"],
        "sell_date": merged["date_sell"],
        "buy_price": merged["price_buy"],
        "sell_price": merged["price_sell"],
        "shares": merged["shares_sell"],
        "notional": merged["notional"],
        "exit_kind": merged["side_sell"].map(EXIT_KIND).fillna(merged["side_sell"]),
    })

    out["gross_ret"] = out["sell_price"] / out["buy_price"] - 1.0
    # 净收益：买入含费、卖出扣费
    out["net_ret"] = (
        out["sell_price"] * (1.0 - cost_sell)
        / (out["buy_price"] * (1.0 + cost_buy))
    ) - 1.0
    out["pnl"] = out["notional"] * out["net_ret"]
    out["hold_days"] = (out["sell_date"] - out["buy_date"]).dt.days

    if calendar is not None and len(calendar):
        pos = pd.Series(range(len(calendar)), index=pd.DatetimeIndex(calendar))
        out["hold_bars"] = (
            out["sell_date"].map(pos).astype("Int64")
            - out["buy_date"].map(pos).astype("Int64")
        )
    else:
        out["hold_bars"] = pd.NA

    return out.sort_values(["buy_date", "code"]).reset_index(drop=True)


def open_positions(trades: pd.DataFrame) -> pd.DataFrame:
    """区间末仍未平仓的买入（净值含其浮动市值，但没有实现收益）。"""
    if trades.empty:
        return pd.DataFrame(columns=["signal_date", "code", "buy_date", "notional"])
    buys = trades[trades["side"] == "buy"]
    sells = trades[trades["side"] != "buy"]
    sold = set(zip(sells["signal_date"], sells["code"], strict=False))
    mask = [
        (sd, c) not in sold
        for sd, c in zip(buys["signal_date"], buys["code"], strict=False)
    ]
    return buys.loc[mask, ["signal_date", "code", "date", "notional"]].rename(
        columns={"date": "buy_date"}
    ).reset_index(drop=True)


def trade_stats(round_trips: pd.DataFrame) -> dict:
    """逐笔层面的胜率、赔率、收益分布。

    注意这些是**等权到每笔**的统计，与净值层指标不同——净值受仓位与
    资金利用率影响，逐笔统计不受。两者要一起看。
    """
    if round_trips.empty:
        return {"n_trades": 0}

    ret = round_trips["net_ret"]
    wins = ret[ret > 0]
    losses = ret[ret <= 0]
    win_rate = float(len(wins) / len(ret))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff = float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0

    return {
        "n_trades": int(len(ret)),
        "win_rate": win_rate,
        "avg_ret": float(ret.mean()),
        "median_ret": float(ret.median()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        # 期望值：胜率 × 平均盈利 + 败率 × 平均亏损，逐笔口径的 edge
        "expectancy": float(win_rate * avg_win + (1 - win_rate) * avg_loss),
        "ret_std": float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
        "best": float(ret.max()),
        "worst": float(ret.min()),
        "p05": float(ret.quantile(0.05)),
        "p25": float(ret.quantile(0.25)),
        "p75": float(ret.quantile(0.75)),
        "p95": float(ret.quantile(0.95)),
        "total_pnl": float(round_trips["pnl"].sum()),
        "avg_hold_days": float(round_trips["hold_days"].mean()),
        "exit_kind_counts": round_trips["exit_kind"].value_counts().to_dict(),
    }


def stats_by(round_trips: pd.DataFrame, column: str) -> pd.DataFrame:
    """按某列分组的逐笔统计（笔数、胜率、平均收益、合计盈亏）。"""
    if round_trips.empty or column not in round_trips.columns:
        return pd.DataFrame()
    grouped = round_trips.groupby(column, dropna=False)
    frame = pd.DataFrame({
        "n_trades": grouped.size(),
        "win_rate": grouped["net_ret"].apply(lambda s: float((s > 0).mean())),
        "avg_ret": grouped["net_ret"].mean(),
        "median_ret": grouped["net_ret"].median(),
        "total_pnl": grouped["pnl"].sum(),
    })
    return frame.sort_values("n_trades", ascending=False)


def monthly_stats(round_trips: pd.DataFrame) -> pd.DataFrame:
    """按买入月份分组，看逐笔 edge 是否稳定（而非只在某几个月赚钱）。"""
    if round_trips.empty:
        return pd.DataFrame()
    frame = round_trips.copy()
    frame["month"] = frame["buy_date"].dt.to_period("M").astype(str)
    return stats_by(frame, "month").sort_index()


def return_histogram(
    round_trips: pd.DataFrame, bins: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """单笔净收益直方图，返回 (计数, 分箱边界)。"""
    if round_trips.empty:
        return np.array([]), np.array([])
    return np.histogram(round_trips["net_ret"].to_numpy(), bins=bins)
