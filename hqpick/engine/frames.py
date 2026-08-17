"""按执行配置把行情宽表对齐成引擎直接消费的价格视图。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hqpick.data.panel import WideFrames
from hqpick.engine.config import ExecConfig


@dataclass(frozen=True)
class AlignedFrames:
    """对齐到回测交易日的行情视图。

    成交价按 ``ExecConfig`` 选定：买入用 entry_price，卖出用 exit_price。
    复权价用于成交与损益，原始价仅用于涨跌停判断（涨跌停价是原始价口径）。
    估值统一用 ffill 后的复权收盘价——停牌日沿用最近有效价，成交价则不 ffill。
    """

    dates: pd.DatetimeIndex
    entry_adj: pd.DataFrame
    entry_raw: pd.DataFrame
    exit_adj: pd.DataFrame
    exit_raw: pd.DataFrame
    adj_close_raw: pd.DataFrame       # 未 ffill，判断退市/无价用
    adj_close_marked: pd.DataFrame    # ffill，仅估值用
    limit_up: pd.DataFrame
    limit_down: pd.DataFrame
    trade_status: pd.DataFrame


def align_frames(
    wide: WideFrames,
    config: ExecConfig,
    start: pd.Timestamp | None = None,
) -> AlignedFrames:
    """裁剪到 ``start`` 之后的交易日，并选出 entry/exit 成交价视图。"""
    dates = wide.dates
    if start is not None:
        dates = dates[dates >= start]
    if len(dates) == 0:
        raise ValueError("回测区间内没有交易日")

    def align(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(index=dates)

    adj_close_raw = align(wide.adj["close"])
    return AlignedFrames(
        dates=dates,
        entry_adj=align(wide.adj[config.entry_price]),
        entry_raw=align(wide.raw[config.entry_price]),
        exit_adj=align(wide.adj[config.exit_price]),
        exit_raw=align(wide.raw[config.exit_price]),
        adj_close_raw=adj_close_raw,
        adj_close_marked=adj_close_raw.ffill(),
        limit_up=align(wide.limit_up),
        limit_down=align(wide.limit_down),
        trade_status=align(wide.trade_status),
    )
