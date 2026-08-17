"""合成行情构造器：用可控数据验证执行规则。"""

from __future__ import annotations

import pandas as pd

from hqpick.data.panel import WideFrames


def make_wide(
    dates: list[str],
    codes: list[str],
    close: dict[str, list[float | None]],
    open_: dict[str, list[float | None]] | None = None,
    vwap: dict[str, list[float | None]] | None = None,
    limit_up: dict[str, list[float | None]] | None = None,
    limit_down: dict[str, list[float | None]] | None = None,
    status: dict[str, list[str | None]] | None = None,
) -> WideFrames:
    """构造 WideFrames。未指定的开盘/VWAP 取收盘价，涨跌停默认 ±10%（不触发）。

    价格视为已复权（复权价 = 原始价），便于手算校验。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))

    def frame(data: dict, default_from: dict | None = None) -> pd.DataFrame:
        source = data if data is not None else default_from
        return pd.DataFrame({c: source[c] for c in codes}, index=idx, dtype=float)

    close_df = frame(close)
    open_df = frame(open_, close) if open_ is not None else close_df.copy()
    vwap_df = frame(vwap, close) if vwap is not None else close_df.copy()

    if limit_up is None:
        up_df = close_df * 1.10
    else:
        up_df = frame(limit_up)
    if limit_down is None:
        down_df = close_df * 0.90
    else:
        down_df = frame(limit_down)

    if status is None:
        status_df = pd.DataFrame("交易", index=idx, columns=codes)
        status_df = status_df.where(close_df.notna())
    else:
        status_df = pd.DataFrame({c: status[c] for c in codes}, index=idx)

    return WideFrames(
        adj={"open": open_df, "close": close_df, "vwap": vwap_df},
        raw={"open": open_df.copy(), "close": close_df.copy(), "vwap": vwap_df.copy()},
        limit_up=up_df,
        limit_down=down_df,
        trade_status=status_df,
    )


def picks_frame(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """[(date, code), ...] → 选股长表。"""
    return pd.DataFrame(
        {
            "date": pd.to_datetime([d for d, _ in pairs]),
            "code": [c for _, c in pairs],
        }
    )
