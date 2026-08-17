"""买卖时点口径：T 日信号 → T+entry_offset 买 → 再持有 H 日卖。"""

from __future__ import annotations

import pandas as pd
import pytest

from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester
from tests.helpers import make_wide, picks_frame


def _run(wide, picks, **kwargs):
    return PickBacktester(ExecConfig(**kwargs)).run(picks, wide)


def test_buy_day_and_sell_day_offsets():
    """H=2、entry_offset=1：T=01-01 信号 → 01-02 买 → 01-04 卖。"""
    dates = [f"2024-01-0{d}" for d in range(1, 9)]
    wide = make_wide(dates, ["A"], {"A": [10.0] * 8})
    result = _run(wide, picks_frame([("2024-01-01", "A")]), hold_days=2)

    trades = result.trades
    assert list(trades["side"]) == ["buy", "sell"]
    assert trades.loc[0, "date"] == pd.Timestamp("2024-01-02")
    assert trades.loc[1, "date"] == pd.Timestamp("2024-01-04")
    assert trades["signal_date"].nunique() == 1


@pytest.mark.parametrize(
    ("hold_days", "entry_offset", "buy_day", "sell_day"),
    [
        (1, 1, "2024-01-02", "2024-01-03"),
        (2, 1, "2024-01-02", "2024-01-04"),
        (5, 1, "2024-01-02", "2024-01-07"),
        (2, 2, "2024-01-03", "2024-01-05"),
    ],
)
def test_timing_grid(hold_days, entry_offset, buy_day, sell_day):
    dates = [f"2024-01-0{d}" for d in range(1, 9)]
    wide = make_wide(dates, ["A"], {"A": [10.0] * 8})
    result = _run(
        wide, picks_frame([("2024-01-01", "A")]),
        hold_days=hold_days, entry_offset=entry_offset,
    )
    trades = result.trades
    assert trades.loc[0, "date"] == pd.Timestamp(buy_day)
    assert trades.loc[1, "date"] == pd.Timestamp(sell_day)


def test_entry_offset_zero_rejected():
    """entry_offset=0 会让信号日当天成交，构成前视，必须拒绝。"""
    with pytest.raises(ValueError, match="entry_offset"):
        ExecConfig(entry_offset=0)


def test_entry_close_exit_close_holds_exactly_h_days():
    """entry=close、H=2：01-02 收盘买、01-04 收盘卖，收益即两日涨幅。"""
    dates = [f"2024-01-0{d}" for d in range(1, 6)]
    close = {"A": [10.0, 10.0, 11.0, 12.0, 12.0]}
    wide = make_wide(dates, ["A"], close)
    result = _run(
        wide, picks_frame([("2024-01-01", "A")]),
        hold_days=2, entry_price="close", exit_price="close",
        cost_buy=0.0, cost_sell=0.0,
    )
    trades = result.trades
    assert trades.loc[0, "price"] == pytest.approx(10.0)   # 01-02 收盘
    assert trades.loc[1, "price"] == pytest.approx(12.0)   # 01-04 收盘
    # 全仓 1/H=1/2 资金投入，标的涨 20% → 组合涨 10%
    assert result.nav.iloc[-1] == pytest.approx(1.10, abs=1e-6)


def test_signal_too_late_is_unexecutable():
    """信号日在末尾、买入日超出区间 → 计入 unexecutable，不成交。"""
    dates = ["2024-01-01", "2024-01-02"]
    wide = make_wide(dates, ["A"], {"A": [10.0, 10.0]})
    result = _run(wide, picks_frame([("2024-01-02", "A")]), hold_days=2)
    assert result.exec_stats["unexecutable_signal_days"] == ["2024-01-02"]
    assert result.trades.empty
