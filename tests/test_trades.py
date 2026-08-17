"""逐笔归因：成交配对与统计。"""

from __future__ import annotations

import pandas as pd
import pytest

from hqpick.analysis.trades import (
    build_round_trips,
    monthly_stats,
    open_positions,
    return_histogram,
    stats_by,
    trade_stats,
)
from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester
from tests.helpers import make_wide, picks_frame

DATES = [f"2024-01-{d:02d}" for d in range(1, 15)]
N = len(DATES)


def _flow(rows: list[tuple]) -> pd.DataFrame:
    """构造成交流水：(date, signal_date, code, side, price, shares, notional)。"""
    return pd.DataFrame(
        rows,
        columns=["date", "signal_date", "code", "side", "price", "shares", "notional"],
    ).assign(
        date=lambda d: pd.to_datetime(d["date"]),
        signal_date=lambda d: pd.to_datetime(d["signal_date"]),
    )


def test_round_trip_pairs_buy_and_sell():
    flow = _flow([
        ("2024-01-02", "2024-01-01", "A", "buy", 10.0, 100.0, 1000.0),
        ("2024-01-04", "2024-01-01", "A", "sell", 12.0, 100.0, 1200.0),
    ])
    rt = build_round_trips(flow)

    assert len(rt) == 1
    row = rt.iloc[0]
    assert row["buy_date"] == pd.Timestamp("2024-01-02")
    assert row["sell_date"] == pd.Timestamp("2024-01-04")
    assert row["gross_ret"] == pytest.approx(0.20)
    assert row["exit_kind"] == "正常到期"
    assert row["hold_days"] == 2


def test_net_return_applies_asymmetric_costs():
    flow = _flow([
        ("2024-01-02", "2024-01-01", "A", "buy", 10.0, 100.0, 1000.0),
        ("2024-01-04", "2024-01-01", "A", "sell", 10.0, 100.0, 1000.0),
    ])
    rt = build_round_trips(flow, cost_buy=0.001, cost_sell=0.002)

    # 平价卖出，净收益 = (1-2‰)/(1+1‰) - 1 ≈ -3‰
    assert rt.iloc[0]["gross_ret"] == pytest.approx(0.0)
    assert rt.iloc[0]["net_ret"] == pytest.approx(0.998 / 1.001 - 1, abs=1e-9)


def test_same_code_across_signal_days_stays_separate():
    """同一只票在不同信号日各买一次 → 两笔独立交易，不能串配。"""
    flow = _flow([
        ("2024-01-02", "2024-01-01", "A", "buy", 10.0, 100.0, 1000.0),
        ("2024-01-03", "2024-01-02", "A", "buy", 11.0, 100.0, 1100.0),
        ("2024-01-04", "2024-01-01", "A", "sell", 12.0, 100.0, 1200.0),
        ("2024-01-05", "2024-01-02", "A", "sell", 9.0, 100.0, 900.0),
    ])
    rt = build_round_trips(flow)

    assert len(rt) == 2
    by_signal = rt.set_index(rt["signal_date"].dt.strftime("%m-%d"))
    assert by_signal.loc["01-01", "gross_ret"] == pytest.approx(0.20)
    assert by_signal.loc["01-02", "gross_ret"] == pytest.approx(9.0 / 11.0 - 1)


def test_exit_kinds_labelled():
    flow = _flow([
        ("2024-01-02", "2024-01-01", "A", "buy", 10.0, 100.0, 1000.0),
        ("2024-01-05", "2024-01-01", "A", "sell_deferred", 9.0, 100.0, 900.0),
        ("2024-01-02", "2024-01-01", "B", "buy", 10.0, 100.0, 1000.0),
        ("2024-01-06", "2024-01-01", "B", "sell_delist", 5.0, 100.0, 500.0),
    ])
    rt = build_round_trips(flow)
    assert set(rt["exit_kind"]) == {"顺延卖出", "退市核销"}


def test_hold_bars_uses_trading_calendar():
    calendar = pd.DatetimeIndex(pd.to_datetime(DATES))
    flow = _flow([
        ("2024-01-02", "2024-01-01", "A", "buy", 10.0, 100.0, 1000.0),
        ("2024-01-05", "2024-01-01", "A", "sell", 10.0, 100.0, 1000.0),
    ])
    rt = build_round_trips(flow, calendar=calendar)
    assert rt.iloc[0]["hold_bars"] == 3


def test_open_positions_excluded_from_round_trips():
    flow = _flow([
        ("2024-01-02", "2024-01-01", "A", "buy", 10.0, 100.0, 1000.0),
        ("2024-01-04", "2024-01-01", "A", "sell", 12.0, 100.0, 1200.0),
        ("2024-01-13", "2024-01-12", "B", "buy", 10.0, 100.0, 1000.0),
    ])
    rt = build_round_trips(flow)
    op = open_positions(flow)

    assert len(rt) == 1
    assert len(op) == 1
    assert op.iloc[0]["code"] == "B"


def test_trade_stats_win_rate_and_payoff():
    flow = _flow([
        ("2024-01-02", "2024-01-01", "A", "buy", 10.0, 100.0, 1000.0),
        ("2024-01-04", "2024-01-01", "A", "sell", 12.0, 100.0, 1200.0),   # +20%
        ("2024-01-02", "2024-01-01", "B", "buy", 10.0, 100.0, 1000.0),
        ("2024-01-04", "2024-01-01", "B", "sell", 11.0, 100.0, 1100.0),   # +10%
        ("2024-01-02", "2024-01-01", "C", "buy", 10.0, 100.0, 1000.0),
        ("2024-01-04", "2024-01-01", "C", "sell", 9.0, 100.0, 900.0),     # -10%
    ])
    st = trade_stats(build_round_trips(flow))

    assert st["n_trades"] == 3
    assert st["win_rate"] == pytest.approx(2 / 3)
    assert st["avg_win"] == pytest.approx(0.15)
    assert st["avg_loss"] == pytest.approx(-0.10)
    assert st["payoff_ratio"] == pytest.approx(1.5)
    assert st["expectancy"] == pytest.approx(2 / 3 * 0.15 + 1 / 3 * -0.10)
    assert st["best"] == pytest.approx(0.20)
    assert st["worst"] == pytest.approx(-0.10)


def test_stats_by_and_monthly():
    flow = _flow([
        ("2024-01-02", "2024-01-01", "A", "buy", 10.0, 100.0, 1000.0),
        ("2024-01-04", "2024-01-01", "A", "sell", 12.0, 100.0, 1200.0),
        ("2024-02-02", "2024-02-01", "B", "buy", 10.0, 100.0, 1000.0),
        ("2024-02-05", "2024-02-01", "B", "sell", 9.0, 100.0, 900.0),
    ])
    rt = build_round_trips(flow)

    by_kind = stats_by(rt, "exit_kind")
    assert by_kind.loc["正常到期", "n_trades"] == 2

    monthly = monthly_stats(rt)
    assert list(monthly.index) == ["2024-01", "2024-02"]
    assert monthly.loc["2024-01", "win_rate"] == 1.0
    assert monthly.loc["2024-02", "win_rate"] == 0.0


def test_empty_inputs_are_safe():
    empty = _flow([]).iloc[0:0]
    rt = build_round_trips(empty)
    assert rt.empty
    assert trade_stats(rt) == {"n_trades": 0}
    assert stats_by(rt, "exit_kind").empty
    assert monthly_stats(rt).empty
    counts, edges = return_histogram(rt)
    assert len(counts) == 0 and len(edges) == 0


def test_round_trips_from_real_engine_run():
    """引擎产出的流水必须能被完整配对，不留悬挂买入。"""
    wide = make_wide(
        DATES, ["A", "B"],
        close={"A": [10.0] * N, "B": [10.0] * N},
        limit_up={c: [999.0] * N for c in "AB"},
        limit_down={c: [0.01] * N for c in "AB"},
    )
    picks = picks_frame([(d, c) for d in DATES[:8] for c in ("A", "B")])
    cfg = ExecConfig(hold_days=2, n_buckets=5)
    result = PickBacktester(cfg).run(picks, wide)

    rt = build_round_trips(
        result.trades, cfg.cost_buy, cfg.cost_sell, calendar=result.nav.index
    )
    n_buys = (result.trades["side"] == "buy").sum()
    n_open = len(open_positions(result.trades))

    assert len(rt) + n_open == n_buys
    assert (rt["hold_bars"] == 2).all()      # H=2，正常到期
    st = trade_stats(rt)
    assert st["n_trades"] == len(rt)
