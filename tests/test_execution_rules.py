"""执行规则：涨停买不进、跌停/停牌顺延、退市核销、现金耗尽。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester
from tests.helpers import make_wide, picks_frame

DATES = [f"2024-01-0{d}" for d in range(1, 9)]


def _run(wide, picks, **kwargs):
    return PickBacktester(ExecConfig(**kwargs)).run(picks, wide)


def test_limit_up_on_buy_day_skips_and_keeps_cash():
    """买入日成交价触涨停 → 放弃该票，资金留现金，NAV 不变。"""
    # 01-02 开盘价 11 == 涨停价 11 → 买不进
    wide = make_wide(
        DATES, ["A"],
        close={"A": [10.0] * 8},
        open_={"A": [10.0, 11.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]},
        limit_up={"A": [11.0] * 8},
    )
    result = _run(wide, picks_frame([("2024-01-01", "A")]), hold_days=2)

    assert result.trades.empty
    assert result.exec_stats["buy_fail_breakdown"]["limit_up"] == 1
    assert result.nav.iloc[-1] == pytest.approx(1.0)
    assert result.exec_stats["final_cash_pct"] == pytest.approx(1.0)


def test_suspended_on_buy_day_skips():
    status = {"A": ["交易", "停牌", "交易", "交易", "交易", "交易", "交易", "交易"]}
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * 8}, status=status)
    result = _run(wide, picks_frame([("2024-01-01", "A")]), hold_days=2)

    assert result.trades.empty
    assert result.exec_stats["buy_fail_breakdown"]["suspended"] == 1


def test_limit_down_on_sell_day_defers_to_next_day():
    """卖出日收盘跌停 → 顺延到下一可交易日，side 标记为 sell_deferred。"""
    # 买入 01-02(开盘10)，到期 01-04 收盘 9 == 跌停价 → 顺延到 01-05
    limit_down = {"A": [9.0] * 8}
    close = {"A": [10.0, 10.0, 10.0, 9.0, 9.5, 9.5, 9.5, 9.5]}
    wide = make_wide(DATES, ["A"], close=close, limit_down=limit_down)
    result = _run(wide, picks_frame([("2024-01-01", "A")]), hold_days=2)

    sells = result.trades[result.trades["side"].str.startswith("sell")]
    assert len(sells) == 1
    assert sells.iloc[0]["side"] == "sell_deferred"
    assert sells.iloc[0]["date"] == pd.Timestamp("2024-01-05")
    assert sells.iloc[0]["price"] == pytest.approx(9.5)
    assert result.exec_stats["sell_defer_breakdown"]["limit_down"] == 1


def test_suspended_on_sell_day_defers():
    status = {"A": ["交易"] * 3 + ["停牌"] + ["交易"] * 4}
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * 8}, status=status)
    result = _run(wide, picks_frame([("2024-01-01", "A")]), hold_days=2)

    sells = result.trades[result.trades["side"].str.startswith("sell")]
    assert sells.iloc[0]["side"] == "sell_deferred"
    assert sells.iloc[0]["date"] == pd.Timestamp("2024-01-05")
    assert result.exec_stats["sell_defer_breakdown"]["suspended"] == 1


def test_delisted_position_force_sold_at_last_valid_price():
    """整行消失（价格与状态同时缺失）→ 当日按最近有效价强制核销。"""
    close = {"A": [10.0, 10.0, 12.0, None, None, None, None, None]}
    status = {"A": ["交易", "交易", "交易", None, None, None, None, None]}
    wide = make_wide(DATES, ["A"], close=close, status=status)
    result = _run(
        wide, picks_frame([("2024-01-01", "A")]),
        hold_days=5, cost_buy=0.0, cost_sell=0.0,
    )

    sells = result.trades[result.trades["side"] == "sell_delist"]
    assert len(sells) == 1
    assert sells.iloc[0]["date"] == pd.Timestamp("2024-01-04")   # 消失当日
    assert sells.iloc[0]["price"] == pytest.approx(12.0)          # 前一日有效价
    assert result.exec_stats["delist_forced_count"] == 1
    # H=5、开盘买收盘卖 → 桶数 6，1/6 资金买入、标的涨 20% → 组合 +3.33%
    assert result.nav.iloc[-1] == pytest.approx(1.0 + 0.20 / 6, abs=1e-6)


def test_suspension_is_not_treated_as_delisting():
    """停牌日价格缺失但 trade_status 仍在 → 走顺延而非退市核销。"""
    close = {"A": [10.0, 10.0, 10.0, None, 10.0, 10.0, 10.0, 10.0]}
    status = {"A": ["交易"] * 3 + ["停牌"] + ["交易"] * 4}
    wide = make_wide(DATES, ["A"], close=close, status=status)
    result = _run(wide, picks_frame([("2024-01-01", "A")]), hold_days=2)

    assert result.exec_stats["delist_forced_count"] == 0
    assert result.exec_stats["sell_defer_count"] >= 1
    sells = result.trades[result.trades["side"].str.startswith("sell")]
    assert sells.iloc[0]["side"] == "sell_deferred"


def test_cash_exhausted_skips_bucket():
    """到期桶卖不出导致现金耗尽 → 后续信号整桶跳过并计数。"""
    # H=1：每日一桶用满全部资金；卖出日全程跌停无法回款
    limit_down = {"A": [10.0] * 8}          # 收盘价恒等于跌停价 → 永远卖不出
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * 8}, limit_down=limit_down)
    picks = picks_frame([(d, "A") for d in DATES[:4]])
    result = _run(wide, picks, hold_days=1)

    assert result.exec_stats["no_cash_skip_days"] >= 1
    assert result.exec_stats["sell_defer_breakdown"]["limit_down"] >= 1


def test_bucket_count_equals_hold_days_in_steady_state():
    """每日有信号时，稳态下同时在场桶数 = H，现金占比趋近 0。"""
    wide = make_wide(DATES, ["A", "B"], close={"A": [10.0] * 8, "B": [10.0] * 8})
    picks = picks_frame([(d, c) for d in DATES for c in ("A", "B")])
    result = _run(wide, picks, hold_days=3)

    # 末日现金占比应显著低于 1（资金已铺开到 3 个桶）
    assert result.exec_stats["final_cash_pct"] < 0.4
    assert result.exec_stats["avg_holding_count"] > 0


def test_asymmetric_costs_applied():
    """买 1‰ / 卖 2‰ 非对称费率：平价市场下净值恰好损失 3‰ × 仓位。"""
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * 8})
    result = _run(
        wide, picks_frame([("2024-01-01", "A")]),
        hold_days=2, cost_buy=0.001, cost_sell=0.002,
    )
    # H=2、开盘买收盘卖 → 桶数 3，1/3 资金参与往返成本 (1‰ + 2‰)
    assert result.nav.iloc[-1] == pytest.approx(1.0 - 0.003 / 3, abs=2e-5)


def test_variable_daily_pick_count():
    """每日选股数可变（含 0 只的日子），引擎不报错且桶内等权。"""
    wide = make_wide(DATES, ["A", "B"], close={"A": [10.0] * 8, "B": [10.0] * 8})
    picks = picks_frame(
        [("2024-01-01", "A"), ("2024-01-03", "A"), ("2024-01-03", "B")]
    )
    result = _run(wide, picks, hold_days=2, cost_buy=0.0, cost_sell=0.0)

    buys = result.trades[result.trades["side"] == "buy"]
    day2 = buys[buys["signal_date"] == pd.Timestamp("2024-01-03")]
    assert len(day2) == 2
    # 桶内等权：两票出资额相等
    assert day2["notional"].iloc[0] == pytest.approx(day2["notional"].iloc[1])
    assert np.isfinite(result.nav).all()
