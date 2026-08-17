"""isolated 资金模式：各份独立滚动、用满不留余、空仓重新等分。"""

from __future__ import annotations

import pytest

from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester
from tests.helpers import make_wide, picks_frame

DATES = [f"2024-01-{d:02d}" for d in range(1, 15)]
N = len(DATES)


def _run(wide, picks, **kwargs):
    kwargs.setdefault("capital_mode", "isolated")
    return PickBacktester(ExecConfig(**kwargs)).run(picks, wide)


def test_default_mode_is_isolated():
    assert ExecConfig().capital_mode == "isolated"


def test_sleeve_invests_its_entire_cash():
    """建仓用满该 sleeve 全部现金，不留余额。"""
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * N})
    result = _run(
        wide, picks_frame([("2024-01-01", "A")]),
        hold_days=2, cost_buy=0.0, cost_sell=0.0, initial_value=900_000,
    )
    buys = result.trades[result.trades["side"] == "buy"]
    # 桶数 3 → 每份 30 万，首次建仓投满 30 万
    assert buys["notional"].sum() == pytest.approx(300_000)


def test_winner_and_loser_sleeves_roll_independently():
    """涨到 110 的那份下次投 110，跌到 90 的那份下次投 90。

    构造：sleeve0 买 A（+10%），sleeve1 买 B（-10%），各自卖出后再次建仓，
    投入金额应等于各自滚动后的资金，互不影响。
    """
    # 桶数 2（开盘买、开盘卖 → 回款当日复用），初始 100 万 → 每份 50 万
    close = {
        "A": [10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        "B": [10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        "C": [10] * N,
    }
    open_ = {
        "A": [10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        "B": [10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        "C": [10] * N,
    }
    wide = make_wide(
        DATES, ["A", "B", "C"], close=close, open_=open_,
        limit_up={c: [999.0] * N for c in "ABC"},
        limit_down={c: [0.01] * N for c in "ABC"},
    )
    # d1→A（sleeve0，d2 开盘买 10、d4 开盘卖 11 → +10%）
    # d2→B（sleeve1，d3 开盘买 10、d5 开盘卖 9  → -10%）
    # d4→C（sleeve0 释放后复用）  d5→C（sleeve1 释放后复用）
    picks = picks_frame([
        ("2024-01-01", "A"), ("2024-01-02", "B"),
        ("2024-01-04", "C"), ("2024-01-05", "C"),
    ])
    result = _run(
        wide, picks, hold_days=2, entry_price="open", exit_price="open",
        cost_buy=0.0, cost_sell=0.0, initial_value=1_000_000,
    )

    buys = result.trades[result.trades["side"] == "buy"]
    by_day = buys.set_index(buys["date"].dt.strftime("%m-%d"))["notional"]

    assert by_day["01-02"] == pytest.approx(500_000)   # sleeve0 首投 50 万
    assert by_day["01-03"] == pytest.approx(500_000)   # sleeve1 首投 50 万
    # sleeve0 赚了 10% → 55 万全投；sleeve1 亏了 10% → 45 万全投
    assert by_day["01-05"] == pytest.approx(550_000)
    assert by_day["01-06"] == pytest.approx(450_000)


def test_all_flat_triggers_equal_resplit():
    """某日全部 sleeve 空仓 → 现金加总重新等分。"""
    close = {
        "A": [10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        "C": [10] * N,
    }
    open_ = {
        "A": [10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        "C": [10] * N,
    }
    wide = make_wide(
        DATES, ["A", "C"], close=close, open_=open_,
        limit_up={c: [999.0] * N for c in "AC"},
        limit_down={c: [0.01] * N for c in "AC"},
    )
    # 只在 d1 建一仓（sleeve0，赚 10%），卖出后全体空仓 → 重新等分；
    # 隔几天再建仓，投入应是等分后的金额而非某一份的历史余额
    picks = picks_frame([("2024-01-01", "A"), ("2024-01-08", "C")])
    result = _run(
        wide, picks, hold_days=2, cost_buy=0.0, cost_sell=0.0,
        initial_value=900_000,
    )

    assert result.exec_stats["sleeve_reset_count"] >= 1
    buys = result.trades[result.trades["side"] == "buy"]
    later = buys[buys["date"] >= "2024-01-09"]
    # 总资产 = 90 万 + A 那一份 30 万赚的 10% = 93 万，等分成 3 份 → 31 万
    assert later["notional"].sum() == pytest.approx(310_000)


def test_no_free_sleeve_skips_without_touching_others():
    """所有 sleeve 都被占用时跳过当日信号，不挪用其他份的钱。"""
    # 全程跌停卖不出 → sleeve 永久占用
    wide = make_wide(
        DATES, ["A"], close={"A": [10.0] * N},
        limit_down={"A": [10.0] * N},
    )
    picks = picks_frame([(d, "A") for d in DATES])
    result = _run(wide, picks, hold_days=2)

    assert result.exec_stats["no_free_sleeve_days"] > 0
    # 只建了 3 个桶（= sleeve 数），之后全部跳过
    buys = result.trades[result.trades["side"] == "buy"]
    assert buys["date"].nunique() == 3


def test_unfilled_cash_stays_in_its_sleeve():
    """涨停买不进的钱留在该 sleeve，下次它建仓时一并投出。"""
    # d2 开盘涨停买不进；d5 恢复正常
    wide = make_wide(
        DATES, ["A"],
        close={"A": [10.0] * N},
        open_={"A": [10, 11] + [10.0] * (N - 2)},
        limit_up={"A": [11.0] + [11.0] * (N - 1)},
        limit_down={"A": [0.01] * N},
    )
    picks = picks_frame([("2024-01-01", "A"), ("2024-01-05", "A")])
    result = _run(
        wide, picks, hold_days=2, cost_buy=0.0, cost_sell=0.0,
        initial_value=900_000,
    )

    assert result.exec_stats["buy_fail_breakdown"]["limit_up"] == 1
    buys = result.trades[result.trades["side"] == "buy"]
    # 首次买不进，钱没花掉；后一次仍按该份的全额 30 万投出
    assert buys["notional"].sum() == pytest.approx(300_000)


def test_two_modes_agree_in_flat_market():
    """平价市场、单份资金时两种模式净值一致（退化一致性）。"""
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * N},
                     limit_up={"A": [999.0] * N}, limit_down={"A": [0.01] * N})
    picks = picks_frame([(d, "A") for d in DATES])

    kw = dict(hold_days=1, n_buckets=1, entry_price="close", exit_price="close")
    iso = _run(wide, picks, capital_mode="isolated", **kw)
    shr = _run(wide, picks, capital_mode="shared", **kw)

    assert iso.nav.iloc[-1] == pytest.approx(shr.nav.iloc[-1], rel=1e-9)


def test_isolated_invests_more_than_shared_in_rising_market():
    """上涨行情里 isolated 投得更满。

    shared 的预算基准是**昨收总资产**（滞后一天），价格上涨时当日现金已高于
    该基准，于是买不满、留下余额；isolated 直接用该 sleeve 的当前现金投满。
    这是两种模式的定义差异，不是执行失败。
    """
    close = {"A": [10 + i for i in range(N)]}      # 单调上涨
    wide = make_wide(DATES, ["A"], close=close, limit_up={"A": [999.0] * N},
                     limit_down={"A": [0.01] * N})
    picks = picks_frame([(d, "A") for d in DATES])

    kw = dict(hold_days=1, n_buckets=1, entry_price="close", exit_price="close",
              cost_buy=0.0, cost_sell=0.0)
    iso = _run(wide, picks, capital_mode="isolated", **kw)
    shr = _run(wide, picks, capital_mode="shared", **kw)

    assert iso.nav.iloc[-1] > shr.nav.iloc[-1]
    assert iso.exec_stats["final_cash_pct"] == pytest.approx(0.0, abs=1e-9)
    assert shr.exec_stats["final_cash_pct"] > 0.0
