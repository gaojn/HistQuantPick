"""slots 资金模式：可用现金按剩余空槽均摊。

    budget = 可用现金 / (N - 已占用槽数)
"""

from __future__ import annotations

import pandas as pd
import pytest

from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester
from tests.helpers import make_wide, picks_frame

DATES = [f"2024-01-{d:02d}" for d in range(1, 15)]
N = len(DATES)


def _run(wide, picks, **kwargs):
    kwargs.setdefault("capital_mode", "slots")
    return PickBacktester(ExecConfig(**kwargs)).run(picks, wide)


def test_default_mode_is_slots():
    assert ExecConfig().capital_mode == "slots"


def test_first_entry_is_cash_over_n():
    """首次建仓 = 现金 / N，全空时天然等分。"""
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * N})
    result = _run(
        wide, picks_frame([("2024-01-01", "A")]),
        hold_days=2, cost_buy=0.0, cost_sell=0.0, initial_value=900_000,
    )
    buys = result.trades[result.trades["side"] == "buy"]
    # 槽数 3、全空 → 90 万 / 3 = 30 万
    assert buys["notional"].sum() == pytest.approx(300_000)


def test_budget_equals_cash_over_free_slots():
    """逐日验证 budget = 可用现金 / (N − 已占用槽数)，且最后一个空槽全投。

    N=3、平价市场、初始 90 万，连续三日建仓：
        d2 空槽 3 → 90/3 = 30 万
        d3 空槽 2 → 60/2 = 30 万
        d4 空槽 1 → 30/1 = 30 万（最后一个空槽，现金全投、不留余）
    """
    wide = make_wide(
        DATES, ["A"], close={"A": [10.0] * N},
        limit_up={"A": [999.0] * N}, limit_down={"A": [0.01] * N},
    )
    picks = picks_frame([(d, "A") for d in DATES[:3]])
    result = _run(
        wide, picks, hold_days=2, cost_buy=0.0, cost_sell=0.0,
        initial_value=900_000,
    )

    buys = result.trades[result.trades["side"] == "buy"]
    by_day = buys.set_index(buys["date"].dt.strftime("%m-%d"))["notional"]
    assert by_day["01-02"] == pytest.approx(300_000)
    assert by_day["01-03"] == pytest.approx(300_000)
    assert by_day["01-04"] == pytest.approx(300_000)
    # 三个槽全满 → 现金归零，没有余留
    assert result.exec_stats["no_free_slot_days"] == 0


def test_profit_flows_into_next_entry():
    """盈利回笼后按空槽均摊：只剩一个空槽时，赚到的钱全部投出去。

    N=2、开盘买开盘卖：d2 投 50 万买 A，A 涨 10% 后 d4 开盘卖回 55 万；
    d4 建仓时另一个槽仍被 B 占着 → 空槽 1 → 把当时现金全投。
    """
    close = {
        "A": [10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        "B": [10] * N,
        "C": [10] * N,
    }
    open_ = {
        "A": [10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        "B": [10] * N,
        "C": [10] * N,
    }
    wide = make_wide(
        DATES, ["A", "B", "C"], close=close, open_=open_,
        limit_up={c: [999.0] * N for c in "ABC"},
        limit_down={c: [0.01] * N for c in "ABC"},
    )
    # d1→A（d2 开盘买 10、d4 开盘卖 11）；d3→B（d4 开盘买，占住另一个槽）
    # d3→C 与 A 同日回笼后建仓
    picks = picks_frame([
        ("2024-01-01", "A"), ("2024-01-02", "B"), ("2024-01-03", "C"),
    ])
    result = _run(
        wide, picks, hold_days=2, entry_price="open", exit_price="open",
        cost_buy=0.0, cost_sell=0.0, initial_value=1_000_000,
    )

    buys = result.trades[result.trades["side"] == "buy"]
    by_day = buys.set_index(buys["date"].dt.strftime("%m-%d"))["notional"]
    assert by_day["01-02"] == pytest.approx(500_000)     # 空槽 2 → 100/2
    assert by_day["01-03"] == pytest.approx(500_000)     # 空槽 1 → 现金全投
    # d4：A 开盘卖回 55 万，B 仍占一槽 → 空槽 1 → 55 万全投，盈利滚进下一笔
    assert by_day["01-04"] == pytest.approx(550_000)


def test_all_flat_resplits_equally():
    """全部空仓后再建仓 = 总现金 / N，天然等分，无需特殊重置逻辑。"""
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
    # 只在 d1 建一仓（赚 10%），卖出后全体空仓；隔几天再建仓，
    # 投入应是「当前总现金 / N」
    picks = picks_frame([("2024-01-01", "A"), ("2024-01-08", "C")])
    result = _run(
        wide, picks, hold_days=2, cost_buy=0.0, cost_sell=0.0,
        initial_value=900_000,
    )

    buys = result.trades[result.trades["side"] == "buy"]
    later = buys[buys["date"] >= "2024-01-09"]
    # 总现金 = 90 万 + 30 万那笔赚的 10% = 93 万，空槽 3 → 93/3 = 31 万
    assert later["notional"].sum() == pytest.approx(310_000)


def test_no_free_slot_skips_entry():
    """所有槽位都被占用时跳过当日信号，不超配。"""
    # 全程跌停卖不出 → 槽位永久占用
    wide = make_wide(
        DATES, ["A"], close={"A": [10.0] * N},
        limit_down={"A": [10.0] * N},
    )
    picks = picks_frame([(d, "A") for d in DATES])
    result = _run(wide, picks, hold_days=2)

    assert result.exec_stats["no_free_slot_days"] > 0
    # 只建了 3 个桶（= 槽数 N），之后全部跳过
    buys = result.trades[result.trades["side"] == "buy"]
    assert buys["date"].nunique() == 3


def test_unfilled_cash_returns_to_pool():
    """涨停买不进的钱留在现金池，下次建仓时一并均摊出去。"""
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
    # 首次买不进，钱没花掉；后一次全空 → 90/3 = 30 万
    assert buys["notional"].sum() == pytest.approx(300_000)


def test_two_modes_agree_in_flat_market():
    """平价市场、单份资金时两种模式净值一致（退化一致性）。"""
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * N},
                     limit_up={"A": [999.0] * N}, limit_down={"A": [0.01] * N})
    picks = picks_frame([(d, "A") for d in DATES])

    kw = dict(hold_days=1, n_buckets=1, entry_price="close", exit_price="close")
    slots_res = _run(wide, picks, capital_mode="slots", **kw)
    shr = _run(wide, picks, capital_mode="shared", **kw)

    assert slots_res.nav.iloc[-1] == pytest.approx(shr.nav.iloc[-1], rel=1e-9)


def test_slots_invests_more_than_shared_in_rising_market():
    """上涨行情里 slots 投得更满。

    shared 的预算基准是**昨收总资产**（滞后一天），价格上涨时当日现金已高于
    该基准，于是买不满、留下余额；slots 直接按当前现金均摊、投满。
    这是两种模式的定义差异，不是执行失败。
    """
    close = {"A": [10 + i for i in range(N)]}      # 单调上涨
    wide = make_wide(DATES, ["A"], close=close, limit_up={"A": [999.0] * N},
                     limit_down={"A": [0.01] * N})
    picks = picks_frame([(d, "A") for d in DATES])

    kw = dict(hold_days=1, n_buckets=1, entry_price="close", exit_price="close",
              cost_buy=0.0, cost_sell=0.0)
    slots_res = _run(wide, picks, capital_mode="slots", **kw)
    shr = _run(wide, picks, capital_mode="shared", **kw)

    assert slots_res.nav.iloc[-1] > shr.nav.iloc[-1]
    assert slots_res.exec_stats["final_cash_pct"] == pytest.approx(0.0, abs=1e-9)
    assert shr.exec_stats["final_cash_pct"] > 0.0


def test_stuck_position_locks_slot_but_not_cash():
    """卡仓只占住槽位，剩余现金照常均摊到其余空槽——这是 slots 相对固定份额的关键改进。"""
    # A 永远跌停卖不出；B 正常
    close = {"A": [10.0] * N, "B": [10.0] * N}
    wide = make_wide(
        DATES, ["A", "B"], close=close,
        limit_up={c: [999.0] * N for c in "AB"},
        limit_down={"A": [10.0] * N, "B": [0.01] * N},
    )
    picks = picks_frame([("2024-01-01", "A")] + [(d, "B") for d in DATES[1:]])
    result = _run(wide, picks, hold_days=2, cost_buy=0.0, cost_sell=0.0,
                  initial_value=900_000)

    # A 桶永久占一个槽，其市值被冻结
    assert result.exec_stats["sell_defer_breakdown"]["limit_down"] > 0
    assert result.exec_stats["avg_frozen_pct"] > 0
    # 但现金没有被锁死：剩下 2 个槽仍在正常轮转（周期 H+1=3 天 → 约 2/3 的日子能建仓）
    buys = result.trades[result.trades["side"] == "buy"]
    assert buys["date"].nunique() >= 8


def test_writeoff_disabled_by_default():
    """默认无限顺延，不做任何强制核销。"""
    wide = make_wide(
        DATES, ["A"], close={"A": [10.0] * N},
        limit_up={"A": [999.0] * N}, limit_down={"A": [10.0] * N},
    )
    result = _run(wide, picks_frame([("2024-01-01", "A")]), hold_days=2)

    assert result.exec_stats["writeoff_stuck_days"] == 0
    assert result.exec_stats["writeoff_forced_count"] == 0
    assert result.exec_stats["max_frozen_pct"] > 0          # 一直冻结着


def test_writeoff_after_threshold_releases_slot():
    """开启 --writeoff-stuck-days 后，卡仓超时按最近有效价核销并释放槽位。"""
    wide = make_wide(
        DATES, ["A"], close={"A": [10.0] * N},
        limit_up={"A": [999.0] * N}, limit_down={"A": [10.0] * N},
    )
    result = _run(
        wide, picks_frame([("2024-01-01", "A")]), hold_days=2,
        writeoff_stuck_days=3, cost_buy=0.0, cost_sell=0.0,
    )

    assert result.exec_stats["writeoff_forced_count"] == 1
    writeoffs = result.trades[result.trades["side"] == "sell_writeoff"]
    assert len(writeoffs) == 1
    assert writeoffs.iloc[0]["price"] == pytest.approx(10.0)
    # 到期日 01-04，卡满 3 天后于 01-07 核销
    assert writeoffs.iloc[0]["date"] == pd.Timestamp("2024-01-07")
