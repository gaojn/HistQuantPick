"""资金守恒：独立于引擎内部记账，从成交流水反推期末资产核对。

方法来自一次外部独立代码审计（见 docs/实验记录.md 之外的工程记录——本文件是
其交付物）：引擎自己算出的 nav 是"cash + Σ持仓市值"，这两个量都是引擎内部
状态，直接拿来比对等于自己验证自己。真正有意义的核对是完全绕开引擎的内部
状态，只用它公开输出的 ``trades`` 流水 + 外部行情 ``wide.adj["close"]``，
独立重算一遍期末现金与持仓市值，看是否与引擎报告的 nav 一致。

重算公式（对照 hqpick/engine/replay.py 的实现推导）::

    买入现金流出 = price × shares × (1 + cost_buy)   # alloc = notional×(1+cost_buy)
    卖出现金流入 = price × shares × (1 - cost_sell)   # 含 sell / sell_deferred /
                                                        # sell_delist / sell_writeoff，
                                                        # 四种卖出路径费率口径一致
    期末现金     = 初始资金 − Σ买入流出 + Σ卖出流入
    期末持仓市值 = Σ_code (Σ买入股数 − Σ卖出股数) × 最终 ffill 收盘价
    期末权益     = 期末现金 + 期末持仓市值

这个值应精确等于 ``result.nav.iloc[-1] * initial_value``（浮点噪声级误差内）。
一旦引擎某个分支多算/少算了一笔现金流（比如某条卖出路径漏了扣费，或者
退市核销重复入账），这里会直接暴露，而不会像"nav == cash+holdings"那种
同义反复的断言一样被内部实现的自洽性掩盖。
"""

from __future__ import annotations

import pandas as pd
import pytest

from hqpick.data.panel import WideFrames
from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester, RunResult
from tests.helpers import make_wide, picks_frame

DATES = [f"2024-01-{d:02d}" for d in range(1, 21)]
N = len(DATES)


def _independent_equity(
    wide: WideFrames, result: RunResult, initial_value: float
) -> float:
    """只用 trades 流水与外部行情独立重算期末权益，不碰引擎内部状态。"""
    trades = result.trades
    cfg = result.config

    buys = trades[trades["side"] == "buy"]
    sells = trades[trades["side"] != "buy"]

    cash_out = (buys["price"] * buys["shares"] * (1 + cfg.cost_buy)).sum()
    cash_in = (sells["price"] * sells["shares"] * (1 - cfg.cost_sell)).sum()
    cash = initial_value - cash_out + cash_in

    net_shares = (
        buys.groupby("code")["shares"].sum()
        .subtract(sells.groupby("code")["shares"].sum(), fill_value=0.0)
    )
    net_shares = net_shares[net_shares.abs() > 1e-9]

    final_prices = wide.adj["close"].ffill().iloc[-1]
    holdings_value = sum(
        shares * final_prices.get(code, 0.0)
        for code, shares in net_shares.items()
        if pd.notna(final_prices.get(code))
    )
    return cash + holdings_value


def _assert_conserved(wide: WideFrames, result: RunResult, initial_value: float) -> None:
    independent = _independent_equity(wide, result, initial_value)
    reported = float(result.nav.iloc[-1]) * initial_value
    assert independent == pytest.approx(reported, rel=1e-9, abs=1e-6), (
        f"独立重算权益 {independent:.4f} 与引擎报告 nav×初始资金 {reported:.4f} 不一致，"
        f"说明某条成交路径的现金流记账有误"
    )


def _run(wide, picks, **kwargs) -> tuple[RunResult, float]:
    initial_value = kwargs.pop("initial_value", 1_000_000.0)
    cfg = ExecConfig(initial_value=initial_value, **kwargs)
    result = PickBacktester(cfg).run(picks, wide)
    return result, initial_value


def test_conserved_normal_multi_stock_flow():
    """基本场景：多只股票、多次建仓，全程无异常。"""
    wide = make_wide(
        DATES, ["A", "B", "C"],
        close={
            "A": [10 + 0.3 * i for i in range(N)],
            "B": [20 - 0.2 * i for i in range(N)],
            "C": [15.0] * N,
        },
    )
    picks = picks_frame([(d, c) for d in DATES[:12] for c in ("A", "B", "C")])
    result, iv = _run(wide, picks, hold_days=3, n_buckets=4)
    _assert_conserved(wide, result, iv)


def test_conserved_with_limit_up_blocking_buys():
    """部分买入因涨停失败，未成交的信号不产生现金流，仍应守恒。"""
    wide = make_wide(
        DATES, ["A", "B"],
        close={"A": [10.0] * N, "B": [10.0] * N},
        open_={"A": [10.0, 11.0] + [10.0] * (N - 2), "B": [10.0] * N},
        limit_up={"A": [11.0] * N, "B": [999.0] * N},
    )
    picks = picks_frame([(d, c) for d in DATES[:10] for c in ("A", "B")])
    result, iv = _run(wide, picks, hold_days=2, n_buckets=3)
    assert result.exec_stats["buy_fail_breakdown"]["limit_up"] > 0
    _assert_conserved(wide, result, iv)


def test_conserved_with_limit_down_deferred_sells():
    """卖出因跌停顺延多日，顺延期间不产生现金流，最终成交后应守恒。"""
    close = {"A": [10.0, 10.0, 10.0, 9.0, 9.0, 9.5, 10.0] + [10.0] * (N - 7)}
    wide = make_wide(
        DATES, ["A"], close=close,
        limit_down={"A": [9.0] * N},
    )
    picks = picks_frame([("2024-01-01", "A")])
    result, iv = _run(wide, picks, hold_days=2, n_buckets=2)
    assert result.exec_stats["sell_defer_breakdown"]["limit_down"] > 0
    _assert_conserved(wide, result, iv)


def test_conserved_with_suspension():
    # 买入日=index1，due_idx=1+hold_days(3)=index4；停牌窗口 index3~5 覆盖到期日
    status = {"A": ["交易"] * 3 + ["停牌"] * 3 + ["交易"] * (N - 6)}
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * N}, status=status)
    picks = picks_frame([("2024-01-01", "A")])
    result, iv = _run(wide, picks, hold_days=3, n_buckets=2)
    assert result.exec_stats["sell_defer_breakdown"]["suspended"] > 0
    _assert_conserved(wide, result, iv)


def test_conserved_with_delisting():
    """退市强制核销走的是独立于正常卖出的现金流路径，单独验证守恒。"""
    close = {"A": [10.0, 11.0, 12.0] + [None] * (N - 3)}
    wide = make_wide(DATES, ["A"], close=close)
    picks = picks_frame([("2024-01-01", "A")])
    result, iv = _run(wide, picks, hold_days=5, n_buckets=3)
    assert result.exec_stats["delist_forced_count"] > 0
    _assert_conserved(wide, result, iv)


def test_conserved_with_writeoff_enabled():
    """卡仓超时强制核销（假设口径）也要守恒——它走的是第四条独立卖出路径。"""
    wide = make_wide(
        DATES, ["A"], close={"A": [10.0] * N},
        limit_down={"A": [10.0] * N},
    )
    picks = picks_frame([("2024-01-01", "A")])
    result, iv = _run(
        wide, picks, hold_days=2, n_buckets=2, writeoff_stuck_days=3,
    )
    assert result.exec_stats["writeoff_forced_count"] > 0
    _assert_conserved(wide, result, iv)


def test_conserved_shared_capital_mode():
    wide = make_wide(
        DATES, ["A", "B"],
        close={"A": [10 + 0.1 * i for i in range(N)], "B": [10.0] * N},
    )
    picks = picks_frame([(d, c) for d in DATES[:10] for c in ("A", "B")])
    result, iv = _run(wide, picks, hold_days=2, n_buckets=3, capital_mode="shared")
    _assert_conserved(wide, result, iv)


@pytest.mark.parametrize("cost_buy,cost_sell", [(0.001, 0.002), (0.0, 0.0), (0.003, 0.0)])
def test_conserved_across_cost_combinations(cost_buy, cost_sell):
    """非对称、零费率、单边零费率——费率参数不应影响守恒性。"""
    wide = make_wide(
        DATES, ["A"], close={"A": [10 + 0.2 * i for i in range(N)]},
    )
    picks = picks_frame([(d, "A") for d in DATES[:8]])
    result, iv = _run(
        wide, picks, hold_days=2, n_buckets=3, cost_buy=cost_buy, cost_sell=cost_sell,
    )
    _assert_conserved(wide, result, iv)


@pytest.mark.parametrize(
    "entry_price,exit_price",
    [("open", "close"), ("open", "open"), ("close", "close"), ("close", "open")],
)
def test_conserved_across_entry_exit_combinations(entry_price, exit_price):
    """四种买卖时点组合（含 sell_before_buy 两种取值）都应守恒。"""
    wide = make_wide(
        DATES, ["A"], close={"A": [10 + 0.15 * i for i in range(N)]},
    )
    picks = picks_frame([(d, "A") for d in DATES[:10]])
    result, iv = _run(
        wide, picks, hold_days=2, n_buckets=2,
        entry_price=entry_price, exit_price=exit_price,
    )
    _assert_conserved(wide, result, iv)


def test_conserved_with_single_slot():
    """N=1 是最容易暴露除零/超配 bug 的边界，单独验证。"""
    wide = make_wide(
        DATES, ["A"], close={"A": [10 + 0.1 * i for i in range(N)]},
    )
    picks = picks_frame([(d, "A") for d in DATES[:10]])
    result, iv = _run(wide, picks, hold_days=1, n_buckets=1)
    _assert_conserved(wide, result, iv)
