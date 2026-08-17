"""资金分桶与同日执行次序。"""

from __future__ import annotations

import pytest

from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester
from tests.helpers import make_wide, picks_frame

DATES = [f"2024-01-{d:02d}" for d in range(1, 21)]


def _run(wide, picks, **kwargs):
    return PickBacktester(ExecConfig(**kwargs)).run(picks, wide)


@pytest.mark.parametrize(
    ("entry_price", "exit_price", "hold_days", "expected_buckets", "sell_first"),
    [
        ("open", "close", 2, 3, False),     # 卖出晚于买入 → 多一桶资金在途
        ("open", "close", 5, 6, False),
        ("open", "open", 2, 2, True),       # 同为开盘竞价 → 回款当日复用
        ("close", "close", 3, 3, True),
        ("close", "open", 2, 2, True),      # 卖出严格早于买入
        ("open", "vwap", 2, 3, False),
        ("vwap", "vwap", 2, 3, False),      # vwap 无单一时点，保守按先买后卖
    ],
)
def test_bucket_count_derivation(
    entry_price, exit_price, hold_days, expected_buckets, sell_first
):
    cfg = ExecConfig(
        hold_days=hold_days, entry_price=entry_price, exit_price=exit_price
    )
    assert cfg.buckets == expected_buckets
    assert cfg.sell_before_buy is sell_first


def test_explicit_n_buckets_overrides_derivation():
    cfg = ExecConfig(hold_days=2, n_buckets=2)
    assert cfg.buckets == 2
    assert cfg.label.endswith("_b2slots")      # slots = 默认资金模式
    assert ExecConfig(
        hold_days=2, n_buckets=2, capital_mode="shared"
    ).label.endswith("_b2shared")


def test_derived_buckets_keep_entry_continuous():
    """自动桶数（H+1）消除建仓断续，强制 1/H 则周期性现金耗尽。

    注意：两者的**平均**现金占比都约 1/(H+1)——开盘买、收盘卖时总有一桶回款
    在途，这是口径的物理下限，改桶数降不下来。H+1 换来的是每日预算稳定、
    不出现「跳过一天→次日现金翻倍」的震荡。要真正吃掉这部分闲置，
    只能把卖出时点提前到开盘（见 test_sell_before_buy_recycles_cash_fully）。
    """
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * 20})
    picks = picks_frame([(d, "A") for d in DATES])

    auto = _run(wide, picks, hold_days=2, capital_mode="shared")
    forced = _run(wide, picks, hold_days=2, n_buckets=2, capital_mode="shared")

    assert auto.exec_stats["n_buckets"] == 3
    assert forced.exec_stats["n_buckets"] == 2
    # 自动桶数：每日都能足额建仓
    assert auto.exec_stats["no_cash_skip_days"] == 0
    assert auto.exec_stats["underfunded_buy_days"] == 0
    # 强制 1/H：回款跨 H+1 个建仓时点 → 周期性建仓不足（桶照建但金额远低于目标）
    assert forced.exec_stats["underfunded_buy_days"] > 0
    assert forced.exec_stats["avg_funding_gap"] > 0.2
    # 平均现金占比同为结构下限 1/(H+1)，不因桶数而改善
    assert auto.exec_stats["avg_cash_pct"] == pytest.approx(1 / 3, abs=0.08)
    assert auto.exec_stats["structural_cash_pct"] == pytest.approx(1 / 3)


def test_sell_before_buy_recycles_cash_fully():
    """开盘买 + 开盘卖：回款当日复用，稳态可满仓，结构性闲置为 0。"""
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * 20})
    picks = picks_frame([(d, "A") for d in DATES])
    result = _run(
        wide, picks, hold_days=2, entry_price="open", exit_price="open",
        capital_mode="shared",
    )

    assert result.exec_stats["n_buckets"] == 2
    assert result.exec_stats["sell_before_buy"] is True
    assert result.exec_stats["structural_cash_pct"] == 0.0
    assert result.exec_stats["final_cash_pct"] == pytest.approx(0.0, abs=1e-6)
    assert result.exec_stats["no_cash_skip_days"] == 0


def test_no_skip_days_with_derived_buckets():
    """自动桶数下每日都有信号也不该出现现金耗尽跳过。"""
    wide = make_wide(DATES, ["A", "B"], close={"A": [10.0] * 20, "B": [10.0] * 20})
    picks = picks_frame([(d, c) for d in DATES for c in ("A", "B")])
    result = _run(wide, picks, hold_days=3, capital_mode="shared")

    assert result.exec_stats["n_buckets"] == 4
    assert result.exec_stats["no_cash_skip_days"] == 0
