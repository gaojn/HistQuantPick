"""分年、分月收益拆解。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hqpick.analysis.metrics import calc_metrics, equal_weight_benchmark
from hqpick.analysis.periodic import (
    month_of_year_stats,
    monthly_matrix,
    monthly_returns,
    yearly_stats,
)
from hqpick.engine.config import ExecConfig


def _daily(start: str, periods: int, value: float = 0.0) -> pd.Series:
    idx = pd.bdate_range(start, periods=periods)
    return pd.Series(value, index=idx)


def test_monthly_returns_compound_within_month():
    idx = pd.to_datetime(["2024-01-10", "2024-01-20", "2024-02-10"])
    ret = pd.Series([0.10, 0.10, -0.05], index=idx)
    monthly = monthly_returns(ret)

    assert monthly.iloc[0] == pytest.approx(1.10 * 1.10 - 1)
    assert monthly.iloc[1] == pytest.approx(-0.05)


def test_monthly_matrix_shape_and_year_total():
    ret = _daily("2023-01-02", 520, 0.001)          # 跨两年
    matrix = monthly_matrix(ret)

    assert list(matrix.columns[:12]) == [f"{m}月" for m in range(1, 13)]
    assert "全年" in matrix.columns
    assert set(matrix.index) == {2023, 2024}
    # 全年 = 该年日收益复利
    year_2023 = ret[ret.index.year == 2023]
    assert matrix.loc[2023, "全年"] == pytest.approx((1 + year_2023).prod() - 1)


def test_monthly_matrix_missing_months_are_nan():
    """只有部分月份有数据时，其余月份留空而不是填 0。"""
    idx = pd.to_datetime(["2024-03-05", "2024-07-05"])
    matrix = monthly_matrix(pd.Series([0.01, 0.02], index=idx))

    assert matrix.loc[2024, "3月"] == pytest.approx(0.01)
    assert np.isnan(matrix.loc[2024, "1月"])


def test_monthly_matrix_excess_column_uses_nav_ratio():
    ret = _daily("2024-01-02", 60, 0.002)
    bm = _daily("2024-01-02", 60, 0.001)
    matrix = monthly_matrix(ret, bm)

    total = (1 + ret).prod() - 1
    bm_total = (1 + bm).prod() - 1
    assert matrix.loc[2024, "全年超额"] == pytest.approx(
        (1 + total) / (1 + bm_total) - 1
    )


def test_yearly_stats_columns_and_values():
    ret = _daily("2023-01-02", 520, 0.001)
    bm = _daily("2023-01-02", 520, 0.0005)
    frame = yearly_stats(ret, bm)

    for col in ("交易日", "收益", "波动", "最大回撤", "Sharpe", "日胜率", "基准", "超额"):
        assert col in frame.columns
    assert set(frame.index) == {2023, 2024}
    assert (frame["收益"] > 0).all()
    assert (frame["超额"] > 0).all()
    # 恒定正收益 → 无回撤
    assert frame["最大回撤"].abs().max() < 1e-12


def test_yearly_stats_attaches_trade_metrics_by_buy_year():
    ret = _daily("2023-01-02", 520, 0.001)
    round_trips = pd.DataFrame({
        "buy_date": pd.to_datetime(["2023-03-01", "2023-05-01", "2024-02-01"]),
        "net_ret": [0.10, -0.05, 0.20],
    })
    frame = yearly_stats(ret, None, round_trips)

    assert frame.loc[2023, "交易笔数"] == 2
    assert frame.loc[2023, "逐笔胜率"] == pytest.approx(0.5)
    assert frame.loc[2024, "交易笔数"] == 1
    assert frame.loc[2024, "平均单笔"] == pytest.approx(0.20)


def test_month_of_year_stats_aggregates_across_years():
    idx = pd.to_datetime(["2023-01-10", "2024-01-10", "2023-06-10"])
    ret = pd.Series([0.05, -0.03, 0.01], index=idx)
    frame = month_of_year_stats(ret)

    assert frame.loc["1月", "样本年数"] == 2
    assert frame.loc["1月", "平均月收益"] == pytest.approx((0.05 - 0.03) / 2)
    assert frame.loc["1月", "为正比例"] == pytest.approx(0.5)
    assert frame.loc["6月", "样本年数"] == 1


def test_empty_inputs_are_safe():
    empty = pd.Series(dtype=float)
    assert monthly_returns(empty).empty
    assert monthly_matrix(empty).empty
    assert yearly_stats(empty).empty
    assert month_of_year_stats(empty).empty


def test_default_config_is_t1_open_to_t2_close_with_two_slots():
    """默认口径：T+1 开盘买、T+2 收盘卖，槽位 2。"""
    cfg = ExecConfig()

    assert cfg.hold_days == 1
    assert cfg.entry_offset == 1
    assert cfg.entry_price == "open"
    assert cfg.exit_price == "close"
    assert cfg.exit_offset == 2
    assert cfg.buckets == 2               # 开盘买收盘卖 → H+1
    assert cfg.label == "T+1open_hold1_T+2close_b2slots"


def test_metrics_record_actual_risk_free_rate():
    ret = pd.Series([0.01, -0.005, 0.008], index=pd.bdate_range("2024-01-02", periods=3))

    metrics = calc_metrics(ret, risk_free=0.07)

    assert metrics["risk_free"] == pytest.approx(0.07)


def test_equal_weight_benchmark_does_not_forward_fill_missing_quotes():
    idx = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame(
        {"A": [10.0, 11.0, None, 13.0], "B": [10.0, 10.0, 10.0, 10.0]}, index=idx
    )

    benchmark = equal_weight_benchmark(prices, idx)

    # A 在缺失报价后恢复时不能把两日累计涨幅记到恢复当日。
    assert benchmark.iloc[2] == pytest.approx(0.0)
    assert benchmark.iloc[3] == pytest.approx(0.0)
