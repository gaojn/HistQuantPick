"""交易分布：行业与市值构成。"""

from __future__ import annotations

import pandas as pd
import pytest

from hqpick.analysis.exposure import (
    MV_LABELS,
    attach_attributes,
    industry_stats,
    mv_bucket_stats,
    mv_summary,
)


def _round_trips(rows: list[tuple]) -> pd.DataFrame:
    """(buy_date, code, net_ret, pnl, notional)。"""
    frame = pd.DataFrame(
        rows, columns=["buy_date", "code", "net_ret", "pnl", "notional"]
    )
    frame["buy_date"] = pd.to_datetime(frame["buy_date"])
    return frame


def _attrs(rows: list[tuple]) -> pd.DataFrame:
    """(date, code, float_mv[亿元], industry_l1)。"""
    frame = pd.DataFrame(rows, columns=["date", "code", "float_mv", "industry_l1"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_attach_puts_stock_into_right_mv_bucket():
    rt = _round_trips([
        ("2024-01-02", "A", 0.10, 1000.0, 10000.0),
        ("2024-01-02", "B", -0.05, -500.0, 10000.0),
    ])
    attrs = _attrs([
        ("2024-01-02", "A", 20.0, "机械"),
        ("2024-01-02", "B", 150.0, "银行"),
    ])
    out = attach_attributes(rt, attrs)

    assert out.set_index("code").loc["A", "mv_bucket"] == "<30亿"
    assert out.set_index("code").loc["B", "mv_bucket"] == "100-300亿"
    assert list(out["industry_l1"]) == ["机械", "银行"]


def test_attach_uses_buy_date_cross_section():
    """同一只票在不同买入日可能落在不同档位，各按当日取值。"""
    rt = _round_trips([
        ("2024-01-02", "A", 0.0, 0.0, 10000.0),
        ("2024-06-03", "A", 0.0, 0.0, 10000.0),
    ])
    attrs = _attrs([
        ("2024-01-02", "A", 25.0, "机械"),
        ("2024-06-03", "A", 120.0, "机械"),
    ])
    out = attach_attributes(rt, attrs).sort_values("buy_date")

    assert list(out["mv_bucket"]) == ["<30亿", "100-300亿"]


def test_unmatched_rows_get_na_and_are_dropped_from_stats():
    rt = _round_trips([
        ("2024-01-02", "A", 0.10, 100.0, 10000.0),
        ("2024-01-02", "ZZZ", -0.10, -100.0, 10000.0),    # 属性表里没有
    ])
    attrs = _attrs([("2024-01-02", "A", 60.0, "医药")])
    out = attach_attributes(rt, attrs)

    assert out["industry_l1"].isna().sum() == 1
    # 未匹配的行不进统计，但也不能让统计崩
    assert industry_stats(out)["交易笔数"].sum() == 1
    assert mv_bucket_stats(out)["交易笔数"].sum() == 1


def test_industry_stats_columns_and_ordering():
    rt = _round_trips([
        ("2024-01-02", "A", 0.10, 100.0, 10000.0),
        ("2024-01-02", "B", -0.05, -50.0, 10000.0),
        ("2024-01-03", "C", 0.02, 20.0, 10000.0),
    ])
    attrs = _attrs([
        ("2024-01-02", "A", 60.0, "医药"),
        ("2024-01-02", "B", 60.0, "医药"),
        ("2024-01-03", "C", 60.0, "银行"),
    ])
    out = industry_stats(attach_attributes(rt, attrs))

    assert list(out.index) == ["医药", "银行"]          # 按笔数降序
    assert out.loc["医药", "交易笔数"] == 2
    assert out.loc["医药", "逐笔胜率"] == pytest.approx(0.5)
    assert out.loc["医药", "笔数占比"] == pytest.approx(2 / 3)
    assert out.loc["银行", "合计盈亏"] == pytest.approx(20.0)


def test_industry_stats_top_folds_rest_into_other():
    rows, attrs_rows = [], []
    for i in range(6):
        code = f"S{i}"
        rows.append(("2024-01-02", code, 0.01 * (i + 1), 10.0, 10000.0))
        attrs_rows.append(("2024-01-02", code, 60.0, f"行业{i}"))
    # 让行业0 有两笔，确保排序稳定
    rows.append(("2024-01-03", "S0b", 0.05, 50.0, 10000.0))
    attrs_rows.append(("2024-01-03", "S0b", 60.0, "行业0"))

    out = industry_stats(
        attach_attributes(_round_trips(rows), _attrs(attrs_rows)), top=2
    )
    assert out.index[0] == "行业0"
    assert "其他" in out.index[-1]
    assert out["交易笔数"].sum() == 7


def test_mv_bucket_stats_sorted_small_to_large():
    rows, attrs_rows = [], []
    for i, mv in enumerate([10.0, 40.0, 70.0, 200.0, 500.0, 2000.0]):
        code = f"S{i}"
        rows.append(("2024-01-02", code, 0.01, 10.0, 10000.0))
        attrs_rows.append(("2024-01-02", code, mv, "机械"))
    out = mv_bucket_stats(attach_attributes(_round_trips(rows), _attrs(attrs_rows)))

    assert list(out.index) == MV_LABELS       # 小到大，不按笔数排
    assert out["笔数占比"].sum() == pytest.approx(1.0)


def test_mv_summary_quantiles():
    rows = [("2024-01-02", f"S{i}", 0.0, 0.0, 10000.0) for i in range(5)]
    attrs_rows = [
        ("2024-01-02", f"S{i}", mv, "机械")
        for i, mv in enumerate([10.0, 20.0, 30.0, 40.0, 50.0])
    ]
    out = mv_summary(attach_attributes(_round_trips(rows), _attrs(attrs_rows)))

    assert out["中位"] == pytest.approx(30.0)
    assert out["均值"] == pytest.approx(30.0)


def test_empty_inputs_are_safe():
    empty = _round_trips([])
    out = attach_attributes(empty, _attrs([]))
    assert industry_stats(out).empty
    assert mv_bucket_stats(out).empty
    assert mv_summary(out) == {}
