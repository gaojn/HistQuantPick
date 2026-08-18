"""反转族信号：因子口径与前视防线。"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from hqpick.signals.reversal import (
    base_universe,
    build_reversal_picks,
    load_reversal_factors,
    select_top,
)

DATES = [date(2024, 1, d) for d in range(1, 32)]


def _write_cache(tmp_path, rows: list[dict]) -> str:
    frame = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    frame.write_parquet(tmp_path / "ashare_daily_2024.parquet")
    return str(tmp_path)


def _row(day, code, close: float, **kw) -> dict:
    base = {
        "code": code, "date": day, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "pre_close": close,
        "limit_up": close * 1.5, "limit_down": close * 0.5,
        "vwap": close, "adj_open": close, "adj_close": close, "adj_vwap": close,
        "volume": 1000.0, "amount": 1e6, "turnover": 2.0, "trade_status": "交易",
        "is_st": 0, "list_days": 500, "float_mv": 1e6, "industry_l1": "机械",
    }
    base.update(kw)
    return base


def _ramp(code: str, prices: list[float], **kw) -> list[dict]:
    return [_row(DATES[i], code, p, **kw) for i, p in enumerate(prices)]


def test_ret5_is_five_day_return_including_today():
    """ret5 = close / close[-5] − 1，含当日收盘。"""
    prices = [10.0] * 10 + [9.0]        # 第 11 天跌到 9
    rows = _ramp("A", prices)
    factors = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    factors = factors.sort(["code", "date"]).with_columns(
        (pl.col("close") / pl.col("close").shift(5).over("code") - 1.0).alias("ret5")
    )
    assert factors["ret5"][10] == pytest.approx(9.0 / 10.0 - 1)


def test_reversal_picks_worst_performers(tmp_path):
    """选 5 日跌幅最大的票。"""
    rows = []
    rows += _ramp("FALL_BIG", [10.0] * 6 + [7.0] * 5)      # 跌 30%
    rows += _ramp("FALL_SMALL", [10.0] * 6 + [9.5] * 5)    # 跌 5%
    rows += _ramp("RISE", [10.0] * 6 + [12.0] * 5)         # 涨
    cache = _write_cache(tmp_path, rows)

    picks = build_reversal_picks(
        DATES[10], DATES[10], lookback=5, max_per_day=1, cache_dir=cache
    )
    assert picks["code"].to_list() == ["FALL_BIG"]


def test_reversal_respects_max_per_day(tmp_path):
    rows = []
    for i in range(8):
        rows += _ramp(f"S{i}", [10.0] * 6 + [9.0 - i * 0.1] * 5)
    cache = _write_cache(tmp_path, rows)

    picks = build_reversal_picks(
        DATES[10], DATES[10], max_per_day=3, cache_dir=cache
    )
    assert len(picks) == 3


def test_liquidity_filter_uses_cross_section_percentile(tmp_path):
    """流动性过滤按当日截面分位，不是绝对阈值。

    三只票成交额都在基础门槛（5000 万元 = 5e4 千元）之上，只是相对高低不同；
    20 日均额需要 20 天样本，所以前 20 天先铺平。
    """
    rows = []
    for code, amount in (("LOW", 6e4), ("MID", 5e6), ("HIGH", 9e6)):
        rows += [_row(DATES[i], code, 10.0, amount=amount) for i in range(20)]
    day = DATES[20]
    # 跌幅：LOW 最多、MID 次之、HIGH 最少
    rows += [
        _row(day, "LOW", 7.0, amount=6e4),
        _row(day, "MID", 8.0, amount=5e6),
        _row(day, "HIGH", 9.0, amount=9e6),
    ]
    cache = _write_cache(tmp_path, rows)

    no_filter = build_reversal_picks(day, day, max_per_day=1, cache_dir=cache)
    assert no_filter["code"].to_list() == ["LOW"]        # 裸反转选到流动性最差的

    filtered = build_reversal_picks(
        day, day, max_per_day=1, min_liquidity_pct=0.5, cache_dir=cache
    )
    assert filtered["code"].to_list() == ["MID"]         # 分位门槛挡掉 LOW


def test_mom20_filter_keeps_only_uptrend(tmp_path):
    """min_ret20=0 只保留 20 日仍上涨的票（趋势中的回调）。"""
    rows = []
    # DOWNTREND: 20 日累计下跌；PULLBACK: 20 日仍上涨但近 5 日回调
    rows += _ramp("DOWNTREND", [20.0] * 6 + [15.0] * 10 + [10.0] * 15)
    rows += _ramp("PULLBACK", [10.0] * 6 + [15.0] * 10 + [13.0] * 15)
    cache = _write_cache(tmp_path, rows)

    day = DATES[25]
    both = build_reversal_picks(day, day, max_per_day=5, cache_dir=cache)
    assert set(both["code"].to_list()) == {"DOWNTREND", "PULLBACK"}

    uptrend = build_reversal_picks(
        day, day, max_per_day=5, min_ret20=0.0, cache_dir=cache
    )
    assert uptrend["code"].to_list() == ["PULLBACK"]


def test_float_mv_filter_in_yi_yuan(tmp_path):
    rows = []
    rows += _ramp("SMALL", [10.0] * 6 + [7.0] * 5, float_mv=2e5)     # 20 亿
    rows += _ramp("BIG", [10.0] * 6 + [8.0] * 5, float_mv=2e6)       # 200 亿
    cache = _write_cache(tmp_path, rows)

    picks = build_reversal_picks(
        DATES[10], DATES[10], max_per_day=5, min_float_mv=100, cache_dir=cache
    )
    assert picks["code"].to_list() == ["BIG"]


def test_base_universe_excludes_untradable(tmp_path):
    """停牌 / ST / 次新 / 涨停 / 成交额过小都不进 universe。"""
    rows = []
    rows += _ramp("OK", [10.0] * 6 + [7.0] * 5)
    rows += _ramp("SUSPENDED", [10.0] * 6 + [7.0] * 5, trade_status="停牌")
    rows += _ramp("ST", [10.0] * 6 + [7.0] * 5, is_st=1)
    rows += _ramp("NEW", [10.0] * 6 + [7.0] * 5, list_days=30)
    rows += _ramp("TINY", [10.0] * 6 + [7.0] * 5, amount=1e3)
    cache = _write_cache(tmp_path, rows)

    picks = build_reversal_picks(
        DATES[10], DATES[10], max_per_day=10, cache_dir=cache
    )
    assert picks["code"].to_list() == ["OK"]


def test_limit_up_stocks_excluded(tmp_path):
    """当日涨停的票剔除——T+1 大概率买不进，留着只会制造逆向选择。"""
    rows = _ramp("LU", [10.0] * 6 + [7.0] * 4)
    # 最后一天涨停收盘
    rows.append(_row(DATES[10], "LU", 7.7, limit_up=7.7, pre_close=7.0))
    rows += _ramp("NORMAL", [10.0] * 6 + [8.0] * 5)
    cache = _write_cache(tmp_path, rows)

    picks = build_reversal_picks(
        DATES[10], DATES[10], max_per_day=5, cache_dir=cache
    )
    assert "LU" not in picks["code"].to_list()


def test_volume_ratio_average_excludes_current_day(tmp_path):
    """放量倍数的均量右对齐到 T−1，不含当日。"""
    rows = [_row(DATES[i], "A", 10.0, volume=1000.0) for i in range(20)]
    rows.append(_row(DATES[20], "A", 9.0, volume=1500.0))
    cache = _write_cache(tmp_path, rows)

    day = DATES[20]
    assert len(build_reversal_picks(
        day, day, min_volume_ratio=1.49, cache_dir=cache)) == 1
    assert len(build_reversal_picks(
        day, day, min_volume_ratio=1.51, cache_dir=cache)) == 0


def test_lookback_one_uses_single_day_drop(tmp_path):
    rows = []
    # 5 日跌幅相同，但单日跌幅不同
    rows += _ramp("SLOW", [10.0, 9.6, 9.2, 8.8, 8.4, 8.0] + [8.0] * 5)
    rows += _ramp("SUDDEN", [10.0, 10.0, 10.0, 10.0, 10.0, 8.0] + [8.0] * 5)
    cache = _write_cache(tmp_path, rows)

    day = DATES[5]
    picks = build_reversal_picks(day, day, lookback=1, max_per_day=1, cache_dir=cache)
    assert picks["code"].to_list() == ["SUDDEN"]


def test_unsupported_lookback_rejected():
    with pytest.raises(ValueError, match="lookback"):
        build_reversal_picks(DATES[0], DATES[1], lookback=3)


def test_select_top_returns_date_code_only(tmp_path):
    rows = _ramp("A", [10.0] * 6 + [7.0] * 5)
    cache = _write_cache(tmp_path, rows)
    factors = load_reversal_factors(DATES[10], DATES[10], cache_dir=cache)

    picks = select_top(
        factors, DATES[10], score=-pl.col("ret5"),
        universe=base_universe(), max_per_day=5,
    )
    assert list(picks.columns) == ["date", "code"]
