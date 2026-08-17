"""信号生成器：口径与前视防线（用合成缓存，不碰真实数据）。"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from hqpick.signals.limit_up import build_limit_up_picks
from hqpick.signals.lower_shadow import build_lower_shadow_picks

DATES = [date(2024, 1, d) for d in range(1, 32)]


def _write_cache(tmp_path, rows: list[dict]) -> str:
    """写一份最小合成缓存 ashare_daily_2024.parquet。"""
    frame = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    frame.write_parquet(tmp_path / "ashare_daily_2024.parquet")
    return str(tmp_path)


def _row(day, code, **kw) -> dict:
    base = {
        "code": code, "date": day, "open": 10.0, "high": 10.5, "low": 9.5,
        "close": 10.0, "pre_close": 10.0, "limit_up": 11.0, "limit_down": 9.0,
        "vwap": 10.0, "adj_open": 10.0, "adj_close": 10.0, "adj_vwap": 10.0,
        "volume": 1000.0, "amount": 1e6, "trade_status": "交易",
        "is_st": 0, "list_days": 500, "float_mv": 1e6,     # 万元 → 100 亿
        "industry_l1": "机械",
    }
    base.update(kw)
    return base


def test_lower_shadow_ranks_longer_shadow_and_higher_volume_first(tmp_path):
    """同日截面里，下影线更长且更放量的票排前面。"""
    rows = []
    for day in DATES[:25]:
        # 建立均量基线
        rows.append(_row(day, "FLAT", low=10.0, volume=1000.0))
        rows.append(_row(day, "SHADOW_BIG", low=10.0, volume=1000.0))
        rows.append(_row(day, "SHADOW_SMALL", low=10.0, volume=1000.0))
    signal_day = DATES[25]
    rows += [
        _row(signal_day, "FLAT", low=10.0, volume=1000.0),                  # 无影线
        _row(signal_day, "SHADOW_BIG", low=9.0, volume=3000.0),             # 长影+大量
        _row(signal_day, "SHADOW_SMALL", low=9.7, volume=1100.0),           # 短影+小量
    ]
    cache = _write_cache(tmp_path, rows)

    picks = build_lower_shadow_picks(
        signal_day, signal_day, max_per_day=1, volume_window=20,
        min_amount=0.0, cache_dir=cache,
    )
    assert picks["code"].to_list() == ["SHADOW_BIG"]


def test_lower_shadow_respects_max_per_day(tmp_path):
    rows = []
    codes = [f"S{i}" for i in range(8)]
    for day in DATES[:25]:
        rows += [_row(day, c, low=10.0) for c in codes]
    signal_day = DATES[25]
    rows += [
        _row(signal_day, c, low=9.0 + i * 0.05, volume=2000.0)
        for i, c in enumerate(codes)
    ]
    cache = _write_cache(tmp_path, rows)

    picks = build_lower_shadow_picks(
        signal_day, signal_day, max_per_day=5, volume_window=20,
        min_amount=0.0, cache_dir=cache,
    )
    assert len(picks) == 5


def test_lower_shadow_average_volume_excludes_current_day(tmp_path):
    """均量窗口右对齐到 T−1：当日放量不能把自己的均量抬高（防自相关）。

    构造：前 20 日量恒为 1000，信号日量 1500 → 放量倍数应恰为 1.5。
    若均量含当日，倍数会小于 1.5。
    """
    rows = [_row(day, "A", low=10.0, volume=1000.0) for day in DATES[:20]]
    signal_day = DATES[20]
    rows.append(_row(signal_day, "A", low=9.5, volume=1500.0))
    cache = _write_cache(tmp_path, rows)

    # 放量阈值 1.49 能选中，1.51 选不中 → 倍数落在 (1.49, 1.51)
    assert len(build_lower_shadow_picks(
        signal_day, signal_day, volume_window=20, min_volume_ratio=1.49,
        min_amount=0.0, cache_dir=cache,
    )) == 1
    assert len(build_lower_shadow_picks(
        signal_day, signal_day, volume_window=20, min_volume_ratio=1.51,
        min_amount=0.0, cache_dir=cache,
    )) == 0


def test_lower_shadow_shadow_ratio_uses_body_low(tmp_path):
    """下影线量的是实体下沿到最低价，不是收盘价到最低价。

    阳线 open=9.8 < close=10.0，实体下沿为 9.8；low=9.6 → 影线 2%。
    若误用 close 会算成 4%，阈值 3% 时就会被错误选入。
    """
    rows = [_row(day, "A", low=10.0, volume=1000.0) for day in DATES[:20]]
    signal_day = DATES[20]
    rows.append(_row(
        signal_day, "A", open=9.8, close=10.0, low=9.6, pre_close=10.0,
        volume=1500.0,
    ))
    cache = _write_cache(tmp_path, rows)

    kw = dict(volume_window=20, min_volume_ratio=1.0, min_amount=0.0, cache_dir=cache)
    assert len(build_lower_shadow_picks(signal_day, signal_day, min_shadow=0.019, **kw)) == 1
    assert len(build_lower_shadow_picks(signal_day, signal_day, min_shadow=0.021, **kw)) == 0


@pytest.mark.parametrize(
    ("field", "value", "kwargs"),
    [
        ("trade_status", "停牌", {}),
        ("is_st", 1, {}),
        ("list_days", 30, {}),
        ("close", 11.0, {}),                       # 涨停 → 默认剔除
        ("float_mv", 2e5, {"min_float_mv": 50}),   # 20 亿 < 50 亿门槛
    ],
)
def test_lower_shadow_filters(tmp_path, field, value, kwargs):
    rows = [_row(day, "A", low=10.0, volume=1000.0) for day in DATES[:20]]
    signal_day = DATES[20]
    rows.append(_row(signal_day, "A", low=9.0, volume=2000.0, **{field: value}))
    cache = _write_cache(tmp_path, rows)

    picks = build_lower_shadow_picks(
        signal_day, signal_day, volume_window=20, min_amount=0.0,
        cache_dir=cache, **kwargs,
    )
    assert picks.is_empty()


def test_lower_shadow_min_float_mv_uses_yi_yuan(tmp_path):
    """min_float_mv 参数单位是亿元，缓存 float_mv 是万元。"""
    rows = [_row(day, "A", low=10.0, volume=1000.0) for day in DATES[:20]]
    signal_day = DATES[20]
    # float_mv = 6e5 万元 = 60 亿
    rows.append(_row(signal_day, "A", low=9.0, volume=2000.0, float_mv=6e5))
    cache = _write_cache(tmp_path, rows)

    kw = dict(volume_window=20, min_amount=0.0, cache_dir=cache)
    assert len(build_lower_shadow_picks(
        signal_day, signal_day, min_float_mv=50, **kw)) == 1
    assert len(build_lower_shadow_picks(
        signal_day, signal_day, min_float_mv=100, **kw)) == 0


def test_lower_shadow_returns_empty_when_nothing_qualifies(tmp_path):
    rows = [_row(day, "A", low=10.0, volume=1000.0) for day in DATES[:22]]
    cache = _write_cache(tmp_path, rows)

    picks = build_lower_shadow_picks(
        DATES[20], DATES[21], volume_window=20, min_amount=0.0, cache_dir=cache
    )
    assert picks.is_empty()
    assert set(picks.columns) == {"date", "code"}


def test_limit_up_min_amount_is_in_thousand_yuan(tmp_path):
    """min_amount 单位是千元（缓存口径），别当成元。"""
    day = DATES[10]
    rows = [_row(day, "A", close=11.0, limit_up=11.0, amount=1e5)]   # 1e5 千元 = 1 亿元
    cache = _write_cache(tmp_path, rows)

    assert len(build_limit_up_picks(day, day, min_amount=5e4, cache_dir=cache)) == 1
    assert len(build_limit_up_picks(day, day, min_amount=2e5, cache_dir=cache)) == 0


def test_limit_up_consecutive_requires_streak(tmp_path):
    """consecutive=2 只在「当日与前一日都涨停」时命中。"""
    rows = []
    # A: 第 10、11 天连续涨停；B: 只有第 11 天涨停
    for i, day in enumerate(DATES[:14]):
        a_lu = i in (9, 10)
        b_lu = i == 10
        rows.append(_row(day, "A", close=11.0 if a_lu else 10.0, limit_up=11.0))
        rows.append(_row(day, "B", close=11.0 if b_lu else 10.0, limit_up=11.0))
    cache = _write_cache(tmp_path, rows)

    single = build_limit_up_picks(DATES[10], DATES[10], cache_dir=cache)
    assert set(single["code"].to_list()) == {"A", "B"}          # 当日都涨停

    double = build_limit_up_picks(
        DATES[10], DATES[10], consecutive=2, cache_dir=cache
    )
    assert double["code"].to_list() == ["A"]                    # 只有 A 连续两天


def test_limit_up_consecutive_broken_by_suspension(tmp_path):
    """中间停牌打断连续，不算连板。"""
    rows = []
    for i, day in enumerate(DATES[:14]):
        if i == 9:
            rows.append(_row(day, "A", close=11.0, limit_up=11.0, trade_status="停牌"))
        else:
            rows.append(_row(day, "A", close=11.0 if i == 10 else 10.0, limit_up=11.0))
    cache = _write_cache(tmp_path, rows)

    picks = build_limit_up_picks(
        DATES[10], DATES[10], consecutive=2, cache_dir=cache
    )
    assert picks.is_empty()


def test_limit_up_consecutive_three_days(tmp_path):
    rows = []
    for i, day in enumerate(DATES[:14]):
        rows.append(_row(day, "A", close=11.0 if i in (8, 9, 10) else 10.0, limit_up=11.0))
    cache = _write_cache(tmp_path, rows)

    assert len(build_limit_up_picks(DATES[10], DATES[10], consecutive=3, cache_dir=cache)) == 1
    assert build_limit_up_picks(DATES[10], DATES[10], consecutive=4, cache_dir=cache).is_empty()


def test_limit_up_consecutive_rejects_zero():
    with pytest.raises(ValueError, match="consecutive"):
        build_limit_up_picks(DATES[0], DATES[1], consecutive=0)
