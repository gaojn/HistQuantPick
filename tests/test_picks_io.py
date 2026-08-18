"""选股表入口：date / code 的容错与防呆。"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester, normalize_picks
from hqpick.picks import check_code_overlap, coerce_dates, load_picks, normalize_frame
from tests.helpers import make_wide

DATES = [f"2024-01-{d:02d}" for d in range(1, 10)]
TARGET = pd.Timestamp("2024-01-02")


@pytest.mark.parametrize(
    "value",
    [
        dt.date(2024, 1, 2),
        dt.datetime(2024, 1, 2),
        pd.Timestamp("2024-01-02"),
        "2024-01-02",
        "2024/01/02",
        "20240102",
        20240102,          # int：pandas 默认会当成 Unix 纳秒，必须按 YYYYMMDD 解析
        20240102.0,        # float 同上
    ],
)
def test_coerce_dates_accepts_every_common_form(value):
    assert coerce_dates([value]).iloc[0] == TARGET


def test_numeric_date_is_never_treated_as_unix_timestamp():
    """20240102 必须是 2024-01-02，不能是 1970-01-01。"""
    out = coerce_dates([20240102])
    assert out.iloc[0].year == 2024
    assert out.iloc[0] != pd.Timestamp("1970-01-01 00:00:00.020240102")


def test_numeric_date_out_of_range_rejected():
    with pytest.raises(ValueError, match="YYYYMMDD"):
        coerce_dates([1704153600])          # 真正的 Unix 秒级时间戳


def test_mixed_string_forms_in_one_column():
    out = coerce_dates(["2024-01-02", "20240103", "2024/01/04"])
    assert list(out) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]


def test_normalize_frame_stringifies_and_strips_codes():
    frame = pd.DataFrame({
        "date": [20240102, 20240102],
        "code": ["  600000.SH ", 1],
    })
    out = normalize_frame(frame)
    assert list(out["code"]) == ["1", "600000.SH"]
    assert out["date"].iloc[0] == TARGET


def test_normalize_frame_drops_extra_columns_and_dupes():
    frame = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02"],
        "code": ["600000.SH", "600000.SH"],
        "score": [1.0, 2.0],
    })
    out = normalize_frame(frame)
    assert list(out.columns) == ["date", "code"]
    assert len(out) == 1


def test_normalize_frame_requires_both_columns():
    with pytest.raises(ValueError, match=r"\[date, code\]"):
        normalize_frame(pd.DataFrame({"date": ["2024-01-02"]}))


def test_code_overlap_zero_match_raises_with_hint():
    """代码全对不上时必须报错——否则回测静默跑出一条恒为 1 的净值。"""
    with pytest.raises(ValueError, match="缺交易所后缀|完全对不上"):
        check_code_overlap(["600000", "000001"], ["600000.SH", "000001.SZ"])


def test_code_overlap_partial_is_allowed():
    """部分对不上是正常的（退市、区间外），只告警不报错。"""
    ratio = check_code_overlap(
        ["600000.SH", "999999.SH"], ["600000.SH", "000001.SZ"]
    )
    assert ratio == pytest.approx(0.5)


def test_engine_rejects_codes_without_suffix():
    """端到端：缺后缀的代码在引擎入口就被拦下，而不是跑出空回测。"""
    wide = make_wide(DATES, ["600000.SH"], close={"600000.SH": [10.0] * 9})
    picks = pd.DataFrame({"date": ["2024-01-01"], "code": [600000]})

    with pytest.raises(ValueError, match="缺交易所后缀|完全对不上"):
        PickBacktester(ExecConfig()).run(picks, wide)


def test_engine_accepts_numeric_dates_end_to_end():
    """date 传 YYYYMMDD 整数也能正常回测。"""
    wide = make_wide(DATES, ["600000.SH"], close={"600000.SH": [10.0] * 9})
    picks = pd.DataFrame({"date": [20240101], "code": ["600000.SH"]})

    result = PickBacktester(ExecConfig()).run(picks, wide)
    assert len(result.trades) > 0


def test_normalize_picks_still_validates_calendar():
    wide = make_wide(DATES, ["A"], close={"A": [10.0] * 9})
    picks = pd.DataFrame({"date": ["2023-06-01"], "code": ["A"]})

    with pytest.raises(ValueError, match="不在行情交易日历"):
        normalize_picks(picks, wide.dates)


def test_load_picks_roundtrip_csv_and_parquet(tmp_path):
    frame = pd.DataFrame({"date": [20240102, 20240103], "code": ["600000.SH", "1"]})

    csv_path = tmp_path / "picks.csv"
    frame.to_csv(csv_path, index=False)
    from_csv = load_picks(csv_path)

    pq_path = tmp_path / "picks.parquet"
    frame.to_parquet(pq_path, index=False)
    from_pq = load_picks(pq_path)

    assert from_csv["date"].iloc[0] == TARGET
    assert list(from_csv["code"]) == list(from_pq["code"])
    pd.testing.assert_frame_equal(from_csv, from_pq)
