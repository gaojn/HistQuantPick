"""参数网格扫描。"""

from __future__ import annotations

from datetime import date

import pytest

from hqpick.grid import GridSpec, format_grid, run_grid
from tests.helpers import make_wide, picks_frame

DATES = [f"2024-01-{d:02d}" for d in range(1, 15)]
N = len(DATES)


@pytest.fixture
def wide():
    return make_wide(
        DATES, ["A", "B"],
        close={"A": [10.0] * N, "B": [10.0] * N},
        limit_up={c: [999.0] * N for c in "AB"},
        limit_down={c: [0.01] * N for c in "AB"},
    )


@pytest.fixture
def picks():
    return picks_frame([(d, c) for d in DATES for c in ("A", "B")])


def test_spec_expands_cartesian_product():
    spec = GridSpec(hold_days=[1, 2, 3], n_buckets=[5, 10], entry_price=["open", "close"])
    configs = spec.configs()
    assert len(configs) == 3 * 2 * 2
    assert {c.hold_days for c in configs} == {1, 2, 3}
    assert {c.buckets for c in configs} == {5, 10}


def test_grid_runs_all_configs(wide, picks):
    spec = GridSpec(hold_days=[1, 2], n_buckets=[4, 6])
    frame = run_grid(picks, date(2024, 1, 1), date(2024, 1, 14), spec, wide=wide)

    assert len(frame) == 4
    assert set(frame["N"]) == {4, 6}
    assert set(frame["hold_days"]) == {1, 2}
    assert (frame["策略"] == "策略").all()
    for col in ("资金利用率", "年化", "最大回撤", "超额IR", "无空槽跳过天数"):
        assert col in frame.columns


def test_grid_with_matched_baseline_returns_distribution_summary(wide, picks):
    baseline_paths = [
        picks_frame([(d, c) for d in DATES for c in ("A", "B")]),
        picks_frame([(d, c) for d in DATES for c in ("B", "A")]),
    ]
    spec = GridSpec(hold_days=[2], n_buckets=[5])
    frame, paths = run_grid(
        picks, date(2024, 1, 1), date(2024, 1, 14), spec,
        baseline_picks=baseline_paths, wide=wide, return_baseline_paths=True,
    )

    assert len(frame) == 2
    assert set(frame["策略"]) == {"策略", "随机基线"}
    # 同一配置下两行的口径必须完全一致，否则比的不是选股
    assert frame["config"].nunique() == 1
    strategy = frame.loc[frame["策略"] == "策略"].iloc[0]
    assert strategy["基线样本数"] == 2
    assert 0.0 < strategy["年化经验p"] <= 1.0
    assert len(paths) == 2
    assert set(paths["基线路径"]) == {1, 2}


def test_grid_rejects_baseline_with_different_schedule(wide, picks):
    baseline = picks_frame([(d, "A") for d in DATES])

    with pytest.raises(ValueError, match="信号日期或每日选股数"):
        run_grid(
            picks, date(2024, 1, 1), date(2024, 1, 14),
            GridSpec(hold_days=[2], n_buckets=[5]),
            baseline_picks=baseline, wide=wide,
        )


def test_larger_n_lowers_utilization(wide, picks):
    """同一策略下 N 越大、资金利用率越低——N 是仓位旋钮。"""
    spec = GridSpec(hold_days=[2], n_buckets=[4, 8, 16])
    frame = run_grid(picks, date(2024, 1, 1), date(2024, 1, 14), spec, wide=wide)
    util = frame.sort_values("N")["资金利用率"].tolist()

    assert util[0] > util[1] > util[2]


def test_format_grid_renders_key_columns(wide, picks):
    spec = GridSpec(hold_days=[2], n_buckets=[5])
    frame = run_grid(picks, date(2024, 1, 1), date(2024, 1, 14), spec, wide=wide)
    text = format_grid(frame)

    assert "资金利用率" in text
    assert "无空槽跳过天数" in text
    assert "%" in text


def test_format_grid_handles_empty():
    import pandas as pd
    assert "空" in format_grid(pd.DataFrame())
