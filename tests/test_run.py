"""编排层集成测试：picks → run_backtest → save_artifacts 全链路接线。

engine 内部逻辑已由 test_timing/test_execution_rules 等覆盖，这里只锁
run.py 把各层结果正确传递、落盘产物齐全——防止参数顺序错、metrics key
错、report 拿错 bm_nav 这类接线 bug 不被测试拦住。
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hqpick import run as run_mod
from hqpick.engine.config import ExecConfig
from tests.helpers import picks_frame


def test_run_backtest_returns_result_and_metrics_with_benchmark(monkeypatch, flat_market):
    monkeypatch.setattr(run_mod, "load_wide_frames", lambda *a, **kw: flat_market)
    picks = picks_frame([("2024-01-01", "A")])

    result, metrics = run_mod.run_backtest(
        picks, date(2024, 1, 1), date(2024, 1, 8), config=ExecConfig(),
    )

    assert result.trades["side"].tolist() == ["buy", "sell"]
    assert metrics["benchmark"] == "equal_weight"
    assert "_bm_nav" in metrics
    assert metrics["n_days"] == len(result.nav)


def test_run_backtest_rejects_empty_window(monkeypatch, flat_market):
    monkeypatch.setattr(run_mod, "load_wide_frames", lambda *a, **kw: flat_market)
    picks = picks_frame([("2024-01-01", "A")])

    with pytest.raises(ValueError, match="没有选股信号"):
        run_mod.run_backtest(picks, date(2024, 2, 1), date(2024, 2, 8))


def test_save_artifacts_writes_all_files_and_pops_bm_nav(monkeypatch, flat_market, tmp_path):
    monkeypatch.setattr(run_mod, "load_wide_frames", lambda *a, **kw: flat_market)
    picks = picks_frame([("2024-01-01", "A")])
    result, metrics = run_mod.run_backtest(picks, date(2024, 1, 1), date(2024, 1, 8))

    out = run_mod.save_artifacts(result, metrics, tmp_path / "out", picks=picks)

    assert out == tmp_path / "out"
    for name in (
        "nav.csv", "trades.csv", "round_trips.csv",
        "metrics.json", "exec_stats.json", "report.html", "picks.parquet",
    ):
        assert (out / name).exists(), name

    nav_frame = pd.read_csv(out / "nav.csv")
    assert list(nav_frame.columns) == ["date", "nav", "daily_ret"]
    assert len(nav_frame) == len(result.nav)

    # save_artifacts 会 pop 掉 "_bm_nav"，落盘的 metrics.json 里不应再出现
    assert "_bm_nav" not in metrics


def test_save_artifacts_without_report_skips_html(monkeypatch, flat_market, tmp_path):
    monkeypatch.setattr(run_mod, "load_wide_frames", lambda *a, **kw: flat_market)
    picks = picks_frame([("2024-01-01", "A")])
    result, metrics = run_mod.run_backtest(picks, date(2024, 1, 1), date(2024, 1, 8))

    out = run_mod.save_artifacts(result, metrics, tmp_path / "out", report=False)

    assert not (out / "report.html").exists()
    assert not (out / "picks.parquet").exists()
    assert (out / "nav.csv").exists()
