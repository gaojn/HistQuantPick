"""HTML 报告生成。"""

from __future__ import annotations

import pytest

from hqpick.analysis.metrics import calc_metrics, equal_weight_benchmark
from hqpick.analysis.report import ReportInputs, render_report, save_report
from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester
from tests.helpers import make_wide, picks_frame

DATES = [f"2024-01-{d:02d}" for d in range(1, 15)]
N = len(DATES)


@pytest.fixture
def run_result():
    close = {"A": [10 + (i % 3) for i in range(N)], "B": [10.0] * N}
    wide = make_wide(
        DATES, ["A", "B"], close=close,
        limit_up={c: [999.0] * N for c in "AB"},
        limit_down={c: [0.01] * N for c in "AB"},
    )
    picks = picks_frame([(d, c) for d in DATES[:10] for c in ("A", "B")])
    cfg = ExecConfig(hold_days=2, n_buckets=5)
    result = PickBacktester(cfg).run(picks, wide)
    bm = equal_weight_benchmark(wide.adj["close"], result.nav.index)
    metrics = calc_metrics(result.daily_ret, bm, cfg.risk_free)
    metrics["benchmark"] = "equal_weight"
    return result, metrics, bm, picks, cfg


def _inputs(run_result, **kwargs) -> ReportInputs:
    result, metrics, bm, picks, cfg = run_result
    defaults = dict(
        nav=result.nav, daily_ret=result.daily_ret, metrics=metrics,
        exec_stats=result.exec_stats, trades=result.trades,
        config_label=cfg.label, bm_nav=(1 + bm).cumprod(), picks=picks,
        cost_buy=cfg.cost_buy, cost_sell=cfg.cost_sell,
    )
    defaults.update(kwargs)
    return ReportInputs(**defaults)


def test_report_is_self_contained(run_result):
    """无外部资源引用——单个 html 可直接发给别人。"""
    doc = render_report(_inputs(run_result))

    assert doc.startswith("<!doctype html>")
    for forbidden in ("<script", "http://", "https://", "cdn."):
        assert forbidden not in doc, f"报告不应引用外部资源: {forbidden}"


def test_report_contains_all_sections(run_result):
    doc = render_report(_inputs(run_result))
    for section in ("净值表现", "执行质量", "逐笔归因", "回测口径"):
        assert section in doc


def test_report_renders_key_metrics_and_charts(run_result):
    doc = render_report(_inputs(run_result))

    assert "年化收益" in doc and "最大回撤" in doc
    assert "胜率" in doc and "盈亏比" in doc
    assert "<svg" in doc                      # 净值/回撤/直方图
    assert doc.count("<svg") >= 3
    assert "资金利用率" in doc and "槽位数 N" in doc


def test_report_warns_on_skipped_entries():
    """槽位占满导致跳过时，报告必须显式提示仓位不可比。"""
    wide = make_wide(
        DATES, ["A"], close={"A": [10.0] * N},
        limit_up={"A": [999.0] * N}, limit_down={"A": [10.0] * N},   # 永远卖不出
    )
    picks = picks_frame([(d, "A") for d in DATES])
    cfg = ExecConfig(hold_days=2, n_buckets=3)
    result = PickBacktester(cfg).run(picks, wide)
    doc = render_report(ReportInputs(
        nav=result.nav, daily_ret=result.daily_ret, metrics={},
        exec_stats=result.exec_stats, trades=result.trades, config_label=cfg.label,
    ))

    assert result.exec_stats["no_free_slot_days"] > 0
    assert "未能建仓" in doc
    assert "比的是仓位不是选股" in doc


def test_report_flags_writeoff_as_assumption():
    """启用卡仓核销时必须标注为假设口径。"""
    wide = make_wide(
        DATES, ["A"], close={"A": [10.0] * N},
        limit_up={"A": [999.0] * N}, limit_down={"A": [10.0] * N},
    )
    picks = picks_frame([("2024-01-01", "A")])
    cfg = ExecConfig(hold_days=2, writeoff_stuck_days=3)
    result = PickBacktester(cfg).run(picks, wide)
    doc = render_report(ReportInputs(
        nav=result.nav, daily_ret=result.daily_ret, metrics={},
        exec_stats=result.exec_stats, trades=result.trades, config_label=cfg.label,
    ))

    assert "假设口径" in doc
    assert "并未真的回笼" in doc


def test_report_handles_no_round_trips():
    """一笔都没成交也不能崩。"""
    wide = make_wide(
        DATES, ["A"], close={"A": [10.0] * N},
        open_={"A": [11.0] * N}, limit_up={"A": [11.0] * N},   # 全程涨停买不进
    )
    picks = picks_frame([("2024-01-01", "A")])
    cfg = ExecConfig(hold_days=2)
    result = PickBacktester(cfg).run(picks, wide)
    doc = render_report(ReportInputs(
        nav=result.nav, daily_ret=result.daily_ret, metrics={},
        exec_stats=result.exec_stats, trades=result.trades, config_label=cfg.label,
    ))

    assert "没有完整的往返交易" in doc


def test_save_report_writes_file(run_result, tmp_path):
    path = save_report(_inputs(run_result), tmp_path / "sub" / "report.html")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")
    assert len(content) > 2000


def test_html_escapes_untrusted_title(run_result):
    doc = render_report(_inputs(run_result, title='<img src=x onerror="alert(1)">'))
    assert "<img src=x" not in doc
    assert "&lt;img" in doc
