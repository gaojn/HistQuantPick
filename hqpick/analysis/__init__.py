"""绩效分析层。"""

from hqpick.analysis.metrics import calc_metrics, equal_weight_benchmark, max_drawdown
from hqpick.analysis.report import ReportInputs, render_report, save_report
from hqpick.analysis.trades import build_round_trips, open_positions, trade_stats

__all__ = [
    "ReportInputs",
    "build_round_trips",
    "calc_metrics",
    "equal_weight_benchmark",
    "max_drawdown",
    "open_positions",
    "render_report",
    "save_report",
    "trade_stats",
]
