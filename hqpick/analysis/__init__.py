"""绩效分析层。"""

from hqpick.analysis.metrics import calc_metrics, equal_weight_benchmark, max_drawdown
from hqpick.analysis.periodic import month_of_year_stats, monthly_matrix, yearly_stats
from hqpick.analysis.report import ReportInputs, render_report, save_report
from hqpick.analysis.trades import build_round_trips, open_positions, trade_stats

__all__ = [
    "ReportInputs",
    "build_round_trips",
    "calc_metrics",
    "equal_weight_benchmark",
    "max_drawdown",
    "month_of_year_stats",
    "monthly_matrix",
    "open_positions",
    "render_report",
    "save_report",
    "trade_stats",
    "yearly_stats",
]
