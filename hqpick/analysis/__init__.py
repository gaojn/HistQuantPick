"""绩效分析层。"""

from hqpick.analysis.exposure import (
    attach_attributes,
    industry_stats,
    load_attributes,
    mv_bucket_stats,
)
from hqpick.analysis.metrics import calc_metrics, equal_weight_benchmark, max_drawdown
from hqpick.analysis.periodic import month_of_year_stats, monthly_matrix, yearly_stats
from hqpick.analysis.report import ReportInputs, render_report, save_report
from hqpick.analysis.trades import build_round_trips, open_positions, trade_stats

__all__ = [
    "ReportInputs",
    "attach_attributes",
    "build_round_trips",
    "calc_metrics",
    "equal_weight_benchmark",
    "industry_stats",
    "load_attributes",
    "max_drawdown",
    "mv_bucket_stats",
    "month_of_year_stats",
    "monthly_matrix",
    "open_positions",
    "render_report",
    "save_report",
    "trade_stats",
    "yearly_stats",
]
