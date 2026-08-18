"""选股列表 → 回测 → 落盘产物的编排层，供 CLI 与脚本复用。"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from hqpick.analysis.metrics import calc_metrics, equal_weight_benchmark
from hqpick.analysis.report import ReportInputs, save_report
from hqpick.analysis.trades import build_round_trips
from hqpick.data.panel import load_wide_frames
from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester, RunResult
from hqpick.picks import load_picks, normalize_frame

logger = logging.getLogger(__name__)

__all__ = ["load_picks", "run_backtest", "save_artifacts"]


def run_backtest(
    picks: pd.DataFrame | pl.DataFrame,
    start: date,
    end: date,
    config: ExecConfig | None = None,
    cache_dir: Path | str | None = None,
) -> tuple[RunResult, dict]:
    """执行回测，返回 (RunResult, metrics)。

    行情区间取 [start, end]；picks 中落在区间外的信号日会被裁掉。
    """
    if isinstance(picks, pl.DataFrame):
        picks = picks.to_pandas()
    picks = normalize_frame(picks)
    mask = (picks["date"] >= pd.Timestamp(start)) & (picks["date"] <= pd.Timestamp(end))
    picks = picks.loc[mask]
    if picks.empty:
        raise ValueError(f"[{start}, {end}] 区间内没有选股信号")

    wide = load_wide_frames(start, end, cache_dir=cache_dir)
    result = PickBacktester(config).run(picks, wide)

    bm_ret = equal_weight_benchmark(wide.adj["close"], result.nav.index)
    metrics = calc_metrics(result.daily_ret, bm_ret, (config or ExecConfig()).risk_free)
    metrics["benchmark"] = "equal_weight"
    metrics["_bm_nav"] = (1.0 + bm_ret).cumprod()
    return result, metrics


def save_artifacts(
    result: RunResult,
    metrics: dict,
    out_dir: Path | str,
    picks: pd.DataFrame | None = None,
    report: bool = True,
    title: str | None = None,
    attributes: pd.DataFrame | None = None,
) -> Path:
    """落盘 nav / trades / round_trips / metrics / exec_stats / report.html。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bm_nav = metrics.pop("_bm_nav", None)

    nav_frame = pd.DataFrame(
        {
            "date": result.nav.index,
            "nav": result.nav.to_numpy(),
            "daily_ret": result.daily_ret.to_numpy(),
        }
    )
    nav_frame.to_csv(out / "nav.csv", index=False)
    result.trades.to_csv(out / "trades.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out / "exec_stats.json").write_text(
        json.dumps(result.exec_stats, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    round_trips = build_round_trips(
        result.trades, result.config.cost_buy, result.config.cost_sell,
        calendar=result.nav.index,
    )
    round_trips.to_csv(out / "round_trips.csv", index=False)

    if picks is not None:
        picks.to_parquet(out / "picks.parquet", index=False)

    if report:
        save_report(
            ReportInputs(
                nav=result.nav, daily_ret=result.daily_ret, metrics=metrics,
                exec_stats=result.exec_stats, trades=result.trades,
                config_label=result.config.label, bm_nav=bm_nav, picks=picks,
                attributes=attributes,
                title=title or f"选股回测报告 · {result.config.label}",
                cost_buy=result.config.cost_buy, cost_sell=result.config.cost_sell,
            ),
            out / "report.html",
        )
    logger.info("产物已写入 %s", out)
    return out
