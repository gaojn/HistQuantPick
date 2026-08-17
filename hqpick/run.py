"""选股列表 → 回测 → 落盘产物的编排层，供 CLI 与脚本复用。"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from hqpick.analysis.metrics import calc_metrics, equal_weight_benchmark
from hqpick.data.panel import load_wide_frames
from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester, RunResult

logger = logging.getLogger(__name__)


def load_picks(path: str | Path) -> pd.DataFrame:
    """读取选股长表（parquet / csv），只保留 [date, code]。"""
    p = Path(path)
    df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
    if not {"date", "code"}.issubset(df.columns):
        raise ValueError(f"选股文件须包含 [date, code] 列，当前列为 {list(df.columns)}")
    out = df[["date", "code"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str)
    return out.drop_duplicates().sort_values(["date", "code"]).reset_index(drop=True)


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
    picks = picks.copy()
    picks["date"] = pd.to_datetime(picks["date"])
    mask = (picks["date"] >= pd.Timestamp(start)) & (picks["date"] <= pd.Timestamp(end))
    picks = picks.loc[mask]
    if picks.empty:
        raise ValueError(f"[{start}, {end}] 区间内没有选股信号")

    wide = load_wide_frames(start, end, cache_dir=cache_dir)
    result = PickBacktester(config).run(picks, wide)

    bm_ret = equal_weight_benchmark(wide.adj["close"], result.nav.index)
    metrics = calc_metrics(result.daily_ret, bm_ret, (config or ExecConfig()).risk_free)
    metrics["benchmark"] = "equal_weight"
    return result, metrics


def save_artifacts(
    result: RunResult,
    metrics: dict,
    out_dir: Path | str,
    picks: pd.DataFrame | None = None,
) -> Path:
    """落盘 nav / trades / metrics / exec_stats，返回输出目录。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

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
    if picks is not None:
        picks.to_parquet(out / "picks.parquet", index=False)
    logger.info("产物已写入 %s", out)
    return out
