"""A 股日频行情面板加载（与 HistQuantOpt 共享同一份 parquet 缓存）。

缓存文件：``<cache_dir>/ashare_daily_<year>.parquet``。
``limit_up`` / ``limit_down`` 直接用缓存真实字段，不在本模块推算。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from hqpick.constants import DEFAULT_CACHE_DIR

# 回测必需列：三种成交价（复权+原始）、涨跌停价、交易状态
BACKTEST_COLUMNS: tuple[str, ...] = (
    "code",
    "date",
    "open",
    "close",
    "vwap",
    "adj_open",
    "adj_close",
    "adj_vwap",
    "limit_up",
    "limit_down",
    "trade_status",
)


def _years_in_range(t1: date, t2: date) -> list[int]:
    if t1 > t2:
        raise ValueError(f"起始日 {t1} 不能晚于截止日 {t2}")
    return list(range(t1.year, t2.year + 1))


def _cache_files(t1: date, t2: date, cache_dir: Path) -> list[Path]:
    paths: list[Path] = []
    missing: list[int] = []
    for year in _years_in_range(t1, t2):
        path = cache_dir / f"ashare_daily_{year}.parquet"
        if path.exists():
            paths.append(path)
        else:
            missing.append(year)
    if missing:
        raise FileNotFoundError(
            f"缺少本地缓存: {', '.join(str(y) for y in missing)} 年。"
            f"请确认 {cache_dir} 下存在对应年份的 parquet。"
        )
    return paths


def load_panel(
    t1: date,
    t2: date,
    columns: Sequence[str] | None = None,
    cache_dir: Path | str | None = None,
) -> pl.DataFrame:
    """加载 [t1, t2] 闭区间的行情长表，按 (date, code) 升序。"""
    cache_path = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    files = _cache_files(t1, t2, cache_path)
    logical = list(dict.fromkeys(["code", "date", *(columns or BACKTEST_COLUMNS)]))

    available = set(pl.scan_parquet(files[0]).collect_schema().names())
    missing = [c for c in logical if c not in available]
    if missing:
        raise KeyError(f"缓存缺少列 {missing}；可用列: {sorted(available)}")

    return (
        pl.scan_parquet(files)
        .select(logical)
        .with_columns(pl.col("date").cast(pl.Date))
        .filter((pl.col("date") >= t1) & (pl.col("date") <= t2))
        .collect()
        .sort(["date", "code"])
    )


@dataclass(frozen=True)
class WideFrames:
    """回测所需的全部行情宽表（index=交易日，columns=股票代码）。"""

    adj: dict[str, pd.DataFrame]      # 'open'/'close'/'vwap' → 复权价
    raw: dict[str, pd.DataFrame]      # 'open'/'close'/'vwap' → 原始价
    limit_up: pd.DataFrame
    limit_down: pd.DataFrame
    trade_status: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.adj["close"].index)


def _pivot(panel: pl.DataFrame, col: str) -> pd.DataFrame:
    wide = (
        panel.select(["date", "code", col])
        .to_pandas()
        .pivot(index="date", columns="code", values=col)
        .sort_index()
    )
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    return wide


def load_wide_frames(
    t1: date,
    t2: date,
    cache_dir: Path | str | None = None,
) -> WideFrames:
    """加载并 pivot 成回测引擎直接消费的宽表集合。"""
    panel = load_panel(t1, t2, cache_dir=cache_dir)
    return WideFrames(
        adj={f: _pivot(panel, f"adj_{f}") for f in ("open", "close", "vwap")},
        raw={f: _pivot(panel, f) for f in ("open", "close", "vwap")},
        limit_up=_pivot(panel, "limit_up"),
        limit_down=_pivot(panel, "limit_down"),
        trade_status=_pivot(panel, "trade_status"),
    )
