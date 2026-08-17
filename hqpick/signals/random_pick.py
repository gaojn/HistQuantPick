"""随机选股信号：管线冒烟与 alpha 显著性对照的 baseline。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from hqpick.constants import SUSPENDED
from hqpick.data.panel import load_panel

_COLUMNS = ("code", "date", "close", "trade_status", "is_st", "list_days")


def build_random_picks(
    t1: date,
    t2: date,
    n_per_day: int = 10,
    seed: int = 42,
    exclude_st: bool = True,
    min_list_days: int = 120,
    cache_dir: Path | str | None = None,
) -> pl.DataFrame:
    """每个交易日从可交易股票中随机抽 ``n_per_day`` 只，返回 [date, code]。

    随机基线的意义：同一套执行口径下跑一遍，任何策略的净值都必须显著
    跑赢它，否则「超额」只是执行规则或市场 beta 带来的假象。
    """
    panel = load_panel(t1, t2, columns=_COLUMNS, cache_dir=cache_dir)

    cond = pl.col("close").is_not_null() & (pl.col("trade_status") != SUSPENDED)
    if exclude_st:
        cond = cond & (pl.col("is_st") == 0)
    if min_list_days > 0:
        cond = cond & (pl.col("list_days") >= min_list_days)
    pool = panel.filter(cond).select(["date", "code"]).sort(["date", "code"])

    rng = np.random.default_rng(seed)
    rows: list[pl.DataFrame] = []
    for (day,), group in pool.group_by(["date"], maintain_order=True):
        codes = group["code"].to_list()
        take = min(n_per_day, len(codes))
        if take == 0:
            continue
        chosen = rng.choice(codes, size=take, replace=False)
        rows.append(pl.DataFrame({"date": [day] * take, "code": sorted(chosen)}))

    if not rows:
        return pl.DataFrame({"date": [], "code": []})
    return pl.concat(rows).sort(["date", "code"])
