"""交易分布：所交易股票的行业与市值构成。

用**买入日**的截面属性（成交时点的规模与所属行业），回答两个问题：
钱实际押在哪些行业上，以及押在多大市值的票上。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from hqpick.data.panel import load_panel

# 流通市值分档（亿元）。缓存里 float_mv 单位是万元，需除以 1e4。
MV_BINS = [0, 30, 50, 100, 300, 1000, float("inf")]
MV_LABELS = ["<30亿", "30-50亿", "50-100亿", "100-300亿", "300-1000亿", "≥1000亿"]

ATTRIBUTE_COLUMNS = ("code", "date", "float_mv", "industry_l1")


def load_attributes(
    t1: date, t2: date, cache_dir: Path | str | None = None
) -> pd.DataFrame:
    """加载 [date, code, float_mv, industry_l1] 长表，float_mv 换算成亿元。"""
    panel = load_panel(t1, t2, columns=ATTRIBUTE_COLUMNS, cache_dir=cache_dir)
    frame = panel.select(["date", "code", "float_mv", "industry_l1"]).to_pandas()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["float_mv"] = frame["float_mv"] / 1e4        # 万元 → 亿元
    return frame


def attach_attributes(
    round_trips: pd.DataFrame, attributes: pd.DataFrame
) -> pd.DataFrame:
    """按 (买入日, 代码) 给逐笔交易挂上行业与市值，并分好市值档。"""
    if round_trips.empty:
        return round_trips.assign(industry_l1=None, float_mv=None, mv_bucket=None)

    attrs = attributes.rename(columns={"date": "buy_date"})
    out = round_trips.merge(attrs, on=["buy_date", "code"], how="left")
    out["mv_bucket"] = pd.cut(
        out["float_mv"], bins=MV_BINS, labels=MV_LABELS, right=False
    )
    return out


def _group_stats(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = frame.groupby(column, dropna=False, observed=False)
    out = pd.DataFrame({
        "交易笔数": grouped.size(),
        "笔数占比": grouped.size() / len(frame),
        "资金占比": grouped["notional"].sum() / frame["notional"].sum(),
        "逐笔胜率": grouped["net_ret"].apply(lambda s: float((s > 0).mean())),
        "平均单笔": grouped["net_ret"].mean(),
        "合计盈亏": grouped["pnl"].sum(),
    })
    return out[out["交易笔数"] > 0]


def industry_stats(round_trips: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    """按中信一级行业统计，按笔数降序；``top`` 给定时其余并入「其他」。"""
    if round_trips.empty or "industry_l1" not in round_trips.columns:
        return pd.DataFrame()
    frame = round_trips.dropna(subset=["industry_l1"])
    if frame.empty:
        return pd.DataFrame()

    out = _group_stats(frame, "industry_l1").sort_values("交易笔数", ascending=False)
    if top is not None and len(out) > top:
        head = out.head(top)
        tail = out.tail(len(out) - top)
        other = pd.DataFrame(
            {
                "交易笔数": [tail["交易笔数"].sum()],
                "笔数占比": [tail["笔数占比"].sum()],
                "资金占比": [tail["资金占比"].sum()],
                # 其余行业合并后的胜率按笔数加权
                "逐笔胜率": [
                    float((tail["逐笔胜率"] * tail["交易笔数"]).sum() / tail["交易笔数"].sum())
                ],
                "平均单笔": [
                    float((tail["平均单笔"] * tail["交易笔数"]).sum() / tail["交易笔数"].sum())
                ],
                "合计盈亏": [tail["合计盈亏"].sum()],
            },
            index=[f"其他（{len(tail)} 个行业）"],
        )
        out = pd.concat([head, other])
    out.index.name = "行业"
    return out


def mv_bucket_stats(round_trips: pd.DataFrame) -> pd.DataFrame:
    """按流通市值分档统计，档位从小到大排列。"""
    if round_trips.empty or "mv_bucket" not in round_trips.columns:
        return pd.DataFrame()
    frame = round_trips.dropna(subset=["mv_bucket"])
    if frame.empty:
        return pd.DataFrame()

    out = _group_stats(frame, "mv_bucket")
    out = out.reindex([lab for lab in MV_LABELS if lab in out.index])
    out.index.name = "流通市值"
    return out


def mv_summary(round_trips: pd.DataFrame) -> dict:
    """所交易股票流通市值的分位数（亿元）。"""
    if round_trips.empty or "float_mv" not in round_trips.columns:
        return {}
    mv = round_trips["float_mv"].dropna()
    if mv.empty:
        return {}
    return {
        "中位": float(mv.median()),
        "均值": float(mv.mean()),
        "p05": float(mv.quantile(0.05)),
        "p25": float(mv.quantile(0.25)),
        "p75": float(mv.quantile(0.75)),
        "p95": float(mv.quantile(0.95)),
    }
