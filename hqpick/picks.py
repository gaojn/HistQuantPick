"""选股表的加载与归一化：``[date, code]`` 长表的容错入口。

选股表是本包唯一的策略输入，来源五花八门（自己的脚本、Excel 导出、数据库
查询），date 与 code 的类型经常不一致。这里统一归一化，并对两类**静默失败**
做防呆：

- ``date`` 传成整数 ``20240102``：pandas 默认按 Unix 纳秒解析，会得到
  1970-01-01，后续只会报「区间内没有信号」，指不到真正原因；
- ``code`` 缺交易所后缀（``600000`` 而非 ``600000.SH``）：与行情面板一个都
  匹配不上，回测照跑，净值恒为 1，错误只藏在 ``exec_stats.no_quote`` 里。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 数值型日期按 YYYYMMDD 解释的合理范围
_YYYYMMDD_MIN = 19000101
_YYYYMMDD_MAX = 21001231

# 未匹配比例超过这个值只告警，不报错（退市、区间外的票属正常情况）
_UNMATCHED_WARN_RATIO = 0.5


def coerce_dates(values) -> pd.Series:
    """把各种形式的日期统一成 datetime64。

    支持 ``datetime`` / ``date`` / ``Timestamp``、``"2024-01-02"`` /
    ``"2024/01/02"`` / ``"20240102"`` 字符串，以及 ``20240102`` 整数或浮点；
    同一列里混用多种字符串格式也能解析。

    数值型一律按 YYYYMMDD 解释——**绝不当作 Unix 时间戳**。pandas 默认会把
    ``20240102`` 读成 1970-01-01 00:00:00.020240102，那会让后续只报出
    「区间内没有信号」，完全指不到真正的原因。
    """
    series = pd.Series(values)
    if series.empty:
        return pd.to_datetime(series)

    if pd.api.types.is_numeric_dtype(series):
        numeric = series.dropna()
        as_int = numeric.round().astype("int64")
        if not as_int.between(_YYYYMMDD_MIN, _YYYYMMDD_MAX).all():
            bad = as_int[~as_int.between(_YYYYMMDD_MIN, _YYYYMMDD_MAX)]
            raise ValueError(
                f"数值型日期只支持 YYYYMMDD 形式（如 20240102），"
                f"下列取值超出范围: {bad.head(3).tolist()}。"
                f"若本意是 Unix 时间戳，请先自行转成 datetime 再传入。"
            )
        return pd.to_datetime(series.round().astype("Int64").astype(str),
                              format="%Y%m%d")

    try:
        return pd.to_datetime(series)
    except (ValueError, TypeError):
        # 同一列里混了多种字符串格式（"2024-01-02" 与 "20240103"）时逐元素推断
        return pd.to_datetime(series, format="mixed")


def coerce_codes(values) -> pd.Series:
    """代码统一成去空白的字符串。"""
    return pd.Series(values).astype(str).str.strip()


def normalize_frame(picks: pd.DataFrame) -> pd.DataFrame:
    """校验并归一化选股长表，只保留 [date, code]，去重后按 (date, code) 排序。"""
    if not {"date", "code"}.issubset(picks.columns):
        raise ValueError(
            f"选股表必须包含 [date, code] 两列，当前列为 {list(picks.columns)}"
        )
    out = pd.DataFrame({
        "date": coerce_dates(picks["date"]),
        "code": coerce_codes(picks["code"]),
    })
    return out.drop_duplicates().sort_values(["date", "code"]).reset_index(drop=True)


def check_code_overlap(codes, universe_codes) -> float:
    """检查选股代码与行情面板的匹配情况，返回未匹配比例。

    一个都匹配不上时直接报错——这几乎总是代码格式问题（缺 .SH/.SZ 后缀），
    继续跑只会得到一条恒为 1 的净值曲线。
    """
    picked = set(codes)
    universe = set(universe_codes)
    if not picked:
        return 0.0

    unmatched = picked - universe
    ratio = len(unmatched) / len(picked)

    if not (picked & universe):
        sample_picked = sorted(picked)[:3]
        sample_universe = sorted(universe)[:3]
        raise ValueError(
            f"选股表里的代码与行情面板完全对不上（{len(picked)} 个代码无一匹配）。"
            f"选股表示例: {sample_picked}；行情面板示例: {sample_universe}。"
            f"常见原因是缺交易所后缀——代码应形如 600000.SH / 000001.SZ。"
        )
    if ratio > _UNMATCHED_WARN_RATIO:
        logger.warning(
            "选股表有 %.1f%% 的代码在行情面板中不存在（示例 %s），"
            "这些信号会全部记为 no_quote 买入失败，请确认代码格式与区间是否正确。",
            ratio * 100, sorted(unmatched)[:3],
        )
    return ratio


def load_picks(path: str | Path) -> pd.DataFrame:
    """读取选股长表（parquet / csv），归一化成 [date, code]。"""
    p = Path(path)
    frame = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
    return normalize_frame(frame)
