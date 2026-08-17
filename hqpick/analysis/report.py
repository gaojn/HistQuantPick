"""自包含 HTML 回测报告：净值、执行质量、逐笔归因。

不依赖任何外部资源（无 CDN、无图片文件），图表用内联 SVG 画，
生成的单个 .html 可直接发给别人。
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hqpick.analysis.exposure import (
    attach_attributes,
    industry_stats,
    mv_bucket_stats,
    mv_summary,
)
from hqpick.analysis.periodic import (
    month_of_year_stats,
    monthly_matrix,
    yearly_stats,
)
from hqpick.analysis.trades import (
    build_round_trips,
    monthly_stats,
    open_positions,
    return_histogram,
    stats_by,
    trade_stats,
)

_CSS = """
:root { --ink:#1a1a1a; --muted:#6b6b6b; --line:#e2e2de; --bg:#fff;
        --pos:#1d9e75; --neg:#d84a3f; --accent:#2a78d6; }
* { box-sizing:border-box; }
body { margin:0; padding:32px 24px 64px; background:var(--bg); color:var(--ink);
       font:15px/1.65 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif; }
.wrap { max-width:1080px; margin:0 auto; }
h1 { font-size:22px; font-weight:600; margin:0 0 4px; }
h2 { font-size:17px; font-weight:600; margin:40px 0 12px; padding-bottom:6px;
     border-bottom:1px solid var(--line); }
h3 { font-size:14px; font-weight:600; margin:24px 0 8px; color:var(--muted); }
.sub { color:var(--muted); font-size:13px; margin:0 0 8px; }
.cards { display:grid; gap:10px; margin:16px 0;
         grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }
.card { border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.card .k { font-size:12px; color:var(--muted); }
.card .v { font-size:20px; font-weight:600; margin-top:2px;
           font-variant-numeric:tabular-nums; }
table { border-collapse:collapse; width:100%; font-size:13px; margin:8px 0 4px; }
th,td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line);
        font-variant-numeric:tabular-nums; white-space:nowrap; }
th { color:var(--muted); font-weight:500; }
th:first-child,td:first-child { text-align:left; }
.pos { color:var(--pos); } .neg { color:var(--neg); }
td.heat { font-weight:500; }
td.na { color:#c3c2b7; }
.scroll { overflow-x:auto; }
.note { background:#fbf7e8; border:1px solid #ecdca6; border-radius:8px;
        padding:10px 14px; font-size:13px; margin:16px 0; }
.warn { background:#fdecea; border-color:#f3bdb6; }
svg { display:block; max-width:100%; }
figure { margin:12px 0 0; }
figcaption { font-size:12px; color:var(--muted); margin-top:4px; }
"""


def _fmt_pct(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v * 100:.{digits}f}%"


def _fmt_num(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:,.{digits}f}"


def _signed(v, formatter=_fmt_pct) -> str:
    cls = "pos" if (v or 0) > 0 else "neg" if (v or 0) < 0 else ""
    return f'<span class="{cls}">{formatter(v)}</span>'


def _card(key: str, value: str) -> str:
    return (
        f'<div class="card"><div class="k">{html.escape(key)}</div>'
        f'<div class="v">{value}</div></div>'
    )


def _table(frame: pd.DataFrame, formatters: dict | None = None) -> str:
    if frame is None or frame.empty:
        return '<p class="sub">（无数据）</p>'
    formatters = formatters or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = []
    for idx, row in frame.iterrows():
        cells = [f"<td>{html.escape(str(idx))}</td>"]
        for col in frame.columns:
            fn = formatters.get(col, lambda v: html.escape(str(v)))
            cells.append(f"<td>{fn(row[col])}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    name = html.escape(str(frame.index.name or ""))
    return (
        f'<div class="scroll"><table><thead><tr><th>{name}</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _line_chart(
    series: dict[str, pd.Series],
    height: int = 240,
    width: int = 1040,
    colors: dict[str, str] | None = None,
    baseline: float | None = None,
) -> str:
    """多条时间序列折线图（内联 SVG）。"""
    series = {
        k: v.dropna() for k, v in series.items()
        if v is not None and len(v.dropna())
    }
    if not series:
        return ""
    colors = colors or {}
    pad_l, pad_r, pad_t, pad_b = 52, 12, 12, 26
    all_vals = pd.concat(list(series.values()))
    lo, hi = float(all_vals.min()), float(all_vals.max())
    if baseline is not None:
        lo, hi = min(lo, baseline), max(hi, baseline)
    if hi - lo < 1e-12:
        hi, lo = hi + 1, lo - 1
    span = hi - lo

    first = next(iter(series.values()))
    idx = first.index
    n = len(idx)

    def x(i: int) -> float:
        return pad_l + (width - pad_l - pad_r) * (i / max(n - 1, 1))

    def y(v: float) -> float:
        return pad_t + (height - pad_t - pad_b) * (1 - (v - lo) / span)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    # 横向网格与刻度
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = lo + span * frac
        yy = y(val)
        parts.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}" '
            f'stroke="#e2e2de" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{yy + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6b6b6b">{val:.2f}</text>'
        )
    if baseline is not None:
        yy = y(baseline)
        parts.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}" '
            f'stroke="#9b9b96" stroke-width="1" stroke-dasharray="4 3"/>'
        )
    # 日期刻度
    for frac in (0.0, 0.5, 1.0):
        i = int((n - 1) * frac)
        parts.append(
            f'<text x="{x(i):.1f}" y="{height - 8}" text-anchor="middle" '
            f'font-size="11" fill="#6b6b6b">{idx[i].strftime("%Y-%m")}</text>'
        )
    # 折线
    for name, s in series.items():
        pts = " ".join(
            f"{x(i):.1f},{y(float(v)):.1f}" for i, v in enumerate(s.to_numpy())
        )
        color = colors.get(name, "#2a78d6")
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round"/>'
        )
    parts.append("</svg>")

    legend = " ".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px">'
        f'<span style="width:10px;height:10px;border-radius:2px;background:'
        f'{colors.get(name, "#2a78d6")}"></span>{html.escape(name)}</span>'
        for name in series
    )
    return (
        '<figure><div style="font-size:12px;color:#6b6b6b;margin-bottom:6px">'
        f'{legend}</div>'
        f"{''.join(parts)}</figure>"
    )


def _histogram(counts: np.ndarray, edges: np.ndarray, height: int = 200,
               width: int = 1040) -> str:
    """单笔收益分布直方图，零轴左右分色。"""
    if len(counts) == 0:
        return ""
    pad_l, pad_r, pad_t, pad_b = 52, 12, 12, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    top = float(counts.max()) or 1.0
    bw = plot_w / len(counts)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" '
        f'y2="{pad_t + plot_h}" stroke="#c3c2b7" stroke-width="1"/>'
    )
    for i, c in enumerate(counts):
        h = plot_h * (float(c) / top)
        xx = pad_l + i * bw
        mid = (edges[i] + edges[i + 1]) / 2
        color = "#1d9e75" if mid > 0 else "#d84a3f"
        parts.append(
            f'<rect x="{xx + 0.5:.1f}" y="{pad_t + plot_h - h:.1f}" '
            f'width="{max(bw - 1, 1):.1f}" height="{h:.1f}" fill="{color}" rx="1"/>'
        )
    # 零轴
    zero_frac = (0 - edges[0]) / (edges[-1] - edges[0]) if edges[-1] > edges[0] else 0
    if 0 <= zero_frac <= 1:
        zx = pad_l + plot_w * zero_frac
        parts.append(
            f'<line x1="{zx:.1f}" y1="{pad_t}" x2="{zx:.1f}" y2="{pad_t + plot_h}" '
            f'stroke="#1a1a1a" stroke-width="1" stroke-dasharray="3 3"/>'
        )
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = edges[0] + (edges[-1] - edges[0]) * frac
        parts.append(
            f'<text x="{pad_l + plot_w * frac:.1f}" y="{height - 8}" '
            f'text-anchor="middle" font-size="11" fill="#6b6b6b">{val * 100:.1f}%</text>'
        )
    parts.append(
        f'<text x="{pad_l - 8}" y="{pad_t + 10}" text-anchor="end" font-size="11" '
        f'fill="#6b6b6b">{int(top)}</text>'
    )
    parts.append("</svg>")
    return f"<figure>{''.join(parts)}<figcaption>单笔净收益分布（笔数）</figcaption></figure>"


def _heat_cell(v, scale: float) -> str:
    """月度矩阵单元格：正绿负红，深浅按 |v|/scale。"""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '<td class="na">—</td>'
    v = float(v)
    if abs(v) < 1e-12:
        return '<td class="heat">0.00%</td>'
    alpha = min(abs(v) / scale, 1.0) * 0.42 if scale > 0 else 0.0
    rgb = "29,158,117" if v > 0 else "216,74,63"
    ink = "#0f6e56" if v > 0 else "#a32d2d"
    return (
        f'<td class="heat" style="background:rgba({rgb},{alpha:.3f});color:{ink}">'
        f"{v * 100:.2f}%</td>"
    )


def _matrix_table(matrix: pd.DataFrame) -> str:
    """月度收益矩阵，按数值上色。"""
    if matrix is None or matrix.empty:
        return '<p class="sub">（无数据）</p>'
    vals = matrix.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    scale = float(np.percentile(np.abs(finite), 90)) if len(finite) else 0.0
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in matrix.columns)
    rows = []
    for idx, row in matrix.iterrows():
        cells = "".join(_heat_cell(row[c], scale) for c in matrix.columns)
        rows.append(f"<tr><td>{html.escape(str(idx))}</td>{cells}</tr>")
    return (
        f'<div class="scroll"><table><thead><tr><th>年份</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _barh(labels: list[str], values: list[float], width: int = 1040,
          row_h: int = 22, color: str = "#2a78d6") -> str:
    """横向条形图：标签 + 条 + 数值，用于占比构成。"""
    if not labels:
        return ""
    pad_l, pad_r = 132, 56
    top = max(values) or 1.0
    height = row_h * len(labels) + 8
    plot_w = width - pad_l - pad_r

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for i, (lab, val) in enumerate(zip(labels, values, strict=False)):
        y = i * row_h + 4
        bar_w = plot_w * (float(val) / top)
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + row_h * 0.65:.0f}" text-anchor="end" '
            f'font-size="12" fill="#1a1a1a">{html.escape(str(lab))}</text>'
        )
        parts.append(
            f'<rect x="{pad_l}" y="{y + 3}" width="{max(bar_w, 1):.1f}" '
            f'height="{row_h - 8}" fill="{color}" rx="2"/>'
        )
        parts.append(
            f'<text x="{pad_l + bar_w + 6:.1f}" y="{y + row_h * 0.65:.0f}" '
            f'font-size="11" fill="#6b6b6b">{val * 100:.1f}%</text>'
        )
    parts.append("</svg>")
    return f"<figure>{''.join(parts)}</figure>"


@dataclass
class ReportInputs:
    """报告所需的全部数据。"""

    nav: pd.Series
    daily_ret: pd.Series
    metrics: dict
    exec_stats: dict
    trades: pd.DataFrame
    config_label: str
    bm_nav: pd.Series | None = None
    picks: pd.DataFrame | None = None
    attributes: pd.DataFrame | None = None    # [date, code, float_mv, industry_l1]
    title: str = "选股回测报告"
    cost_buy: float = 0.0
    cost_sell: float = 0.0


def _drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def _section_overview(inp: ReportInputs) -> str:
    m = inp.metrics
    cards = [
        _card("年化收益", _signed(m.get("ann_return"))),
        _card("年化波动", _fmt_pct(m.get("ann_vol"))),
        _card("Sharpe", _fmt_num(m.get("sharpe"))),
        _card("最大回撤", _fmt_pct(m.get("max_drawdown"))),
        _card("超额 IR", _fmt_num(m.get("information_ratio"))),
        _card("年化超额", _signed(m.get("ann_excess_return"))),
        _card("日胜率", _fmt_pct(m.get("win_rate_daily"), 1)),
        _card("交易日数", f"{m.get('n_days', 0)}"),
    ]
    return f'<div class="cards">{"".join(cards)}</div>'


def _section_nav(inp: ReportInputs) -> str:
    series = {"策略净值": inp.nav}
    colors = {"策略净值": "#2a78d6", "基准（全市场等权）": "#9b9b96", "回撤": "#d84a3f"}
    if inp.bm_nav is not None and len(inp.bm_nav):
        series["基准（全市场等权）"] = inp.bm_nav
    chart = _line_chart(series, colors=colors, baseline=1.0)
    dd = _drawdown(inp.nav)
    dd_chart = _line_chart({"回撤": dd}, height=140, colors=colors, baseline=0.0)
    return chart + dd_chart


def _section_execution(inp: ReportInputs) -> str:
    st = inp.exec_stats
    util = 1.0 - st.get("avg_cash_pct", 0.0)
    cards = [
        _card("资金利用率", _fmt_pct(util, 1)),
        _card("平均持仓数", f"{st.get('avg_holding_count', 0):.0f}"),
        _card("槽位数 N", f"{st.get('n_buckets', '—')}"),
        _card("无空槽跳过", f"{st.get('no_free_slot_days', 0)} 天"),
        _card("冻结市值均值", _fmt_pct(st.get("avg_frozen_pct", 0.0), 2)),
        _card("买入失败", f"{st.get('buy_fail_count', 0)} 笔"),
        _card("卖出顺延", f"{st.get('sell_defer_count', 0)} 票·天"),
        _card("退市核销", f"{st.get('delist_forced_count', 0)} 笔"),
    ]
    bf = st.get("buy_fail_breakdown", {})
    sd = st.get("sell_defer_breakdown", {})
    detail = pd.DataFrame(
        {
            "买入失败（笔）": [
                bf.get("limit_up", 0), bf.get("suspended", 0), bf.get("no_quote", 0), "—",
            ],
            "卖出受阻（票·天）": [
                sd.get("limit_down", 0), sd.get("suspended", 0), "—", sd.get("no_price", 0),
            ],
        },
        index=["涨停/跌停", "停牌", "无行情", "无有效价"],
    )
    detail.index.name = "原因"

    notes = []
    if st.get("no_free_slot_days", 0) > 0:
        notes.append(
            f'<div class="note">槽位被占满导致 <b>{st["no_free_slot_days"]}</b> 天的信号未能建仓，'
            f"实际仓位低于设定。与其他配置比较前请先对齐这个数字，否则比的是仓位不是选股。</div>"
        )
    if st.get("writeoff_forced_count", 0) > 0:
        notes.append(
            f'<div class="note warn">本次启用了卡仓超时核销（{st.get("writeoff_stuck_days")} 天），'
            f'强制核销 <b>{st["writeoff_forced_count"]}</b> 笔。这是<b>假设口径</b>——'
            f"资金并未真的回笼，结论不可直接用于实盘判断。</div>"
        )
    if st.get("stale_value_pct", 0) > 0.02:
        notes.append(
            '<div class="note warn">末日有 '
            f'{_fmt_pct(st["stale_value_pct"])} 的净值靠陈旧价估值。</div>'
        )
    return f'<div class="cards">{"".join(cards)}</div>' + "".join(notes) + _table(detail)


def _section_trades(inp: ReportInputs) -> str:
    rt = build_round_trips(
        inp.trades, inp.cost_buy, inp.cost_sell, calendar=inp.nav.index
    )
    st = trade_stats(rt)
    if not st.get("n_trades"):
        return '<p class="sub">（本次回测没有完整的往返交易）</p>'

    op = open_positions(inp.trades)
    cards = [
        _card("完整交易", f"{st['n_trades']:,}"),
        _card("胜率", _fmt_pct(st["win_rate"], 1)),
        _card("平均单笔", _signed(st["avg_ret"])),
        _card("中位单笔", _signed(st["median_ret"])),
        _card("平均盈利", _signed(st["avg_win"])),
        _card("平均亏损", _signed(st["avg_loss"])),
        _card("盈亏比", _fmt_num(st["payoff_ratio"])),
        _card("期望值", _signed(st["expectancy"])),
    ]
    dist = pd.DataFrame(
        {"单笔净收益": [st["worst"], st["p05"], st["p25"], st["median_ret"],
                     st["p75"], st["p95"], st["best"]]},
        index=["最差", "5%", "25%", "中位", "75%", "95%", "最好"],
    )
    dist.index.name = "分位"

    counts, edges = return_histogram(rt)
    by_kind = stats_by(rt, "exit_kind")
    by_month = monthly_stats(rt)

    pct = lambda v: _signed(v)                                      # noqa: E731
    pct1 = lambda v: _fmt_pct(v, 1)                                 # noqa: E731
    money = lambda v: _signed(v, lambda x: _fmt_num(x, 0))          # noqa: E731
    fmt = {"win_rate": pct1, "avg_ret": pct, "median_ret": pct, "total_pnl": money}

    open_note = (
        f'<div class="note">另有 <b>{len(op)}</b> 笔买入在区间末尚未平仓，'
        f"其浮动盈亏计入净值但不进逐笔统计。</div>" if len(op) else ""
    )
    return (
        f'<div class="cards">{"".join(cards)}</div>'
        + open_note
        + _histogram(counts, edges)
        + "<h3>分位分布</h3>" + _table(dist, {"单笔净收益": pct})
        + "<h3>按卖出方式</h3>" + _table(by_kind, fmt)
        + "<h3>按买入月份</h3>" + _table(by_month, fmt)
    )


def _section_periodic(inp: ReportInputs) -> str:
    rt = build_round_trips(
        inp.trades, inp.cost_buy, inp.cost_sell, calendar=inp.nav.index
    )
    bm_ret = inp.bm_nav.pct_change().fillna(0.0) if inp.bm_nav is not None else None
    yearly = yearly_stats(
        inp.daily_ret, bm_ret, rt, inp.metrics.get("risk_free", 0.02)
    )
    matrix = monthly_matrix(inp.daily_ret, bm_ret)
    moy = month_of_year_stats(inp.daily_ret)

    pct = lambda v: _signed(v)                                      # noqa: E731
    pct1 = lambda v: _fmt_pct(v, 1)                                 # noqa: E731
    num = lambda v: _fmt_num(v)                                     # noqa: E731
    count = lambda v: f"{int(v):,}" if pd.notna(v) else "—"         # noqa: E731
    yearly_fmt = {
        "收益": pct, "基准": pct, "超额": pct, "波动": pct1,
        "最大回撤": pct1, "Sharpe": num, "日胜率": pct1,
        "交易日": count, "交易笔数": count,
        "逐笔胜率": pct1, "平均单笔": pct,
    }
    moy_fmt = {
        "平均月收益": pct, "中位月收益": pct, "为正比例": pct1,
        "最好": pct, "最差": pct, "样本年数": count,
    }
    return (
        "<h3>分年表现</h3>" + _table(yearly, yearly_fmt)
        + "<h3>月度收益矩阵</h3>" + _matrix_table(matrix)
        + '<h3>月份效应</h3><p class="sub">每个自然月份只有很少几年的样本，'
        + "只用来看收益是否集中在少数月份，不能据此下季节性结论。</p>"
        + _table(moy, moy_fmt)
    )


def _section_exposure(inp: ReportInputs) -> str:
    if inp.attributes is None or inp.attributes.empty:
        return (
            '<p class="sub">（未提供行业/市值数据，跳过该节。'
            "命令行运行时会自动加载。）</p>"
        )
    rt = build_round_trips(
        inp.trades, inp.cost_buy, inp.cost_sell, calendar=inp.nav.index
    )
    if rt.empty:
        return '<p class="sub">（没有完整的往返交易）</p>'

    rt = attach_attributes(rt, inp.attributes)
    ind = industry_stats(rt, top=15)
    mv = mv_bucket_stats(rt)
    summary = mv_summary(rt)

    pct = lambda v: _signed(v)                                      # noqa: E731
    pct1 = lambda v: _fmt_pct(v, 1)                                 # noqa: E731
    money = lambda v: _signed(v, lambda x: _fmt_num(x, 0))          # noqa: E731
    count = lambda v: f"{int(v):,}" if pd.notna(v) else "—"         # noqa: E731
    fmt = {
        "交易笔数": count, "笔数占比": pct1, "资金占比": pct1,
        "逐笔胜率": pct1, "平均单笔": pct, "合计盈亏": money,
    }

    cards = ""
    if summary:
        cards = '<div class="cards">' + "".join(
            _card(k, f"{v:,.0f} 亿")
            for k, v in (
                ("市值中位", summary["中位"]), ("市值均值", summary["均值"]),
                ("5% 分位", summary["p05"]), ("25% 分位", summary["p25"]),
                ("75% 分位", summary["p75"]), ("95% 分位", summary["p95"]),
            )
        ) + "</div>"

    ind_chart = ""
    if not ind.empty:
        ind_chart = _barh(list(ind.index), ind["资金占比"].tolist())
    mv_chart = ""
    if not mv.empty:
        mv_chart = _barh(list(mv.index), mv["资金占比"].tolist(), color="#eb6834")

    missing = ""
    n_missing = int(rt["industry_l1"].isna().sum())
    if n_missing:
        missing = (
            f'<div class="note">有 {n_missing} 笔交易未匹配到行业/市值'
            f"（占 {n_missing / len(rt):.1%}），已从本节统计中剔除。</div>"
        )
    return (
        "<h3>流通市值分布</h3>" + cards + mv_chart + _table(mv, fmt)
        + "<h3>行业分布（按资金占比）</h3>" + ind_chart + _table(ind, fmt)
        + missing
    )


def _section_config(inp: ReportInputs) -> str:
    st = inp.exec_stats
    rows = {
        "口径": st.get("config", inp.config_label),
        "持有期 H": st.get("hold_days"),
        "买入": f"T+{st.get('entry_offset')} {st.get('entry_price')}",
        "卖出": (
            f"T+{st.get('entry_offset', 0) + st.get('hold_days', 0)} "
            f"{st.get('exit_price')}"
        ),
        "资金模式": st.get("capital_mode"),
        "槽位数 N": st.get("n_buckets"),
        "同日先卖后买": "是" if st.get("sell_before_buy") else "否",
        "卡仓核销阈值": (
            f"{st['writeoff_stuck_days']} 天"
            if st.get("writeoff_stuck_days") else "关闭（无限顺延）"
        ),
        "基准": inp.metrics.get("benchmark", "—"),
    }
    if inp.picks is not None and len(inp.picks):
        days = inp.picks["date"].nunique()
        rows["信号"] = (
            f"{len(inp.picks):,} 条 / {days} 个交易日"
            f"（日均 {len(inp.picks) / days:.1f} 只）"
        )
    frame = pd.DataFrame({"值": list(rows.values())}, index=list(rows.keys()))
    frame.index.name = "参数"
    return _table(frame)


def render_report(inp: ReportInputs) -> str:
    """生成完整 HTML 字符串。"""
    period = (
        f"{inp.nav.index[0].strftime('%Y-%m-%d')} ~ "
        f"{inp.nav.index[-1].strftime('%Y-%m-%d')}"
        if len(inp.nav) else "—"
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(inp.title)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{html.escape(inp.title)}</h1>
<p class="sub">{html.escape(period)} · {html.escape(inp.config_label)}</p>
<h2>净值表现</h2>
{_section_overview(inp)}
{_section_nav(inp)}
<h2>分年与分月</h2>
{_section_periodic(inp)}
<h2>执行质量</h2>
{_section_execution(inp)}
<h2>逐笔归因</h2>
{_section_trades(inp)}
<h2>交易分布</h2>
{_section_exposure(inp)}
<h2>回测口径</h2>
{_section_config(inp)}
</div></body></html>
"""


def save_report(inp: ReportInputs, path: str | Path) -> Path:
    """渲染并写出 HTML 文件。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(inp), encoding="utf-8")
    return out
