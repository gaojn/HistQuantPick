"""选股列表回测引擎：T 日信号 → T+entry_offset 买入 → 再持有 H 日卖出。

语义
----
- T 日（信号日）收盘后给出选股列表（可为 0~任意只，每日只数可变）；
- 买入日 = T + ``entry_offset``（默认 T+1），按 ``entry_price`` 成交，桶内等权；
- 卖出日 = 买入日 + ``hold_days``（默认再持有 2 日），按 ``exit_price`` 成交；
- 资金按槽位均摊（默认 slots 模式）：建仓金额 = 可用现金 / (N - 已占用槽数)。
  一次建仓占一个槽、卖光释放；全空时天然等分，只剩一个空槽时全投。
  N 与 H 解耦：H 管持有多久，N 管单次下多重。详见 hqpick.engine.capital。
- 同日执行次序由成交时点决定：卖出不晚于买入时先卖后买（回款当日复用），
  否则先买后卖。

执行规则（与 HistQuantOpt 同口径）
--------------------------------
- 买入日成交价涨停（raw ≥ limit_up × 99.9%）或停牌：放弃该票，资金留现金；
- 卖出日成交价跌停（raw ≤ limit_down × 100.1%）或停牌：顺延到下一可交易日；
- 退市（当日面板中该票整行消失，价格与交易状态同时缺失）：不论是否到期，
  当日按最近有效收盘价强制核销、扣卖出费；
- 卖出受阻超过 writeoff_stuck_days（默认 0=不启用）：按最近有效价强制核销，
  属**假设口径**（钱并未真的回笼），仅供敏感性分析；
- 估值用 ffill 后的复权收盘价，成交价不 ffill；
- 成本非对称：买入 ``cost_buy``、卖出 ``cost_sell``。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hqpick.constants import LIMIT_DOWN_TOL, LIMIT_UP_TOL, SUSPENDED
from hqpick.data.panel import WideFrames
from hqpick.engine.capital import NO_FREE_SLOT, make_book
from hqpick.engine.config import ExecConfig
from hqpick.engine.frames import AlignedFrames, align_frames
from hqpick.engine.state import Bucket, ReplayRecords, ReplayState
from hqpick.picks import check_code_overlap, normalize_frame

TRADE_COLUMNS = [
    "date", "signal_date", "code", "side", "price", "shares", "notional",
]


@dataclass(frozen=True)
class RunResult:
    """一次回测的原始产物（指标计算交给 analysis 层）。"""

    nav: pd.Series                 # 组合净值（起点 1.0）
    daily_ret: pd.Series
    turnover: pd.Series            # 成交日 → 双边成交额 / 当日总资产
    holding_counts: pd.Series
    trades: pd.DataFrame
    exec_stats: dict
    config: ExecConfig


def normalize_picks(
    picks: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    universe_codes=None,
) -> dict[pd.Timestamp, list[str]]:
    """长表 [date, code] → {信号日: 代码列表}。

    date 支持 datetime / 字符串 / YYYYMMDD 数值，code 统一成字符串；
    给了 ``universe_codes`` 时还会校验代码与行情面板对得上（防止缺后缀
    导致的静默空回测）。信号日必须落在交易日历内。
    """
    frame = normalize_frame(picks)
    if universe_codes is not None:
        check_code_overlap(frame["code"], universe_codes)

    missing = pd.DatetimeIndex(frame["date"].unique()).difference(calendar)
    if not missing.empty:
        preview = ", ".join(day.strftime("%Y-%m-%d") for day in missing[:5])
        suffix = f"（共 {len(missing)} 日）" if len(missing) > 5 else ""
        raise ValueError(f"信号日不在行情交易日历中: {preview}{suffix}")

    return {day: sorted(group["code"].tolist()) for day, group in frame.groupby("date")}


class PickBacktester:
    """选股列表 → 分桶重叠持仓回测。"""

    def __init__(self, config: ExecConfig | None = None) -> None:
        self.config = config or ExecConfig()

    def run(self, picks: pd.DataFrame, wide: WideFrames) -> RunResult:
        cfg = self.config
        picks_map = normalize_picks(
            picks, wide.dates, universe_codes=wide.adj["close"].columns
        )
        if not picks_map:
            raise ValueError("选股表为空，无法回测")

        frames = align_frames(wide, cfg, start=min(picks_map))
        state = ReplayState(equity=cfg.initial_value)
        state.book = make_book(cfg.capital_mode, cfg.buckets, cfg.initial_value)
        records = ReplayRecords(port_values=pd.Series(0.0, index=frames.dates))
        self._replay(frames, picks_map, state, records)

        nav = records.port_values / cfg.initial_value
        trades = pd.DataFrame(records.trades, columns=TRADE_COLUMNS)
        return RunResult(
            nav=nav,
            daily_ret=nav.pct_change(fill_method=None).fillna(0.0),
            turnover=pd.Series(records.turnover, name="turnover", dtype=float),
            holding_counts=pd.Series(
                records.holding_counts, name="holdings", dtype=float
            ),
            trades=trades,
            exec_stats=self._build_exec_stats(state, records, frames),
            config=cfg,
        )

    # ── 主循环 ─────────────────────────────────────────────

    def _replay(
        self,
        frames: AlignedFrames,
        picks_map: dict[pd.Timestamp, list[str]],
        state: ReplayState,
        records: ReplayRecords,
    ) -> None:
        cfg = self.config

        def do_buy(i: int) -> float:
            """建仓：entry_offset 个交易日前的信号在今日成交。"""
            if i < cfg.entry_offset:
                return 0.0
            signal_day = frames.dates[i - cfg.entry_offset]
            codes = picks_map.get(signal_day)
            if not codes:
                return 0.0
            return self._buy_bucket(frames, state, records, i, signal_day, codes)

        def do_sell(i: int) -> float:
            """退市核销 + 到期卖出（含此前顺延未卖出的）。"""
            return (
                self._force_sell_delisted(frames, state, records, i)
                + self._sell_due_buckets(frames, state, records, i)
            )

        # 同日执行次序由成交时点决定：卖出不晚于买入时，回款当日即可复用
        steps = (do_sell, do_buy) if cfg.sell_before_buy else (do_buy, do_sell)

        for i, day in enumerate(frames.dates):
            traded = sum(step(i) for step in steps)

            # 收盘估值
            equity = state.book.cash + self._holdings_value(frames, state, day)
            records.port_values.iloc[i] = equity
            records.holding_counts[day] = sum(
                len(b.holdings) for b in state.book.buckets
            )
            if equity > 1e-8:
                records.cash_ratios.append(state.book.cash / equity)
                records.frozen_ratios.append(
                    self._stuck_value(frames, state, day, i) / equity
                )
                if traded > 0:
                    records.turnover[day] = traded / equity
            state.book.on_close(i)
            state.equity = equity

        # 信号日太靠后、买入日超出回测区间的信号无法执行
        last_idx = len(frames.dates) - 1
        date_pos = {d: k for k, d in enumerate(frames.dates)}
        state.unexecutable_signals = [
            day for day in sorted(picks_map)
            if date_pos.get(day, -1) + cfg.entry_offset > last_idx
        ]

    def _buy_bucket(
        self,
        frames: AlignedFrames,
        state: ReplayState,
        records: ReplayRecords,
        i: int,
        signal_day: pd.Timestamp,
        codes: list[str],
    ) -> float:
        """按资金账本给出的预算建一桶，桶内等权。返回成交额。

        预算来源随 capital_mode 而异：slots 取「可用现金 / 剩余空槽数」，
        shared 取「总资产 / 桶数」并受现金约束。
        """
        cfg = self.config
        day = frames.dates[i]
        plan = state.book.plan(state.equity, i)
        if not plan.ok:
            if plan.reject == NO_FREE_SLOT:
                # 槽位全被占用（含卡仓）→ 跳过，账本内部已计数
                return 0.0
            # 现金耗尽（到期桶因跌停/停牌未能回款）→ 本期信号整桶跳过
            state.no_cash_skip_days += 1
            return 0.0
        if plan.target > 1e-8 and plan.budget < plan.target * 0.99:
            # 建仓不足：比整桶跳过更隐蔽，桶照建但金额远低于目标
            state.underfunded_buy_days += 1
            state.funding_gap_sum += 1.0 - plan.budget / plan.target
        alloc = plan.budget / len(codes)

        bucket = state.book.open_bucket(
            plan, signal_day=signal_day, buy_day=day, due_idx=i + cfg.hold_days
        )
        traded = 0.0
        px_adj_row = frames.entry_adj.loc[day]
        px_raw_row = frames.entry_raw.loc[day]
        limit_up = frames.limit_up.loc[day]
        status = frames.trade_status.loc[day]

        for code in codes:
            px_adj = px_adj_row.get(code)
            px_raw = px_raw_row.get(code)
            if pd.isna(px_adj) or pd.isna(px_raw) or px_adj <= 0:
                state.buy_fail_no_quote += 1
                continue
            if status.get(code) == SUSPENDED:
                state.buy_fail_suspended += 1
                continue
            up = limit_up.get(code)
            if pd.notna(up) and px_raw >= up * LIMIT_UP_TOL:
                state.buy_fail_limit_up += 1
                continue

            notional = alloc / (1.0 + cfg.cost_buy)   # 含费出资 = alloc
            shares = notional / float(px_adj)
            bucket.holdings[code] = bucket.holdings.get(code, 0.0) + shares
            state.book.charge(plan, alloc)
            traded += notional
            records.trades.append(
                {
                    "date": day, "signal_date": signal_day, "code": code,
                    "side": "buy", "price": float(px_adj),
                    "shares": shares, "notional": notional,
                }
            )

        state.book.discard_empty()
        return traded

    def _force_sell_delisted(
        self,
        frames: AlignedFrames,
        state: ReplayState,
        records: ReplayRecords,
        i: int,
    ) -> float:
        """退市核销：面板中整行消失的持仓按最近有效价强制卖出。

        与停牌的区别：停牌日面板仍有该票的行（trade_status='停牌'），走顺延；
        只有价格与交易状态**同时**缺失才判定为退市。
        """
        day = frames.dates[i]
        close_row = frames.adj_close_raw.loc[day]
        status_row = frames.trade_status.loc[day]
        marked = frames.adj_close_marked.loc[day]
        traded = 0.0

        for bucket in state.book.buckets:
            for code in list(bucket.holdings):
                if pd.notna(close_row.get(code)) or pd.notna(status_row.get(code)):
                    continue                      # 仍在面板中（含停牌）
                px = marked.get(code)
                if pd.isna(px) or px <= 0:
                    continue                      # 无任何历史有效价，无法核销
                shares = bucket.holdings.pop(code)
                notional = shares * float(px)
                state.book.credit(bucket, notional * (1.0 - self.config.cost_sell))
                state.delist_forced_count += 1
                traded += notional
                records.trades.append(
                    {
                        "date": day, "signal_date": bucket.signal_day, "code": code,
                        "side": "sell_delist", "price": float(px),
                        "shares": shares, "notional": notional,
                    }
                )

        state.book.discard_empty()
        return traded

    def _sell_due_buckets(
        self,
        frames: AlignedFrames,
        state: ReplayState,
        records: ReplayRecords,
        i: int,
    ) -> float:
        """卖出所有到期（due_idx ≤ i）的桶；跌停/停牌顺延到下一交易日。"""
        day = frames.dates[i]
        px_adj_row = frames.exit_adj.loc[day]
        px_raw_row = frames.exit_raw.loc[day]
        limit_down = frames.limit_down.loc[day]
        status = frames.trade_status.loc[day]
        traded = 0.0

        writeoff_days = self.config.writeoff_stuck_days
        marked = frames.adj_close_marked.loc[day]

        for bucket in state.book.buckets:
            if bucket.due_idx > i or not bucket.holdings:
                continue
            deferred = bucket.due_idx < i
            stuck_days = i - bucket.due_idx
            for code in list(bucket.holdings):
                px_adj = px_adj_row.get(code)
                px_raw = px_raw_row.get(code)
                blocked: str | None = None
                if pd.isna(px_adj) or px_adj <= 0:
                    blocked = "no_price"
                elif status.get(code) == SUSPENDED:
                    blocked = "suspended"
                else:
                    down = limit_down.get(code)
                    if pd.notna(down) and pd.notna(px_raw) and px_raw <= down * LIMIT_DOWN_TOL:
                        blocked = "limit_down"

                if blocked is not None:
                    # 卖出受阻超时 → 按最近有效价强制核销（假设口径，默认关闭）
                    if writeoff_days > 0 and stuck_days >= writeoff_days:
                        px_stuck = marked.get(code)
                        if pd.notna(px_stuck) and px_stuck > 0:
                            shares = bucket.holdings.pop(code)
                            notional = shares * float(px_stuck)
                            state.book.credit(
                                bucket, notional * (1.0 - self.config.cost_sell)
                            )
                            state.writeoff_forced_count += 1
                            traded += notional
                            records.trades.append(
                                {
                                    "date": day, "signal_date": bucket.signal_day,
                                    "code": code, "side": "sell_writeoff",
                                    "price": float(px_stuck), "shares": shares,
                                    "notional": notional,
                                }
                            )
                            continue
                    if blocked == "no_price":
                        state.sell_defer_no_price += 1
                    elif blocked == "suspended":
                        state.sell_defer_suspended += 1
                    else:
                        state.sell_defer_limit_down += 1
                    continue

                shares = bucket.holdings.pop(code)
                notional = shares * float(px_adj)
                state.book.credit(bucket, notional * (1.0 - self.config.cost_sell))
                traded += notional
                records.trades.append(
                    {
                        "date": day, "signal_date": bucket.signal_day, "code": code,
                        "side": "sell_deferred" if deferred else "sell",
                        "price": float(px_adj), "shares": shares, "notional": notional,
                    }
                )

        state.book.discard_empty()
        return traded

    @staticmethod
    def _bucket_value(frames: AlignedFrames, day, buckets: Iterable[Bucket]) -> float:
        """给定桶集合的持仓市值（ffill 后的复权收盘价估值，从不 ffill 成交价）。"""
        marked = frames.adj_close_marked.loc[day]
        value = 0.0
        for bucket in buckets:
            for code, shares in bucket.holdings.items():
                px = marked.get(code)
                if pd.notna(px):
                    value += shares * float(px)
        return value

    @classmethod
    def _stuck_value(cls, frames: AlignedFrames, state: ReplayState, day, i: int) -> float:
        """已过到期日却仍未卖出的持仓市值——被停牌/跌停冻结的那部分资金。

        判定口径须与 ``_sell_due_buckets`` 一致（``due_idx <= i`` 即到期）：
        到期当天若卖出尝试已失败（跌停/停牌），当天收盘时就该计入冻结，
        不能等到次日才计入——否则每次卡仓都会少算最早的一天。
        """
        due = (b for b in state.book.buckets if b.due_idx <= i)
        return cls._bucket_value(frames, day, due)

    @classmethod
    def _holdings_value(cls, frames: AlignedFrames, state: ReplayState, day) -> float:
        return cls._bucket_value(frames, day, state.book.buckets)

    def _build_exec_stats(
        self,
        state: ReplayState,
        records: ReplayRecords,
        frames: AlignedFrames,
    ) -> dict:
        final_nav = float(records.port_values.iloc[-1])
        last_day = frames.dates[-1]
        raw_last = frames.adj_close_raw.loc[last_day]
        marked_last = frames.adj_close_marked.loc[last_day]

        stuck_count = 0
        stuck_value = 0.0
        open_positions = 0
        for bucket in state.book.buckets:
            for code, shares in bucket.holdings.items():
                if shares <= 1e-10:
                    continue
                open_positions += 1
                if pd.isna(raw_last.get(code)) and pd.notna(marked_last.get(code)):
                    stuck_count += 1
                    stuck_value += shares * float(marked_last.get(code))

        cfg = self.config
        return {
            "config": cfg.label,
            "hold_days": cfg.hold_days,
            "entry_offset": cfg.entry_offset,
            "entry_price": cfg.entry_price,
            "exit_price": cfg.exit_price,
            "capital_mode": cfg.capital_mode,
            "n_buckets": cfg.buckets,
            "sell_before_buy": cfg.sell_before_buy,
            # 该口径下现金闲置的理论下限：卖出晚于买入时，永远有一桶回款在途
            "structural_cash_pct": 0.0 if cfg.sell_before_buy else 1.0 / cfg.buckets,
            "buy_fail_count": state.buy_fail_count,
            "buy_fail_breakdown": {
                "limit_up": state.buy_fail_limit_up,
                "suspended": state.buy_fail_suspended,
                "no_quote": state.buy_fail_no_quote,
            },
            "sell_defer_count": state.sell_defer_count,
            "sell_defer_breakdown": {
                "limit_down": state.sell_defer_limit_down,
                "suspended": state.sell_defer_suspended,
                "no_price": state.sell_defer_no_price,
            },
            "delist_forced_count": state.delist_forced_count,
            "writeoff_forced_count": state.writeoff_forced_count,
            "writeoff_stuck_days": cfg.writeoff_stuck_days,
            "avg_frozen_pct": (
                float(np.mean(records.frozen_ratios)) if records.frozen_ratios else 0.0
            ),
            "max_frozen_pct": (
                float(np.max(records.frozen_ratios)) if records.frozen_ratios else 0.0
            ),
            "no_cash_skip_days": state.no_cash_skip_days,
            "no_free_slot_days": getattr(state.book, "no_free_slot_days", 0),
            "underfunded_buy_days": state.underfunded_buy_days,
            "avg_funding_gap": (
                state.funding_gap_sum / state.underfunded_buy_days
                if state.underfunded_buy_days else 0.0
            ),
            "unexecutable_signal_days": [
                d.strftime("%Y-%m-%d") for d in state.unexecutable_signals
            ],
            "avg_cash_pct": (
                float(np.mean(records.cash_ratios)) if records.cash_ratios else 0.0
            ),
            "final_cash_pct": float(state.book.cash / final_nav) if final_nav > 1e-8 else 0.0,
            "avg_holding_count": (
                float(np.mean(list(records.holding_counts.values())))
                if records.holding_counts else 0.0
            ),
            "open_position_count": open_positions,
            "open_bucket_count": len(state.book.buckets),
            "delisted_stuck_count": stuck_count,
            "stale_value_pct": (
                float(stuck_value / final_nav) if final_nav > 1e-8 else 0.0
            ),
        }
