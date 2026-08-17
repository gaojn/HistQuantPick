"""逐日重放过程中的账户状态与执行统计。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Bucket:
    """一次信号建仓形成的持仓桶。"""

    signal_day: pd.Timestamp
    buy_day: pd.Timestamp
    due_idx: int                                              # 应卖出的交易日序号（含）
    holdings: dict[str, float] = field(default_factory=dict)  # code → 股数


@dataclass
class ReplayState:
    """逐日推进中的账户状态。资金归属由 ``book`` 负责，这里只放执行计数。"""

    equity: float                                   # 上一收盘总资产（shared 模式的预算基准）
    book: Any = None                                # CapitalBook，由 PickBacktester 注入

    buy_fail_suspended: int = 0      # 买入日停牌
    buy_fail_limit_up: int = 0       # 买入日成交价涨停
    buy_fail_no_quote: int = 0       # 买入日无行情（未上市/已退市/代码错误）
    sell_defer_limit_down: int = 0   # 卖出日跌停顺延（按被阻断的票·天计）
    sell_defer_suspended: int = 0    # 卖出日停牌顺延
    sell_defer_no_price: int = 0     # 行仍在但无有效价（长期停牌等）
    delist_forced_count: int = 0     # 退市强制核销笔数
    writeoff_forced_count: int = 0   # 卖出受阻超时强制核销笔数（假设口径）
    no_cash_skip_days: int = 0       # 现金耗尽导致整桶跳过的天数
    underfunded_buy_days: int = 0    # 现金不足、建仓金额低于目标预算的天数（shared）
    funding_gap_sum: float = 0.0     # 累计资金缺口 / 目标预算
    unexecutable_signals: list[pd.Timestamp] = field(default_factory=list)

    @property
    def buy_fail_count(self) -> int:
        return self.buy_fail_suspended + self.buy_fail_limit_up + self.buy_fail_no_quote

    @property
    def sell_defer_count(self) -> int:
        return (
            self.sell_defer_limit_down
            + self.sell_defer_suspended
            + self.sell_defer_no_price
        )


@dataclass
class ReplayRecords:
    """逐日重放产生的时间序列记录。"""

    port_values: pd.Series
    turnover: dict = field(default_factory=dict)              # 成交日 → 换手率
    cash_ratios: list[float] = field(default_factory=list)
    frozen_ratios: list[float] = field(default_factory=list)   # 卡仓持仓占净值比
    holding_counts: dict = field(default_factory=dict)        # 交易日 → 持仓票数
    trades: list[dict] = field(default_factory=list)
