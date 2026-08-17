"""回测执行配置。"""

from __future__ import annotations

from dataclasses import dataclass

from hqpick.constants import (
    DEFAULT_COST_BUY,
    DEFAULT_COST_SELL,
    DEFAULT_INITIAL_VALUE,
    DEFAULT_RISK_FREE,
    PRICE_FIELDS,
    PRICE_TIME_ORDER,
)


@dataclass(frozen=True)
class ExecConfig:
    """执行口径。

    时点语义（signal_idx = 信号日 T 在交易日历中的序号）::

        买入日 = T + entry_offset            默认 T+1
        卖出日 = 买入日 + hold_days           默认 T+1+H

    即 ``hold_days=H`` 表示买入日到卖出日之间正好跨 H 个交易日；
    H=2、entry_price=open、exit_price=close 即「T+1 开盘买、T+3 收盘卖」。

    资金分桶
    --------
    每日建仓预算 = 前收盘总资产 / ``n_buckets``。桶数默认由资金周转周期推导，
    **不一定等于 H**：一笔钱从买入时点占用到卖出时点，若卖出时点晚于买入时点
    （如开盘买、收盘卖），卖出当日的建仓已经用不上这笔回款，于是资金要跨
    H+1 个建仓时点，桶数应为 H+1；只有当卖出时点不晚于买入时点（如开盘买、
    开盘卖）时，回款能当日复用，桶数才等于 H。

    按 H 分桶而回款跨 H+1 天，会在启动阶段错位一次并永久欠一桶资金，
    退化成断续的部分建仓。要严格按 1/H 分配可显式传 ``n_buckets=H``。

    Attributes
    ----------
    hold_days    : 持有期 H（交易日）
    entry_offset : 信号日到买入日的交易日偏移，必须 ≥1（禁止 T 日成交，防前视）
    entry_price  : 买入成交价字段 open / close / vwap
    exit_price   : 卖出成交价字段 open / close / vwap
    n_buckets    : 资金分桶数；None 表示按上述规则自动推导
    capital_mode : isolated（默认，各份独立滚动、空仓重新等分）/ shared（共享现金池）
    cost_buy     : 买入费率（默认 1‰）
    cost_sell    : 卖出费率（默认 2‰，含印花税）
    initial_value: 初始资金（元）
    risk_free    : 年化无风险利率（Sharpe 用）
    """

    hold_days: int = 2
    entry_offset: int = 1
    entry_price: str = "open"
    exit_price: str = "close"
    n_buckets: int | None = None
    capital_mode: str = "isolated"
    cost_buy: float = DEFAULT_COST_BUY
    cost_sell: float = DEFAULT_COST_SELL
    initial_value: float = DEFAULT_INITIAL_VALUE
    risk_free: float = DEFAULT_RISK_FREE

    def __post_init__(self) -> None:
        if self.hold_days < 1:
            raise ValueError(f"hold_days 须 ≥ 1，当前为 {self.hold_days}")
        if self.entry_offset < 1:
            raise ValueError(
                f"entry_offset 须 ≥ 1（T 日信号最早只能 T+1 成交），当前为 {self.entry_offset}"
            )
        for name, value in (("entry_price", self.entry_price), ("exit_price", self.exit_price)):
            if value not in PRICE_FIELDS:
                raise ValueError(f"{name} 须为 {sorted(PRICE_FIELDS)} 之一，当前为 {value!r}")
        if self.capital_mode not in ("isolated", "shared"):
            raise ValueError(
                f"capital_mode 须为 isolated / shared，当前为 {self.capital_mode!r}"
            )
        if self.n_buckets is not None and self.n_buckets < 1:
            raise ValueError(f"n_buckets 须 ≥ 1，当前为 {self.n_buckets}")
        if not 0.0 <= self.cost_buy < 1.0 or not 0.0 <= self.cost_sell < 1.0:
            raise ValueError("费率须落在 [0, 1) 区间")
        if self.initial_value <= 0:
            raise ValueError("initial_value 须为正数")

    @property
    def sell_before_buy(self) -> bool:
        """同一交易日是否先卖后买（卖出回款当日即可用于建仓）。

        仅当卖出时点严格早于、或同为集合竞价时点（open/open、close/close）时成立。
        vwap 是全天均价，无法保证卖在买前，一律保守地按先买后卖处理。
        """
        entry_order = PRICE_TIME_ORDER[self.entry_price]
        exit_order = PRICE_TIME_ORDER[self.exit_price]
        if exit_order < entry_order:
            return True
        return exit_order == entry_order and self.exit_price != "vwap"

    @property
    def buckets(self) -> int:
        """实际使用的资金分桶数。"""
        if self.n_buckets is not None:
            return self.n_buckets
        return self.hold_days if self.sell_before_buy else self.hold_days + 1

    @property
    def exit_offset(self) -> int:
        """信号日到卖出日的交易日偏移。"""
        return self.entry_offset + self.hold_days

    @property
    def label(self) -> str:
        """人类可读的口径标签，如 ``T+1open_hold2_T+3close_b3``。"""
        return (
            f"T+{self.entry_offset}{self.entry_price}"
            f"_hold{self.hold_days}"
            f"_T+{self.exit_offset}{self.exit_price}"
            f"_b{self.buckets}{'i' if self.capital_mode == 'isolated' else 's'}"
        )
