"""资金管理：共享池（shared）与独立份额（isolated）两种模式。

shared
    所有建仓共用一个现金池，每日预算 = 前收盘总资产 / 桶数。盈亏在桶之间
    互相传导：某桶浮盈会抬高后续所有桶的建仓规模。

slots（默认）
    资金按槽位均摊：``budget = 可用现金 / (N - 已占用槽数)``。一次建仓占一个槽，
    卖光释放。全空时天然等分，只剩一个空槽时全投；某只票卡住只占住槽位，
    不锁死现金。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from hqpick.engine.state import Bucket

# 建仓被拒的原因
NO_CASH = "no_cash"
NO_FREE_SLOT = "no_free_slot"


@dataclass
class BuyPlan:
    """一次建仓的预算与归属。"""

    target: float                 # 目标预算
    budget: float                 # 实际可用预算
    reject: str | None = None     # 无法建仓时的原因

    @property
    def ok(self) -> bool:
        return self.reject is None and self.budget > 1e-8


class CapitalBook(ABC):
    """资金账本：决定每次建仓的预算来源，以及卖出回笼的去向。"""

    def __init__(self, n_slots: int, initial_value: float) -> None:
        self.n_slots = n_slots
        self.initial_value = initial_value

    @property
    @abstractmethod
    def cash(self) -> float:
        """当前总现金。"""

    @property
    @abstractmethod
    def buckets(self) -> list[Bucket]:
        """全部在场持仓桶。"""

    @abstractmethod
    def plan(self, equity: float, day_idx: int) -> BuyPlan:
        """规划本次建仓的预算与归属。"""

    @abstractmethod
    def open_bucket(
        self, plan: BuyPlan, signal_day: pd.Timestamp, buy_day: pd.Timestamp, due_idx: int
    ) -> Bucket:
        """按规划建一个空桶并挂进账本。"""

    @abstractmethod
    def charge(self, plan: BuyPlan, amount: float) -> None:
        """建仓扣款。"""

    @abstractmethod
    def credit(self, bucket: Bucket, amount: float) -> None:
        """卖出回笼资金到该桶所属的资金池。"""

    @abstractmethod
    def discard_empty(self) -> None:
        """清理已无持仓的桶。"""

    @abstractmethod
    def on_close(self, day_idx: int) -> None:
        """收盘钩子：释放已清空的资金份额、必要时重新等分。"""


class SharedBook(CapitalBook):
    """共享现金池：每日预算 = 前收盘总资产 / 桶数。"""

    def __init__(self, n_slots: int, initial_value: float) -> None:
        super().__init__(n_slots, initial_value)
        self._cash = float(initial_value)
        self._buckets: list[Bucket] = []

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def buckets(self) -> list[Bucket]:
        return self._buckets

    def plan(self, equity: float, day_idx: int) -> BuyPlan:
        target = equity / self.n_slots
        budget = min(target, self._cash)
        reject = NO_CASH if budget <= 1e-8 else None
        return BuyPlan(target=target, budget=budget, reject=reject)

    def open_bucket(self, plan, signal_day, buy_day, due_idx) -> Bucket:
        bucket = Bucket(signal_day=signal_day, buy_day=buy_day, due_idx=due_idx)
        self._buckets.append(bucket)
        return bucket

    def charge(self, plan: BuyPlan, amount: float) -> None:
        self._cash -= amount

    def credit(self, bucket: Bucket, amount: float) -> None:
        self._cash += amount

    def discard_empty(self) -> None:
        self._buckets = [b for b in self._buckets if b.holdings]

    def on_close(self, day_idx: int) -> None:
        """共享池无份额概念，收盘无需处理。"""


class SlotBook(CapitalBook):
    """按剩余空槽均摊可用现金（默认）。

    ``budget = 可用现金 / (N - 已占用槽数)``

    一次建仓占一个槽，卖光后释放。这一条式子自动覆盖了几种情形：

    - 全部空仓时 k=0 → ``现金/N``，天然等分，无需额外的重置逻辑；
    - 只剩一个空槽时 → 现金全投，赚到 110 万的那轮就投 110 万；
    - 某只票卡住（停牌/跌停卖不出）时该槽持续占用，但**现金不被锁死**，
      剩余资金照样均摊到其余空槽；
    - 先买后卖的口径下，当日到期尚未卖出的桶仍占槽，分母自然少一个，
      「回款在途」不必再单独建模。

    N 与持有期 H 解耦：H 决定持有多久，N 决定单次下多重。N < H+1 时槽位
    不够周转，会显式跳过建仓（计入 ``no_free_slot_days``）而不是悄悄错位。
    """

    def __init__(self, n_slots: int, initial_value: float) -> None:
        super().__init__(n_slots, initial_value)
        self._cash = float(initial_value)
        self._buckets: list[Bucket] = []
        self.no_free_slot_days = 0

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def buckets(self) -> list[Bucket]:
        return self._buckets

    @property
    def occupied(self) -> int:
        """已占用槽数 = 仍有持仓的桶数。"""
        return sum(1 for b in self._buckets if b.holdings)

    def plan(self, equity: float, day_idx: int) -> BuyPlan:
        free_slots = self.n_slots - self.occupied
        if free_slots <= 0:
            # 槽位全被占用（含卡仓）→ 本期信号跳过，不超配
            self.no_free_slot_days += 1
            return BuyPlan(target=0.0, budget=0.0, reject=NO_FREE_SLOT)
        budget = self._cash / free_slots
        reject = NO_CASH if budget <= 1e-8 else None
        return BuyPlan(target=budget, budget=budget, reject=reject)

    def open_bucket(self, plan, signal_day, buy_day, due_idx) -> Bucket:
        bucket = Bucket(signal_day=signal_day, buy_day=buy_day, due_idx=due_idx)
        self._buckets.append(bucket)
        return bucket

    def charge(self, plan: BuyPlan, amount: float) -> None:
        self._cash -= amount

    def credit(self, bucket: Bucket, amount: float) -> None:
        self._cash += amount

    def discard_empty(self) -> None:
        self._buckets = [b for b in self._buckets if b.holdings]

    def on_close(self, day_idx: int) -> None:
        """槽位在持仓卖光时即释放，收盘无需额外处理。"""
        self.discard_empty()


def make_book(mode: str, n_slots: int, initial_value: float) -> CapitalBook:
    """按模式构造资金账本。"""
    if mode == "slots":
        return SlotBook(n_slots, initial_value)
    if mode == "shared":
        return SharedBook(n_slots, initial_value)
    raise ValueError(f"capital_mode 须为 slots / shared，当前为 {mode!r}")
