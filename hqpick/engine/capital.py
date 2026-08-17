"""资金管理：共享池（shared）与独立份额（isolated）两种模式。

shared
    所有建仓共用一个现金池，每日预算 = 前收盘总资产 / 桶数。盈亏在桶之间
    互相传导：某桶浮盈会抬高后续所有桶的建仓规模。

isolated（默认）
    资金切成 N 份互不透支的 sleeve，每份自己滚动：建仓时用该 sleeve 的
    **全部现金**（不留余额），卖出回笼也只进它自己账上。涨到 110 万的那份
    下次就投 110 万，跌到 90 万的那份下次就投 90 万。
    收盘时若全部 sleeve 都空仓，则把现金加总重新等分，重新开始滚动。
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
    owner: int | None = None      # sleeve 序号；shared 模式为 None
    reject: str | None = None     # 无法建仓时的原因

    @property
    def ok(self) -> bool:
        return self.reject is None and self.budget > 1e-8


@dataclass
class Sleeve:
    """一份独立滚动的资金。"""

    index: int
    cash: float
    bucket: Bucket | None = None   # None 表示空闲，可用于建仓
    free_since: int = -1           # 上次释放时的交易日序号，用于 FIFO 轮转

    @property
    def is_free(self) -> bool:
        """持仓已清空即视为空闲——先卖后买的口径下，当日卖出后可立即复用。"""
        return self.bucket is None or not self.bucket.holdings


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
        return BuyPlan(target=target, budget=budget, owner=None, reject=reject)

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


class IsolatedBook(CapitalBook):
    """独立份额：N 个 sleeve 各自滚动，空仓时重新等分。"""

    def __init__(self, n_slots: int, initial_value: float) -> None:
        super().__init__(n_slots, initial_value)
        share = float(initial_value) / n_slots
        self.sleeves = [Sleeve(index=i, cash=share) for i in range(n_slots)]
        self.no_free_slot_days = 0
        self.reset_count = 0
        self._day = 0

    @property
    def cash(self) -> float:
        return sum(s.cash for s in self.sleeves)

    @property
    def buckets(self) -> list[Bucket]:
        return [s.bucket for s in self.sleeves if s.bucket is not None]

    def plan(self, equity: float, day_idx: int) -> BuyPlan:
        free = [s for s in self.sleeves if s.is_free]
        if not free:
            # 到期桶因跌停/停牌卖不出，sleeve 仍被占用；不挪用其他 sleeve 的钱
            self.no_free_slot_days += 1
            return BuyPlan(target=0.0, budget=0.0, reject=NO_FREE_SLOT)

        # FIFO：优先启用空闲最久的一份，保证轮转均匀
        sleeve = min(free, key=lambda s: (s.free_since, s.index))
        budget = sleeve.cash          # 用满该份全部现金，不留余额
        reject = NO_CASH if budget <= 1e-8 else None
        return BuyPlan(target=budget, budget=budget, owner=sleeve.index, reject=reject)

    def open_bucket(self, plan, signal_day, buy_day, due_idx) -> Bucket:
        bucket = Bucket(
            signal_day=signal_day, buy_day=buy_day, due_idx=due_idx, sleeve=plan.owner
        )
        self.sleeves[plan.owner].bucket = bucket
        return bucket

    def charge(self, plan: BuyPlan, amount: float) -> None:
        self.sleeves[plan.owner].cash -= amount

    def credit(self, bucket: Bucket, amount: float) -> None:
        self.sleeves[bucket.sleeve].cash += amount

    def discard_empty(self) -> None:
        """持仓清空的 sleeve 立即解绑，当日即可再次建仓。"""
        for sleeve in self.sleeves:
            if sleeve.bucket is not None and not sleeve.bucket.holdings:
                sleeve.bucket = None
                sleeve.free_since = self._day     # 记录释放时点，供 FIFO 轮转

    def on_close(self, day_idx: int) -> None:
        self._day = day_idx
        self.discard_empty()

        # 全部空仓 → 现金加总重新等分，重新开始滚动
        if not all(s.is_free for s in self.sleeves):
            return
        total = self.cash
        if total <= 1e-8:
            return
        spread = max(s.cash for s in self.sleeves) - min(s.cash for s in self.sleeves)
        if spread <= 1e-6:
            return                       # 已经等分，无需重置（启动期不误计）
        share = total / self.n_slots
        for sleeve in self.sleeves:
            sleeve.cash = share
        self.reset_count += 1


def make_book(mode: str, n_slots: int, initial_value: float) -> CapitalBook:
    """按模式构造资金账本。"""
    if mode == "isolated":
        return IsolatedBook(n_slots, initial_value)
    if mode == "shared":
        return SharedBook(n_slots, initial_value)
    raise ValueError(f"capital_mode 须为 isolated / shared，当前为 {mode!r}")
