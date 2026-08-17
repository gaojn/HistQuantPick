"""pytest fixtures。"""

from __future__ import annotations

import pytest

from hqpick.data.panel import WideFrames
from tests.helpers import make_wide


@pytest.fixture
def flat_market() -> WideFrames:
    """8 个交易日、2 只股票，价格恒为 10 的平价市场。"""
    dates = [f"2024-01-0{d}" for d in range(1, 9)]
    return make_wide(
        dates=dates,
        codes=["A", "B"],
        close={"A": [10.0] * 8, "B": [10.0] * 8},
    )
