"""全局常量与默认路径。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "output"

# 行情只读本地 ashare 年片：默认 workspace 的 cache/data（与本包同级目录）。
# 仅共享数据目录，不 import 其他项目代码。可用 HQPICK_CACHE 覆盖。
_DEFAULT_CACHE = ROOT.parent / "cache" / "data"
DEFAULT_CACHE_DIR = Path(
    os.environ.get("HQPICK_CACHE", str(_DEFAULT_CACHE))
).expanduser()

# 涨跌停判断容差（与 HistQuantOpt ExecutionLedger 同口径）
LIMIT_UP_TOL = 0.999
LIMIT_DOWN_TOL = 1.001

# trade_status 取值中唯一不可成交的状态；XD/XR/N 均可正常交易
SUSPENDED = "停牌"

# 成交价字段：逻辑名 → (复权列, 原始列)
PRICE_FIELDS: dict[str, tuple[str, str]] = {
    "open": ("adj_open", "open"),
    "close": ("adj_close", "close"),
    "vwap": ("adj_vwap", "vwap"),
}

# 日内时点先后：决定同一交易日「先卖后买」还是「先买后卖」。
# vwap 是全天均价、不对应单一时点，故不与自身比较（见 ExecConfig.sell_before_buy）。
PRICE_TIME_ORDER: dict[str, int] = {"open": 0, "vwap": 1, "close": 2}

DEFAULT_COST_BUY = 0.001
DEFAULT_COST_SELL = 0.002
DEFAULT_RISK_FREE = 0.02
DEFAULT_INITIAL_VALUE = 1e8

TRADING_DAYS_PER_YEAR = 242
