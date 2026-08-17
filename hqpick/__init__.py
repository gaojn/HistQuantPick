"""HistQuantPick：A 股选股列表回测引擎。

T 日信号 → T+entry_offset 成交 → 再持有 hold_days 个交易日 → 卖出，资金分 H 桶重叠持仓。
涨跌停、停牌、退市按 HistQuantOpt 同口径处理。
"""

from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import PickBacktester, RunResult

__version__ = "0.1.0"
__all__ = ["ExecConfig", "PickBacktester", "RunResult"]
