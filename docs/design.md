# 架构

## 模块

```
hqpick/
├── constants.py           缓存路径、涨跌停容差、价格字段与日内时点次序
├── data/
│   └── panel.py           读 parquet 缓存 → 长表 / WideFrames 宽表集合
├── engine/
│   ├── config.py          ExecConfig：时点、价格、桶数推导、同日次序判定
│   ├── capital.py         资金账本：SlotBook（空槽均摊，默认）/ SharedBook（共享池）
│   ├── frames.py          按配置裁剪日期、选出 entry/exit 价格视图
│   ├── state.py           持仓桶、账户状态、执行计数器
│   └── replay.py          PickBacktester：逐日重放主循环
├── signals/
│   ├── limit_up.py        T 日涨停 / 连续 N 日涨停信号
│   ├── lower_shadow.py    下影线放量信号（截面打分取前 N）
│   ├── random_pick.py     随机基线信号
│   └── reversal.py        短期反转族因子（研究模块，未接入 CLI/__init__，见 docs/实验记录.md）
├── analysis/
│   ├── metrics.py         净值指标、等权基准
│   ├── trades.py          成交流水 → 逐笔完整交易，胜率/赔率/分组统计
│   ├── periodic.py        分年表现、月度收益矩阵、月份效应
│   ├── exposure.py        行业与市值分布（按买入日截面）
│   └── report.py          自包含 HTML 报告（内联 SVG 图表，零外部依赖）
├── picks.py               选股表入口：date/code 归一化与防呆校验
├── grid.py                参数网格扫描：笛卡尔积展开 + 随机基线对照
├── run.py                 编排：加载 → 回测 → 落盘
└── cli.py                 hqpick signal / run / grid
```

## 数据流

```
picks [date, code]  ─┐
                     ├─→ PickBacktester.run ─→ RunResult ─→ calc_metrics ─→ 产物
行情缓存 → WideFrames ┘                          (nav / trades / exec_stats)
```

`WideFrames` 持有三种成交价的复权/原始宽表加涨跌停与交易状态，`align_frames`
按 `ExecConfig` 从中挑出买入价与卖出价视图，引擎只认 `entry_*` / `exit_*`，
不关心具体是开盘还是收盘——新增价格口径只需扩 `PRICE_FIELDS` 与
`PRICE_TIME_ORDER`，引擎无改动。

## 主循环

逐交易日推进，每日四步：

1. **建仓** — `entry_offset` 个交易日前的信号在今日成交，预算由资金账本给出
   （slots 取 `现金/(N−占用槽数)`，shared 取 `min(总资产/桶数, 现金)`），桶内等权
2. **退市核销** — 面板整行消失的持仓按最近有效价强制卖出
3. **到期卖出** — `due_idx ≤ i` 的桶，跌停/停牌顺延
4. **收盘估值** — 现金 + 持仓市值（ffill 复权收盘价），随后触发账本收盘钩子

第 1 步与第 2/3 步的**先后由成交时点决定**：卖出时点不晚于买入时点时先卖后买
（回款当日可复用），否则先买后卖。见 `ExecConfig.sell_before_buy` 与
[method.md](method.md#资金分桶与资金利用率)。

资金归属通过 `CapitalBook` 抽象，引擎只调用 `plan / open_bucket / charge / credit`，
不关心钱从哪来——新增资金模式无需改主循环。

持仓以**桶**（`Bucket`）为单位组织，而非扁平持仓字典：同一只股票可能同时存在于
多个桶中（不同信号日买入、到期日不同），必须分别记账才能各自按期卖出。
估值与权重计算时再跨桶聚合。

## 设计约束

- **不依赖 HistQuantOpt 的代码**，只共享行情缓存目录。执行规则靠文档与测试对齐，
  不靠 import——避免 opt 的优化器/风险模型依赖被拖进来。
- **入口防呆**：选股表的两类静默失败（数值日期被当 Unix 戳、代码缺后缀）
  在 `picks.py` 统一拦截。静默跑出一条恒为 1 的净值曲线比报错危险得多。
- **前视防线在类型层**：`entry_offset ≥ 1` 在 `ExecConfig.__post_init__` 校验，
  构造非法配置直接报错，而不是回测跑完才发现。
- **执行失败必须可见**：买入失败、卖出顺延、建仓不足都分类计数进 `exec_stats`，
  不静默吞掉。净值好看但 `buy_fail` 巨大的回测是不可信的。
- **口径进目录名**：`ExecConfig.label` 编码全部关键参数，不同口径产物不互相覆盖。

## 测试

| 文件 | 覆盖 |
|---|---|
| `tests/helpers.py` | 合成行情构造器（价格视为已复权，便于手算校验） |
| `tests/test_timing.py` | 买卖日偏移、H 与 entry_offset 组合、前视拦截、末尾信号不可执行 |
| `tests/test_execution_rules.py` | 涨停买不进、跌停/停牌顺延、退市核销、停牌≠退市、非对称费率、每日只数可变 |
| `tests/test_buckets.py` | 桶数推导、同日次序、建仓不足诊断、卖出提前到开盘可满仓 |
| `tests/test_picks_io.py` | date 各形式（含 YYYYMMDD 数值）、混合格式、代码缺后缀的拦截 |
| `tests/test_reversal.py` | 反转因子口径（含当日/不含当日窗口）、截面过滤、universe 排除条件 |
| `tests/test_signals.py` | 信号口径与前视防线（均量不含当日、实体下沿、连板判定、各过滤条件、字段单位） |
| `tests/test_exposure.py` | 市值分档、按买入日截面取值、未匹配剔除、行业折叠 |
| `tests/test_periodic.py` | 月收益复利、矩阵缺月留空、超额口径、逐笔按买入年份归属、默认口径锁定 |
| `tests/test_trades.py` | 买卖配对、同票跨信号日不串配、费率、未平仓剔除、分组统计 |
| `tests/test_report.py` | 自包含性（无外部资源）、各节存在、跳过建仓与假设口径的警示、HTML 转义 |
| `tests/test_grid.py` | 网格展开、基线对照口径一致、N 与利用率的单调关系 |
| `tests/test_capital_slots.py` | 空槽均摊公式、最后一槽全投、全空等分、无空槽跳过、卡仓不锁死现金、超时核销、两模式差异 |
