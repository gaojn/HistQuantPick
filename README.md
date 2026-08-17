# HistQuantPick

A 股**选股列表回测引擎**：每日选出任意只数的股票，T+1 成交、持有 H 个交易日卖出，
资金分桶重叠持仓。涨跌停、停牌、退市按贴近真实成交的规则处理。

面向的问题是「我每天有一份选股名单，按 T+1 开盘买、持有 H 天卖出，到底能赚多少」——
不做组合优化、不带风险模型，只回答执行层面的真实收益。

## 与其他仓库的边界

```
HistQuantFactor / HistQuantDEAP   挖掘候选因子
        ↓
HistQuantScreen                   单因子准入筛查（IC / decay / 换手）
        ↓
HistQuantFM / HistQuantCombo      多因子合成 → 打分
        ↓
   ┌────┴────────────────────────────────┐
HistQuantOpt                        HistQuantPick（本包）
组合优化：权重求解、风险模型、          选股列表回测：无优化器、
TE/行业/风格约束、指数增强              分桶重叠持仓、逐笔视角
```

与 HistQuantOpt 只共享**行情缓存目录**，没有代码依赖。执行规则（涨跌停容差、
停牌顺延、退市核销、非对称费率）与 opt 的 `RealisticBacktester` 同口径，
两边净值可直接比较。

## 安装

```bash
pip install -e .
```

需要 Python ≥ 3.10。行情缓存默认读 `~/quant_data/ashare_cache/ashare_daily_<year>.parquet`
（与 HistQuantOpt 共用同一份，不重复同步），可用环境变量 `HQPICK_CACHE` 覆盖。

## 快速开始

```bash
python3 -m pytest -q
```

生成一份选股列表，再回测：

```bash
python3 -m hqpick signal limit-up --start 2024-01-01 --end 2025-12-31 --out output/picks.parquet
```

```bash
python3 -m hqpick run --picks output/picks.parquet --hold-days 2 --start 2024-01-01 --end 2025-12-31
```

自带两个信号生成器：`limit-up`（T 日涨停）与 `random`（随机基线，用于判断
策略超额是否只是执行规则或市场 beta 的假象）。自有策略只需产出
`[date, code]` 长表（parquet/csv），`date` 是**信号日 T**。

## 口径

```
买入日 = T + entry_offset      默认 T+1，必须 ≥1（禁止 T 日成交）
卖出日 = 买入日 + hold_days     默认 T+3（H=2）
```

`hold_days=H` 表示买入日到卖出日之间正好跨 H 个交易日。买入价与卖出价
各自可选 `open / close / vwap`。

**资金默认按份独立滚动**（`--capital-mode isolated`）：切成 N 份互不透支，每份用满自己的
现金建仓、卖出只回自己账上，赚的那份下次投得多、亏的那份投得少；全部空仓时重新等分。
另有 `shared` 共享现金池模式。两者差异与实测对比见 [docs/method.md](docs/method.md#资金模式)。

**资金分桶数不一定等于 H**：开盘买、收盘卖时，卖出当日的回款赶不上当日开盘
建仓，资金要跨 H+1 个建仓时点，桶数自动取 H+1。详见 [docs/method.md](docs/method.md#资金分桶与资金利用率)——
这一节解释了为什么「1/H 资金分配」在开盘买/收盘卖口径下必然留下 1/(H+1) 的
现金闲置，以及怎么消除它。

## 产物

`output/<口径标签>/`：

| 文件 | 内容 |
|---|---|
| `nav.csv` | 净值与日收益 |
| `trades.csv` | 逐笔成交（含 side 区分正常卖出 / 顺延卖出 / 退市核销） |
| `metrics.json` | 年化、波动、Sharpe、回撤、超额 IR、跟踪误差 |
| `exec_stats.json` | 买入失败与卖出顺延的分类计数、现金占比、建仓不足诊断 |

## 文档

- [docs/操作指南.md](docs/操作指南.md) — 新成员必读，命令与参数
- [docs/method.md](docs/method.md) — 口径、执行规则、资金分桶
- [docs/design.md](docs/design.md) — 模块架构
