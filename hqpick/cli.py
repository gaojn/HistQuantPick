"""HistQuantPick 命令行入口。

    hqpick signal limit-up --start 2020-01-01 --end 2026-06-30 --out picks.parquet
    hqpick signal random   --start 2020-01-01 --end 2026-06-30 --n 10 --out picks.parquet
    hqpick run --picks picks.parquet --hold-days 2 --start 2020-01-01 --end 2026-06-30
    hqpick grid --picks picks.parquet --n-buckets 3,5,10 --hold-days 1,2,5 --baseline random
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from hqpick.constants import (
    DEFAULT_COST_BUY,
    DEFAULT_COST_SELL,
    DEFAULT_INITIAL_VALUE,
    DEFAULT_OUT_DIR,
    DEFAULT_RISK_FREE,
)
from hqpick.engine.config import ExecConfig
from hqpick.grid import GridSpec, format_grid, run_grid
from hqpick.run import load_picks, run_backtest, save_artifacts
from hqpick.signals import build_limit_up_picks, build_random_picks

logger = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def cmd_signal(args: argparse.Namespace) -> None:
    if args.kind == "limit-up":
        picks = build_limit_up_picks(
            args.start, args.end,
            exclude_st=not args.include_st,
            min_list_days=args.min_list_days,
            min_amount=args.min_amount,
            max_per_day=args.max_per_day,
            cache_dir=args.cache_dir,
        )
    else:
        picks = build_random_picks(
            args.start, args.end,
            n_per_day=args.n,
            seed=args.seed,
            exclude_st=not args.include_st,
            min_list_days=args.min_list_days,
            cache_dir=args.cache_dir,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    picks.write_parquet(out) if out.suffix == ".parquet" else picks.write_csv(out)

    days = picks["date"].n_unique() if len(picks) else 0
    print(f"信号 {args.kind}: {len(picks)} 条, 覆盖 {days} 个交易日 → {out}")
    if days:
        print(f"日均选股数: {len(picks) / days:.1f}")


def cmd_run(args: argparse.Namespace) -> None:
    config = ExecConfig(
        hold_days=args.hold_days,
        entry_offset=args.entry_offset,
        entry_price=args.entry_price,
        exit_price=args.exit_price,
        n_buckets=args.n_buckets,
        capital_mode=args.capital_mode,
        writeoff_stuck_days=args.writeoff_stuck_days,
        cost_buy=args.cost_buy,
        cost_sell=args.cost_sell,
        initial_value=args.initial_value,
        risk_free=args.risk_free,
    )
    picks = load_picks(args.picks)
    result, metrics = run_backtest(
        picks, args.start, args.end, config=config, cache_dir=args.cache_dir
    )
    out_dir = Path(args.out_dir or (DEFAULT_OUT_DIR / config.label))
    save_artifacts(result, metrics, out_dir, picks=picks)

    print(f"\n口径: {config.label}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    print("\n执行统计:")
    print(json.dumps(result.exec_stats, ensure_ascii=False, indent=2, default=str))
    print(f"\n产物: {out_dir}")


def _int_list(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


def _str_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def cmd_grid(args: argparse.Namespace) -> None:
    picks = load_picks(args.picks)
    spec = GridSpec(
        hold_days=_int_list(args.hold_days),
        n_buckets=_int_list(args.n_buckets) if args.n_buckets else [None],
        entry_price=_str_list(args.entry_price),
        exit_price=_str_list(args.exit_price),
        capital_mode=_str_list(args.capital_mode),
        writeoff_stuck_days=_int_list(args.writeoff_stuck_days),
        fixed={
            "cost_buy": args.cost_buy,
            "cost_sell": args.cost_sell,
            "initial_value": args.initial_value,
            "risk_free": args.risk_free,
        },
    )

    baseline = None
    if args.baseline:
        days = picks["date"].nunique()
        n_per_day = max(1, round(len(picks) / days)) if days else 10
        logger.info("生成随机基线：日均 %d 只，与策略同口径对照", n_per_day)
        baseline = build_random_picks(
            args.start, args.end, n_per_day=n_per_day, seed=args.seed,
            cache_dir=args.cache_dir,
        ).to_pandas()

    frame = run_grid(
        picks, args.start, args.end, spec,
        baseline_picks=baseline, cache_dir=args.cache_dir,
    )

    out = Path(args.out or (DEFAULT_OUT_DIR / "grid_summary.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    print()
    print(format_grid(frame))
    print(f"\n共 {len(frame)} 个配置 → {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hqpick",
        description="A 股选股列表回测：T 日信号 → T+1 成交 → 持有 H 日 → 卖出，资金分 H 桶",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("signal", help="生成选股列表")
    ps.add_argument("kind", choices=["limit-up", "random"], help="信号类型")
    ps.add_argument("--start", type=_parse_date, required=True)
    ps.add_argument("--end", type=_parse_date, required=True)
    ps.add_argument("--out", required=True, help="输出 parquet/csv 路径")
    ps.add_argument("--n", type=int, default=10, help="random: 每日选股数")
    ps.add_argument("--seed", type=int, default=42, help="random: 随机种子")
    ps.add_argument("--max-per-day", type=int, default=None,
                    help="limit-up: 每日最多保留几只（按成交额降序）")
    ps.add_argument("--min-amount", type=float, default=0.0,
                    help="limit-up: 当日最低成交额（元）")
    ps.add_argument("--min-list-days", type=int, default=120, help="最少上市天数，剔次新")
    ps.add_argument("--include-st", action="store_true", help="不剔除 ST")
    ps.add_argument("--cache-dir", default=None, help="行情缓存目录，默认 HQPICK_CACHE")
    ps.set_defaults(func=cmd_signal)

    pr = sub.add_parser("run", help="选股列表 → 回测")
    pr.add_argument("--picks", required=True, help="选股长表 parquet/csv，列 [date, code]")
    pr.add_argument("--hold-days", type=int, default=2,
                    help="持有期 H：买入日后再持有 H 个交易日卖出，资金分 H 桶（默认 2）")
    pr.add_argument("--entry-offset", type=int, default=1,
                    help="信号日到买入日的交易日偏移，须 ≥1（默认 1，即 T+1）")
    pr.add_argument("--entry-price", choices=["open", "close", "vwap"], default="open")
    pr.add_argument("--exit-price", choices=["open", "close", "vwap"], default="close")
    pr.add_argument("--capital-mode", choices=["slots", "shared"], default="slots",
                    help="资金模式：slots=可用现金按剩余空槽均摊（默认）；"
                         "shared=共享现金池，每日按总资产/桶数重算")
    pr.add_argument("--writeoff-stuck-days", type=int, default=0,
                    help="卖出受阻超过该天数后按最近有效价强制核销（敏感性分析用，"
                         "钱并未真的回笼）；0=无限顺延，与真实约束一致")
    pr.add_argument("--n-buckets", type=int, default=None,
                    help="资金分桶数；默认按资金周转周期推导（开盘买/收盘卖为 H+1）")
    pr.add_argument("--start", type=_parse_date, required=True)
    pr.add_argument("--end", type=_parse_date, required=True)
    pr.add_argument("--cost-buy", type=float, default=DEFAULT_COST_BUY)
    pr.add_argument("--cost-sell", type=float, default=DEFAULT_COST_SELL)
    pr.add_argument("--initial-value", type=float, default=DEFAULT_INITIAL_VALUE)
    pr.add_argument("--risk-free", type=float, default=DEFAULT_RISK_FREE)
    pr.add_argument("--out-dir", default=None, help="产物目录，默认 output/<口径标签>")
    pr.add_argument("--cache-dir", default=None, help="行情缓存目录，默认 HQPICK_CACHE")
    pr.set_defaults(func=cmd_run)

    pg = sub.add_parser("grid", help="参数网格扫描（行情只加载一次）")
    pg.add_argument("--picks", required=True, help="选股长表 parquet/csv")
    pg.add_argument("--start", type=_parse_date, required=True)
    pg.add_argument("--end", type=_parse_date, required=True)
    pg.add_argument("--hold-days", default="2", help="逗号分隔，如 1,2,3,5")
    pg.add_argument("--n-buckets", default="", help="逗号分隔，如 3,5,10；留空=按 H 推导")
    pg.add_argument("--entry-price", default="open", help="逗号分隔，如 open,close")
    pg.add_argument("--exit-price", default="close", help="逗号分隔")
    pg.add_argument("--capital-mode", default="slots", help="逗号分隔，如 slots,shared")
    pg.add_argument("--writeoff-stuck-days", default="0", help="逗号分隔")
    pg.add_argument("--baseline", action="store_true",
                    help="同口径跑一遍随机基线做对照（日均只数与策略一致）")
    pg.add_argument("--seed", type=int, default=42, help="随机基线种子")
    pg.add_argument("--cost-buy", type=float, default=DEFAULT_COST_BUY)
    pg.add_argument("--cost-sell", type=float, default=DEFAULT_COST_SELL)
    pg.add_argument("--initial-value", type=float, default=DEFAULT_INITIAL_VALUE)
    pg.add_argument("--risk-free", type=float, default=DEFAULT_RISK_FREE)
    pg.add_argument("--out", default=None, help="汇总 csv 路径，默认 output/grid_summary.csv")
    pg.add_argument("--cache-dir", default=None)
    pg.set_defaults(func=cmd_grid)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
