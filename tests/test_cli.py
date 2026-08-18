"""命令行编排：锁定 signal/run/grid 三个子命令把参数正确传给下层函数。"""

from __future__ import annotations

import pandas as pd
import polars as pl

from hqpick import cli
from hqpick.engine.config import ExecConfig
from hqpick.engine.replay import RunResult


def test_signal_cli_random_forwards_args_and_writes_parquet(monkeypatch, tmp_path):
    captured = {}

    def fake_build_random_picks(start, end, **kwargs):
        captured["start"] = start
        captured["end"] = end
        captured.update(kwargs)
        return pl.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "code": ["A", "B"],
        })

    monkeypatch.setattr(cli, "build_random_picks", fake_build_random_picks)

    out = tmp_path / "picks.parquet"
    args = cli.build_parser().parse_args([
        "signal", "random", "--start", "2024-01-01", "--end", "2024-01-10",
        "--n", "7", "--seed", "3", "--min-list-days", "60", "--out", str(out),
    ])
    args.func(args)

    assert captured["n_per_day"] == 7
    assert captured["seed"] == 3
    assert captured["min_list_days"] == 60
    assert captured["exclude_st"] is True
    assert out.exists()
    assert pl.read_parquet(out)["code"].to_list() == ["A", "B"]


def test_signal_cli_lower_shadow_forwards_args_and_defaults_min_amount(monkeypatch, tmp_path):
    captured = {}

    def fake_build_lower_shadow_picks(start, end, **kwargs):
        captured["start"] = start
        captured["end"] = end
        captured.update(kwargs)
        return pl.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "code": ["A"],
        })

    monkeypatch.setattr(cli, "build_lower_shadow_picks", fake_build_lower_shadow_picks)

    out = tmp_path / "picks.parquet"
    args = cli.build_parser().parse_args([
        "signal", "lower-shadow", "--start", "2024-01-01", "--end", "2024-01-10",
        "--n", "5", "--volume-window", "10", "--min-shadow", "0.03",
        "--min-volume-ratio", "1.5", "--min-float-mv", "20", "--out", str(out),
    ])
    args.func(args)

    assert captured["max_per_day"] == 5
    assert captured["volume_window"] == 10
    assert captured["min_shadow"] == 0.03
    assert captured["min_volume_ratio"] == 1.5
    assert captured["min_float_mv"] == 20
    assert captured["exclude_st"] is True
    # --min-amount 未指定（默认 0.0）时，lower-shadow 兜底成 5000 万元
    assert captured["min_amount"] == 5e4
    assert out.exists()


def test_signal_cli_lower_shadow_uses_explicit_min_amount(monkeypatch, tmp_path):
    captured = {}

    def fake_build_lower_shadow_picks(start, end, **kwargs):
        captured.update(kwargs)
        return pl.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "code": ["A"]})

    monkeypatch.setattr(cli, "build_lower_shadow_picks", fake_build_lower_shadow_picks)

    out = tmp_path / "picks.parquet"
    args = cli.build_parser().parse_args([
        "signal", "lower-shadow", "--start", "2024-01-01", "--end", "2024-01-10",
        "--min-amount", "1e5", "--include-st", "--out", str(out),
    ])
    args.func(args)

    assert captured["min_amount"] == 1e5
    assert captured["exclude_st"] is False


def test_signal_cli_limit_up_forwards_args(monkeypatch, tmp_path):
    captured = {}

    def fake_build_limit_up_picks(start, end, **kwargs):
        captured["start"] = start
        captured["end"] = end
        captured.update(kwargs)
        return pl.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "code": ["A", "B"],
        })

    monkeypatch.setattr(cli, "build_limit_up_picks", fake_build_limit_up_picks)

    out = tmp_path / "picks.parquet"
    args = cli.build_parser().parse_args([
        "signal", "limit-up", "--start", "2024-01-01", "--end", "2024-01-10",
        "--consecutive", "2", "--max-per-day", "30", "--min-amount", "8e4",
        "--min-list-days", "90", "--out", str(out),
    ])
    args.func(args)

    assert captured["consecutive"] == 2
    assert captured["max_per_day"] == 30
    assert captured["min_amount"] == 8e4
    assert captured["min_list_days"] == 90
    assert captured["exclude_st"] is True
    assert out.exists()
    assert pl.read_parquet(out)["code"].to_list() == ["A", "B"]


def test_run_cli_builds_config_and_saves_artifacts(monkeypatch, tmp_path):
    picks = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "code": ["A"]})
    captured = {}

    monkeypatch.setattr(cli, "load_picks", lambda _: picks)

    fake_result = RunResult(
        nav=pd.Series([1.0]), daily_ret=pd.Series([0.0]),
        turnover=pd.Series(dtype=float), holding_counts=pd.Series(dtype=float),
        trades=pd.DataFrame(), exec_stats={}, config=ExecConfig(hold_days=2),
    )

    def fake_run_backtest(picks_arg, start, end, config, cache_dir):
        captured["picks"] = picks_arg
        captured["config"] = config
        return fake_result, {"ann_return": 0.1}

    def fake_save_artifacts(result, metrics, out_dir, **kwargs):
        captured["out_dir"] = out_dir
        captured["save_kwargs"] = kwargs
        return out_dir

    monkeypatch.setattr(cli, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(cli, "save_artifacts", fake_save_artifacts)
    monkeypatch.setattr(cli, "load_attributes", lambda *a, **kw: pd.DataFrame())

    out_dir = tmp_path / "custom_out"
    args = cli.build_parser().parse_args([
        "run", "--picks", "picks.parquet", "--start", "2024-01-01", "--end", "2024-01-10",
        "--hold-days", "2", "--capital-mode", "shared", "--out-dir", str(out_dir),
    ])
    args.func(args)

    assert captured["picks"].equals(picks)
    assert captured["config"].hold_days == 2
    assert captured["config"].capital_mode == "shared"
    assert captured["out_dir"] == out_dir
    assert captured["save_kwargs"]["picks"].equals(picks)
    assert captured["save_kwargs"]["report"] is True


def test_grid_cli_requests_matched_multi_path_baseline(monkeypatch, tmp_path):
    picks = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-04"]),
        "code": ["A", "B", "C"],
    })
    captured = {}

    monkeypatch.setattr(cli, "load_picks", lambda _: picks)

    def fake_baselines(reference, n_paths, seed, cache_dir):
        captured["reference"] = reference
        captured["n_paths"] = n_paths
        captured["seed"] = seed
        return [pl.from_pandas(reference) for _ in range(n_paths)]

    def fake_run_grid(*args, baseline_picks, return_baseline_paths, **kwargs):
        captured["baseline_picks"] = baseline_picks
        captured["return_baseline_paths"] = return_baseline_paths
        return pd.DataFrame({"策略": ["策略"]}), pd.DataFrame({"基线路径": [1, 2, 3]})

    monkeypatch.setattr(cli, "build_matched_random_baselines", fake_baselines)
    monkeypatch.setattr(cli, "run_grid", fake_run_grid)
    monkeypatch.setattr(cli, "format_grid", lambda _: "grid summary")

    out = tmp_path / "grid.csv"
    args = cli.build_parser().parse_args([
        "grid", "--picks", "picks.parquet", "--start", "2024-01-01", "--end", "2024-01-10",
        "--baseline", "--baseline-sims", "3", "--seed", "9", "--out", str(out),
    ])
    args.func(args)

    assert captured["reference"].equals(picks)
    assert captured["n_paths"] == 3
    assert captured["seed"] == 9
    assert captured["return_baseline_paths"] is True
    assert len(captured["baseline_picks"]) == 3
    assert out.exists()
    paths = pd.read_csv(tmp_path / "grid_baseline_paths.csv")
    assert paths["随机种子"].to_list() == [9, 10, 11]
