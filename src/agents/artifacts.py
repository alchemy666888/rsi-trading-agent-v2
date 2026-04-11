from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agents.state import AgentState


def write_report(state: AgentState) -> str:
    config = state["config"]
    run_metrics = state["performance"]["run_metrics"]
    benchmark_metrics = state["performance"]["benchmark_metrics"]
    split_metadata = state["split_metadata"]
    risk_cfg = config.get("risk", {})

    lines = [
        "# BTC Model-Based Research Prototype Report",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Asset: {config['asset']['symbol']}",
        f"- Timeframe: {config['asset']['timeframe']}",
        f"- Dataset mode: {config['dataset']['source_mode']}",
        f"- Run ID: {state['run_id']}",
        "",
        "## Held-Out Simulation",
        "",
        f"- OOS Bars: {split_metadata['oos_start']}..{split_metadata['oos_end']}",
        f"- Completed Cycles: {state.get('cursor', split_metadata['oos_start']) - split_metadata['oos_start']}",
        f"- Paused: {state.get('paused', False)}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Sharpe Ratio | {float(run_metrics['sharpe']):.4f} |",
        f"| Max Drawdown | {float(run_metrics['max_drawdown']) * 100.0:.2f}% |",
        f"| Total Return | {float(run_metrics['total_return']) * 100.0:.2f}% |",
        f"| Win Rate | {float(run_metrics['win_rate']) * 100.0:.2f}% |",
        f"| Trade Count | {int(run_metrics['trade_count'])} |",
        "",
        "## Walk-Forward Benchmark",
        "",
        f"- Train Bars: {benchmark_metrics.get('settings', {}).get('train_bars', 'n/a')}",
        f"- Validation Bars: {benchmark_metrics.get('settings', {}).get('validation_bars', 'n/a')}",
        f"- Test Bars: {benchmark_metrics.get('settings', {}).get('test_bars', 'n/a')}",
        f"- Purge Bars: {benchmark_metrics.get('settings', {}).get('purge_bars', 'n/a')}",
        f"- Signal Delay Bars: {benchmark_metrics.get('settings', {}).get('signal_delay_bars', 'n/a')}",
        "",
        "| Mode | Mean Sharpe | Mean Max Drawdown | Mean Total Return | Mean Win Rate | Folds |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for mode in ["expanding", "rolling"]:
        mode_metrics = benchmark_metrics.get(mode, {})
        lines.append(
            "| "
            f"{mode.title()} | "
            f"{float(mode_metrics.get('mean_sharpe', 0.0)):.4f} | "
            f"{float(mode_metrics.get('mean_max_drawdown', 0.0)) * 100.0:.2f}% | "
            f"{float(mode_metrics.get('mean_total_return', 0.0)) * 100.0:.2f}% | "
            f"{float(mode_metrics.get('mean_win_rate', 0.0)) * 100.0:.2f}% | "
            f"{len(mode_metrics.get('folds', []))} |"
        )

    lines.extend(
        [
            "",
            "## Calibration Rule",
            "",
            f"- {state.get('shap_rule', 'No SHAP rule available.')}",
            "",
            "## Active Risk Rules",
            "",
            f"- Max abs position: {risk_cfg.get('max_abs_position', 1)}",
            f"- Max turnover per bar: {risk_cfg.get('max_turnover_per_bar', 1)}",
            f"- Max drawdown pause: {float(risk_cfg.get('max_drawdown_pause', 1.0)) * 100.0:.2f}%",
            f"- Block high volatility: {bool(risk_cfg.get('block_high_volatility', False))}",
            f"- Stop loss pct: {float(risk_cfg.get('stop_loss_pct', 0.0)) * 100.0:.2f}%",
            f"- Take profit pct: {float(risk_cfg.get('take_profit_pct', 0.0)) * 100.0:.2f}%",
            "",
            "## Optimization Provenance",
            "",
        ]
    )

    optimization_events = state.get("optimization_events", [])
    if optimization_events:
        lines.extend(
            [
                "| Event | Split | Objective | Value | Long Threshold | Short Threshold | Future OOS Only |",
                "|---:|---|---|---:|---:|---:|---|",
            ]
        )
        for idx, event in enumerate(optimization_events, start=1):
            lines.append(
                "| "
                f"{idx} | "
                f"{event.get('split', 'n/a')} | "
                f"{event.get('objective_name', 'n/a')} | "
                f"{float(event.get('objective_value', 0.0)):.4f} | "
                f"{float(event.get('best_params', {}).get('long_threshold', 0.0)):.4f} | "
                f"{float(event.get('best_params', {}).get('short_threshold', 0.0)):.4f} | "
                f"{event.get('affects_future_oos_only', False)} |"
            )
    else:
        lines.append("- No optimization events recorded.")

    return "\n".join(lines) + "\n"


def persist_run_artifacts(project_root: Path, state: AgentState) -> Path:
    artifact_dir = project_root / state["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report_text = write_report(state)
    (artifact_dir / "report.md").write_text(report_text, encoding="utf-8")
    (artifact_dir / "config.yaml").write_text(yaml.safe_dump(state["config"], sort_keys=False), encoding="utf-8")
    (artifact_dir / "dataset_metadata.json").write_text(
        json.dumps(state["dataset_metadata"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "split_metadata.json").write_text(
        json.dumps(state["split_metadata"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "strategy_params.json").write_text(
        json.dumps(state["strategy_params"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "optimization_events.json").write_text(
        json.dumps(state.get("optimization_events", []), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "feature_importances.json").write_text(
        json.dumps(state.get("feature_importances", []), indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "trades.json").write_text(
        json.dumps(state.get("trades", []), indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "trade_history_buffer.json").write_text(
        json.dumps(state.get("trade_history_buffer", []), indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "equity_curve.json").write_text(
        json.dumps(state.get("equity_curve", []), indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "returns.json").write_text(
        json.dumps(state.get("returns", []), indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "benchmark_metrics.json").write_text(
        json.dumps(state["performance"]["benchmark_metrics"], indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "run_metrics.json").write_text(
        json.dumps(state["performance"]["run_metrics"], indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "shap_rule.txt").write_text(state.get("shap_rule", ""), encoding="utf-8")
    return artifact_dir
