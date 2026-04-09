from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.graph import build_agent_graph
from agents.state import AgentState


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def configure_logging(config: dict[str, Any]) -> None:
    level_name = str(config["logging"]["level"]).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def write_markdown_report(state: AgentState) -> Path:
    config = state["config"]
    report_path = PROJECT_ROOT / str(config["runtime"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)

    performance = state["performance"]
    sharpe = float(performance["sharpe"])
    drawdown = float(performance["max_drawdown"]) * 100.0
    total_return = float(performance["total_return"]) * 100.0
    win_rate = float(performance["win_rate"]) * 100.0
    shap_rule = state.get(
        "shap_rule",
        "If RSI rises above 55, long probability tends to increase.",
    )
    optimization_events = state.get("optimization_events", [])
    walk_forward_metrics = performance.get("walk_forward_metrics", {})
    feature_importances = state.get("feature_importances", [])

    # Walk-forward metrics are the trustworthy out-of-sample measure
    wf_overall = walk_forward_metrics.get("overall", {}) if walk_forward_metrics else {}
    wf_sharpe = float(wf_overall.get("mean_sharpe", 0.0))
    wf_accuracy = float(wf_overall.get("mean_accuracy", 0.0)) * 100.0

    lines = [
        "# Week 1 Trading Agent Report (LightGBM + Walk-Forward)",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Asset: {config['asset']['symbol']}",
        f"- Timeframe: {config['asset']['timeframe']}",
        f"- Completed Cycles: {state.get('cycle_count', 0)}",
        "",
        "## Out-of-Sample Performance (Walk-Forward)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| WF Mean Sharpe | {wf_sharpe:.4f} |",
        f"| WF Mean Accuracy | {wf_accuracy:.2f}% |",
        "",
        "## Live Simulation Performance (post-training data only)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Sharpe Ratio | {sharpe:.4f} |",
        f"| Max Drawdown | {drawdown:.2f}% |",
        f"| Total Return | {total_return:.2f}% |",
        f"| Win Rate | {win_rate:.2f}% |",
        "",
        "## SHAP Rule",
        "",
        f"- {shap_rule}",
        "",
        "## Walk-Forward Backtest",
        "",
    ]

    if walk_forward_metrics:
        settings = walk_forward_metrics.get("settings", {})
        lines.extend(
            [
                f"- Train Bars: {settings.get('train_bars', 'n/a')}",
                f"- Test Bars: {settings.get('test_bars', 'n/a')}",
                f"- Step Bars: {settings.get('step_bars', 'n/a')}",
                "",
                "| Mode | Mean Sharpe | Mean Max Drawdown | Mean Total Return | Mean Accuracy | Folds |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in ["expanding", "rolling"]:
            mode_metrics = walk_forward_metrics.get(mode, {})
            lines.append(
                "| "
                f"{mode.title()} | "
                f"{float(mode_metrics.get('mean_sharpe', 0.0)):.4f} | "
                f"{float(mode_metrics.get('mean_max_drawdown', 0.0)) * 100.0:.2f}% | "
                f"{float(mode_metrics.get('mean_total_return', 0.0)) * 100.0:.2f}% | "
                f"{float(mode_metrics.get('mean_accuracy', 0.0)) * 100.0:.2f}% | "
                f"{len(mode_metrics.get('folds', []))} |"
            )
        overall = walk_forward_metrics.get("overall", {})
        lines.extend(
            [
                "",
                "- Overall:",
                f"  - Mean Sharpe: {float(overall.get('mean_sharpe', 0.0)):.4f}",
                f"  - Mean Max Drawdown: {float(overall.get('mean_max_drawdown', 0.0)) * 100.0:.2f}%",
                f"  - Mean Total Return: {float(overall.get('mean_total_return', 0.0)) * 100.0:.2f}%",
                f"  - Mean Accuracy: {float(overall.get('mean_accuracy', 0.0)) * 100.0:.2f}%",
                "",
            ]
        )
    else:
        lines.extend(["- Walk-forward metrics not available yet.", ""])

    lines.extend(
        [
            "## Top 10 LightGBM Feature Importances",
            "",
        ]
    )
    if feature_importances:
        lines.extend(["| Rank | Feature | Importance |", "|---:|---|---:|"])
        for rank, item in enumerate(feature_importances[:10], start=1):
            lines.append(
                "| "
                f"{rank} | "
                f"{item.get('feature', 'n/a')} | "
                f"{float(item.get('importance', 0.0)):.4f} |"
            )
        lines.append("")
    else:
        lines.extend(["- Feature importances not available.", ""])

    lines.extend(
        [
        "## Optimization Events",
        "",
        ]
    )

    if optimization_events:
        lines.extend(
            [
                "| Cycle | Optuna Objective | Long Threshold | Short Threshold |",
                "|---:|---:|---:|---:|",
            ]
        )
        for event in optimization_events:
            params = event["best_params"]
            lines.append(
                "| "
                f"{event['cycle']} | "
                f"{float(event['optuna_objective']):.4f} | "
                f"{float(params['long_threshold']):.4f} | "
                f"{float(params['short_threshold']):.4f} |"
            )
    else:
        lines.append("- No optimization event triggered.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_initial_state(config: dict[str, Any]) -> AgentState:
    return {
        "config": config,
        "cycle_count": 0,
        "cursor": 0,
        "done": False,
        "position": 0,
        "previous_position": 0,
        "last_action": "HOLD",
        "lightgbm_model": None,
        "feature_columns": [],
        "feature_importances": [],
        "equity": float(config["simulation"]["initial_equity"]),
        "equity_curve": [],
        "returns": [],
        "trades": [],
        "replay_buffer": [],
        "strategy_params": {
            "long_threshold": float(config["simulation"]["long_threshold"]),
            "short_threshold": float(config["simulation"]["short_threshold"]),
        },
        "optimization_events": [],
        "performance": {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "walk_forward_metrics": {},
        },
        "shap_rule": "",
    }


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    configure_logging(config)
    logger = logging.getLogger("run_mvp")

    graph = build_agent_graph()
    initial_state = build_initial_state(config)
    final_state: AgentState = graph.invoke(initial_state, config={"recursion_limit": 15000})

    report_path = write_markdown_report(final_state)
    logger.info("MVP run complete. Cycles=%s", final_state.get("cycle_count", 0))
    logger.info(
        "Final metrics: sharpe=%.4f, max_drawdown=%.4f",
        float(final_state["performance"]["sharpe"]),
        float(final_state["performance"]["max_drawdown"]),
    )
    logger.info("Markdown report written to %s", report_path)


if __name__ == "__main__":
    main()
