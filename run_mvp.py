from __future__ import annotations

import json
import logging
import hashlib
import subprocess
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

SHAP_RULES_PATH = PROJECT_ROOT / "data" / "shap_rules.json"


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


def load_shap_rules() -> list[str]:
    """Load persisted SHAP rules from previous sessions, if available."""
    if SHAP_RULES_PATH.exists():
        try:
            rules = json.loads(SHAP_RULES_PATH.read_text(encoding="utf-8"))
            if isinstance(rules, list):
                return rules
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_shap_rules(rules: list[str]) -> None:
    """Persist SHAP rules to disk so the knowledge base survives across sessions."""
    SHAP_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHAP_RULES_PATH.write_text(json.dumps(rules, indent=2), encoding="utf-8")


def evaluate_readiness(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config", {}).get("readiness", {})
    min_returns = int(cfg.get("min_returns_for_sharpe", 1000))
    min_walk_forward_folds = int(cfg.get("min_walk_forward_folds", 5))
    require_positive_wf_sharpe = bool(cfg.get("require_positive_wf_sharpe", True))

    returns_count = len(state.get("returns", []))
    wf = state.get("performance", {}).get("walk_forward_metrics", {})
    expanding_folds = len(wf.get("expanding", {}).get("folds", []))
    rolling_folds = len(wf.get("rolling", {}).get("folds", []))
    overall_sharpe = float(wf.get("overall", {}).get("mean_sharpe", 0.0))

    reasons: list[str] = []
    if returns_count < min_returns:
        reasons.append(f"Insufficient returns for reliable Sharpe: {returns_count} < {min_returns}.")
    if max(expanding_folds, rolling_folds) < min_walk_forward_folds:
        reasons.append(
            "Insufficient walk-forward folds: "
            f"expanding={expanding_folds}, rolling={rolling_folds}, "
            f"required at least {min_walk_forward_folds} in one mode."
        )
    if require_positive_wf_sharpe and overall_sharpe <= 0.0:
        reasons.append(f"Walk-forward mean Sharpe not positive: {overall_sharpe:.4f}.")

    return {
        "go_ahead": len(reasons) == 0,
        "checks": {
            "min_returns_for_sharpe": min_returns,
            "min_walk_forward_folds": min_walk_forward_folds,
            "require_positive_wf_sharpe": require_positive_wf_sharpe,
        },
        "observed": {
            "returns_count": returns_count,
            "expanding_folds": expanding_folds,
            "rolling_folds": rolling_folds,
            "wf_mean_sharpe": overall_sharpe,
        },
        "reasons": reasons,
    }


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
    shap_rules = state.get("shap_rules", [])
    optimization_events = state.get("optimization_events", [])
    walk_forward_metrics = performance.get("walk_forward_metrics", {})
    feature_importances = state.get("feature_importances", [])
    run_metadata = state.get("run_metadata", {})
    data_metadata = state.get("data_metadata", {})
    model_hyperparameters = run_metadata.get("model_hyperparameters", {})
    strategy_params = state.get("strategy_params", {})
    returns_count = len(state.get("returns", []))
    readiness = state.get("readiness", {})

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
        "## Run Metadata",
        "",
        f"- Git Commit Hash: `{run_metadata.get('git_commit_hash', 'unknown')}`",
        f"- Config Hash (SHA-256): `{run_metadata.get('config_hash', 'unknown')}`",
        f"- Data Source: `{data_metadata.get('data_source', 'unknown')}`",
        f"- Data File Path: `{data_metadata.get('data_file_path', 'unknown')}`",
        f"- Data Row Count: {data_metadata.get('data_row_count', 'n/a')}",
        f"- First Timestamp (UTC): {run_metadata.get('first_timestamp_utc', 'n/a')}",
        f"- Last Timestamp (UTC): {run_metadata.get('last_timestamp_utc', 'n/a')}",
        "",
        "### Model Hyperparameters",
        "",
        "```json",
        json.dumps(model_hyperparameters, indent=2, sort_keys=True),
        "```",
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

    if returns_count < 1000:
        lines.extend(
            [
                "> [!WARNING]",
                f"> Reliability warning: only {returns_count} returns were observed. "
                "Sharpe and related risk-adjusted metrics are unstable below 1,000 returns.",
                "",
            ]
        )

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
            "## Methodology",
            "",
            f"- Walk-Forward Train Bars: {walk_forward_metrics.get('settings', {}).get('train_bars', 'n/a')}",
            f"- Walk-Forward Test Bars: {walk_forward_metrics.get('settings', {}).get('test_bars', 'n/a')}",
            f"- Walk-Forward Step Bars: {walk_forward_metrics.get('settings', {}).get('step_bars', 'n/a')}",
            f"- Walk-Forward Embargo Gap: {walk_forward_metrics.get('settings', {}).get('embargo_gap', config.get('walk_forward', {}).get('embargo_gap', 'n/a'))}",
            f"- Trading Costs: slippage={config['simulation'].get('slippage_bps', 'n/a')} bps, "
            f"funding_rate_per_8h={config['simulation'].get('funding_rate_per_8h', 0.0)}",
            f"- Decision Thresholds: long={float(strategy_params.get('long_threshold', config['simulation']['long_threshold'])):.4f}, "
            f"short={float(strategy_params.get('short_threshold', config['simulation']['short_threshold'])):.4f}",
            "",
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

    # SHAP Knowledge Base
    lines.extend(["## SHAP Knowledge Base", ""])
    if shap_rules:
        lines.append(f"*{len(shap_rules)} unique rules accumulated across sessions.*")
        lines.append("")
        for i, rule in enumerate(shap_rules[-20:], start=1):  # show last 20
            lines.append(f"{i}. {rule}")
        lines.append("")
    else:
        lines.extend(["- No rules accumulated yet.", ""])

    lines.extend(
        [
        "## Readiness Gate",
        "",
        f"- GO_AHEAD: **{bool(readiness.get('go_ahead', False))}**",
        "",
        "### Gate Checks",
        "",
        "```json",
        json.dumps(readiness.get("checks", {}), indent=2, sort_keys=True),
        "```",
        "",
        "### Observed",
        "",
        "```json",
        json.dumps(readiness.get("observed", {}), indent=2, sort_keys=True),
        "```",
        "",
        "### Gate Reasons",
        "",
        ]
    )
    reasons = readiness.get("reasons", [])
    if reasons:
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- All readiness checks passed.")
    lines.extend(["", "## Optimization Events", ""])

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


def build_initial_state(config: dict[str, Any], prior_shap_rules: list[str]) -> AgentState:
    return {
        "config": config,
        "cycle_count": 0,
        "cursor": 0,
        "done": False,
        "position": 0,
        "previous_position": 0,
        "entry_price": None,
        "last_action": "HOLD",
        "lightgbm_model": None,
        "feature_columns": [],
        "feature_importances": [],
        "equity": float(config["simulation"]["initial_equity"]),
        "equity_curve": [],
        "returns": [],
        "trades": [],
        "replay_buffer": [],
        "risk_params": {},
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
        "shap_rules": prior_shap_rules,
    }


def _get_git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _compute_config_hash(config_path: Path) -> str:
    payload = config_path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _to_utc_iso(timestamp_ms: Any) -> str:
    try:
        dt = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return "n/a"


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    configure_logging(config)
    logger = logging.getLogger("run_mvp")

    prior_shap_rules = load_shap_rules()
    if prior_shap_rules:
        logger.info("Loaded %d SHAP rules from previous sessions.", len(prior_shap_rules))

    graph = build_agent_graph()
    initial_state = build_initial_state(config, prior_shap_rules)
    initial_state["run_metadata"] = {
        "git_commit_hash": _get_git_commit_hash(),
        "config_hash": _compute_config_hash(config_path),
        "model_hyperparameters": dict(config.get("model", {})),
    }
    final_state: AgentState = graph.invoke(initial_state, config={"recursion_limit": 15000})
    final_state["run_metadata"] = dict(initial_state["run_metadata"])
    data_meta = final_state.get("data_metadata", {})
    final_state["run_metadata"]["first_timestamp_utc"] = _to_utc_iso(
        data_meta.get("first_timestamp_ms")
    )
    final_state["run_metadata"]["last_timestamp_utc"] = _to_utc_iso(
        data_meta.get("last_timestamp_ms")
    )

    # Persist accumulated SHAP rules for next session
    final_rules = final_state.get("shap_rules", [])
    save_shap_rules(final_rules)
    logger.info("Saved %d SHAP rules to %s", len(final_rules), SHAP_RULES_PATH)
    final_state["readiness"] = evaluate_readiness(final_state)

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
