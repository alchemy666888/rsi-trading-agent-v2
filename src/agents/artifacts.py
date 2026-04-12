from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from agents.decision_audit import decision_row_to_summary
from agents.metrics_utils import (
    compare_benchmark_metrics,
    compute_equity_curve_diagnostics,
    compute_exposure_stats,
    compute_trade_summary_statistics,
    count_events_by_type,
)
from agents.state import AgentState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_utc_iso(timestamp_ms: Any) -> str | None:
    try:
        return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def build_last_state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    current_row = state.get("current_row", {})
    if not isinstance(current_row, dict):
        current_row = {}
    return {
        "cursor": state.get("cursor"),
        "position": state.get("position"),
        "target_position": state.get("target_position"),
        "equity": state.get("equity"),
        "last_action": state.get("last_action"),
        "current_bar_timestamp": current_row.get("timestamp"),
        "current_close": current_row.get("close"),
        "prediction": state.get("prediction"),
        "risk_status": state.get("risk_status"),
        "returns_count": len(state.get("returns", [])),
        "transition_count": len(state.get("trades", [])),
    }


def _compute_pause_durations_seconds(events: list[dict[str, Any]]) -> list[float]:
    starts: list[datetime] = []
    durations: list[float] = []
    for event in events:
        event_type = str(event.get("event_type", ""))
        timestamp_raw = event.get("timestamp_utc")
        if not isinstance(timestamp_raw, str):
            continue
        try:
            timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
        except Exception:
            continue
        if event_type == "drawdown_pause_activated":
            starts.append(timestamp)
        elif event_type == "drawdown_pause_cleared" and starts:
            start = starts.pop(0)
            durations.append(max(0.0, (timestamp - start).total_seconds()))
    return durations


def _normalize_error_info(error_info: dict[str, Any] | None) -> dict[str, Any]:
    info = dict(error_info or {})
    return {
        "error_type": str(info.get("error_type", "n/a")),
        "error_message": str(info.get("error_message", "n/a")),
        "traceback": str(info.get("traceback", "")),
        "failed_at_utc": str(info.get("failed_at_utc", _utc_now())),
    }


def _build_benchmark_sufficiency_payload(benchmark_metrics: dict[str, Any]) -> dict[str, Any]:
    sufficiency = dict(benchmark_metrics.get("sufficiency", {}))
    expanding = dict(sufficiency.get("expanding", {}))
    rolling = dict(sufficiency.get("rolling", {}))
    return {
        "generated_at_utc": _utc_now(),
        "overall_sufficient": bool(sufficiency.get("overall_sufficient", False)),
        "acceptance_thresholds": {
            "min_folds_per_mode": int(sufficiency.get("min_folds_per_mode", 0)),
            "min_total_test_bars": int(sufficiency.get("min_total_test_bars", 0)),
        },
        "expanding": {
            "fold_count": int(expanding.get("fold_count", 0)),
            "total_test_bars": int(expanding.get("total_test_bars", 0)),
            "sufficient": bool(expanding.get("sufficient", False)),
            "reasons": list(expanding.get("reasons", [])),
        },
        "rolling": {
            "fold_count": int(rolling.get("fold_count", 0)),
            "total_test_bars": int(rolling.get("total_test_bars", 0)),
            "sufficient": bool(rolling.get("sufficient", False)),
            "reasons": list(rolling.get("reasons", [])),
        },
        "failure_reasons": list(sufficiency.get("reasons", [])),
    }


def _build_regression_ledger_entry(state: AgentState) -> dict[str, Any]:
    run_metadata = dict(state.get("run_metadata", {}))
    dataset_metadata = dict(state.get("dataset_metadata", {}))
    run_metrics = dict(state.get("performance", {}).get("run_metrics", {}))
    benchmark_metrics = dict(state.get("performance", {}).get("benchmark_metrics", {}))
    overall = dict(benchmark_metrics.get("overall", {}))
    optimization_events = list(state.get("optimization_events", []))
    validation_sharpe = 0.0
    if optimization_events:
        validation_sharpe = _safe_float(optimization_events[-1].get("objective_value", 0.0))
    return {
        "run_id": state.get("run_id", "unknown"),
        "generated_at_utc": _utc_now(),
        "dataset_window": {
            "dataset_hash": dataset_metadata.get("dataset_hash"),
            "snapshot_hash": dataset_metadata.get("snapshot_hash"),
            "raw_data_hash": dataset_metadata.get("raw_data_hash"),
            "timestamp_start": dataset_metadata.get("timestamp_start"),
            "timestamp_end": dataset_metadata.get("timestamp_end"),
            "timestamp_start_utc": dataset_metadata.get("timestamp_start_utc"),
            "timestamp_end_utc": dataset_metadata.get("timestamp_end_utc"),
        },
        "metrics": {
            "validation_sharpe": validation_sharpe,
            "walk_forward_mean_sharpe": _safe_float(overall.get("mean_sharpe")),
            "held_out_sharpe": _safe_float(run_metrics.get("sharpe")),
            "max_drawdown": _safe_float(run_metrics.get("max_drawdown")),
            "total_return": _safe_float(run_metrics.get("total_return")),
            "transition_count": _safe_int(run_metrics.get("transition_count")),
            "completed_trade_count": _safe_int(run_metrics.get("completed_trade_count")),
        },
        "thresholds": dict(state.get("strategy_params", {})),
        "git_commit_hash": run_metadata.get("git_commit_hash"),
        "config_hash": run_metadata.get("config_hash"),
    }


def _append_regression_ledger(artifact_root: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    ledger_path = artifact_root / "regression_ledger.json"
    existing_rows: list[dict[str, Any]] = []
    if ledger_path.exists():
        try:
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                existing_rows = [row for row in payload if isinstance(row, dict)]
        except Exception:
            existing_rows = []
    existing_rows.append(entry)
    _write_json(ledger_path, existing_rows, sort_keys=False)
    return existing_rows


def build_report_payload(state: AgentState) -> dict[str, Any]:
    config = dict(state.get("config", {}))
    reporting_cfg = dict(config.get("reporting", {}))
    run_metadata = dict(state.get("run_metadata", {}))
    dataset_metadata = dict(state.get("dataset_metadata", {}))
    split_metadata = dict(state.get("split_metadata", {}))
    run_metrics = dict(state.get("performance", {}).get("run_metrics", {}))
    benchmark_metrics = dict(state.get("performance", {}).get("benchmark_metrics", {}))
    optimization_events = list(state.get("optimization_events", []))
    decision_log = list(state.get("decision_log", []))
    completed_trades = list(state.get("completed_trades", []))
    event_log = list(state.get("event_log", []))
    readiness = dict(state.get("readiness", {}))
    returns = [float(value) for value in state.get("returns", [])]
    equity_curve = [float(value) for value in state.get("equity_curve", [])]
    initial_equity = _safe_float(config.get("simulation", {}).get("initial_equity"), default=1.0)
    timeframe = str(config.get("asset", {}).get("timeframe", "15m"))
    split_counts = dict(run_metadata.get("split_counts", {}))
    if not split_counts:
        split_counts = {
            "total_bars": _safe_int(dataset_metadata.get("rows")),
            "train_bars": max(0, _safe_int(split_metadata.get("train_end")) - _safe_int(split_metadata.get("train_start"))),
            "validation_bars": max(
                0,
                _safe_int(split_metadata.get("validation_end")) - _safe_int(split_metadata.get("validation_start")),
            ),
            "oos_bars": max(0, _safe_int(split_metadata.get("oos_end")) - _safe_int(split_metadata.get("oos_start"))),
        }
    split_counts.setdefault("total_bars", _safe_int(dataset_metadata.get("rows")))

    trade_summary = compute_trade_summary_statistics(completed_trades)
    run_transition_count = _safe_int(run_metrics.get("transition_count", len(state.get("trades", []))))
    run_completed_trade_count = _safe_int(run_metrics.get("completed_trade_count", len(completed_trades)))
    bar_win_rate = _safe_float(run_metrics.get("bar_win_rate", 0.0))
    completed_trade_win_rate = _safe_float(
        run_metrics.get("completed_trade_win_rate", trade_summary.get("win_rate", 0.0))
    )
    exposure_stats = compute_exposure_stats(decision_log)
    equity_diagnostics = compute_equity_curve_diagnostics(
        returns,
        equity_curve,
        initial_equity,
        timeframe=timeframe,
    )
    event_counts = count_events_by_type(event_log)
    benchmark_comparison = compare_benchmark_metrics(run_metrics, benchmark_metrics)
    pause_durations = _compute_pause_durations_seconds(event_log)
    error_info = _normalize_error_info(state.get("error_info")) if state.get("error_info") else None

    risk_blocked_trades = 0
    for row in decision_log:
        constraints = row.get("risk_constraints_applied", {})
        if isinstance(constraints, dict) and constraints.get("blocked_high_volatility", False):
            risk_blocked_trades += 1

    data_quality_notes = []
    missing_values_total = dataset_metadata.get("missing_values_total")
    if missing_values_total is not None:
        data_quality_notes.append(f"Missing values filled during feature prep: {int(missing_values_total)}.")
    else:
        data_quality_notes.append("Missing value summary unavailable.")
    data_quality_notes.append(
        f"Feature columns available for model: {len(state.get('feature_columns', []))}."
    )
    data_quality_notes.append(
        "Execution convention: signal observed at close(t), executed at close(t + delay), "
        "position earns returns from close(t + delay) to close(t + delay + 1)."
    )
    data_quality_notes.append(
        "Risk checks (drawdown pause, volatility block, stop-loss, take-profit) are evaluated at signal time."
    )
    data_quality_notes.append(
        f"Slippage assumption (bps): {_safe_float(config.get('simulation', {}).get('slippage_bps')):.2f}."
    )
    data_quality_notes.append(
        f"Benchmark eligible input: {bool(dataset_metadata.get('benchmark_eligible', False))}."
    )
    if optimization_events:
        latest_event = optimization_events[-1] if isinstance(optimization_events[-1], dict) else {}
        diagnostics = latest_event.get("diagnostics", {})
        if isinstance(diagnostics, dict) and diagnostics:
            one_side_exposure_max = max(
                float(diagnostics.get("long_exposure_pct", 0.0)),
                float(diagnostics.get("short_exposure_pct", 0.0)),
            )
            data_quality_notes.append(
                "Calibration diagnostics: "
                f"degenerate_regime={bool(diagnostics.get('degenerate_regime', False))}, "
                f"transition_count={int(diagnostics.get('transition_count', 0))}, "
                f"min_transition_count={int(diagnostics.get('min_transition_count', 0))}, "
                f"one_side_exposure_max={one_side_exposure_max:.4f}."
            )
    for warning in readiness.get("warnings", []):
        data_quality_notes.append(f"Readiness warning: {warning}")
    evidence = readiness.get("evidence_sufficiency", {})
    if isinstance(evidence, dict):
        data_quality_notes.append(
            "Benchmark sufficiency artifact available: "
            f"{bool(evidence.get('benchmark_sufficiency_artifact_available', False))}."
        )
        data_quality_notes.append(
            "Regression history windows: "
            f"{int(evidence.get('historical_window_count', 0))}/"
            f"{int(evidence.get('minimum_historical_windows', 0))}."
        )

    return {
        "metadata": {
            "run_id": state.get("run_id", "unknown"),
            "generated_at_utc": _utc_now(),
            "report_detail_level": reporting_cfg.get("report_detail_level", "full"),
            "git_commit_hash": run_metadata.get("git_commit_hash", "unknown"),
            "config_hash": run_metadata.get("config_hash", "unknown"),
            "symbol": config.get("asset", {}).get("symbol", "unknown"),
            "timeframe": config.get("asset", {}).get("timeframe", "unknown"),
            "dataset_source": dataset_metadata.get("source_ref"),
            "dataset_mode": dataset_metadata.get("source_mode"),
            "benchmark_eligible": bool(dataset_metadata.get("benchmark_eligible", False)),
            "snapshot_path": dataset_metadata.get("snapshot_path"),
            "snapshot_hash": dataset_metadata.get("snapshot_hash"),
            "raw_data_hash": dataset_metadata.get("raw_data_hash"),
            "dataset_span_start": _to_utc_iso(dataset_metadata.get("timestamp_start")),
            "dataset_span_end": _to_utc_iso(dataset_metadata.get("timestamp_end")),
            "split_counts": split_counts,
        },
        "headline_kpis": {
            "sharpe": _safe_float(run_metrics.get("sharpe")),
            "max_drawdown": _safe_float(run_metrics.get("max_drawdown")),
            "total_return": _safe_float(run_metrics.get("total_return")),
            "bar_win_rate": bar_win_rate,
            "transition_count": run_transition_count,
            "completed_trade_win_rate": completed_trade_win_rate,
            "completed_trade_count": run_completed_trade_count,
            "avg_completed_trade_return": _safe_float(trade_summary.get("avg_trade_return")),
            "profit_factor": trade_summary.get("profit_factor", 0.0),
            "turnover": _safe_float(exposure_stats.get("turnover")),
            "exposure_stats": exposure_stats,
        },
        "calibration_summary": {
            "strategy_params": state.get("strategy_params", {}),
            "objective": optimization_events[-1].get("objective_name", "n/a") if optimization_events else "n/a",
            "best_validation_score": optimization_events[-1].get("objective_value", 0.0) if optimization_events else 0.0,
            "best_validation_score_adjusted": optimization_events[-1].get("objective_value_adjusted", optimization_events[-1].get("objective_value", 0.0))
            if optimization_events
            else 0.0,
            "optimization_trials": len(optimization_events),
            "optimization_events": optimization_events,
        },
        "benchmark_comparison": benchmark_comparison,
        "risk_diagnostics": {
            "blocked_trades": risk_blocked_trades,
            "stop_loss_triggered_count": event_counts.get("stop_loss_triggered", 0),
            "take_profit_triggered_count": event_counts.get("take_profit_triggered", 0),
            "drawdown_pause_activated_count": event_counts.get("drawdown_pause_activated", 0),
            "drawdown_pause_cleared_count": event_counts.get("drawdown_pause_cleared", 0),
            "pause_durations_seconds": pause_durations,
            "max_consecutive_losses": trade_summary.get("max_consecutive_losses", 0),
        },
        "trade_diagnostics": {
            "best_trade": trade_summary.get("best_trade"),
            "worst_trade": trade_summary.get("worst_trade"),
            "longest_hold_bars": trade_summary.get("longest_hold_bars"),
            "average_hold_bars": trade_summary.get("average_hold_bars"),
            "long_vs_short_breakdown": {
                "long_trades": trade_summary.get("long_count", 0),
                "short_trades": trade_summary.get("short_count", 0),
            },
        },
        "equity_diagnostics": equity_diagnostics,
        "data_quality": {
            "notes": data_quality_notes,
            "dataset_metadata": dataset_metadata,
            "split_metadata": split_metadata,
            "readiness": readiness,
        },
        "failure": error_info,
        "event_counts": event_counts,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    kpis = report["headline_kpis"]
    calibration = report["calibration_summary"]
    benchmark = report["benchmark_comparison"]
    risk = report["risk_diagnostics"]
    trade = report["trade_diagnostics"]
    data_quality = report["data_quality"]
    failure = report.get("failure")
    overfit_warnings = benchmark.get("overfit_warnings", [])
    split_counts = metadata.get("split_counts", {})

    lines = [
        "# BTC Model-Based Research Prototype Report",
        "",
        "## Run Metadata",
        "",
        f"- Run ID: {metadata.get('run_id')}",
        f"- Generated (UTC): {metadata.get('generated_at_utc')}",
        f"- Report Detail Level: {metadata.get('report_detail_level')}",
        f"- Git Commit: {metadata.get('git_commit_hash')}",
        f"- Config Hash: {metadata.get('config_hash')}",
        f"- Asset: {metadata.get('symbol')} ({metadata.get('timeframe')})",
        f"- Dataset Source: {metadata.get('dataset_source')} ({metadata.get('dataset_mode')})",
        f"- Snapshot Path: {metadata.get('snapshot_path')}",
        f"- Snapshot Hash: {metadata.get('snapshot_hash')}",
        f"- Raw Data Hash: {metadata.get('raw_data_hash')}",
        f"- Benchmark Eligible Input: {metadata.get('benchmark_eligible')}",
        f"- Dataset Span (UTC): {metadata.get('dataset_span_start')} -> {metadata.get('dataset_span_end')}",
        (
            "- Bars: "
            f"total={split_counts.get('total_bars', 'n/a')}, "
            f"train={split_counts.get('train_bars', 'n/a')}, "
            f"validation={split_counts.get('validation_bars', 'n/a')}, "
            f"oos={split_counts.get('oos_bars', 'n/a')}"
        ),
        "",
        "## Headline KPIs",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Sharpe | {float(kpis.get('sharpe', 0.0)):.4f} |",
        f"| Max Drawdown | {float(kpis.get('max_drawdown', 0.0)) * 100.0:.2f}% |",
        f"| Total Return | {float(kpis.get('total_return', 0.0)) * 100.0:.2f}% |",
        f"| Bar Win Rate | {float(kpis.get('bar_win_rate', 0.0)) * 100.0:.2f}% |",
        f"| Transition Count | {int(kpis.get('transition_count', 0))} |",
        f"| Completed Trade Win Rate | {float(kpis.get('completed_trade_win_rate', 0.0)) * 100.0:.2f}% |",
        f"| Completed Trade Count | {int(kpis.get('completed_trade_count', 0))} |",
        f"| Avg Completed Trade Return | {float(kpis.get('avg_completed_trade_return', 0.0)) * 100.0:.4f}% |",
        f"| Profit Factor | {kpis.get('profit_factor', 0.0)} |",
        f"| Turnover | {float(kpis.get('turnover', 0.0)):.4f} |",
        f"| Long Exposure | {float(kpis.get('exposure_stats', {}).get('long_exposure_pct', 0.0)) * 100.0:.2f}% |",
        f"| Short Exposure | {float(kpis.get('exposure_stats', {}).get('short_exposure_pct', 0.0)) * 100.0:.2f}% |",
        f"| Flat Exposure | {float(kpis.get('exposure_stats', {}).get('flat_exposure_pct', 0.0)) * 100.0:.2f}% |",
        "",
        "## Calibration Summary",
        "",
        f"- Tuned Thresholds: {json.dumps(calibration.get('strategy_params', {}), sort_keys=True)}",
        f"- Objective: {calibration.get('objective')}",
        f"- Best Validation Score: {float(calibration.get('best_validation_score', 0.0)):.4f}",
        f"- Best Validation Score (Adjusted): {float(calibration.get('best_validation_score_adjusted', calibration.get('best_validation_score', 0.0))):.4f}",
        f"- Optimization Events Recorded: {int(calibration.get('optimization_trials', 0))}",
        "",
        "## Benchmark Comparison",
        "",
        f"- Delta Sharpe (Held-Out - Walk-Forward Mean): {float(benchmark.get('held_out_vs_walk_forward', {}).get('delta_sharpe', 0.0)):.4f}",
        f"- Delta Total Return (Held-Out - Walk-Forward Mean): {float(benchmark.get('held_out_vs_walk_forward', {}).get('delta_total_return', 0.0)) * 100.0:.2f}%",
    ]

    if overfit_warnings:
        lines.append("- Overfit Warnings:")
        for warning in overfit_warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("- Overfit Warnings: none triggered.")

    lines.extend(
        [
            "",
            "## Risk Diagnostics",
            "",
            f"- Blocked Trades: {int(risk.get('blocked_trades', 0))}",
            f"- Stop Loss Trigger Count: {int(risk.get('stop_loss_triggered_count', 0))}",
            f"- Take Profit Trigger Count: {int(risk.get('take_profit_triggered_count', 0))}",
            f"- Drawdown Pause Activated Count: {int(risk.get('drawdown_pause_activated_count', 0))}",
            f"- Drawdown Pause Cleared Count: {int(risk.get('drawdown_pause_cleared_count', 0))}",
            f"- Pause Durations (seconds): {risk.get('pause_durations_seconds', [])}",
            f"- Max Consecutive Losses: {int(risk.get('max_consecutive_losses', 0))}",
            "",
            "## Trade Diagnostics",
            "",
            f"- Best Trade: {trade.get('best_trade')}",
            f"- Worst Trade: {trade.get('worst_trade')}",
            f"- Longest Hold (bars): {trade.get('longest_hold_bars')}",
            f"- Average Hold (bars): {float(trade.get('average_hold_bars', 0.0)):.2f}",
            f"- Long/Short Breakdown: {trade.get('long_vs_short_breakdown')}",
            "",
            "## Data Quality / Readiness",
            "",
        ]
    )
    for note in data_quality.get("notes", []):
        lines.append(f"- {note}")

    readiness = data_quality.get("readiness", {})
    if not isinstance(readiness, dict):
        readiness = {}
    engineering_validity = readiness.get("engineering_validity", {})
    if not isinstance(engineering_validity, dict):
        engineering_validity = {}
    research_validity = readiness.get("research_validity", {})
    if not isinstance(research_validity, dict):
        research_validity = {}
    evidence_sufficiency = readiness.get("evidence_sufficiency", {})
    if not isinstance(evidence_sufficiency, dict):
        evidence_sufficiency = {}
    hard_blockers = readiness.get("hard_blockers", [])

    lines.extend(["", "## Phase Gate", ""])
    lines.append(f"- Phase Gate Decision: {readiness.get('phase_gate_decision', 'unknown')}")
    lines.append(f"- Fine-Tuning Gate Open: {readiness.get('fine_tuning_gate_open', False)}")
    lines.append(f"- Engineering Valid: {bool(engineering_validity.get('green', False))}")
    lines.append(f"- Research Valid: {bool(research_validity.get('green', False))}")
    lines.append(f"- Evidence Sufficient: {bool(evidence_sufficiency.get('green', False))}")
    lines.append(
        f"- Execution Semantics Consistent: {bool(engineering_validity.get('execution_semantics_consistent', False))}"
    )
    lines.append(
        f"- KPI Schema Consistent: {bool(engineering_validity.get('kpi_schema_consistent', False))}"
    )
    lines.append(
        f"- Snapshot-Based Input: {bool(evidence_sufficiency.get('snapshot_based', False))}"
    )
    lines.append(
        f"- Walk-Forward Sufficient: {bool(evidence_sufficiency.get('walk_forward_sufficient', False))}"
    )
    lines.append(
        "- Benchmark Sufficiency Artifact Available: "
        f"{bool(evidence_sufficiency.get('benchmark_sufficiency_artifact_available', False))}"
    )
    lines.append(
        "- Historical Window Count: "
        f"{int(evidence_sufficiency.get('historical_window_count', 0))} "
        f"(minimum {int(evidence_sufficiency.get('minimum_historical_windows', 0))})"
    )
    if isinstance(hard_blockers, list) and hard_blockers:
        for blocker in hard_blockers:
            if isinstance(blocker, dict):
                lines.append(f"- Hard Blocker [{blocker.get('code', 'unknown')}]: {blocker.get('message', 'n/a')}")
    else:
        lines.append("- Hard Blocker: none")

    lines.extend(["", "## Failure / Fallbacks", ""])
    if failure:
        lines.append(f"- Error Type: {failure.get('error_type')}")
        lines.append(f"- Error Message: {failure.get('error_message')}")
        lines.append(f"- Failed At (UTC): {failure.get('failed_at_utc')}")
    else:
        lines.append("- No failures captured.")

    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=sort_keys), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _to_csv_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    normalized_records = [
        {key: _to_csv_scalar(value) for key, value in record.items()}
        for record in records
    ]
    pl.DataFrame(normalized_records).write_csv(path)


def _stage_artifact_dir_non_blocking(project_root: Path, artifact_dir: Path, *, enabled: bool) -> None:
    if not enabled:
        return
    if not (project_root / ".git").exists():
        return
    try:
        rel_artifact_dir = artifact_dir.relative_to(project_root)
    except Exception:
        return
    try:
        subprocess.run(
            ["git", "add", str(rel_artifact_dir)],
            cwd=project_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


def persist_run_artifacts(project_root: Path, state: AgentState) -> Path:
    artifact_dir = project_root / str(state.get("artifact_dir", "artifacts/unknown-run"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = artifact_dir.parent

    report_payload = build_report_payload(state)
    report_transition_count = _safe_int(report_payload.get("headline_kpis", {}).get("transition_count"))
    report_completed_trade_count = _safe_int(report_payload.get("headline_kpis", {}).get("completed_trade_count"))
    report_bar_win_rate = _safe_float(report_payload.get("headline_kpis", {}).get("bar_win_rate"))
    report_completed_trade_win_rate = _safe_float(
        report_payload.get("headline_kpis", {}).get("completed_trade_win_rate")
    )
    run_metrics = dict(state.get("performance", {}).get("run_metrics", {}))
    trade_rows_count = len(list(state.get("trades", [])))
    completed_trade_rows_count = len(list(state.get("completed_trades", [])))
    run_transition_count = _safe_int(run_metrics.get("transition_count", trade_rows_count))
    run_completed_trade_count = _safe_int(run_metrics.get("completed_trade_count", completed_trade_rows_count))
    run_bar_win_rate = _safe_float(run_metrics.get("bar_win_rate"))
    run_completed_trade_win_rate = _safe_float(run_metrics.get("completed_trade_win_rate"))
    if not state.get("error_info"):
        if {"win_rate", "trade_count"}.intersection(set(run_metrics.keys())):
            raise AssertionError(
                "Ambiguous legacy KPI keys detected in run_metrics; expected explicit schema only."
            )
        if report_transition_count != run_transition_count:
            raise AssertionError(
                "Report transition_count mismatch: "
                f"report={report_transition_count}, run_metrics={run_transition_count}"
            )
        if report_completed_trade_count != run_completed_trade_count:
            raise AssertionError(
                "Report completed_trade_count mismatch: "
                f"report={report_completed_trade_count}, run_metrics={run_completed_trade_count}"
            )
        if abs(report_bar_win_rate - run_bar_win_rate) > 1e-12:
            raise AssertionError(
                "Report bar_win_rate mismatch: "
                f"report={report_bar_win_rate}, run_metrics={run_bar_win_rate}"
            )
        if abs(report_completed_trade_win_rate - run_completed_trade_win_rate) > 1e-12:
            raise AssertionError(
                "Report completed_trade_win_rate mismatch: "
                f"report={report_completed_trade_win_rate}, run_metrics={run_completed_trade_win_rate}"
            )
        if trade_rows_count != run_transition_count:
            raise AssertionError(
                "Trade rows mismatch: "
                f"trades={trade_rows_count}, run_metrics={run_transition_count}"
            )
        if completed_trade_rows_count != run_completed_trade_count:
            raise AssertionError(
                "Completed trade rows mismatch: "
                f"completed_trades={completed_trade_rows_count}, run_metrics={run_completed_trade_count}"
            )

    report_text = render_markdown_report(report_payload)
    (artifact_dir / "report.md").write_text(report_text, encoding="utf-8")

    reporting_cfg = dict(state.get("config", {}).get("reporting", {}))
    write_report_json = bool(reporting_cfg.get("write_report_json", True))
    persist_csv_exports = bool(reporting_cfg.get("persist_csv_exports", True))
    auto_stage_artifacts = bool(reporting_cfg.get("auto_stage_artifacts", False))

    if write_report_json:
        _write_json(artifact_dir / "report.json", report_payload)

    if "config" in state:
        (artifact_dir / "config.yaml").write_text(yaml.safe_dump(state["config"], sort_keys=False), encoding="utf-8")

    _write_json(artifact_dir / "dataset_metadata.json", dict(state.get("dataset_metadata", {})))
    _write_json(artifact_dir / "split_metadata.json", dict(state.get("split_metadata", {})))
    _write_json(artifact_dir / "strategy_params.json", dict(state.get("strategy_params", {})))
    _write_json(artifact_dir / "optimization_events.json", list(state.get("optimization_events", [])))
    _write_json(artifact_dir / "feature_importances.json", list(state.get("feature_importances", [])), sort_keys=False)
    _write_json(artifact_dir / "trades.json", list(state.get("trades", [])), sort_keys=False)
    _write_json(artifact_dir / "trade_history_buffer.json", list(state.get("trade_history_buffer", [])), sort_keys=False)
    _write_json(artifact_dir / "completed_trades.json", list(state.get("completed_trades", [])), sort_keys=False)
    _write_json(artifact_dir / "equity_curve.json", list(state.get("equity_curve", [])), sort_keys=False)
    _write_json(artifact_dir / "returns.json", list(state.get("returns", [])), sort_keys=False)
    benchmark_metrics_payload = dict(state.get("performance", {}).get("benchmark_metrics", {}))
    _write_json(artifact_dir / "benchmark_metrics.json", benchmark_metrics_payload, sort_keys=False)
    _write_json(
        artifact_dir / "benchmark_sufficiency.json",
        _build_benchmark_sufficiency_payload(benchmark_metrics_payload),
        sort_keys=False,
    )
    _write_json(artifact_dir / "run_metrics.json", dict(state.get("performance", {}).get("run_metrics", {})), sort_keys=False)
    _write_json(artifact_dir / "readiness.json", dict(state.get("readiness", {})), sort_keys=False)
    _write_json(artifact_dir / "event_counts.json", report_payload.get("event_counts", {}))
    _write_json(artifact_dir / "event_log.json", list(state.get("event_log", [])), sort_keys=False)
    (artifact_dir / "shap_rule.txt").write_text(str(state.get("shap_rule", "")), encoding="utf-8")

    decision_log = list(state.get("decision_log", []))
    _write_jsonl(artifact_dir / "decision_log.jsonl", decision_log)
    _write_json(artifact_dir / "decision_log.json", decision_log, sort_keys=False)

    regression_entry = _build_regression_ledger_entry(state)
    _write_json(artifact_dir / "regression_ledger_entry.json", regression_entry, sort_keys=False)
    if not state.get("error_info"):
        ledger_snapshot = _append_regression_ledger(artifact_root, regression_entry)
        _write_json(artifact_dir / "regression_ledger_snapshot.json", ledger_snapshot, sort_keys=False)

    if persist_csv_exports:
        _write_csv_records(artifact_dir / "trades.csv", list(state.get("trades", [])))
        _write_csv_records(
            artifact_dir / "equity_curve.csv",
            [
                {"cycle": idx + 1, "equity": float(value)}
                for idx, value in enumerate(state.get("equity_curve", []))
            ],
        )
        cumulative = 1.0
        return_rows: list[dict[str, Any]] = []
        for idx, value in enumerate(state.get("returns", []), start=1):
            ret = float(value)
            cumulative *= 1.0 + ret
            return_rows.append(
                {
                    "cycle": idx,
                    "strategy_return": ret,
                    "cumulative_return": cumulative - 1.0,
                }
            )
        _write_csv_records(artifact_dir / "returns.csv", return_rows)

        optimization_rows: list[dict[str, Any]] = []
        for idx, event in enumerate(state.get("optimization_events", []), start=1):
            best_params = event.get("best_params", {}) if isinstance(event, dict) else {}
            diagnostics = event.get("diagnostics", {}) if isinstance(event, dict) else {}
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            optimization_rows.append(
                {
                    "event_index": idx,
                    "split": event.get("split"),
                    "objective_name": event.get("objective_name"),
                    "objective_value": event.get("objective_value"),
                    "objective_value_adjusted": event.get("objective_value_adjusted", event.get("objective_value")),
                    "long_threshold": best_params.get("long_threshold"),
                    "short_threshold": best_params.get("short_threshold"),
                    "affects_future_oos_only": event.get("affects_future_oos_only"),
                    "guard_enabled": diagnostics.get("guard_enabled"),
                    "degenerate_regime": diagnostics.get("degenerate_regime"),
                    "transition_count": diagnostics.get("transition_count"),
                    "min_transition_count": diagnostics.get("min_transition_count"),
                    "activity_penalty": diagnostics.get("activity_penalty"),
                    "concentration_penalty": diagnostics.get("concentration_penalty"),
                    "penalty_total": diagnostics.get("penalty_total"),
                    "raw_sharpe": diagnostics.get("raw_sharpe"),
                }
            )
        _write_csv_records(artifact_dir / "optimization_events.csv", optimization_rows)
        _write_csv_records(
            artifact_dir / "decision_summary.csv",
            [decision_row_to_summary(row) for row in decision_log],
        )

    if state.get("error_info"):
        error_info = _normalize_error_info(state.get("error_info"))
        _write_json(artifact_dir / "error_summary.json", error_info)
        (artifact_dir / "traceback.txt").write_text(error_info.get("traceback", ""), encoding="utf-8")
        _write_json(
            artifact_dir / "last_state_snapshot.json",
            dict(state.get("last_state_snapshot", build_last_state_snapshot(state))),
            sort_keys=False,
        )

    _stage_artifact_dir_non_blocking(
        project_root,
        artifact_dir,
        enabled=auto_stage_artifacts,
    )

    return artifact_dir
