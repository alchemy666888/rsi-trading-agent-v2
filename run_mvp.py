from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.artifacts import build_last_state_snapshot, persist_run_artifacts
from agents.graph import build_agent_graph
from agents.logging_utils import setup_run_logging, utc_now_iso
from agents.setup import prepare_experiment
from agents.state import AgentState


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def build_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


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


def _relative_artifact_dir(config: dict[str, Any], run_id: str) -> str:
    artifact_root = Path(str(config.get("runtime", {}).get("artifact_output_dir", "artifacts")))
    return str(artifact_root / run_id)


def _artifact_root_from_run_dir(project_root: Path, artifact_dir: str) -> Path:
    return (project_root / artifact_dir).parent


def _load_regression_ledger(project_root: Path, artifact_dir: str) -> list[dict[str, Any]]:
    ledger_path = _artifact_root_from_run_dir(project_root, artifact_dir) / "regression_ledger.json"
    if not ledger_path.exists():
        return []
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _build_error_info(exc: Exception) -> dict[str, Any]:
    return {
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "failed_at_utc": utc_now_iso(),
    }


def _auto_stage_run_artifacts(project_root: Path, artifact_path: Path, config: dict[str, Any]) -> None:
    reporting_cfg = dict(config.get("reporting", {}))
    if not bool(reporting_cfg.get("auto_stage_artifacts", False)):
        return
    if not (project_root / ".git").exists():
        return
    try:
        rel_artifact_path = artifact_path.relative_to(project_root)
    except Exception:
        rel_artifact_path = artifact_path
    try:
        subprocess.run(
            ["git", "add", str(rel_artifact_path)],
            cwd=project_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


def evaluate_readiness(state: AgentState) -> dict[str, Any]:
    config = dict(state.get("config", {}))
    readiness_cfg = dict(config.get("readiness", {}))
    optimization_events = list(state.get("optimization_events", []))
    validation_sharpe = 0.0
    latest_optimization_event: dict[str, Any] = {}
    if optimization_events:
        latest_optimization_event = dict(optimization_events[-1])
        validation_sharpe = float(latest_optimization_event.get("objective_value", 0.0))

    benchmark_metrics = dict(state.get("performance", {}).get("benchmark_metrics", {}))
    overall = dict(benchmark_metrics.get("overall", {}))
    walk_forward_mean_sharpe = float(overall.get("mean_sharpe", 0.0))
    run_metrics = dict(state.get("performance", {}).get("run_metrics", {}))
    held_out_sharpe = float(run_metrics.get("sharpe", 0.0))
    dataset_metadata = dict(state.get("dataset_metadata", {}))
    source_mode = str(dataset_metadata.get("source_mode", "unknown"))

    warnings: list[str] = []
    triple_negative_sharpe = validation_sharpe < 0.0 and walk_forward_mean_sharpe < 0.0 and held_out_sharpe < 0.0
    if triple_negative_sharpe:
        warnings.append("Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.")

    optimization_diagnostics = latest_optimization_event.get("diagnostics", {})
    if not isinstance(optimization_diagnostics, dict):
        optimization_diagnostics = {}
    degenerate_threshold_calibration = bool(optimization_diagnostics.get("degenerate_regime", False))
    if degenerate_threshold_calibration:
        warnings.append(
            "Validation threshold calibration selected a degenerate activity/exposure regime."
        )

    required_kpi_fields = {
        "bar_win_rate",
        "transition_count",
        "completed_trade_win_rate",
        "completed_trade_count",
    }
    legacy_kpi_fields = {"win_rate", "trade_count"}
    has_required_kpis = required_kpi_fields.issubset(set(run_metrics.keys()))
    has_legacy_kpis = bool(legacy_kpi_fields.intersection(set(run_metrics.keys())))
    transition_count_matches = int(run_metrics.get("transition_count", -1)) == len(list(state.get("trades", [])))
    completed_trade_count_matches = int(run_metrics.get("completed_trade_count", -1)) == len(
        list(state.get("completed_trades", []))
    )
    kpi_schema_consistent = (
        has_required_kpis
        and not has_legacy_kpis
        and transition_count_matches
        and completed_trade_count_matches
    )

    trade_rows = [row for row in state.get("trades", []) if isinstance(row, dict)]
    misaligned_trade_events = 0
    for trade in trade_rows:
        if "bar_timestamp" not in trade or "execution_timestamp" not in trade:
            continue
        if int(trade.get("bar_timestamp", -1)) != int(trade.get("execution_timestamp", -2)):
            misaligned_trade_events += 1
    execution_semantics_consistent = misaligned_trade_events == 0

    sufficiency = dict(benchmark_metrics.get("sufficiency", {}))
    benchmark_sufficiency_artifact_available = bool(sufficiency)
    walk_forward_sufficient = bool(sufficiency.get("overall_sufficient", False)) if benchmark_sufficiency_artifact_available else False
    snapshot_based = source_mode == "snapshot"
    regression_history = [row for row in state.get("regression_history", []) if isinstance(row, dict)]
    regression_history_entry_count = len(regression_history)
    regression_tracking_available = regression_history_entry_count > 0
    min_historical_windows = max(1, int(readiness_cfg.get("min_historical_windows", 5)))
    current_window_key = "|".join(
        [
            str(dataset_metadata.get("snapshot_hash", "n/a")),
            str(dataset_metadata.get("raw_data_hash", "n/a")),
            str(dataset_metadata.get("dataset_hash", "n/a")),
            str(dataset_metadata.get("timestamp_start", "n/a")),
            str(dataset_metadata.get("timestamp_end", "n/a")),
        ]
    )
    historical_window_keys: set[str] = set()
    for row in regression_history:
        dataset_window = row.get("dataset_window", {})
        if not isinstance(dataset_window, dict):
            dataset_window = {}
        key = "|".join(
            [
                str(dataset_window.get("snapshot_hash", row.get("snapshot_hash", "n/a"))),
                str(dataset_window.get("raw_data_hash", row.get("raw_data_hash", "n/a"))),
                str(dataset_window.get("dataset_hash", row.get("dataset_hash", "n/a"))),
                str(dataset_window.get("timestamp_start", row.get("timestamp_start", "n/a"))),
                str(dataset_window.get("timestamp_end", row.get("timestamp_end", "n/a"))),
            ]
        )
        historical_window_keys.add(key)
    historical_window_keys.add(current_window_key)
    historical_window_count = len(historical_window_keys)
    multi_window_evidence_sufficient = historical_window_count >= min_historical_windows

    engineering_blockers: list[dict[str, Any]] = []
    if not execution_semantics_consistent:
        engineering_blockers.append(
            {
                "code": "execution_semantics_parity_failure",
                "message": "Trade execution timestamps do not align with execution-bar fills.",
                "misaligned_trade_events": misaligned_trade_events,
            }
        )
    if not kpi_schema_consistent:
        engineering_blockers.append(
            {
                "code": "kpi_schema_inconsistent",
                "message": "Run metrics do not use the required explicit KPI schema.",
                "has_required_kpis": has_required_kpis,
                "has_legacy_kpis": has_legacy_kpis,
                "transition_count_matches": transition_count_matches,
                "completed_trade_count_matches": completed_trade_count_matches,
            }
        )

    research_blockers: list[dict[str, Any]] = []
    if triple_negative_sharpe:
        research_blockers.append(
            {
                "code": "triple_negative_sharpe",
                "message": "Validation, walk-forward, and held-out Sharpe are all negative.",
                "validation_sharpe": validation_sharpe,
                "walk_forward_mean_sharpe": walk_forward_mean_sharpe,
                "held_out_sharpe": held_out_sharpe,
            }
        )
    if degenerate_threshold_calibration:
        research_blockers.append(
            {
                "code": "degenerate_threshold_calibration",
                "message": "Validation threshold optimization converged to a degenerate activity/exposure regime.",
                "diagnostics": optimization_diagnostics,
            }
        )

    evidence_blockers: list[dict[str, Any]] = []
    if not snapshot_based:
        evidence_blockers.append(
            {
                "code": "exchange_mode_not_regression_eligible",
                "message": "Benchmark/readiness runs must use a frozen snapshot input.",
                "source_mode": source_mode,
            }
        )
    if not benchmark_sufficiency_artifact_available:
        evidence_blockers.append(
            {
                "code": "benchmark_sufficiency_missing",
                "message": "Benchmark sufficiency metadata artifact is missing.",
            }
        )
    elif not walk_forward_sufficient:
        evidence_blockers.append(
            {
                "code": "walk_forward_evidence_insufficient",
                "message": "Walk-forward benchmark does not meet minimum fold/bar sufficiency.",
                "sufficiency": sufficiency,
            }
        )
    if not regression_tracking_available:
        evidence_blockers.append(
            {
                "code": "regression_history_missing",
                "message": "No persisted regression history is available.",
                "regression_history_entry_count": regression_history_entry_count,
            }
        )
    if not multi_window_evidence_sufficient:
        evidence_blockers.append(
            {
                "code": "multi_window_history_insufficient",
                "message": "Insufficient historical windows in persisted regression history.",
                "historical_window_count": historical_window_count,
                "minimum_required": min_historical_windows,
            }
        )

    engineering_green = not engineering_blockers
    research_green = not research_blockers
    evidence_green = not evidence_blockers
    hard_blockers = [*engineering_blockers, *research_blockers, *evidence_blockers]
    if hard_blockers:
        warnings.append(
            "Readiness hard blockers are present; do not advance to fine-tuning or next phase."
        )
    if not multi_window_evidence_sufficient:
        warnings.append(
            "Fine-tuning gate remains closed until multi-window regression evidence meets minimum history requirements."
        )

    phase_gate_green = engineering_green and research_green and evidence_green

    return {
        "validation_sharpe": validation_sharpe,
        "walk_forward_mean_sharpe": walk_forward_mean_sharpe,
        "held_out_sharpe": held_out_sharpe,
        "bad_research_performance": {
            "triple_negative_sharpe": triple_negative_sharpe,
        },
        "engineering_validity": {
            "green": engineering_green,
            "execution_semantics_consistent": execution_semantics_consistent,
            "kpi_schema_consistent": kpi_schema_consistent,
            "engineering_blockers": engineering_blockers,
        },
        "research_validity": {
            "green": research_green,
            "triple_negative_sharpe": triple_negative_sharpe,
            "degenerate_threshold_calibration": degenerate_threshold_calibration,
            "research_blockers": research_blockers,
        },
        "evidence_sufficiency": {
            "green": evidence_green,
            "snapshot_based": snapshot_based,
            "walk_forward_sufficient": walk_forward_sufficient,
            "benchmark_sufficiency_artifact_available": benchmark_sufficiency_artifact_available,
            "regression_tracking_available": regression_tracking_available,
            "regression_history_entry_count": regression_history_entry_count,
            "historical_window_count": historical_window_count,
            "minimum_historical_windows": min_historical_windows,
            "multi_window_evidence_sufficient": multi_window_evidence_sufficient,
            "evidence_blockers": evidence_blockers,
        },
        "hard_blockers": hard_blockers,
        "phase_gate_decision": "advance" if phase_gate_green else "do_not_advance",
        "fine_tuning_gate_open": phase_gate_green and multi_window_evidence_sufficient,
        "warnings": warnings,
    }


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    run_id = build_run_id()
    artifact_dir = _relative_artifact_dir(config, run_id)
    artifact_dir_abs = PROJECT_ROOT / artifact_dir
    symbol = str(config.get("asset", {}).get("symbol", "unknown"))
    timeframe = str(config.get("asset", {}).get("timeframe", "unknown"))
    event_logger = setup_run_logging(
        config=config,
        artifact_dir=artifact_dir_abs,
        run_id=run_id,
        symbol=symbol,
        timeframe=timeframe,
    )
    logger = logging.getLogger("run_mvp")

    run_metadata = {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "git_commit_hash": _get_git_commit_hash(),
        "config_hash": _compute_config_hash(config_path),
    }

    event_logger.emit(
        stage="run",
        event_type="config_loaded",
        message="Config loaded from disk.",
        config_path=str(config_path),
    )
    event_logger.emit(
        stage="run",
        event_type="run_started",
        message="Research prototype run started.",
        run_metadata=run_metadata,
    )

    graph = build_agent_graph()
    state_for_failure: AgentState = {
        "config": config,
        "run_id": run_id,
        "artifact_dir": artifact_dir,
        "event_log": event_logger.events,
        "run_metadata": run_metadata,
    }

    try:
        initial_state = prepare_experiment(
            config,
            run_id=run_id,
            artifact_dir=artifact_dir,
            run_metadata=run_metadata,
            event_logger=event_logger,
        )
        state_for_failure = initial_state
        event_logger.emit(
            stage="run",
            event_type="oos_simulation_started",
            message="Out-of-sample simulation loop started.",
            oos_start=initial_state.get("split_metadata", {}).get("oos_start"),
            oos_end=initial_state.get("split_metadata", {}).get("oos_end"),
        )
        final_state: AgentState = graph.invoke(initial_state, config={"recursion_limit": 2000})
        final_state["event_log"] = event_logger.events
        state_for_failure = final_state
        event_logger.emit(
            stage="run",
            event_type="oos_simulation_completed",
            message="Out-of-sample simulation loop completed.",
            completed_cycles=len(final_state.get("returns", [])),
        )
        final_state["regression_history"] = _load_regression_ledger(PROJECT_ROOT, artifact_dir)
        readiness = evaluate_readiness(final_state)
        final_state["readiness"] = readiness
        if readiness.get("warnings") or readiness.get("hard_blockers"):
            event_logger.emit(
                stage="run",
                event_type="readiness_warning",
                message="Readiness warning detected.",
                readiness=readiness,
            )

        event_logger.emit(
            stage="artifacts",
            event_type="artifact_persistence_started",
            message="Artifact persistence started.",
            artifact_dir=artifact_dir,
        )
        persisted_path = persist_run_artifacts(PROJECT_ROOT, final_state)
        event_logger.emit(
            stage="artifacts",
            event_type="artifact_persistence_completed",
            message="Artifact persistence completed.",
            artifact_dir=str(persisted_path),
        )
        event_logger.emit(
            stage="run",
            event_type="run_completed",
            message="Run completed successfully.",
            run_metrics=final_state.get("performance", {}).get("run_metrics", {}),
        )
        _auto_stage_run_artifacts(PROJECT_ROOT, persisted_path, config)
        logger.info("Research prototype run complete. Run ID=%s", run_id)
        logger.info("Artifact bundle written to %s", persisted_path)
        logger.info("Held-out Sharpe=%.4f", float(final_state["performance"]["run_metrics"]["sharpe"]))
        return
    except Exception as exc:
        error_info = _build_error_info(exc)
        state_for_failure["event_log"] = event_logger.events
        state_for_failure["error_info"] = error_info
        state_for_failure["last_state_snapshot"] = build_last_state_snapshot(state_for_failure)
        event_logger.emit(
            stage="run",
            event_type="run_failed",
            message="Run failed with exception.",
            level=logging.ERROR,
            error_type=error_info["error_type"],
            error_message=error_info["error_message"],
        )
        try:
            event_logger.emit(
                stage="artifacts",
                event_type="artifact_persistence_started",
                message="Partial artifact persistence started after failure.",
                artifact_dir=artifact_dir,
            )
            persisted_path = persist_run_artifacts(PROJECT_ROOT, state_for_failure)
            event_logger.emit(
                stage="artifacts",
                event_type="artifact_persistence_completed",
                message="Partial artifact persistence completed.",
                artifact_dir=str(persisted_path),
            )
            _auto_stage_run_artifacts(PROJECT_ROOT, persisted_path, config)
            logger.error("Run failed. Partial artifacts written to %s", persisted_path)
        except Exception:
            logger.exception("Failed to persist partial artifact bundle after run failure.")
        raise


if __name__ == "__main__":
    main()
