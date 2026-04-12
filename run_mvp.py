from __future__ import annotations

import hashlib
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
    optimization_events = list(state.get("optimization_events", []))
    validation_sharpe = 0.0
    if optimization_events:
        validation_sharpe = float(optimization_events[-1].get("objective_value", 0.0))

    benchmark_metrics = dict(state.get("performance", {}).get("benchmark_metrics", {}))
    overall = dict(benchmark_metrics.get("overall", {}))
    walk_forward_mean_sharpe = float(overall.get("mean_sharpe", 0.0))
    held_out_sharpe = float(state.get("performance", {}).get("run_metrics", {}).get("sharpe", 0.0))

    warnings: list[str] = []
    if validation_sharpe < 0.0 and walk_forward_mean_sharpe < 0.0 and held_out_sharpe < 0.0:
        warnings.append(
            "Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative."
        )

    return {
        "validation_sharpe": validation_sharpe,
        "walk_forward_mean_sharpe": walk_forward_mean_sharpe,
        "held_out_sharpe": held_out_sharpe,
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
        readiness = evaluate_readiness(final_state)
        final_state["readiness"] = readiness
        if readiness.get("warnings"):
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
