from __future__ import annotations

import json
import logging
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.artifacts import persist_run_artifacts
from agents.graph import build_agent_graph
from agents.setup import prepare_experiment
from agents.state import AgentState

SHAP_RULES_PATH = PROJECT_ROOT / "data" / "shap_rules.json"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def configure_logging(config: dict[str, Any]) -> None:
    level_name = str(config["logging"]["level"]).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


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

    run_id = build_run_id()
    artifact_root = Path(str(config["runtime"]["artifact_output_dir"]))
    artifact_dir = str(artifact_root / run_id)
    initial_state = prepare_experiment(config, run_id=run_id, artifact_dir=artifact_dir)

    graph = build_agent_graph()
    final_state: AgentState = graph.invoke(initial_state, config={"recursion_limit": 2000})

    persisted_path = persist_run_artifacts(PROJECT_ROOT, final_state)
    logger.info("Research prototype run complete. Run ID=%s", run_id)
    logger.info("Artifact bundle written to %s", persisted_path)
    logger.info("Held-out Sharpe=%.4f", float(final_state["performance"]["run_metrics"]["sharpe"]))


if __name__ == "__main__":
    main()
