from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def normalize_logging_config(config: dict[str, Any]) -> dict[str, Any]:
    logging_cfg = dict(config.get("logging", {}))
    return {
        "level": str(logging_cfg.get("level", "INFO")).upper(),
        "enable_json_logs": bool(logging_cfg.get("enable_json_logs", True)),
        "enable_console_logs": bool(logging_cfg.get("enable_console_logs", True)),
        "json_log_filename": str(logging_cfg.get("json_log_filename", "events.jsonl")),
        "decision_log_enabled": bool(logging_cfg.get("decision_log_enabled", True)),
    }


class JsonlEventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "event_payload", None)
        if payload is None:
            payload = {
                "timestamp_utc": utc_now_iso(),
                "event_type": "log_message",
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
            }
        return json.dumps(_json_safe(payload), sort_keys=True)


class ConsoleEventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "event_payload", None)
        if not isinstance(payload, dict):
            return f"{utc_now_iso()} | {record.levelname} | {record.getMessage()}"

        stage = str(payload.get("stage", "n/a"))
        event_type = str(payload.get("event_type", "n/a"))
        cursor = payload.get("cursor")
        message = str(payload.get("message", record.getMessage()))
        return f"{payload.get('timestamp_utc', utc_now_iso())} | {record.levelname} | {stage}:{event_type} | cursor={cursor} | {message}"


def _extract_market_timestamp(state: dict[str, Any]) -> int | None:
    current_row = state.get("current_row", {})
    if isinstance(current_row, dict) and current_row.get("timestamp") is not None:
        return int(current_row["timestamp"])
    prediction = state.get("prediction", {})
    if isinstance(prediction, dict) and prediction.get("signal_timestamp") is not None:
        return int(prediction["signal_timestamp"])
    return None


@dataclass
class RunEventLogger:
    logger: logging.Logger
    run_id: str
    symbol: str
    timeframe: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def _build_context_from_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        if not state:
            return {}
        prediction = state.get("prediction", {}) if isinstance(state.get("prediction"), dict) else {}
        risk_status = state.get("risk_status", {})
        if not isinstance(risk_status, dict):
            risk_status = {}
        return {
            "cursor": state.get("cursor"),
            "market_timestamp": _extract_market_timestamp(state),
            "position": state.get("position"),
            "target_position": state.get("target_position"),
            "equity": state.get("equity"),
            "prediction_probability": prediction.get("prob_up"),
            "prediction_regime": prediction.get("regime"),
            "risk_status": {
                "paused": risk_status.get("paused"),
                "reasons": risk_status.get("reasons", []),
                "stop_triggered": risk_status.get("stop_triggered"),
                "take_profit_triggered": risk_status.get("take_profit_triggered"),
                "blocked_high_volatility": risk_status.get("blocked_high_volatility"),
            },
            "thresholds": state.get("strategy_params"),
        }

    def emit(
        self,
        *,
        stage: str,
        event_type: str,
        message: str,
        state: dict[str, Any] | None = None,
        level: int = logging.INFO,
        **extra: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp_utc": utc_now_iso(),
            "run_id": self.run_id,
            "stage": stage,
            "event_type": event_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "message": message,
        }
        payload.update(self._build_context_from_state(state))
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        safe_payload = _json_safe(payload)
        if isinstance(safe_payload, dict):
            self.events.append(safe_payload)
            self.logger.log(level, message, extra={"event_payload": safe_payload})
            return safe_payload

        fallback_payload = {
            "timestamp_utc": utc_now_iso(),
            "run_id": self.run_id,
            "stage": stage,
            "event_type": event_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "message": message,
        }
        self.events.append(fallback_payload)
        self.logger.log(level, message, extra={"event_payload": fallback_payload})
        return fallback_payload


_ACTIVE_EVENT_LOGGER: RunEventLogger | None = None


def setup_run_logging(
    config: dict[str, Any],
    artifact_dir: Path,
    run_id: str,
    *,
    symbol: str,
    timeframe: str,
) -> RunEventLogger:
    global _ACTIVE_EVENT_LOGGER
    normalized = normalize_logging_config(config)
    log_level = getattr(logging, normalized["level"], logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_console_enabled = any(getattr(handler, "_rsi_root_console", False) for handler in root_logger.handlers)
    if normalized["enable_console_logs"] and not root_console_enabled:
        root_handler = logging.StreamHandler(stream=sys.stdout)
        root_handler._rsi_root_console = True  # type: ignore[attr-defined]
        root_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root_handler.setLevel(log_level)
        root_logger.addHandler(root_handler)

    event_logger = logging.getLogger(f"agents.events.{run_id}")
    event_logger.handlers.clear()
    event_logger.setLevel(log_level)
    event_logger.propagate = False

    if normalized["enable_console_logs"]:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(ConsoleEventFormatter())
        event_logger.addHandler(console_handler)

    if normalized["enable_json_logs"]:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        json_log_path = artifact_dir / normalized["json_log_filename"]
        json_handler = logging.FileHandler(json_log_path, mode="a", encoding="utf-8")
        json_handler.setLevel(log_level)
        json_handler.setFormatter(JsonlEventFormatter())
        event_logger.addHandler(json_handler)

    if not event_logger.handlers:
        event_logger.addHandler(logging.NullHandler())

    _ACTIVE_EVENT_LOGGER = RunEventLogger(logger=event_logger, run_id=run_id, symbol=symbol, timeframe=timeframe)
    return _ACTIVE_EVENT_LOGGER


def emit_event(
    state: dict[str, Any] | None,
    *,
    stage: str,
    event_type: str,
    message: str,
    level: int = logging.INFO,
    **extra: Any,
) -> dict[str, Any] | None:
    global _ACTIVE_EVENT_LOGGER
    logger: RunEventLogger | None = None
    if state:
        state_logger = state.get("event_logger")
        if isinstance(state_logger, RunEventLogger):
            logger = state_logger
    if logger is None:
        logger = _ACTIVE_EVENT_LOGGER
    if logger is not None:
        return logger.emit(
            stage=stage,
            event_type=event_type,
            message=message,
            state=state,
            level=level,
            **extra,
        )
    return None
