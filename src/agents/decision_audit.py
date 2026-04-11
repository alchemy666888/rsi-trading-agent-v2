from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


DEFAULT_DECISION_FEATURES = [
    "ret_1",
    "ret_5",
    "rsi_14",
    "macd_hist_12_26_9",
    "atr_14",
    "realized_vol_base",
    "volatility_regime",
    "volume",
    "close",
]


def _to_utc_iso(timestamp_ms: Any) -> str | None:
    try:
        dt = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return None


def position_label(position: int) -> str:
    if position > 0:
        return "LONG"
    if position < 0:
        return "SHORT"
    return "FLAT"


def select_feature_snapshot(
    row: dict[str, Any],
    *,
    feature_names: list[str] | None = None,
    max_features: int = 12,
) -> dict[str, Any]:
    candidates = feature_names if feature_names else DEFAULT_DECISION_FEATURES
    snapshot: dict[str, Any] = {}
    for feature in candidates:
        if feature in row and row.get(feature) is not None:
            snapshot[feature] = row.get(feature)
        if len(snapshot) >= max_features:
            break
    return snapshot


def build_decision_audit_row(
    state: dict[str, Any],
    *,
    strategy_return: float,
    realized_pnl_pct: float | None,
    unrealized_pnl_pct: float | None,
) -> dict[str, Any]:
    current_row = state.get("current_row", {})
    if not isinstance(current_row, dict):
        current_row = {}
    prediction = state.get("prediction", {})
    if not isinstance(prediction, dict):
        prediction = {}
    risk_status = state.get("risk_status", {})
    if not isinstance(risk_status, dict):
        risk_status = {}
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}

    reporting_cfg = dict(config.get("reporting", {}))
    feature_names = reporting_cfg.get("decision_feature_columns")
    if not isinstance(feature_names, list):
        feature_names = None

    market_timestamp = current_row.get("timestamp")
    reason_codes = list(risk_status.get("reasons", []))
    if state.get("last_action") == "HOLD" and not reason_codes:
        reason_codes.append("no_trade_signal_or_no_position_change")

    row = {
        "cursor": state.get("cursor"),
        "bar_timestamp": market_timestamp,
        "bar_timestamp_utc": _to_utc_iso(market_timestamp),
        "close_price": current_row.get("close"),
        "selected_features": select_feature_snapshot(
            current_row,
            feature_names=feature_names,
            max_features=int(reporting_cfg.get("decision_feature_count", 12)),
        ),
        "model_output": {
            "prob_up": prediction.get("prob_up"),
            "regime": prediction.get("regime"),
            "source_model": prediction.get("source_model"),
        },
        "thresholds": state.get("strategy_params", {}),
        "pre_risk_proposed_action": state.get("pre_risk_action", "FLAT"),
        "pre_risk_proposed_position": state.get("proposed_position", 0),
        "risk_constraints_applied": {
            "paused": bool(risk_status.get("paused", False)),
            "blocked_high_volatility": bool(risk_status.get("blocked_high_volatility", False)),
            "stop_triggered": bool(risk_status.get("stop_triggered", False)),
            "take_profit_triggered": bool(risk_status.get("take_profit_triggered", False)),
            "reasons": reason_codes,
        },
        "final_action": state.get("last_action"),
        "from_position": state.get("position", 0),
        "target_position": state.get("target_position", 0),
        "resulting_position": state.get("target_position", 0),
        "strategy_return": float(strategy_return),
        "realized_pnl_pct": realized_pnl_pct,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "reason_codes": reason_codes,
    }
    return row


def decision_row_to_summary(row: dict[str, Any]) -> dict[str, Any]:
    model_output = row.get("model_output", {})
    thresholds = row.get("thresholds", {})
    risk_constraints = row.get("risk_constraints_applied", {})
    reason_codes = row.get("reason_codes", [])

    return {
        "cursor": row.get("cursor"),
        "bar_timestamp": row.get("bar_timestamp"),
        "bar_timestamp_utc": row.get("bar_timestamp_utc"),
        "close_price": row.get("close_price"),
        "prob_up": model_output.get("prob_up") if isinstance(model_output, dict) else None,
        "regime": model_output.get("regime") if isinstance(model_output, dict) else None,
        "long_threshold": thresholds.get("long_threshold") if isinstance(thresholds, dict) else None,
        "short_threshold": thresholds.get("short_threshold") if isinstance(thresholds, dict) else None,
        "pre_risk_proposed_action": row.get("pre_risk_proposed_action"),
        "pre_risk_proposed_position": row.get("pre_risk_proposed_position"),
        "final_action": row.get("final_action"),
        "from_position": row.get("from_position"),
        "target_position": row.get("target_position"),
        "strategy_return": row.get("strategy_return"),
        "realized_pnl_pct": row.get("realized_pnl_pct"),
        "unrealized_pnl_pct": row.get("unrealized_pnl_pct"),
        "risk_paused": risk_constraints.get("paused") if isinstance(risk_constraints, dict) else None,
        "risk_stop_triggered": risk_constraints.get("stop_triggered") if isinstance(risk_constraints, dict) else None,
        "risk_take_profit_triggered": risk_constraints.get("take_profit_triggered")
        if isinstance(risk_constraints, dict)
        else None,
        "reason_codes": ",".join(str(code) for code in reason_codes) if isinstance(reason_codes, list) else "",
    }
