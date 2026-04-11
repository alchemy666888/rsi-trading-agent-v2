from __future__ import annotations

from typing import Any

from agents.state import RiskStatus


def build_default_risk_status() -> RiskStatus:
    return {
        "paused": False,
        "reasons": [],
        "capped_position": 0,
        "stop_triggered": False,
        "take_profit_triggered": False,
        "blocked_high_volatility": False,
    }


def clamp_position(target_position: int, max_abs_position: int) -> int:
    return max(-max_abs_position, min(max_abs_position, target_position))


def apply_turnover_limit(current_position: int, target_position: int, max_turnover_per_bar: int) -> int:
    delta = target_position - current_position
    if abs(delta) <= max_turnover_per_bar:
        return target_position
    if delta > 0:
        return current_position + max_turnover_per_bar
    return current_position - max_turnover_per_bar


def evaluate_risk(
    state: dict[str, Any],
    proposed_position: int,
) -> tuple[int, RiskStatus]:
    config = state["config"]
    risk_cfg = config.get("risk", {})
    current_position = int(state.get("position", 0))
    current_close = float(state["current_row"]["close"])
    entry_price = state.get("entry_price")
    max_drawdown = float(state.get("performance", {}).get("run_metrics", {}).get("max_drawdown", 0.0))

    max_abs_position = int(risk_cfg.get("max_abs_position", 1))
    max_turnover_per_bar = int(risk_cfg.get("max_turnover_per_bar", 1))
    pause_drawdown = float(risk_cfg.get("max_drawdown_pause", 1.0))
    stop_loss_pct = float(risk_cfg.get("stop_loss_pct", 0.0))
    take_profit_pct = float(risk_cfg.get("take_profit_pct", 0.0))
    block_high_volatility = bool(risk_cfg.get("block_high_volatility", False))

    status = build_default_risk_status()
    target_position = clamp_position(proposed_position, max_abs_position)
    status["capped_position"] = target_position

    if max_drawdown >= pause_drawdown:
        status["paused"] = True
        status["reasons"].append("max_drawdown_pause")
        return 0, status

    if block_high_volatility and state["prediction"]["regime"] == "high_volatility":
        target_position = 0
        status["blocked_high_volatility"] = True
        status["reasons"].append("high_volatility_block")

    if current_position != 0 and entry_price is not None:
        if current_position > 0:
            pnl_pct = (current_close / float(entry_price)) - 1.0
        else:
            pnl_pct = (float(entry_price) / current_close) - 1.0

        if stop_loss_pct > 0.0 and pnl_pct <= -stop_loss_pct:
            target_position = 0
            status["stop_triggered"] = True
            status["reasons"].append("stop_loss")
        if take_profit_pct > 0.0 and pnl_pct >= take_profit_pct:
            target_position = 0
            status["take_profit_triggered"] = True
            status["reasons"].append("take_profit")

    target_position = apply_turnover_limit(current_position, target_position, max_turnover_per_bar)
    status["capped_position"] = target_position
    return target_position, status
