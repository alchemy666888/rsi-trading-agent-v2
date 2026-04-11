from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agents.decision_audit import build_decision_audit_row, position_label
from agents.evaluation import TIMEFRAME_MINUTES, compute_run_metrics
from agents.logging_utils import emit_event
from agents.modeling import predict_probability
from agents.risk import evaluate_risk
from agents.state import AgentState


def _action_from_transition(current_position: int, target_position: int) -> str:
    if target_position == current_position:
        return "HOLD"
    if target_position == 0:
        return "FLAT"
    if target_position > 0:
        return "LONG"
    return "SHORT"


def _compute_realized_pnl_pct(position: int, entry_price: float | None, close_price: float) -> float | None:
    if position == 0 or entry_price is None:
        return None
    if position > 0:
        return (close_price / float(entry_price)) - 1.0
    return (float(entry_price) / close_price) - 1.0


def data_node(state: AgentState) -> AgentState:
    historical_data = state["historical_data"]
    cursor = int(state["cursor"])
    split_metadata = state["split_metadata"]
    oos_end = int(split_metadata["oos_end"])
    done = bool(state.get("paused", False)) or cursor >= oos_end
    if done:
        emit_event(
            state,
            stage="data",
            event_type="oos_cycle_terminal_state",
            message="Data node reached terminal held-out state.",
            done=done,
            paused=bool(state.get("paused", False)),
        )
        return {"done": True}

    current_row = historical_data.row(cursor, named=True)
    return {"current_row": current_row, "done": False}


def predict_node(state: AgentState) -> AgentState:
    row = state["current_row"]
    prob_up = predict_probability(row, state.get("lightgbm_model"), state.get("feature_columns", []))
    regime = "high_volatility" if float(row.get("volatility_regime", 0.0)) >= 0.5 else "normal"
    next_timestamp = int(state["historical_data"]["timestamp"][int(state["cursor"]) + 1])
    prediction = {
        "prob_up": prob_up,
        "regime": regime,
        "signal_timestamp": int(row["timestamp"]),
        "execution_timestamp": next_timestamp,
        "source_model": "lightgbm_baseline" if state.get("lightgbm_model") is not None else "fallback",
    }
    emit_event(
        state,
        stage="predict",
        event_type="prediction_generated",
        message="Prediction generated for next execution bar.",
        prediction=prediction,
    )
    return {"prediction": prediction}


def risk_node(state: AgentState) -> AgentState:
    strategy_params = state["strategy_params"]
    prediction = state["prediction"]
    proposed_position = 0
    threshold_reason = "prob_between_thresholds"
    if float(prediction["prob_up"]) > float(strategy_params["long_threshold"]):
        proposed_position = 1
        threshold_reason = "prob_above_long_threshold"
    elif float(prediction["prob_up"]) < float(strategy_params["short_threshold"]):
        proposed_position = -1
        threshold_reason = "prob_below_short_threshold"

    pre_risk_action = position_label(proposed_position)
    target_position, risk_status = evaluate_risk(state, proposed_position)
    emit_event(
        state,
        stage="risk",
        event_type="risk_evaluated",
        message="Risk policy evaluated proposed position.",
        proposed_position=proposed_position,
        target_position=target_position,
        pre_risk_action=pre_risk_action,
        threshold_reason=threshold_reason,
    )

    if bool(risk_status.get("stop_triggered", False)):
        emit_event(
            state,
            stage="risk",
            event_type="stop_loss_triggered",
            message="Stop-loss triggered; forcing flat target.",
            proposed_position=proposed_position,
            target_position=target_position,
        )
    if bool(risk_status.get("take_profit_triggered", False)):
        emit_event(
            state,
            stage="risk",
            event_type="take_profit_triggered",
            message="Take-profit triggered; forcing flat target.",
            proposed_position=proposed_position,
            target_position=target_position,
        )

    was_paused = bool(state.get("paused", False))
    is_paused = bool(risk_status.get("paused", False))
    if is_paused and not was_paused:
        emit_event(
            state,
            stage="risk",
            event_type="drawdown_pause_activated",
            message="Drawdown pause activated.",
            proposed_position=proposed_position,
            target_position=target_position,
        )
    if was_paused and not is_paused:
        emit_event(
            state,
            stage="risk",
            event_type="drawdown_pause_cleared",
            message="Drawdown pause cleared.",
            proposed_position=proposed_position,
            target_position=target_position,
        )

    return {
        "target_position": target_position,
        "proposed_position": proposed_position,
        "pre_risk_action": pre_risk_action,
        "risk_status": risk_status,
        "paused": bool(risk_status["paused"]),
    }


def decision_node(state: AgentState) -> AgentState:
    current_position = int(state.get("position", 0))
    target_position = int(state.get("target_position", current_position))
    action = _action_from_transition(current_position, target_position)
    emit_event(
        state,
        stage="decision",
        event_type="decision_made",
        message="Decision node produced final action.",
        action=action,
        current_position=current_position,
        target_position=target_position,
        pre_risk_action=state.get("pre_risk_action"),
        proposed_position=state.get("proposed_position"),
    )
    return {"last_action": action}


def _build_completed_trade(
    state: AgentState,
    *,
    current_position: int,
    close_now: float,
    returns_length: int,
) -> dict[str, Any]:
    entry_cycle = state.get("entry_cycle")
    hold_bars = 0 if entry_cycle is None else max(1, returns_length - int(entry_cycle))
    return {
        "direction": current_position,
        "open_cycle": state.get("entry_cycle"),
        "close_cycle": returns_length,
        "open_timestamp": state.get("entry_timestamp"),
        "close_timestamp": state.get("prediction", {}).get("execution_timestamp"),
        "entry_price": state.get("entry_price"),
        "exit_price": close_now,
        "pnl_pct": _compute_realized_pnl_pct(current_position, state.get("entry_price"), close_now),
        "hold_bars": hold_bars,
        "close_reason": list(state.get("risk_status", {}).get("reasons", [])),
    }


def evaluate_node(state: AgentState) -> AgentState:
    config = state["config"]
    data = state["historical_data"]
    cursor = int(state["cursor"])
    current_position = int(state.get("position", 0))
    target_position = int(state.get("target_position", current_position))

    close_now = float(data["close"][cursor])
    close_next = float(data["close"][cursor + 1])
    market_return = (close_next - close_now) / close_now
    slippage = float(config["simulation"]["slippage_bps"]) / 10000.0
    transaction_cost = slippage * abs(target_position - current_position)
    strategy_return = (target_position * market_return) - transaction_cost

    cycle_count_current = int(state.get("cycle_count", 0)) + 1
    funding_rate = float(config["simulation"].get("funding_rate_per_8h", 0.0))
    tf_str = config.get("asset", {}).get("timeframe", "15m")
    tf_minutes = TIMEFRAME_MINUTES.get(tf_str, 15)
    funding_interval_bars = max(1, int(8 * 60 / tf_minutes))
    if funding_rate > 0.0 and cycle_count_current % funding_interval_bars == 0 and current_position != 0:
        strategy_return -= abs(current_position) * funding_rate

    returns = list(state.get("returns", []))
    returns.append(strategy_return)

    initial_equity = float(config["simulation"]["initial_equity"])
    equity = float(state.get("equity", initial_equity)) * (1.0 + strategy_return)
    equity_curve = list(state.get("equity_curve", []))
    equity_curve.append(equity)

    trades = list(state.get("trades", []))
    trade_history_buffer = list(state.get("trade_history_buffer", []))
    completed_trades = list(state.get("completed_trades", []))
    realized_pnl_pct: float | None = None

    if target_position != current_position:
        trade_event = {
            "cycle": len(returns),
            "bar_timestamp": int(state["current_row"]["timestamp"]),
            "signal_timestamp": int(state["prediction"]["signal_timestamp"]),
            "execution_timestamp": int(state["prediction"]["execution_timestamp"]),
            "action": state["last_action"],
            "from_position": current_position,
            "to_position": target_position,
            "price": close_now,
            "prob_up": float(state["prediction"]["prob_up"]),
            "risk_reasons": state.get("risk_status", {}).get("reasons", []),
        }
        trades.append(trade_event)
        trade_history_buffer.append(
            {
                **trade_event,
                "return": strategy_return,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    position = target_position
    entry_price = state.get("entry_price")
    entry_cycle = state.get("entry_cycle")
    entry_timestamp = state.get("entry_timestamp")
    if target_position != current_position and current_position != 0:
        completed_trade = _build_completed_trade(
            state,
            current_position=current_position,
            close_now=close_now,
            returns_length=len(returns),
        )
        realized_pnl_pct = float(completed_trade.get("pnl_pct", 0.0))
        completed_trades.append(completed_trade)
        emit_event(
            state,
            stage="evaluate",
            event_type="trade_closed",
            message="Position closed.",
            closed_trade=completed_trade,
            realized_pnl_pct=realized_pnl_pct,
        )

    if target_position != current_position:
        entry_price = None if target_position == 0 else close_now
        entry_cycle = None if target_position == 0 else len(returns)
        entry_timestamp = None if target_position == 0 else int(state["prediction"]["execution_timestamp"])
    if target_position != current_position and target_position != 0:
        emit_event(
            state,
            stage="evaluate",
            event_type="trade_opened",
            message="Position opened.",
            from_position=current_position,
            to_position=target_position,
            entry_price=close_now,
        )

    unrealized_pnl_pct = _compute_realized_pnl_pct(position, entry_price, close_now)

    decision_log = list(state.get("decision_log", []))
    decision_log_enabled = bool(
        config.get("logging", {}).get(
            "decision_log_enabled",
            config.get("reporting", {}).get("decision_log_enabled", True),
        )
    )
    if decision_log_enabled:
        decision_row = build_decision_audit_row(
            state,
            strategy_return=strategy_return,
            realized_pnl_pct=realized_pnl_pct,
            unrealized_pnl_pct=unrealized_pnl_pct,
        )
        decision_log.append(decision_row)

    cycle_count = len(returns)
    performance = {
        "run_metrics": compute_run_metrics(returns, equity_curve, initial_equity, trades, timeframe=tf_str),
        "benchmark_metrics": state["performance"]["benchmark_metrics"],
    }
    emit_event(
        state,
        stage="evaluate",
        event_type="step_evaluated",
        message="Held-out step evaluated and metrics updated.",
        strategy_return=strategy_return,
        cycle_count=cycle_count,
    )
    next_cursor = cursor + 1
    done = bool(state.get("paused", False)) or next_cursor >= int(state["split_metadata"]["oos_end"])
    return {
        "cursor": next_cursor,
        "equity": equity,
        "returns": returns,
        "equity_curve": equity_curve,
        "trades": trades,
        "trade_history_buffer": trade_history_buffer[-int(config["simulation"].get("trade_history_limit", 10000)) :],
        "completed_trades": completed_trades,
        "decision_log": decision_log,
        "position": position,
        "entry_price": entry_price,
        "entry_cycle": entry_cycle,
        "entry_timestamp": entry_timestamp,
        "performance": performance,
        "done": done,
    }


def optimize_node(state: AgentState) -> AgentState:
    emit_event(
        state,
        stage="optimize",
        event_type="optimize_cycle",
        message="Optimize node completed cycle.",
        done=bool(state.get("done", False)),
    )
    return {}
