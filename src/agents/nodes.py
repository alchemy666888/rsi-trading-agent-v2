from __future__ import annotations

from datetime import datetime, timezone

from agents.evaluation import TIMEFRAME_MINUTES, compute_run_metrics
from agents.modeling import predict_probability
from agents.risk import evaluate_risk
from agents.state import AgentState


def data_node(state: AgentState) -> AgentState:
    historical_data = state["historical_data"]
    cursor = int(state["cursor"])
    split_metadata = state["split_metadata"]
    oos_end = int(split_metadata["oos_end"])
    done = bool(state.get("paused", False)) or cursor >= oos_end
    if done:
        return {"done": True}

    current_row = historical_data.row(cursor, named=True)
    return {"current_row": current_row, "done": False}


def predict_node(state: AgentState) -> AgentState:
    row = state["current_row"]
    prob_up = predict_probability(row, state.get("lightgbm_model"), state.get("feature_columns", []))
    regime = "high_volatility" if float(row.get("volatility_regime", 0.0)) >= 0.5 else "normal"
    next_timestamp = int(state["historical_data"]["timestamp"][int(state["cursor"]) + 1])
    return {
        "prediction": {
            "prob_up": prob_up,
            "regime": regime,
            "signal_timestamp": int(row["timestamp"]),
            "execution_timestamp": next_timestamp,
            "source_model": "lightgbm_baseline" if state.get("lightgbm_model") is not None else "fallback",
        }
    }


def risk_node(state: AgentState) -> AgentState:
    strategy_params = state["strategy_params"]
    prediction = state["prediction"]
    proposed_position = 0
    if float(prediction["prob_up"]) > float(strategy_params["long_threshold"]):
        proposed_position = 1
    elif float(prediction["prob_up"]) < float(strategy_params["short_threshold"]):
        proposed_position = -1

    target_position, risk_status = evaluate_risk(state, proposed_position)
    return {
        "target_position": target_position,
        "risk_status": risk_status,
        "paused": bool(risk_status["paused"]),
    }


def decision_node(state: AgentState) -> AgentState:
    current_position = int(state.get("position", 0))
    target_position = int(state.get("target_position", current_position))
    action = "HOLD"
    if target_position != current_position:
        action = "LONG" if target_position > 0 else "SHORT" if target_position < 0 else "FLAT"
    return {"last_action": action}


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

    # Perpetual funding cost: charged every 8 h while in a position.
    # Interval in bars adapts to the base timeframe (480 at 1m, 32 at 15m, etc.).
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
    if target_position != current_position:
        trade_event = {
            "cycle": len(returns),
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
    if target_position != current_position:
        entry_price = None if target_position == 0 else close_now

    cycle_count = len(returns)
    performance = {
        "run_metrics": compute_run_metrics(returns, equity_curve, initial_equity, trades, timeframe=tf_str),
        "benchmark_metrics": state["performance"]["benchmark_metrics"],
    }
    next_cursor = cursor + 1
    done = bool(state.get("paused", False)) or next_cursor >= int(state["split_metadata"]["oos_end"])
    return {
        "cursor": next_cursor,
        "equity": equity,
        "returns": returns,
        "equity_curve": equity_curve,
        "trades": trades,
        "trade_history_buffer": trade_history_buffer[-int(config["simulation"].get("trade_history_limit", 10000)) :],
        "position": position,
        "entry_price": entry_price,
        "performance": performance,
        "done": done,
    }


def optimize_node(state: AgentState) -> AgentState:
    return {}
