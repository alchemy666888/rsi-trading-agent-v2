from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import ccxt  # type: ignore[import-untyped]
import numpy as np
import optuna
import polars as pl
import shap
import talib

from agents.state import AgentState

LOGGER = logging.getLogger(__name__)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_features(raw_df: pl.DataFrame) -> pl.DataFrame:
    close = raw_df["close"].to_numpy()
    high = raw_df["high"].to_numpy()
    low = raw_df["low"].to_numpy()

    rsi = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(
        close,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9,
    )
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
    atr = talib.ATR(high, low, close, timeperiod=14)

    feature_df = raw_df.with_columns(
        [
            pl.Series("rsi", rsi),
            pl.Series("macd", macd),
            pl.Series("macd_signal", macd_signal),
            pl.Series("macd_hist", macd_hist),
            pl.Series("bb_upper", bb_upper),
            pl.Series("bb_middle", bb_middle),
            pl.Series("bb_lower", bb_lower),
            pl.Series("atr", atr),
            pl.col("close").pct_change().alias("ret_1"),
            pl.col("close").pct_change(5).alias("ret_5"),
            pl.col("volume").pct_change().alias("vol_chg_1"),
        ]
    )

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "atr",
        "ret_1",
        "ret_5",
        "vol_chg_1",
    ]
    feature_df = feature_df.with_columns(
        [pl.col(col).fill_null(0.0).fill_nan(0.0) for col in numeric_cols]
    )
    return feature_df


def _load_historical_btc_data(config: dict[str, Any]) -> pl.DataFrame:
    asset_cfg = config["asset"]
    api_cfg = config.get("api_keys", {})

    exchange_name = asset_cfg["exchange"]
    exchange_cls = getattr(ccxt, exchange_name)
    exchange_params = {
        "enableRateLimit": True,
        "apiKey": api_cfg.get("ccxt_api_key", ""),
        "secret": api_cfg.get("ccxt_api_secret", ""),
    }
    exchange = exchange_cls(exchange_params)

    ohlcv = exchange.fetch_ohlcv(
        symbol=asset_cfg["symbol"],
        timeframe=asset_cfg["timeframe"],
        limit=int(asset_cfg["fetch_limit"]),
    )
    if not ohlcv:
        raise RuntimeError("No OHLCV rows returned from exchange.")

    raw_df = pl.DataFrame(
        {
            "timestamp": [int(row[0]) for row in ohlcv],
            "open": [float(row[1]) for row in ohlcv],
            "high": [float(row[2]) for row in ohlcv],
            "low": [float(row[3]) for row in ohlcv],
            "close": [float(row[4]) for row in ohlcv],
            "volume": [float(row[5]) for row in ohlcv],
        }
    ).sort("timestamp")

    LOGGER.info("Loaded %s BTC candles from %s", raw_df.height, exchange_name)
    return _build_features(raw_df)


def _prediction_probability(row: dict[str, Any]) -> float:
    rsi = _safe_float(row.get("rsi"), 50.0)
    macd_hist = _safe_float(row.get("macd_hist"), 0.0)
    close = max(_safe_float(row.get("close"), 1.0), 1e-9)
    bb_middle = _safe_float(row.get("bb_middle"), close)
    atr = _safe_float(row.get("atr"), 0.0)

    rsi_signal = (rsi - 50.0) / 50.0
    macd_signal = (macd_hist / close) * 1000.0
    mean_reversion_signal = (close - bb_middle) / close
    atr_signal = atr / close

    linear_score = (
        (1.35 * rsi_signal)
        + (0.90 * macd_signal)
        - (0.65 * mean_reversion_signal)
        - (0.25 * atr_signal)
    )
    return float(np.clip(_sigmoid(linear_score), 0.0, 1.0))


def _compute_performance(
    returns: list[float],
    equity_curve: list[float],
    initial_equity: float,
) -> dict[str, float]:
    if not returns:
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
        }

    returns_array = np.array(returns, dtype=float)
    mean_return = float(returns_array.mean())
    std_return = float(returns_array.std(ddof=1)) if len(returns_array) > 1 else 0.0
    annualization = math.sqrt(365 * 24 * 60)  # 1m BTC bars
    sharpe = (mean_return / std_return) * annualization if std_return > 1e-12 else 0.0

    equity_array = np.array([initial_equity] + equity_curve, dtype=float)
    rolling_max = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array / rolling_max) - 1.0
    max_drawdown = float(abs(np.min(drawdowns)))

    win_rate = float((returns_array > 0.0).mean())
    total_return = float((equity_array[-1] / initial_equity) - 1.0)
    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_return": total_return,
    }


def _run_optuna_tune(state: AgentState) -> tuple[dict[str, float], float]:
    data = state["historical_data"]
    cfg = state["config"]
    simulation_cfg = cfg["simulation"]
    cursor = int(state["cursor"])

    start = max(0, cursor - 400)
    window = data.slice(start, cursor - start + 1)
    closes = window["close"].to_numpy()
    rsis = window["rsi"].to_numpy()
    macd_hists = window["macd_hist"].to_numpy()
    bb_mids = window["bb_middle"].to_numpy()
    atrs = window["atr"].to_numpy()

    if len(closes) < 20:
        default_params = {
            "long_threshold": float(simulation_cfg["long_threshold"]),
            "short_threshold": float(simulation_cfg["short_threshold"]),
        }
        return default_params, 0.0

    slippage = float(simulation_cfg["slippage_bps"]) / 10000.0

    def objective(trial: optuna.Trial) -> float:
        long_threshold = trial.suggest_float("long_threshold", 0.52, 0.72)
        short_threshold = trial.suggest_float("short_threshold", 0.28, 0.48)
        if short_threshold >= long_threshold:
            return -10.0

        position = 0
        previous_position = 0
        strategy_returns: list[float] = []

        for idx in range(len(closes) - 1):
            row = {
                "rsi": float(rsis[idx]),
                "macd_hist": float(macd_hists[idx]),
                "close": float(closes[idx]),
                "bb_middle": float(bb_mids[idx]),
                "atr": float(atrs[idx]),
            }
            prob_up = _prediction_probability(row)
            if prob_up > long_threshold:
                position = 1
            elif prob_up < short_threshold:
                position = -1

            market_return = (float(closes[idx + 1]) - float(closes[idx])) / float(
                closes[idx]
            )
            transaction_cost = slippage * abs(position - previous_position)
            strategy_returns.append((position * market_return) - transaction_cost)
            previous_position = position

        if len(strategy_returns) < 5:
            return -10.0
        arr = np.array(strategy_returns, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        if std <= 1e-12:
            return -10.0
        return float((arr.mean() / std) * math.sqrt(365 * 24 * 60))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(simulation_cfg["optuna_trials"]))
    return {
        "long_threshold": float(study.best_params["long_threshold"]),
        "short_threshold": float(study.best_params["short_threshold"]),
    }, float(study.best_value)


def _derive_shap_rule(state: AgentState) -> str:
    data = state["historical_data"]
    cursor = int(state["cursor"])
    lookback = min(150, cursor + 1)
    segment = data.slice(cursor - lookback + 1, lookback)
    feature_cols = ["rsi", "macd_hist", "ret_1"]
    x = segment.select(feature_cols).to_numpy()

    if len(x) < 20:
        return "If RSI rises above 55, long probability tends to increase."

    background = x[: min(50, len(x) - 1)]
    target = x[-1:]

    def model_fn(values: np.ndarray) -> np.ndarray:
        rsi_signal = (values[:, 0] - 50.0) / 50.0
        macd_signal = values[:, 1] * 1000.0
        ret_signal = values[:, 2] * 500.0
        logits = (1.35 * rsi_signal) + (0.90 * macd_signal) - (0.65 * ret_signal)
        logits = np.clip(logits, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logits))

    try:
        explainer = shap.KernelExplainer(model_fn, background)
        shap_values = explainer.shap_values(target, nsamples=80)
        shap_vector = np.array(shap_values, dtype=float).reshape(-1)
        top_idx = int(np.argmax(np.abs(shap_vector)))
        direction = "increases" if shap_vector[top_idx] >= 0 else "decreases"
        feature = feature_cols[top_idx]
        feature_value = float(target[0, top_idx])
        return (
            f"If {feature} is elevated (current={feature_value:.4f}), "
            f"the model estimate for long probability typically {direction}."
        )
    except Exception as exc:  # pragma: no cover - fallback path
        LOGGER.warning("SHAP rule extraction failed, using fallback rule: %s", exc)
        return "If RSI rises above 55, long probability tends to increase."


def update_lora_adapter(_: AgentState) -> None:
    # MVP stub: keeps the LoRA update hook in place from day one.
    LOGGER.info("LoRA adapter update hook executed (MVP dummy).")


def append_to_replay_buffer(state: AgentState) -> list[dict[str, Any]]:
    replay = list(state.get("replay_buffer", []))
    replay.append(
        {
            "cycle": int(state["cycle_count"]),
            "return": float(state["returns"][-1]),
            "action": state["last_action"],
            "position": int(state["position"]),
            "regime": state["prediction"]["regime"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    limit = int(state["config"]["simulation"]["replay_buffer_limit"])
    return replay[-limit:]


def data_node(state: AgentState) -> AgentState:
    config = state["config"]
    historical_data = state.get("historical_data")
    if historical_data is None:
        historical_data = _load_historical_btc_data(config)
        cursor = int(config["runtime"]["warmup_bars"])
        current_row = historical_data.row(cursor, named=True)
        features_df = historical_data.slice(cursor, 1)
        LOGGER.info(
            "Data node initialized with %s rows; warmup cursor=%s.",
            historical_data.height,
            cursor,
        )
        return {
            "historical_data": historical_data,
            "cursor": cursor,
            "current_row": current_row,
            "features_df": features_df,
            "done": False,
        }

    cursor = int(state["cursor"])
    max_cycles = int(config["runtime"]["max_cycles"])
    cycle_count = int(state.get("cycle_count", 0))
    done = cycle_count >= max_cycles or cursor >= (historical_data.height - 2)
    if done:
        return {"done": True}

    current_row = historical_data.row(cursor, named=True)
    features_df = historical_data.slice(cursor, 1)
    return {
        "current_row": current_row,
        "features_df": features_df,
        "done": False,
    }


def predict_node(state: AgentState) -> AgentState:
    row = state["current_row"]
    prob_up = _prediction_probability(row)
    one_bar_return = abs(_safe_float(row.get("ret_1"), 0.0))
    regime = "high_volatility" if one_bar_return >= 0.002 else "normal"
    return {
        "prediction": {
            "prob_up": prob_up,
            "regime": regime,
        }
    }


def decision_node(state: AgentState) -> AgentState:
    prediction = state["prediction"]
    config = state["config"]
    params = state.get(
        "strategy_params",
        {
            "long_threshold": float(config["simulation"]["long_threshold"]),
            "short_threshold": float(config["simulation"]["short_threshold"]),
        },
    )

    prob_up = float(prediction["prob_up"])
    previous_position = int(state.get("position", 0))
    position = previous_position
    action = "HOLD"

    if prob_up > float(params["long_threshold"]):
        position = 1
        action = "LONG"
    elif prob_up < float(params["short_threshold"]):
        position = -1
        action = "SHORT"

    trades = list(state.get("trades", []))
    if position != previous_position:
        trades.append(
            {
                "cycle": int(state.get("cycle_count", 0)) + 1,
                "timestamp": int(state["current_row"]["timestamp"]),
                "action": action,
                "new_position": position,
                "price": float(state["current_row"]["close"]),
            }
        )

    return {
        "last_action": action,
        "previous_position": previous_position,
        "position": position,
        "trades": trades,
    }


def evaluate_node(state: AgentState) -> AgentState:
    config = state["config"]
    data = state["historical_data"]
    cursor = int(state["cursor"])
    previous_position = int(state["previous_position"])
    position = int(state["position"])

    close_now = float(data["close"][cursor])
    close_next = float(data["close"][cursor + 1])
    market_return = (close_next - close_now) / close_now

    slippage = float(config["simulation"]["slippage_bps"]) / 10000.0
    transaction_cost = slippage * abs(position - previous_position)
    strategy_return = (position * market_return) - transaction_cost

    returns = list(state.get("returns", []))
    returns.append(strategy_return)

    initial_equity = float(config["simulation"]["initial_equity"])
    equity = float(state.get("equity", initial_equity)) * (1.0 + strategy_return)
    equity_curve = list(state.get("equity_curve", []))
    equity_curve.append(equity)

    cycle_count = int(state.get("cycle_count", 0)) + 1
    performance = _compute_performance(returns, equity_curve, initial_equity)
    done = (
        cycle_count >= int(config["runtime"]["max_cycles"])
        or (cursor + 1) >= (data.height - 1)
    )
    return {
        "cycle_count": cycle_count,
        "cursor": cursor + 1,
        "equity": equity,
        "returns": returns,
        "equity_curve": equity_curve,
        "performance": performance,
        "done": done,
    }


def optimize_node(state: AgentState) -> AgentState:
    config = state["config"]
    cycle_count = int(state["cycle_count"])
    performance = state["performance"]
    optimize_every = int(config["simulation"]["optimization_interval_cycles"])
    sharpe_trigger = float(config["simulation"]["sharpe_trigger"])

    should_optimize = (cycle_count % optimize_every == 0) or (
        float(performance["sharpe"]) < sharpe_trigger
    )
    if not should_optimize:
        return {}

    best_params, best_score = _run_optuna_tune(state)
    shap_rule = _derive_shap_rule(state)
    update_lora_adapter(state)
    replay_buffer = append_to_replay_buffer(state)

    optimization_events = list(state.get("optimization_events", []))
    optimization_events.append(
        {
            "cycle": cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "best_params": best_params,
            "optuna_objective": best_score,
            "shap_rule": shap_rule,
        }
    )
    LOGGER.info(
        "Optimization triggered on cycle=%s (sharpe=%.4f).",
        cycle_count,
        float(performance["sharpe"]),
    )
    return {
        "strategy_params": best_params,
        "optimization_events": optimization_events,
        "replay_buffer": replay_buffer,
        "shap_rule": shap_rule,
    }
