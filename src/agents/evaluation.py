from __future__ import annotations

import math
from typing import Any

import numpy as np
import optuna
import polars as pl

from agents.modeling import build_lgbm_params
from agents.modeling import derive_shap_rule as derive_shap_rule_from_model
from agents.risk import apply_turnover_limit, clamp_position
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Maps timeframe strings to bar duration in minutes, used for annualization
# and funding-interval calculations throughout the pipeline.
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
}


def annualization_factor(timeframe: str = "15m") -> float:
    """Return sqrt(bars_per_year) for Sharpe ratio annualization."""
    minutes = TIMEFRAME_MINUTES.get(timeframe, 15)
    return math.sqrt(365 * 24 * 60 / minutes)


def metric_float(metric: Any) -> float:
    arr = np.asarray(metric, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.nan_to_num(arr[0], nan=0.0, posinf=0.0, neginf=0.0))


def compute_run_metrics(
    returns: list[float],
    equity_curve: list[float],
    initial_equity: float,
    trades: list[dict[str, Any]],
    timeframe: str = "15m",
) -> dict[str, float]:
    if not returns:
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "trade_count": 0,
        }

    returns_array = np.array(returns, dtype=float)
    mean_return = float(returns_array.mean())
    std_return = float(returns_array.std(ddof=1)) if len(returns_array) > 1 else 0.0
    ann = annualization_factor(timeframe)
    sharpe = (mean_return / std_return) * ann if std_return > 1e-12 else 0.0

    equity_array = np.array([initial_equity] + equity_curve, dtype=float)
    rolling_max = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array / rolling_max) - 1.0
    max_drawdown = float(abs(np.min(drawdowns)))

    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": float((returns_array > 0.0).mean()),
        "total_return": float((equity_array[-1] / initial_equity) - 1.0),
        "trade_count": len(trades),
    }


def choose_position(prob_up: float, long_threshold: float, short_threshold: float) -> int:
    if prob_up > long_threshold:
        return 1
    if prob_up < short_threshold:
        return -1
    return 0


def simulate_policy(
    close: np.ndarray,
    probs: np.ndarray,
    volatility_regime: np.ndarray,
    strategy_params: dict[str, float],
    risk_cfg: dict[str, Any],
    initial_equity: float,
    slippage_bps: float,
    timestamp: np.ndarray | None = None,
    signal_delay_bars: int = 1,
    funding_rate_per_8h: float = 0.0,
    timeframe: str = "15m",
) -> dict[str, Any]:
    if close.shape[0] < 2:
        return {
            "returns": [],
            "equity_curve": [],
            "trades": [],
            "metrics": compute_run_metrics([], [], initial_equity, [], timeframe=timeframe),
        }

    position = 0
    entry_price: float | None = None
    equity = initial_equity
    returns: list[float] = []
    equity_curve: list[float] = []
    trades: list[dict[str, Any]] = []
    slippage = slippage_bps / 10000.0
    long_threshold = float(strategy_params["long_threshold"])
    short_threshold = float(strategy_params["short_threshold"])
    max_abs_position = int(risk_cfg.get("max_abs_position", 1))
    max_turnover_per_bar = int(risk_cfg.get("max_turnover_per_bar", 1))
    stop_loss_pct = float(risk_cfg.get("stop_loss_pct", 0.0))
    take_profit_pct = float(risk_cfg.get("take_profit_pct", 0.0))
    block_high_volatility = bool(risk_cfg.get("block_high_volatility", False))
    delay_bars = max(0, int(signal_delay_bars))
    funding_rate = max(0.0, float(funding_rate_per_8h))
    tf_minutes = TIMEFRAME_MINUTES.get(timeframe, 15)
    funding_interval_bars = max(1, int(8 * 60 / tf_minutes))
    scheduled_actions: dict[int, tuple[int, int, int]] = {}

    for idx in range(len(close) - 1):
        current_close = float(close[idx])
        desired_position = choose_position(float(probs[idx]), long_threshold, short_threshold)
        desired_position = clamp_position(desired_position, max_abs_position)

        if block_high_volatility and int(volatility_regime[idx]) >= 1:
            desired_position = 0

        if position != 0 and entry_price is not None:
            pnl_pct = ((current_close / entry_price) - 1.0) if position > 0 else ((entry_price / current_close) - 1.0)
            if stop_loss_pct > 0.0 and pnl_pct <= -stop_loss_pct:
                desired_position = 0
            if take_profit_pct > 0.0 and pnl_pct >= take_profit_pct:
                desired_position = 0

        apply_idx = idx + delay_bars
        if apply_idx < len(close) - 1:
            signal_ts = int(timestamp[idx]) if timestamp is not None else idx
            scheduled_actions[apply_idx] = (desired_position, idx, signal_ts)

        target_position = position
        scheduled = scheduled_actions.pop(idx, None)
        signal_idx = idx
        signal_timestamp = int(timestamp[idx]) if timestamp is not None else idx
        if scheduled is not None:
            scheduled_position, signal_idx, signal_timestamp = scheduled
            target_position = apply_turnover_limit(position, scheduled_position, max_turnover_per_bar)

        transaction_cost = slippage * abs(target_position - position)
        next_close = float(close[idx + 1])
        market_return = (next_close - current_close) / current_close
        strategy_return = (target_position * market_return) - transaction_cost
        if funding_rate > 0.0 and (idx + 1) % funding_interval_bars == 0 and position != 0:
            strategy_return -= abs(position) * funding_rate
        equity *= 1.0 + strategy_return
        returns.append(strategy_return)
        equity_curve.append(equity)

        if target_position != position:
            execution_timestamp = int(timestamp[idx + 1]) if timestamp is not None else (idx + 1)
            action = "LONG" if target_position > 0 else "SHORT" if target_position < 0 else "FLAT"
            trades.append(
                {
                    "bar_index": idx,
                    "signal_bar_index": signal_idx,
                    "execution_bar_index": idx + 1,
                    "signal_timestamp": signal_timestamp,
                    "execution_timestamp": execution_timestamp,
                    "timestamp": execution_timestamp,
                    "action": action,
                    "from_position": position,
                    "to_position": target_position,
                    "price": current_close,
                }
            )
            entry_price = None if target_position == 0 else current_close
        position = target_position

    return {
        "returns": returns,
        "equity_curve": equity_curve,
        "trades": trades,
        "metrics": compute_run_metrics(returns, equity_curve, initial_equity, trades, timeframe=timeframe),
    }


def calibrate_thresholds(
    validation_df: pl.DataFrame,
    probs: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    sim_cfg = config["simulation"]
    risk_cfg = config.get("risk", {})
    timeframe = config.get("asset", {}).get("timeframe", "15m")
    close = validation_df["close"].to_numpy()
    regime = validation_df["volatility_regime"].to_numpy()
    timestamps = validation_df["timestamp"].to_numpy()
    initial_equity = float(sim_cfg["initial_equity"])
    slippage_bps = float(sim_cfg["slippage_bps"])
    signal_delay_bars = int(sim_cfg.get("signal_delay_bars", 1))
    funding_rate_per_8h = float(sim_cfg.get("funding_rate_per_8h", 0.0))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "long_threshold": trial.suggest_float("long_threshold", 0.52, 0.75),
            "short_threshold": trial.suggest_float("short_threshold", 0.25, 0.48),
        }
        if params["short_threshold"] >= params["long_threshold"]:
            return -10.0
        result = simulate_policy(
            close,
            probs,
            regime,
            params,
            risk_cfg,
            initial_equity,
            slippage_bps,
            timestamps,
            signal_delay_bars=signal_delay_bars,
            funding_rate_per_8h=funding_rate_per_8h,
            timeframe=timeframe,
        )
        return float(result["metrics"]["sharpe"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(config["simulation"].get("optuna_trials", 15)))

    best_params = {
        "long_threshold": float(study.best_params["long_threshold"]),
        "short_threshold": float(study.best_params["short_threshold"]),
    }
    return best_params, {
        "split": "validation",
        "objective_name": "validation_sharpe",
        "objective_value": float(study.best_value),
        "affects_future_oos_only": True,
    }


def build_walk_forward_folds(config: dict[str, Any], split_metadata: dict[str, int], total_rows: int, mode: str) -> list[dict[str, int]]:
    walk_cfg = config.get("walk_forward", {})
    train_bars = int(walk_cfg.get("train_bars", 400))
    validation_bars = int(walk_cfg.get("validation_bars", 100))
    test_bars = int(walk_cfg.get("test_bars", 100))
    step_bars = int(walk_cfg.get("step_bars", 50))
    purge_bars = int(walk_cfg.get("purge_bars", 5))
    oos_start = int(split_metadata["oos_start"])

    folds: list[dict[str, int]] = []
    test_start = train_bars + validation_bars + purge_bars
    fold_idx = 0
    while test_start + test_bars < oos_start:
        if mode == "expanding":
            train_start = 0
        else:
            train_start = max(0, test_start - purge_bars - validation_bars - train_bars)

        train_end = max(train_start + train_bars, test_start - purge_bars - validation_bars)
        validation_start = train_end
        validation_end = validation_start + validation_bars
        test_start_actual = validation_end + purge_bars
        test_end = test_start_actual + test_bars
        if test_end >= min(oos_start, total_rows - 1):
            break
        folds.append(
            {
                "fold": fold_idx,
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "test_start": test_start_actual,
                "test_end": test_end,
                "purge_bars": purge_bars,
            }
        )
        fold_idx += 1
        test_start += step_bars
    return folds


def run_walk_forward_backtest(
    data: pl.DataFrame,
    feature_cols: list[str],
    config: dict[str, Any],
    split_metadata: dict[str, int],
) -> dict[str, Any]:
    sim_cfg = config["simulation"]
    risk_cfg = config.get("risk", {})
    timeframe = config.get("asset", {}).get("timeframe", "15m")
    initial_equity = float(sim_cfg["initial_equity"])
    slippage_bps = float(sim_cfg["slippage_bps"])
    signal_delay_bars = int(sim_cfg.get("signal_delay_bars", 1))
    funding_rate_per_8h = float(sim_cfg.get("funding_rate_per_8h", 0.0))

    def run_mode(mode: str) -> dict[str, Any]:
        folds = build_walk_forward_folds(config, split_metadata, data.height, mode)
        fold_results: list[dict[str, Any]] = []
        for fold in folds:
            train_df = data.slice(fold["train_start"], fold["train_end"] - fold["train_start"])
            validation_df = data.slice(fold["validation_start"], fold["validation_end"] - fold["validation_start"])
            test_df = data.slice(fold["test_start"], fold["test_end"] - fold["test_start"])
            if min(train_df.height, validation_df.height, test_df.height) < 20:
                continue

            x_train = train_df.select(feature_cols).to_numpy()
            y_train = train_df["target_up"].to_numpy().astype(int)
            if np.unique(y_train).size < 2:
                y_train[-1] = 1 - y_train[-1]

            model = lgb.LGBMClassifier(**build_lgbm_params(config))
            model.fit(x_train, y_train)

            validation_probs = model.predict_proba(validation_df.select(feature_cols).to_numpy())[:, 1]
            best_params, optimization_meta = calibrate_thresholds(validation_df, validation_probs, config)

            test_probs = model.predict_proba(test_df.select(feature_cols).to_numpy())[:, 1]
            simulation = simulate_policy(
                test_df["close"].to_numpy(),
                test_probs,
                test_df["volatility_regime"].to_numpy(),
                best_params,
                risk_cfg,
                initial_equity,
                slippage_bps,
                test_df["timestamp"].to_numpy(),
                signal_delay_bars=signal_delay_bars,
                funding_rate_per_8h=funding_rate_per_8h,
                timeframe=timeframe,
            )
            fold_results.append(
                {
                    **fold,
                    "long_threshold": float(best_params["long_threshold"]),
                    "short_threshold": float(best_params["short_threshold"]),
                    "optimization_meta": optimization_meta,
                    **simulation["metrics"],
                }
            )

        if not fold_results:
            return {
                "folds": [],
                "mean_sharpe": 0.0,
                "mean_max_drawdown": 0.0,
                "mean_total_return": 0.0,
                "mean_win_rate": 0.0,
            }

        return {
            "folds": fold_results,
            "mean_sharpe": float(np.mean([fold["sharpe"] for fold in fold_results])),
            "mean_max_drawdown": float(np.mean([fold["max_drawdown"] for fold in fold_results])),
            "mean_total_return": float(np.mean([fold["total_return"] for fold in fold_results])),
            "mean_win_rate": float(np.mean([fold["win_rate"] for fold in fold_results])),
        }

    expanding = run_mode("expanding")
    rolling = run_mode("rolling")
    overall = {
        "mean_sharpe": float(np.mean([expanding["mean_sharpe"], rolling["mean_sharpe"]])),
        "mean_max_drawdown": float(np.mean([expanding["mean_max_drawdown"], rolling["mean_max_drawdown"]])),
        "mean_total_return": float(np.mean([expanding["mean_total_return"], rolling["mean_total_return"]])),
        "mean_win_rate": float(np.mean([expanding["mean_win_rate"], rolling["mean_win_rate"]])),
    }
    return {
        "settings": {
            "train_bars": int(config.get("walk_forward", {}).get("train_bars", 400)),
            "validation_bars": int(config.get("walk_forward", {}).get("validation_bars", 100)),
            "test_bars": int(config.get("walk_forward", {}).get("test_bars", 100)),
            "step_bars": int(config.get("walk_forward", {}).get("step_bars", 50)),
            "purge_bars": int(config.get("walk_forward", {}).get("purge_bars", 5)),
            "signal_delay_bars": int(config.get("simulation", {}).get("signal_delay_bars", 1)),
        },
        "expanding": expanding,
        "rolling": rolling,
        "overall": overall,
    }


def derive_shap_rule(data: pl.DataFrame, cursor: int, model: Any, feature_cols: list[str]) -> str:
    return derive_shap_rule_from_model(data, cursor, model, feature_cols)
