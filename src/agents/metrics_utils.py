from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def count_events_by_type(events: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(event.get("event_type", "unknown")) for event in events if isinstance(event, dict))
    return dict(counter)


def extract_drawdown_episodes(
    equity_curve: list[float],
    initial_equity: float,
) -> list[dict[str, Any]]:
    if not equity_curve:
        return []

    values = [float(initial_equity)] + [float(value) for value in equity_curve]
    episodes: list[dict[str, Any]] = []
    peak_value = values[0]
    peak_index = 0
    active_episode: dict[str, Any] | None = None

    for idx in range(1, len(values)):
        value = values[idx]
        if value >= peak_value:
            peak_value = value
            peak_index = idx
            if active_episode is not None:
                active_episode["end_idx"] = idx
                active_episode["duration_bars"] = idx - int(active_episode["start_idx"])
                episodes.append(active_episode)
                active_episode = None
            continue

        drawdown = (value / peak_value) - 1.0
        if active_episode is None:
            active_episode = {
                "start_idx": peak_index,
                "trough_idx": idx,
                "end_idx": None,
                "max_drawdown": abs(drawdown),
                "duration_bars": idx - peak_index,
            }
        else:
            if drawdown < -float(active_episode["max_drawdown"]):
                active_episode["max_drawdown"] = abs(drawdown)
                active_episode["trough_idx"] = idx
            active_episode["duration_bars"] = idx - int(active_episode["start_idx"])

    if active_episode is not None:
        episodes.append(active_episode)
    return episodes


def compute_equity_curve_diagnostics(
    returns: list[float],
    equity_curve: list[float],
    initial_equity: float,
) -> dict[str, Any]:
    drawdown_episodes = extract_drawdown_episodes(equity_curve, initial_equity)
    returns_array = np.asarray(returns, dtype=float)
    annualized_vol = float(returns_array.std(ddof=1) * np.sqrt(365 * 24 * 60)) if returns_array.size > 1 else 0.0

    return {
        "bars": len(returns),
        "annualized_volatility": annualized_vol,
        "max_drawdown_episode": max(drawdown_episodes, key=lambda row: row.get("max_drawdown", 0.0), default=None),
        "drawdown_episodes": drawdown_episodes,
        "drawdown_episode_count": len(drawdown_episodes),
    }


def compute_trade_summary_statistics(completed_trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not completed_trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "profit_factor": None,
            "max_consecutive_losses": 0,
            "best_trade": None,
            "worst_trade": None,
            "longest_hold_bars": 0,
            "average_hold_bars": 0.0,
            "long_count": 0,
            "short_count": 0,
        }

    returns = [float(trade.get("pnl_pct", 0.0)) for trade in completed_trades]
    holds = [int(trade.get("hold_bars", 0)) for trade in completed_trades]
    long_count = sum(1 for trade in completed_trades if int(trade.get("direction", 0)) > 0)
    short_count = sum(1 for trade in completed_trades if int(trade.get("direction", 0)) < 0)
    wins = [value for value in returns if value > 0.0]
    losses = [value for value in returns if value < 0.0]
    gross_profit = float(sum(wins))
    gross_loss = abs(float(sum(losses)))

    consecutive_losses = 0
    max_consecutive_losses = 0
    for value in returns:
        if value < 0.0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0

    profit_factor: float | None = None
    if gross_loss > 1e-12:
        profit_factor = gross_profit / gross_loss

    best_trade = max(completed_trades, key=lambda row: float(row.get("pnl_pct", 0.0)))
    worst_trade = min(completed_trades, key=lambda row: float(row.get("pnl_pct", 0.0)))

    return {
        "trade_count": len(completed_trades),
        "win_rate": float(np.mean(np.asarray(returns, dtype=float) > 0.0)),
        "avg_trade_return": float(np.mean(returns)),
        "profit_factor": profit_factor,
        "max_consecutive_losses": max_consecutive_losses,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "longest_hold_bars": max(holds) if holds else 0,
        "average_hold_bars": float(np.mean(holds)) if holds else 0.0,
        "long_count": long_count,
        "short_count": short_count,
    }


def compute_exposure_stats(decision_log: list[dict[str, Any]]) -> dict[str, float]:
    if not decision_log:
        return {
            "long_exposure_pct": 0.0,
            "short_exposure_pct": 0.0,
            "flat_exposure_pct": 0.0,
            "turnover": 0.0,
        }

    resulting_positions = [int(row.get("resulting_position", 0)) for row in decision_log]
    long_exposure = float(np.mean(np.asarray(resulting_positions) > 0))
    short_exposure = float(np.mean(np.asarray(resulting_positions) < 0))
    flat_exposure = float(np.mean(np.asarray(resulting_positions) == 0))

    turnover_numerator = 0.0
    previous = int(decision_log[0].get("from_position", 0))
    for position in resulting_positions:
        turnover_numerator += abs(position - previous)
        previous = position
    turnover = turnover_numerator / max(1, len(resulting_positions))

    return {
        "long_exposure_pct": long_exposure,
        "short_exposure_pct": short_exposure,
        "flat_exposure_pct": flat_exposure,
        "turnover": turnover,
    }


def compare_benchmark_metrics(
    run_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
) -> dict[str, Any]:
    overall = benchmark_metrics.get("overall", {})
    overall_sharpe = float(overall.get("mean_sharpe", 0.0))
    overall_return = float(overall.get("mean_total_return", 0.0))
    run_sharpe = float(run_metrics.get("sharpe", 0.0))
    run_return = float(run_metrics.get("total_return", 0.0))

    warnings: list[str] = []
    if overall_sharpe > 1e-12 and run_sharpe > (overall_sharpe * 2.0):
        warnings.append("Held-out Sharpe is far above walk-forward mean Sharpe; potential overfit risk.")
    if overall_return > 1e-12 and run_return > (overall_return * 2.0):
        warnings.append("Held-out total return is far above walk-forward mean total return.")
    if run_sharpe < 0.0 and overall_sharpe > 0.0:
        warnings.append("Held-out Sharpe is negative while walk-forward mean Sharpe is positive.")
    if run_sharpe < 0.0 and overall_sharpe < 0.0:
        warnings.append("Held-out and walk-forward Sharpe are both negative; strategy is not yet investment-ready.")

    return {
        "held_out_vs_walk_forward": {
            "delta_sharpe": run_sharpe - overall_sharpe,
            "delta_total_return": run_return - overall_return,
            "held_out_sharpe": run_sharpe,
            "walk_forward_mean_sharpe": overall_sharpe,
            "held_out_total_return": run_return,
            "walk_forward_mean_total_return": overall_return,
        },
        "overfit_warnings": warnings,
    }
