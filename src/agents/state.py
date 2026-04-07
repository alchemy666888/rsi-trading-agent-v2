from __future__ import annotations

from typing import Any, Literal, TypedDict

import polars as pl

Action = Literal["LONG", "SHORT", "HOLD"]


class Prediction(TypedDict):
    prob_up: float
    regime: str


class Performance(TypedDict):
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_return: float


class AgentState(TypedDict, total=False):
    config: dict[str, Any]
    cycle_count: int
    cursor: int
    done: bool

    historical_data: pl.DataFrame
    current_row: dict[str, Any]
    features_df: pl.DataFrame

    prediction: Prediction
    last_action: Action
    previous_position: int
    position: int
    entry_price: float | None

    equity: float
    equity_curve: list[float]
    returns: list[float]
    trades: list[dict[str, Any]]
    replay_buffer: list[dict[str, Any]]

    performance: Performance
    strategy_params: dict[str, float]
    optimization_events: list[dict[str, Any]]
    shap_rule: str
