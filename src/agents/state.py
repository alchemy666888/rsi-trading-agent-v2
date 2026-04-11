from __future__ import annotations

from typing import Any, Literal, TypedDict

import polars as pl

Action = Literal["LONG", "SHORT", "FLAT", "HOLD"]


class Prediction(TypedDict):
    prob_up: float
    regime: str
    signal_timestamp: int
    execution_timestamp: int
    source_model: str


class RiskStatus(TypedDict):
    paused: bool
    reasons: list[str]
    capped_position: int
    stop_triggered: bool
    take_profit_triggered: bool
    blocked_high_volatility: bool


class RunMetrics(TypedDict):
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_return: float
    trade_count: int


class Performance(TypedDict):
    run_metrics: RunMetrics
    benchmark_metrics: dict[str, Any]


class SplitMetadata(TypedDict):
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    oos_start: int
    oos_end: int


class AgentState(TypedDict, total=False):
    config: dict[str, Any]
    run_id: str
    artifact_dir: str
    dataset_metadata: dict[str, Any]
    split_metadata: SplitMetadata

    historical_data: pl.DataFrame
    cursor: int
    done: bool
    paused: bool

    current_row: dict[str, Any]
    prediction: Prediction
    risk_status: RiskStatus

    last_action: Action
    position: int
    target_position: int
    entry_price: float | None

    lightgbm_model: Any
    feature_columns: list[str]
    feature_importances: list[dict[str, Any]]

    equity: float
    equity_curve: list[float]
    returns: list[float]
    trades: list[dict[str, Any]]
    trade_history_buffer: list[dict[str, Any]]

<<<<<<< HEAD
    performance: Performance
    risk_params: dict[str, Any]
=======
>>>>>>> 3c91bfa (fine-tune: 2026-04-11-1800.md)
    strategy_params: dict[str, float]
    optimization_events: list[dict[str, Any]]
    performance: Performance
    shap_rule: str
    shap_rules: list[str]
    readiness: dict[str, Any]
    data_metadata: dict[str, Any]
    run_metadata: dict[str, Any]
