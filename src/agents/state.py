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
    total_return: float
    bar_win_rate: float
    transition_count: int
    completed_trade_win_rate: float
    completed_trade_count: int


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
    purge_bars: int
    max_feature_lag: int
    required_embargo_gap: int


class AgentState(TypedDict, total=False):
    config: dict[str, Any]
    run_id: str
    artifact_dir: str
    event_logger: Any
    event_log: list[dict[str, Any]]

    dataset_metadata: dict[str, Any]
    split_metadata: SplitMetadata
    run_metadata: dict[str, Any]
    error_info: dict[str, Any]
    last_state_snapshot: dict[str, Any]

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
    proposed_position: int
    pending_signals: list[dict[str, Any]]
    applied_signal: dict[str, Any] | None
    signal_delay_bars_runtime: int
    pre_risk_action: Action
    entry_price: float | None
    entry_cycle: int | None
    entry_timestamp: int | None

    lightgbm_model: Any
    feature_columns: list[str]
    feature_importances: list[dict[str, Any]]

    equity: float
    equity_curve: list[float]
    returns: list[float]
    trades: list[dict[str, Any]]
    trade_history_buffer: list[dict[str, Any]]
    completed_trades: list[dict[str, Any]]
    decision_log: list[dict[str, Any]]
    cycle_count: int

    strategy_params: dict[str, float]
    optimization_events: list[dict[str, Any]]
    performance: Performance
    shap_rule: str
    shap_rules: list[str]
    readiness: dict[str, Any]
    data_metadata: dict[str, Any]
    run_metadata: dict[str, Any]
