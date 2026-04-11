from __future__ import annotations

from typing import Any

from agents.data import compute_split_metadata, load_historical_data
from agents.evaluation import calibrate_thresholds, run_walk_forward_backtest
from agents.logging_utils import RunEventLogger, utc_now_iso
from agents.modeling import derive_shap_rule, train_lightgbm_baseline
from agents.state import AgentState


def _emit_setup_event(
    event_logger: RunEventLogger | None,
    *,
    event_type: str,
    message: str,
    **extra: Any,
) -> None:
    if event_logger is None:
        return
    event_logger.emit(stage="setup", event_type=event_type, message=message, **extra)


def prepare_experiment(
    config: dict[str, Any],
    run_id: str,
    artifact_dir: str,
    *,
    run_metadata: dict[str, Any] | None = None,
    event_logger: RunEventLogger | None = None,
) -> AgentState:
    historical_data, dataset_metadata = load_historical_data(config)
    _emit_setup_event(
        event_logger,
        event_type="dataset_loaded",
        message="Dataset loaded and sorted for experiment setup.",
        dataset_rows=historical_data.height,
        dataset_columns=historical_data.width,
        dataset_metadata=dataset_metadata,
    )
    _emit_setup_event(
        event_logger,
        event_type="feature_engineering_completed",
        message="Feature engineering completed.",
        feature_count=len([column for column in historical_data.columns if column not in {"timestamp", "dt", "target_up"}]),
    )

    split_metadata = compute_split_metadata(config, historical_data.height)
    split_counts = {
        "total_bars": int(historical_data.height),
        "train_bars": int(split_metadata["train_end"] - split_metadata["train_start"]),
        "validation_bars": int(split_metadata["validation_end"] - split_metadata["validation_start"]),
        "oos_bars": int(split_metadata["oos_end"] - split_metadata["oos_start"]),
    }

    _emit_setup_event(
        event_logger,
        event_type="model_training_started",
        message="LightGBM baseline training started.",
        split_metadata=split_metadata,
    )
    model, feature_columns, feature_importances = train_lightgbm_baseline(historical_data, split_metadata, config)
    _emit_setup_event(
        event_logger,
        event_type="model_training_completed",
        message="LightGBM baseline training completed.",
        feature_count=len(feature_columns),
        top_feature=feature_importances[0]["feature"] if feature_importances else None,
    )

    validation_df = historical_data.slice(
        split_metadata["validation_start"],
        split_metadata["validation_end"] - split_metadata["validation_start"],
    )
    _emit_setup_event(
        event_logger,
        event_type="threshold_calibration_started",
        message="Validation threshold calibration started.",
        validation_rows=validation_df.height,
    )
    validation_probs = model.predict_proba(validation_df.select(feature_columns).to_numpy())[:, 1]
    strategy_params, optimization_meta = calibrate_thresholds(validation_df, validation_probs, config)
    _emit_setup_event(
        event_logger,
        event_type="threshold_calibration_completed",
        message="Validation threshold calibration completed.",
        objective_name=optimization_meta.get("objective_name"),
        objective_value=optimization_meta.get("objective_value"),
        tuned_thresholds=strategy_params,
    )
    _emit_setup_event(
        event_logger,
        event_type="optimization_event_recorded",
        message="Optimization event recorded.",
        optimization_event={**optimization_meta, "best_params": strategy_params},
    )

    _emit_setup_event(
        event_logger,
        event_type="walk_forward_started",
        message="Walk-forward benchmark started.",
    )
    benchmark_metrics = run_walk_forward_backtest(
        historical_data,
        feature_columns,
        config,
        split_metadata,
    )
    _emit_setup_event(
        event_logger,
        event_type="walk_forward_completed",
        message="Walk-forward benchmark completed.",
        benchmark_overall=benchmark_metrics.get("overall", {}),
    )
    shap_rule = derive_shap_rule(historical_data, split_metadata["validation_end"] - 1, model, feature_columns)

    merged_run_metadata = dict(run_metadata or {})
    merged_run_metadata.setdefault("run_id", run_id)
    merged_run_metadata.setdefault("generated_at_utc", utc_now_iso())
    merged_run_metadata["split_counts"] = split_counts

    return {
        "config": config,
        "run_id": run_id,
        "artifact_dir": artifact_dir,
        "event_log": event_logger.events if event_logger is not None else [],
        "dataset_metadata": dataset_metadata,
        "split_metadata": split_metadata,
        "run_metadata": merged_run_metadata,
        "historical_data": historical_data,
        "cursor": split_metadata["oos_start"],
        "done": False,
        "paused": False,
        "position": 0,
        "target_position": 0,
        "proposed_position": 0,
        "pre_risk_action": "FLAT",
        "entry_price": None,
        "entry_cycle": None,
        "entry_timestamp": None,
        "last_action": "HOLD",
        "lightgbm_model": model,
        "feature_columns": feature_columns,
        "feature_importances": feature_importances,
        "equity": float(config["simulation"]["initial_equity"]),
        "equity_curve": [],
        "returns": [],
        "trades": [],
        "trade_history_buffer": [],
        "completed_trades": [],
        "decision_log": [],
        "strategy_params": strategy_params,
        "optimization_events": [
            {
                **optimization_meta,
                "best_params": strategy_params,
            }
        ],
        "performance": {
            "run_metrics": {
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_return": 0.0,
                "trade_count": 0,
            },
            "benchmark_metrics": benchmark_metrics,
        },
        "shap_rule": shap_rule,
    }
