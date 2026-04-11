from __future__ import annotations

from typing import Any

from agents.data import compute_split_metadata, load_historical_data
from agents.evaluation import calibrate_thresholds, run_walk_forward_backtest
from agents.modeling import derive_shap_rule, train_lightgbm_baseline
from agents.state import AgentState


def prepare_experiment(config: dict[str, Any], run_id: str, artifact_dir: str) -> AgentState:
    historical_data, dataset_metadata = load_historical_data(config)
    split_metadata = compute_split_metadata(config, historical_data.height)
    model, feature_columns, feature_importances = train_lightgbm_baseline(historical_data, split_metadata, config)

    validation_df = historical_data.slice(
        split_metadata["validation_start"],
        split_metadata["validation_end"] - split_metadata["validation_start"],
    )
    validation_probs = model.predict_proba(validation_df.select(feature_columns).to_numpy())[:, 1]
    strategy_params, optimization_meta = calibrate_thresholds(validation_df, validation_probs, config)

    benchmark_metrics = run_walk_forward_backtest(
        historical_data,
        feature_columns,
        config,
        split_metadata,
    )
    shap_rule = derive_shap_rule(historical_data, split_metadata["validation_end"] - 1, model, feature_columns)

    return {
        "config": config,
        "run_id": run_id,
        "artifact_dir": artifact_dir,
        "dataset_metadata": dataset_metadata,
        "split_metadata": split_metadata,
        "historical_data": historical_data,
        "cursor": split_metadata["oos_start"],
        "done": False,
        "paused": False,
        "position": 0,
        "target_position": 0,
        "entry_price": None,
        "last_action": "HOLD",
        "lightgbm_model": model,
        "feature_columns": feature_columns,
        "feature_importances": feature_importances,
        "equity": float(config["simulation"]["initial_equity"]),
        "equity_curve": [],
        "returns": [],
        "trades": [],
        "trade_history_buffer": [],
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
