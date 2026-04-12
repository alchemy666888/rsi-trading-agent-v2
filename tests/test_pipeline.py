from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.artifacts import persist_run_artifacts
from agents.data import apply_dataset_row_slice, compute_split_metadata, load_historical_data
from agents.evaluation import (
    build_walk_forward_folds,
    calibrate_thresholds,
    compute_calibration_objective,
    choose_position,
    drop_terminal_supervised_row,
    run_walk_forward_backtest,
    simulate_policy,
)
from agents.features import build_features
from agents.metrics_utils import compare_benchmark_metrics, compute_equity_curve_diagnostics
from agents.modeling import train_lightgbm_baseline
from agents.nodes import data_node, decision_node, evaluate_node, risk_node, runtime_config_node
from agents.risk import evaluate_risk
from run_mvp import evaluate_readiness, resolve_config_path


def build_test_config() -> dict:
    return {
        "asset": {"symbol": "BTC/USDT", "timeframe": "15m", "exchange": "binance", "fetch_limit": 1200},
        "dataset": {"source_mode": "snapshot"},
        "snapshot": {"path": "ignored.parquet", "auto_write": False},
        "runtime": {"max_cycles": 200, "warmup_bars": 120, "artifact_output_dir": "artifacts"},
        "splits": {"train_ratio": 0.6, "validation_ratio": 0.2, "minimum_oos_bars": 150},
        "simulation": {
            "initial_equity": 10000.0,
            "slippage_bps": 5,
            "funding_rate_per_8h": 0.0,
            "signal_delay_bars": 1,
            "optuna_trials": 5,
            "trade_history_limit": 1000,
            "calibration_use_activity_guard": True,
            "calibration_min_transition_count": 8,
            "calibration_max_one_side_exposure": 0.95,
            "calibration_transition_penalty": 8.0,
            "calibration_concentration_penalty": 4.0,
        },
        "risk": {
            "max_abs_position": 1,
            "max_turnover_per_bar": 1,
            "max_drawdown_pause": 0.15,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.05,
            "block_high_volatility": True,
        },
        "features": {
            "include_multi_timeframe": True,
            "max_lag_bars": 16,
            "multi_timeframe_intervals": ["1h", "4h"],
            "timeframe": "15m",
        },
        "walk_forward": {
            "train_bars": 240,
            "validation_bars": 60,
            "test_bars": 40,
            "step_bars": 20,
            "purge_bars": 5,
            "min_folds_per_mode": 1,
            "min_total_test_bars": 40,
        },
        "readiness": {
            "min_historical_windows": 5,
        },
        "logging": {
            "level": "INFO",
            "enable_json_logs": True,
            "enable_console_logs": False,
            "decision_log_enabled": True,
        },
        "reporting": {
            "report_detail_level": "full",
            "write_report_json": True,
            "persist_csv_exports": True,
            "auto_stage_artifacts": False,
            "decision_log_enabled": True,
            "decision_feature_count": 12,
        },
    }


def _build_ohlcv(rows: int, *, spike_start: int | None = None) -> pl.DataFrame:
    timestamps = [1_700_000_000_000 + (idx * 900_000) for idx in range(rows)]
    close = np.linspace(100.0, 100.0 + rows - 1, rows, dtype=float)
    if spike_start is not None:
        close[spike_start:] = close[spike_start:] * 10.0
    open_ = close - 0.5
    high = close + 1.0
    low = close - 1.0
    volume = np.full(rows, 1000.0, dtype=float)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class PipelineTests(unittest.TestCase):
    def _build_artifact_state(self, trade_count: int, completed_trade_count: int) -> dict:
        trades = [{"action": "LONG", "idx": idx} for idx in range(trade_count)]
        completed = [{"direction": 1, "pnl_pct": 0.01, "hold_bars": 2} for _ in range(completed_trade_count)]
        return {
            "config": build_test_config(),
            "run_id": "test-run",
            "artifact_dir": "artifacts/test-run",
            "dataset_metadata": {
                "rows": 100,
                "source_mode": "snapshot",
                "source_ref": "snapshots/unit-test.parquet",
                "benchmark_eligible": True,
                "snapshot_path": "snapshots/unit-test.parquet",
                "snapshot_hash": "abc123",
                "raw_data_hash": "raw123",
            },
            "split_metadata": {
                "train_start": 0,
                "train_end": 60,
                "validation_start": 60,
                "validation_end": 80,
                "oos_start": 80,
                "oos_end": 99,
                "purge_bars": 5,
                "max_feature_lag": 16,
                "required_embargo_gap": 17,
            },
            "strategy_params": {"long_threshold": 0.6, "short_threshold": 0.4},
            "optimization_events": [
                {
                    "split": "validation",
                    "objective_name": "validation_sharpe",
                    "objective_value": 1.23,
                    "best_params": {"long_threshold": 0.6, "short_threshold": 0.4},
                    "affects_future_oos_only": True,
                }
            ],
            "feature_importances": [{"feature": "rsi_14", "importance": 10.0}],
            "trades": trades,
            "trade_history_buffer": list(trades),
            "completed_trades": completed,
            "decision_log": [],
            "event_log": [{"event_type": "run_started"}],
            "equity_curve": [10050.0],
            "returns": [0.005],
            "performance": {
                "run_metrics": {
                    "sharpe": 1.0,
                    "max_drawdown": 0.01,
                    "total_return": 0.005,
                    "bar_win_rate": 1.0,
                    "transition_count": trade_count,
                    "completed_trade_win_rate": 1.0 if completed_trade_count > 0 else 0.0,
                    "completed_trade_count": completed_trade_count,
                },
                "benchmark_metrics": {
                    "settings": {
                        "train_bars": 240,
                        "validation_bars": 60,
                        "test_bars": 40,
                        "purge_bars": 5,
                        "signal_delay_bars": 1,
                    },
                    "expanding": {
                        "mean_sharpe": 0.1,
                        "mean_max_drawdown": 0.02,
                        "mean_total_return": 0.03,
                        "mean_bar_win_rate": 0.6,
                        "fold_count": 1,
                        "total_test_bars": 40,
                        "folds": [1],
                    },
                    "rolling": {
                        "mean_sharpe": 0.2,
                        "mean_max_drawdown": 0.03,
                        "mean_total_return": 0.04,
                        "mean_bar_win_rate": 0.7,
                        "fold_count": 1,
                        "total_test_bars": 40,
                        "folds": [1],
                    },
                    "overall": {
                        "mean_sharpe": 0.15,
                        "mean_max_drawdown": 0.025,
                        "mean_total_return": 0.035,
                        "mean_bar_win_rate": 0.65,
                    },
                    "sufficiency": {
                        "min_folds_per_mode": 1,
                        "min_total_test_bars": 40,
                        "expanding": {"fold_count": 1, "total_test_bars": 40, "sufficient": True, "reasons": []},
                        "rolling": {"fold_count": 1, "total_test_bars": 40, "sufficient": True, "reasons": []},
                        "overall_sufficient": True,
                        "reasons": [],
                    },
                },
            },
            "shap_rule": "Test rule",
            "cursor": 90,
            "paused": False,
            "run_metadata": {
                "git_commit_hash": "abc",
                "config_hash": "xyz",
                "split_counts": {"total_bars": 100, "train_bars": 60, "validation_bars": 20, "oos_bars": 19},
            },
        }

    def test_split_metadata_keeps_oos_after_validation(self) -> None:
        config = build_test_config()
        split_metadata = compute_split_metadata(config, total_rows=1200)
        self.assertLess(split_metadata["train_end"], split_metadata["validation_end"])
        self.assertGreaterEqual(split_metadata["oos_start"], split_metadata["validation_end"])
        self.assertGreater(split_metadata["oos_end"], split_metadata["oos_start"])
        self.assertGreaterEqual(
            split_metadata["oos_start"],
            split_metadata["validation_end"] + max(split_metadata["purge_bars"], split_metadata["required_embargo_gap"]),
        )

    def test_apply_dataset_row_slice_respects_requested_bounds(self) -> None:
        raw_df = _build_ohlcv(100)
        sliced, meta = apply_dataset_row_slice(raw_df, {"row_slice": {"start": 10, "end": 30}})
        self.assertEqual(sliced.height, 20)
        self.assertEqual(int(sliced["timestamp"][0]), int(raw_df["timestamp"][10]))
        self.assertEqual(int(sliced["timestamp"][-1]), int(raw_df["timestamp"][29]))
        self.assertTrue(meta["row_slice_enabled"])
        self.assertEqual(meta["row_slice_start"], 10)
        self.assertEqual(meta["row_slice_end"], 30)

    def test_apply_dataset_row_slice_rejects_invalid_bounds(self) -> None:
        raw_df = _build_ohlcv(50)
        with self.assertRaises(ValueError):
            apply_dataset_row_slice(raw_df, {"row_slice": {"start": 25, "end": 25}})
        with self.assertRaises(ValueError):
            apply_dataset_row_slice(raw_df, {"row_slice": {"start": -1, "end": 10}})

    def test_load_historical_data_applies_snapshot_row_slice(self) -> None:
        raw_df = _build_ohlcv(200)
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = Path(tmp_dir) / "snapshot.parquet"
            raw_df.write_parquet(snapshot_path)
            config = build_test_config()
            config["dataset"] = {
                "source_mode": "snapshot",
                "row_slice": {"start": 20, "end": 140},
            }
            config["snapshot"] = {"path": str(snapshot_path), "auto_write": False}
            feature_df, metadata = load_historical_data(config)
            self.assertEqual(feature_df.height, 120)
            self.assertTrue(metadata["row_slice_enabled"])
            self.assertEqual(metadata["row_slice_start"], 20)
            self.assertEqual(metadata["row_slice_end"], 140)
            self.assertEqual(int(metadata["timestamp_start"]), int(raw_df["timestamp"][20]))
            self.assertEqual(int(metadata["timestamp_end"]), int(raw_df["timestamp"][139]))

    def test_resolve_config_path_supports_relative_and_absolute_inputs(self) -> None:
        relative = resolve_config_path("config/config.yaml")
        absolute = resolve_config_path(str(PROJECT_ROOT / "config" / "config.yaml"))
        expected = (PROJECT_ROOT / "config" / "config.yaml").resolve()
        self.assertEqual(relative, expected)
        self.assertEqual(absolute, expected)

    def test_drop_terminal_supervised_row_trims_last_bar(self) -> None:
        df = pl.DataFrame({"timestamp": [1, 2, 3], "close": [10.0, 11.0, 12.0]})
        trimmed = drop_terminal_supervised_row(df)
        self.assertEqual(trimmed.height, 2)
        self.assertEqual(trimmed["timestamp"].to_list(), [1, 2])

    def test_train_lightgbm_baseline_avoids_split_boundary_label_leakage(self) -> None:
        feature_df = pl.DataFrame(
            {
                "timestamp": [1, 2, 3, 4],
                "dt": [1, 2, 3, 4],
                "close": [1.0, 2.0, 1.0, 100.0],
                # Row index 2 would leak a label from validation if used.
                "target_up": [1, 0, 1, 0],
            }
        )
        split_metadata = {"train_start": 0, "train_end": 3}
        captured: dict[str, np.ndarray] = {}

        class _DummyBooster:
            def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
                return np.array([1.0], dtype=float)

        class _DummyLGBMClassifier:
            def __init__(self, **kwargs):
                self.booster_ = _DummyBooster()

            def fit(self, x: np.ndarray, y: np.ndarray) -> None:
                captured["x"] = np.asarray(x)
                captured["y"] = np.asarray(y)

        with patch("agents.modeling.lgb.LGBMClassifier", _DummyLGBMClassifier):
            train_lightgbm_baseline(feature_df, split_metadata, build_test_config())

        self.assertIn("x", captured)
        self.assertIn("y", captured)
        self.assertEqual(int(captured["x"].shape[0]), 2)
        self.assertEqual(captured["y"].tolist(), [1, 0])

    def test_walk_forward_training_avoids_fold_boundary_label_leakage(self) -> None:
        rows = 300
        data = pl.DataFrame(
            {
                "timestamp": np.arange(1_700_000_000_000, 1_700_000_000_000 + rows, dtype=np.int64),
                "close": np.linspace(100.0, 200.0, rows, dtype=float),
                "volatility_regime": np.zeros(rows, dtype=int),
            }
        )
        config = build_test_config()
        config["walk_forward"].update(
            {
                "train_bars": 60,
                "validation_bars": 20,
                "test_bars": 20,
                "step_bars": 20,
                "purge_bars": 5,
            }
        )
        split_metadata = {
            "train_start": 0,
            "train_end": 180,
            "validation_start": 180,
            "validation_end": 220,
            "oos_start": 260,
            "oos_end": 299,
            "purge_bars": 5,
            "max_feature_lag": 16,
            "required_embargo_gap": 17,
        }
        trained_rows: list[int] = []

        class _DummyLGBMClassifier:
            def __init__(self, **kwargs):
                self.booster_ = None

            def fit(self, x: np.ndarray, y: np.ndarray) -> None:
                trained_rows.append(int(np.asarray(x).shape[0]))

            def predict_proba(self, x: np.ndarray) -> np.ndarray:
                arr = np.asarray(x)
                return np.column_stack(
                    [
                        np.full(arr.shape[0], 0.3, dtype=float),
                        np.full(arr.shape[0], 0.7, dtype=float),
                    ]
                )

        with patch("agents.evaluation.lgb.LGBMClassifier", _DummyLGBMClassifier), patch(
            "agents.evaluation.calibrate_thresholds",
            return_value=(
                {"long_threshold": 0.6, "short_threshold": 0.4},
                {"split": "validation", "objective_name": "validation_sharpe", "objective_value": 0.0},
            ),
        ), patch(
            "agents.evaluation.simulate_policy",
            return_value={
                "returns": [],
                "equity_curve": [],
                "trades": [],
                "metrics": {"sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0, "bar_win_rate": 0.0},
            },
        ):
            benchmark = run_walk_forward_backtest(data, feature_cols=["close"], config=config, split_metadata=split_metadata)

        expected_rows = [
            (int(fold["train_end"]) - int(fold["train_start"]) - 1)
            for fold in [*benchmark["expanding"]["folds"], *benchmark["rolling"]["folds"]]
        ]
        self.assertTrue(expected_rows)
        self.assertEqual(trained_rows, expected_rows)

    def test_simulate_policy_drawdown_pause_matches_runtime(self) -> None:
        close = np.array([100.0, 90.0, 80.0, 70.0, 60.0], dtype=float)
        probs = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=float)
        volatility_regime = np.zeros_like(close, dtype=int)
        timestamps = np.array([1, 2, 3, 4, 5], dtype=int)
        strategy_params = {"long_threshold": 0.6, "short_threshold": 0.4}
        config = build_test_config()
        config["simulation"]["signal_delay_bars"] = 0
        config["simulation"]["slippage_bps"] = 0.0
        config["risk"]["max_drawdown_pause"] = 0.05
        config["risk"]["block_high_volatility"] = False
        config["risk"]["stop_loss_pct"] = 0.0
        config["risk"]["take_profit_pct"] = 0.0

        simulation = simulate_policy(
            close=close,
            probs=probs,
            volatility_regime=volatility_regime,
            strategy_params=strategy_params,
            risk_cfg=config["risk"],
            initial_equity=10000.0,
            slippage_bps=0.0,
            timestamp=timestamps,
            signal_delay_bars=0,
            funding_rate_per_8h=0.0,
            timeframe="15m",
        )
        self.assertTrue(simulation["paused"])

        runtime_state = {
            "config": config,
            "historical_data": pl.DataFrame({"timestamp": timestamps, "close": close}),
            "split_metadata": {"oos_end": len(close) - 1},
            "cursor": 0,
            "done": False,
            "paused": False,
            "position": 0,
            "target_position": 0,
            "proposed_position": 0,
            "pending_signals": [],
            "applied_signal": None,
            "signal_delay_bars_runtime": 0,
            "entry_price": None,
            "entry_cycle": None,
            "entry_timestamp": None,
            "last_action": "HOLD",
            "strategy_params": strategy_params,
            "prediction": {
                "prob_up": 0.5,
                "regime": "normal",
                "signal_timestamp": 0,
                "execution_timestamp": 0,
                "source_model": "unit_test",
            },
            "current_row": {"timestamp": int(timestamps[0]), "close": float(close[0]), "rsi_14": 50.0},
            "risk_status": {
                "paused": False,
                "reasons": [],
                "stop_triggered": False,
                "take_profit_triggered": False,
                "blocked_high_volatility": False,
            },
            "returns": [],
            "equity_curve": [],
            "trades": [],
            "trade_history_buffer": [],
            "completed_trades": [],
            "decision_log": [],
            "equity": 10000.0,
            "performance": {"run_metrics": {}, "benchmark_metrics": {}},
        }

        for idx in range(len(close) - 1):
            runtime_state["cursor"] = idx
            runtime_state["current_row"] = {
                "timestamp": int(timestamps[idx]),
                "close": float(close[idx]),
                "rsi_14": 50.0,
            }
            runtime_state["prediction"] = {
                "prob_up": float(probs[idx]),
                "regime": "normal",
                "signal_timestamp": int(timestamps[idx]),
                "execution_timestamp": int(timestamps[idx]),
                "source_model": "unit_test",
            }
            runtime_state.update(risk_node(runtime_state))
            runtime_state.update(decision_node(runtime_state))
            runtime_state.update(evaluate_node(runtime_state))
            if bool(runtime_state.get("paused", False)):
                break

        self.assertEqual(len(simulation["returns"]), len(runtime_state["returns"]))
        self.assertTrue(np.allclose(simulation["returns"], runtime_state["returns"], atol=1e-12))
        self.assertTrue(np.allclose(simulation["equity_curve"], runtime_state["equity_curve"], atol=1e-9))
        self.assertEqual(len(simulation["trades"]), len(runtime_state["trades"]))

    def test_signal_delay_shift_execution_timestamp_and_changes_return(self) -> None:
        close = pl.Series("close", [100.0, 110.0, 100.0, 110.0, 100.0]).to_numpy()
        probs = pl.Series("prob", [0.9, 0.9, 0.1, 0.1, 0.1]).to_numpy()
        volatility_regime = pl.Series("regime", [0, 0, 0, 0, 0]).to_numpy()
        timestamps = pl.Series("timestamp", [1, 2, 3, 4, 5]).to_numpy()
        kwargs = {
            "close": close,
            "probs": probs,
            "volatility_regime": volatility_regime,
            "strategy_params": {"long_threshold": 0.6, "short_threshold": 0.4},
            "risk_cfg": {
                "max_abs_position": 1,
                "max_turnover_per_bar": 2,
                "block_high_volatility": False,
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.0,
            },
            "initial_equity": 10000.0,
            "slippage_bps": 0.0,
            "timestamp": timestamps,
        }
        delay_0 = simulate_policy(**kwargs, signal_delay_bars=0)
        delay_1 = simulate_policy(**kwargs, signal_delay_bars=1)
        self.assertTrue(delay_0["trades"])
        self.assertTrue(delay_1["trades"])
        trade_pairs = min(len(delay_0["trades"]), len(delay_1["trades"]))
        for idx in range(trade_pairs):
            self.assertEqual(
                delay_1["trades"][idx]["execution_timestamp"] - delay_0["trades"][idx]["execution_timestamp"],
                1,
            )
        for trade in delay_1["trades"]:
            execution_idx = int(trade["execution_bar_index"])
            self.assertEqual(int(trade["execution_timestamp"]), int(timestamps[execution_idx]))
            self.assertAlmostEqual(float(trade["price"]), float(close[execution_idx]), places=12)
        self.assertNotAlmostEqual(sum(delay_0["returns"]), sum(delay_1["returns"]), places=9)

    def test_signal_delay_roundtrip(self) -> None:
        close = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0], dtype=float)
        probs = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1], dtype=float)
        volatility_regime = np.zeros_like(close, dtype=int)
        timestamps = np.array([10, 20, 30, 40, 50, 60], dtype=int)
        strategy_params = {"long_threshold": 0.6, "short_threshold": 0.4}
        risk_cfg = {
            "max_abs_position": 1,
            "max_turnover_per_bar": 2,
            "block_high_volatility": False,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
        }
        simulation = simulate_policy(
            close=close,
            probs=probs,
            volatility_regime=volatility_regime,
            strategy_params=strategy_params,
            risk_cfg=risk_cfg,
            initial_equity=10000.0,
            slippage_bps=0.0,
            timestamp=timestamps,
            signal_delay_bars=1,
            funding_rate_per_8h=0.0,
            timeframe="15m",
        )

        runtime_cfg = build_test_config()
        runtime_cfg["simulation"]["slippage_bps"] = 0.0
        runtime_cfg["simulation"]["signal_delay_bars"] = 1
        runtime_cfg["risk"]["max_turnover_per_bar"] = 2
        runtime_cfg["risk"]["block_high_volatility"] = False
        runtime_cfg["risk"]["stop_loss_pct"] = 0.0
        runtime_cfg["risk"]["take_profit_pct"] = 0.0
        runtime_state = {
            "config": runtime_cfg,
            "historical_data": pl.DataFrame({"timestamp": timestamps, "close": close}),
            "split_metadata": {"oos_end": len(close) - 1},
            "cursor": 0,
            "done": False,
            "paused": False,
            "position": 0,
            "target_position": 0,
            "proposed_position": 0,
            "pending_signals": [],
            "applied_signal": None,
            "signal_delay_bars_runtime": 1,
            "entry_price": None,
            "entry_cycle": None,
            "entry_timestamp": None,
            "last_action": "HOLD",
            "strategy_params": strategy_params,
            "risk_status": {
                "paused": False,
                "reasons": [],
                "stop_triggered": False,
                "take_profit_triggered": False,
                "blocked_high_volatility": False,
            },
            "current_row": {"timestamp": int(timestamps[0]), "close": float(close[0]), "rsi_14": 50.0},
            "prediction": {
                "prob_up": 0.5,
                "regime": "normal",
                "signal_timestamp": int(timestamps[0]),
                "execution_timestamp": int(timestamps[1]),
                "source_model": "unit_test",
            },
            "returns": [],
            "equity_curve": [],
            "trades": [],
            "trade_history_buffer": [],
            "completed_trades": [],
            "decision_log": [],
            "equity": 10000.0,
            "performance": {"run_metrics": {}, "benchmark_metrics": {}},
        }

        for idx in range(len(close) - 1):
            runtime_state["cursor"] = idx
            runtime_state["current_row"] = {"timestamp": int(timestamps[idx]), "close": float(close[idx]), "rsi_14": 50.0}
            runtime_state["target_position"] = choose_position(
                float(probs[idx]),
                strategy_params["long_threshold"],
                strategy_params["short_threshold"],
            )
            execution_idx = min(idx + runtime_state["signal_delay_bars_runtime"], len(close) - 2)
            runtime_state["prediction"] = {
                "prob_up": float(probs[idx]),
                "regime": "normal",
                "signal_timestamp": int(timestamps[idx]),
                "execution_timestamp": int(timestamps[execution_idx]),
                "source_model": "unit_test",
            }
            runtime_state["risk_status"] = {
                "paused": False,
                "reasons": [],
                "stop_triggered": False,
                "take_profit_triggered": False,
                "blocked_high_volatility": False,
            }
            runtime_state.update(decision_node(runtime_state))
            runtime_state.update(evaluate_node(runtime_state))
            if runtime_state.get("done", False):
                break

        self.assertEqual(len(simulation["returns"]), len(runtime_state["returns"]))
        self.assertTrue(np.allclose(simulation["returns"], runtime_state["returns"], atol=1e-12))
        self.assertTrue(np.allclose(simulation["equity_curve"], runtime_state["equity_curve"], atol=1e-9))
        self.assertEqual(len(simulation["trades"]), len(runtime_state["trades"]))
        self.assertEqual(
            [int(trade["execution_timestamp"]) for trade in simulation["trades"]],
            [int(trade["execution_timestamp"]) for trade in runtime_state["trades"]],
        )
        self.assertEqual(
            [int(trade["signal_timestamp"]) for trade in simulation["trades"]],
            [int(trade["signal_timestamp"]) for trade in runtime_state["trades"]],
        )
        self.assertEqual(len(simulation["completed_trades"]), len(runtime_state["completed_trades"]))
        for sim_trade in simulation["trades"]:
            self.assertEqual(int(sim_trade["bar_timestamp"]), int(sim_trade["execution_timestamp"]))
        for runtime_trade in runtime_state["trades"]:
            self.assertEqual(int(runtime_trade["bar_timestamp"]), int(runtime_trade["execution_timestamp"]))

    def test_simulate_policy_applies_funding_cost(self) -> None:
        close = np.array([100.0] * 10, dtype=float)
        probs = np.array([0.9] * 10, dtype=float)
        volatility_regime = np.zeros(10, dtype=int)
        result = simulate_policy(
            close=close,
            probs=probs,
            volatility_regime=volatility_regime,
            strategy_params={"long_threshold": 0.6, "short_threshold": 0.4},
            risk_cfg={
                "max_abs_position": 1,
                "max_turnover_per_bar": 1,
                "block_high_volatility": False,
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.0,
            },
            initial_equity=10000.0,
            slippage_bps=0.0,
            timestamp=np.arange(1, 11, dtype=int),
            signal_delay_bars=0,
            funding_rate_per_8h=0.01,
            timeframe="1h",
        )
        self.assertEqual(len(result["returns"]), 9)
        self.assertLess(result["returns"][7], 0.0)
        self.assertAlmostEqual(sum(result["returns"]), -0.01, places=9)

    def test_stop_loss_flatten_is_delayed_by_signal_delay(self) -> None:
        timestamps = np.array([10, 20, 30, 40, 50], dtype=int)
        result = simulate_policy(
            close=np.array([100.0, 100.0, 90.0, 89.0, 89.0], dtype=float),
            probs=np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=float),
            volatility_regime=np.zeros(5, dtype=int),
            strategy_params={"long_threshold": 0.6, "short_threshold": 0.4},
            risk_cfg={
                "max_abs_position": 1,
                "max_turnover_per_bar": 2,
                "block_high_volatility": False,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.0,
            },
            initial_equity=10000.0,
            slippage_bps=0.0,
            timestamp=timestamps,
            signal_delay_bars=1,
            funding_rate_per_8h=0.0,
            timeframe="15m",
        )
        self.assertGreaterEqual(len(result["trades"]), 2)
        self.assertEqual(result["trades"][0]["action"], "LONG")
        self.assertEqual(result["trades"][1]["action"], "FLAT")
        self.assertEqual(int(result["trades"][1]["execution_timestamp"]), int(timestamps[3]))
        self.assertEqual(
            int(result["trades"][1]["execution_timestamp"]) - int(timestamps[2]),
            int(timestamps[1] - timestamps[0]),
        )

    def test_take_profit_flatten_is_delayed_by_signal_delay(self) -> None:
        timestamps = np.array([10, 20, 30, 40, 50], dtype=int)
        result = simulate_policy(
            close=np.array([100.0, 100.0, 110.0, 111.0, 111.0], dtype=float),
            probs=np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=float),
            volatility_regime=np.zeros(5, dtype=int),
            strategy_params={"long_threshold": 0.6, "short_threshold": 0.4},
            risk_cfg={
                "max_abs_position": 1,
                "max_turnover_per_bar": 2,
                "block_high_volatility": False,
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.05,
            },
            initial_equity=10000.0,
            slippage_bps=0.0,
            timestamp=timestamps,
            signal_delay_bars=1,
            funding_rate_per_8h=0.0,
            timeframe="15m",
        )
        self.assertGreaterEqual(len(result["trades"]), 2)
        self.assertEqual(result["trades"][0]["action"], "LONG")
        self.assertEqual(result["trades"][1]["action"], "FLAT")
        self.assertEqual(int(result["trades"][1]["execution_timestamp"]), int(timestamps[3]))

    def test_drawdown_pause_and_high_vol_block(self) -> None:
        state = {
            "config": build_test_config(),
            "position": 1,
            "entry_price": 100.0,
            "current_row": {"close": 95.0},
            "prediction": {"regime": "high_volatility"},
            "performance": {"run_metrics": {"max_drawdown": 0.20}},
        }
        target_position, risk_status = evaluate_risk(state, proposed_position=1)
        self.assertEqual(target_position, 0)
        self.assertTrue(risk_status["paused"])
        self.assertIn("max_drawdown_pause", risk_status["reasons"])

    def test_walk_forward_folds_include_purge_gap(self) -> None:
        config = build_test_config()
        split_metadata = compute_split_metadata(config, total_rows=1200)
        folds = build_walk_forward_folds(config, split_metadata, total_rows=1200, mode="rolling")
        self.assertTrue(folds)
        for fold in folds:
            self.assertEqual(fold["test_start"] - fold["validation_end"], config["walk_forward"]["purge_bars"])

    def test_walk_forward_sufficiency_flags_insufficient_evidence(self) -> None:
        config = build_test_config()
        config["walk_forward"]["min_folds_per_mode"] = 99
        config["walk_forward"]["min_total_test_bars"] = 9999
        rows = 1200
        raw = _build_ohlcv(rows)
        feature_df = build_features(raw, feature_cfg=config["features"])
        feature_cols = [col for col in feature_df.columns if col not in {"timestamp", "dt", "target_up"}]
        split_metadata = compute_split_metadata(config, total_rows=feature_df.height)
        benchmark = run_walk_forward_backtest(feature_df, feature_cols, config, split_metadata)
        self.assertIn("sufficiency", benchmark)
        self.assertFalse(bool(benchmark["sufficiency"]["overall_sufficient"]))

    def test_artifact_bundle_writes_expected_files(self) -> None:
        state = self._build_artifact_state(trade_count=1, completed_trade_count=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = persist_run_artifacts(Path(tmp_dir), state)
            self.assertTrue((artifact_path / "report.md").exists())
            self.assertTrue((artifact_path / "report.json").exists())
            self.assertTrue((artifact_path / "benchmark_metrics.json").exists())
            self.assertTrue((artifact_path / "benchmark_sufficiency.json").exists())
            self.assertTrue((artifact_path / "readiness.json").exists())
            self.assertTrue((artifact_path / "regression_ledger_entry.json").exists())
            self.assertTrue((artifact_path / "regression_ledger_snapshot.json").exists())
            self.assertTrue((artifact_path / "decision_log.jsonl").exists())
            self.assertTrue((artifact_path / "decision_summary.csv").exists())
            self.assertTrue((artifact_path / "returns.csv").exists())
            self.assertTrue((artifact_path.parent / "regression_ledger.json").exists())
            report_text = (artifact_path / "report.md").read_text(encoding="utf-8")
            self.assertIn("Run Metadata", report_text)
            self.assertIn("Headline KPIs", report_text)
            benchmark_payload = json.loads((artifact_path / "benchmark_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("expanding", benchmark_payload)

    def test_regression_ledger_appends_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state_a = self._build_artifact_state(trade_count=1, completed_trade_count=1)
            state_a["run_id"] = "run-a"
            state_a["artifact_dir"] = "artifacts/run-a"
            state_a["dataset_metadata"]["timestamp_start"] = 1_700_000_000_000
            state_a["dataset_metadata"]["timestamp_end"] = 1_700_001_000_000
            persist_run_artifacts(root, state_a)

            state_b = self._build_artifact_state(trade_count=1, completed_trade_count=1)
            state_b["run_id"] = "run-b"
            state_b["artifact_dir"] = "artifacts/run-b"
            state_b["dataset_metadata"]["timestamp_start"] = 1_700_010_000_000
            state_b["dataset_metadata"]["timestamp_end"] = 1_700_011_000_000
            persist_run_artifacts(root, state_b)

            ledger = json.loads((root / "artifacts" / "regression_ledger.json").read_text(encoding="utf-8"))
            self.assertEqual(len(ledger), 2)
            self.assertEqual(ledger[0]["run_id"], "run-a")
            self.assertEqual(ledger[1]["run_id"], "run-b")

    def test_report_trade_count_prefers_run_metrics(self) -> None:
        state = self._build_artifact_state(trade_count=3, completed_trade_count=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = persist_run_artifacts(Path(tmp_dir), state)
            report_text = (artifact_path / "report.md").read_text(encoding="utf-8")
            self.assertIn("| Transition Count | 3 |", report_text)

    def test_artifact_persistence_asserts_trade_count_mismatch(self) -> None:
        state = self._build_artifact_state(trade_count=2, completed_trade_count=2)
        state["trades"] = [{"action": "LONG"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(AssertionError):
                persist_run_artifacts(Path(tmp_dir), state)

    def test_artifact_persistence_rejects_legacy_kpi_schema(self) -> None:
        state = self._build_artifact_state(trade_count=2, completed_trade_count=1)
        state["performance"]["run_metrics"] = {
            "sharpe": 0.1,
            "max_drawdown": 0.01,
            "total_return": 0.02,
            "win_rate": 0.5,
            "trade_count": 2,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(AssertionError):
                persist_run_artifacts(Path(tmp_dir), state)

    def test_calibrate_thresholds_propagates_signal_delay(self) -> None:
        config = build_test_config()
        config["simulation"]["signal_delay_bars"] = 2
        config["simulation"]["optuna_trials"] = 2
        validation_df = pl.DataFrame(
            {
                "close": np.linspace(100.0, 200.0, 50, dtype=float),
                "volatility_regime": np.zeros(50, dtype=int),
                "timestamp": np.arange(1_700_000_000_000, 1_700_000_000_000 + (50 * 900_000), 900_000, dtype=int),
            }
        )
        probs = np.full(50, 0.7, dtype=float)
        delays: list[int] = []

        def _fake_simulate(*args, **kwargs):
            delays.append(int(kwargs.get("signal_delay_bars", -1)))
            return {"metrics": {"sharpe": 0.0}, "returns": [], "equity_curve": [], "trades": []}

        with patch("agents.evaluation.simulate_policy", side_effect=_fake_simulate):
            calibrate_thresholds(validation_df, probs, config)
        self.assertTrue(delays)
        self.assertEqual(set(delays), {2})

    def test_compute_calibration_objective_penalizes_low_activity(self) -> None:
        sim_cfg = dict(build_test_config()["simulation"])
        sim_cfg["calibration_min_transition_count"] = 12
        score, diagnostics = compute_calibration_objective(
            sharpe=0.25,
            transition_count=2,
            long_exposure_pct=0.40,
            short_exposure_pct=0.30,
            sim_cfg=sim_cfg,
            usable_rows=200,
        )
        self.assertLess(score, 0.25)
        self.assertGreater(float(diagnostics["activity_penalty"]), 0.0)
        self.assertTrue(bool(diagnostics["degenerate_regime"]))

    def test_compute_calibration_objective_penalizes_one_sided_exposure(self) -> None:
        sim_cfg = dict(build_test_config()["simulation"])
        sim_cfg["calibration_min_transition_count"] = 1
        sim_cfg["calibration_max_one_side_exposure"] = 0.80
        score, diagnostics = compute_calibration_objective(
            sharpe=0.15,
            transition_count=15,
            long_exposure_pct=0.92,
            short_exposure_pct=0.03,
            sim_cfg=sim_cfg,
            usable_rows=200,
        )
        self.assertLess(score, 0.15)
        self.assertGreater(float(diagnostics["concentration_penalty"]), 0.0)
        self.assertTrue(bool(diagnostics["degenerate_regime"]))

    def test_multi_timeframe_future_spike_does_not_leak(self) -> None:
        cfg = dict(build_test_config()["features"])
        cfg["multi_timeframe_intervals"] = ["1h", "4h"]
        cfg["max_lag_bars"] = 4
        baseline = build_features(_build_ohlcv(120), feature_cfg=cfg)
        spiked = build_features(_build_ohlcv(120, spike_start=96), feature_cfg=cfg)
        tf_cols = [col for col in baseline.columns if col.startswith("tf_")]
        self.assertTrue(tf_cols)
        safe_rows = baseline.filter(pl.col("timestamp") < int(_build_ohlcv(120)["timestamp"][92]))
        safe_rows_spiked = spiked.filter(pl.col("timestamp") < int(_build_ohlcv(120)["timestamp"][92]))
        for col in tf_cols:
            self.assertTrue(np.allclose(safe_rows[col].to_numpy(), safe_rows_spiked[col].to_numpy(), atol=1e-12))

    def test_data_node_enforces_purge_and_embargo(self) -> None:
        historical_data = pl.DataFrame(
            {
                "timestamp": [1_700_000_000_000 + (idx * 900_000) for idx in range(64)],
                "close": [100.0 + idx for idx in range(64)],
            }
        )
        state = {
            "historical_data": historical_data,
            "cursor": 12,
            "paused": False,
            "split_metadata": {
                "train_start": 0,
                "train_end": 20,
                "validation_start": 20,
                "validation_end": 30,
                "oos_start": 31,
                "oos_end": 63,
                "purge_bars": 5,
                "max_feature_lag": 16,
                "required_embargo_gap": 17,
            },
        }
        out = data_node(state)
        expected_cursor = 30 + max(5, 17)
        self.assertEqual(out["cursor"], expected_cursor)
        self.assertEqual(out["current_row"]["timestamp"], int(historical_data["timestamp"][expected_cursor]))

    def test_runtime_config_node_reads_signal_delay(self) -> None:
        out = runtime_config_node({"config": {"simulation": {"signal_delay_bars": 3}}})
        self.assertEqual(out["signal_delay_bars_runtime"], 3)

    def test_split_metadata_records_feature_lag_and_embargo(self) -> None:
        config = build_test_config()
        metadata = compute_split_metadata(config, total_rows=1200)
        self.assertEqual(metadata["max_feature_lag"], config["features"]["max_lag_bars"])
        self.assertGreaterEqual(metadata["required_embargo_gap"], metadata["purge_bars"])

    def test_split_metadata_snapshot_for_1000_bars_matches_expected_counts(self) -> None:
        config = build_test_config()
        config["runtime"]["warmup_bars"] = 240
        config["splits"]["minimum_oos_bars"] = 200
        config["features"]["max_lag_bars"] = 16
        config["walk_forward"]["purge_bars"] = 5
        metadata = compute_split_metadata(config, total_rows=1000)
        self.assertEqual(metadata["train_end"], 600)
        self.assertEqual(metadata["validation_end"], 800)
        self.assertEqual(metadata["required_embargo_gap"], 17)
        self.assertEqual(metadata["oos_start"], 817)
        self.assertEqual(metadata["oos_end"], 999)
        self.assertEqual(metadata["oos_end"] - metadata["oos_start"], 182)

    def test_equity_curve_diagnostics_respect_timeframe(self) -> None:
        returns = [0.01, -0.02, 0.015, -0.005]
        equity_curve = [10100.0, 9898.0, 10046.47, 9996.23765]
        diag_1m = compute_equity_curve_diagnostics(returns, equity_curve, 10000.0, timeframe="1m")
        diag_15m = compute_equity_curve_diagnostics(returns, equity_curve, 10000.0, timeframe="15m")
        self.assertGreater(diag_1m["annualized_volatility"], diag_15m["annualized_volatility"])
        self.assertAlmostEqual(
            float(diag_1m["annualized_volatility"]) / float(diag_15m["annualized_volatility"]),
            float(np.sqrt(15.0)),
            places=10,
        )

    def test_compare_benchmark_metrics_warns_on_negative_sharpe_pair(self) -> None:
        out = compare_benchmark_metrics(
            {"sharpe": -1.0, "total_return": -0.10},
            {"overall": {"mean_sharpe": -0.5, "mean_total_return": -0.03}},
        )
        self.assertTrue(any("both negative" in message for message in out["overfit_warnings"]))

    def test_evaluate_readiness_rejects_triple_negative_sharpe(self) -> None:
        readiness = evaluate_readiness(
            {
                "config": build_test_config(),
                "dataset_metadata": {"source_mode": "snapshot"},
                "regression_history": [
                    {
                        "dataset_window": {
                            "snapshot_hash": "sha-a",
                            "raw_data_hash": "raw-a",
                            "dataset_hash": "ds-a",
                            "timestamp_start": 1,
                            "timestamp_end": 2,
                        }
                    }
                ],
                "optimization_events": [{"objective_value": -0.7}],
                "trades": [],
                "completed_trades": [],
                "performance": {
                    "run_metrics": {
                        "sharpe": -1.2,
                        "bar_win_rate": 0.0,
                        "transition_count": 0,
                        "completed_trade_win_rate": 0.0,
                        "completed_trade_count": 0,
                    },
                    "benchmark_metrics": {
                        "overall": {"mean_sharpe": -0.8},
                        "sufficiency": {"overall_sufficient": True},
                    },
                },
            }
        )
        blocker_codes = {row["code"] for row in readiness["hard_blockers"]}
        self.assertIn("triple_negative_sharpe", blocker_codes)
        self.assertEqual(readiness["phase_gate_decision"], "do_not_advance")
        self.assertFalse(readiness["fine_tuning_gate_open"])
        self.assertTrue(readiness["warnings"])

    def test_evaluate_readiness_blocks_exchange_mode(self) -> None:
        readiness = evaluate_readiness(
            {
                "config": build_test_config(),
                "dataset_metadata": {"source_mode": "exchange"},
                "regression_history": [
                    {
                        "dataset_window": {
                            "snapshot_hash": "sha-a",
                            "raw_data_hash": "raw-a",
                            "dataset_hash": "ds-a",
                            "timestamp_start": 1,
                            "timestamp_end": 2,
                        }
                    }
                ],
                "optimization_events": [{"objective_value": 0.2}],
                "trades": [],
                "completed_trades": [],
                "performance": {
                    "run_metrics": {
                        "sharpe": 0.1,
                        "bar_win_rate": 0.5,
                        "transition_count": 0,
                        "completed_trade_win_rate": 0.0,
                        "completed_trade_count": 0,
                    },
                    "benchmark_metrics": {
                        "overall": {"mean_sharpe": 0.05},
                        "sufficiency": {"overall_sufficient": True},
                    },
                },
            }
        )
        blocker_codes = {row["code"] for row in readiness["hard_blockers"]}
        self.assertIn("exchange_mode_not_regression_eligible", blocker_codes)
        self.assertFalse(readiness["fine_tuning_gate_open"])

    def test_evaluate_readiness_rejects_missing_regression_history(self) -> None:
        readiness = evaluate_readiness(
            {
                "config": build_test_config(),
                "dataset_metadata": {"source_mode": "snapshot"},
                "optimization_events": [{"objective_value": 0.2}],
                "trades": [],
                "completed_trades": [],
                "performance": {
                    "run_metrics": {
                        "sharpe": 0.1,
                        "bar_win_rate": 0.5,
                        "transition_count": 0,
                        "completed_trade_win_rate": 0.0,
                        "completed_trade_count": 0,
                    },
                    "benchmark_metrics": {
                        "overall": {"mean_sharpe": 0.05},
                        "sufficiency": {"overall_sufficient": True},
                    },
                },
            }
        )
        blocker_codes = {row["code"] for row in readiness["hard_blockers"]}
        self.assertIn("regression_history_missing", blocker_codes)
        self.assertEqual(readiness["phase_gate_decision"], "do_not_advance")

    def test_evaluate_readiness_rejects_single_window_evidence(self) -> None:
        config = build_test_config()
        config["readiness"]["min_historical_windows"] = 3
        readiness = evaluate_readiness(
            {
                "config": config,
                "dataset_metadata": {
                    "source_mode": "snapshot",
                    "snapshot_hash": "sha-current",
                    "raw_data_hash": "raw-current",
                    "dataset_hash": "ds-current",
                    "timestamp_start": 10,
                    "timestamp_end": 20,
                },
                "regression_history": [
                    {
                        "dataset_window": {
                            "snapshot_hash": "sha-current",
                            "raw_data_hash": "raw-current",
                            "dataset_hash": "ds-current",
                            "timestamp_start": 10,
                            "timestamp_end": 20,
                        }
                    }
                ],
                "optimization_events": [{"objective_value": 0.2}],
                "trades": [],
                "completed_trades": [],
                "performance": {
                    "run_metrics": {
                        "sharpe": 0.1,
                        "bar_win_rate": 0.5,
                        "transition_count": 0,
                        "completed_trade_win_rate": 0.0,
                        "completed_trade_count": 0,
                    },
                    "benchmark_metrics": {
                        "overall": {"mean_sharpe": 0.05},
                        "sufficiency": {"overall_sufficient": True},
                    },
                },
            }
        )
        blocker_codes = {row["code"] for row in readiness["hard_blockers"]}
        self.assertIn("multi_window_history_insufficient", blocker_codes)
        self.assertFalse(readiness["fine_tuning_gate_open"])

    def test_evaluate_readiness_rejects_degenerate_threshold_calibration(self) -> None:
        config = build_test_config()
        config["readiness"]["min_historical_windows"] = 1
        readiness = evaluate_readiness(
            {
                "config": config,
                "dataset_metadata": {
                    "source_mode": "snapshot",
                    "snapshot_hash": "sha-current",
                    "raw_data_hash": "raw-current",
                    "dataset_hash": "ds-current",
                    "timestamp_start": 10,
                    "timestamp_end": 20,
                },
                "regression_history": [
                    {
                        "dataset_window": {
                            "snapshot_hash": "sha-current",
                            "raw_data_hash": "raw-current",
                            "dataset_hash": "ds-current",
                            "timestamp_start": 10,
                            "timestamp_end": 20,
                        }
                    }
                ],
                "optimization_events": [
                    {
                        "objective_value": 0.2,
                        "diagnostics": {
                            "guard_enabled": True,
                            "degenerate_regime": True,
                            "transition_count": 1,
                            "min_transition_count": 12,
                        },
                    }
                ],
                "trades": [],
                "completed_trades": [],
                "performance": {
                    "run_metrics": {
                        "sharpe": 0.1,
                        "bar_win_rate": 0.5,
                        "transition_count": 0,
                        "completed_trade_win_rate": 0.0,
                        "completed_trade_count": 0,
                    },
                    "benchmark_metrics": {
                        "overall": {"mean_sharpe": 0.05},
                        "sufficiency": {"overall_sufficient": True},
                    },
                },
            }
        )
        blocker_codes = {row["code"] for row in readiness["hard_blockers"]}
        self.assertIn("degenerate_threshold_calibration", blocker_codes)
        self.assertEqual(readiness["phase_gate_decision"], "do_not_advance")

    def test_development_plan_matches_default_timeframe(self) -> None:
        plan_text = (PROJECT_ROOT / "docs" / "development-plan.md").read_text(encoding="utf-8")
        self.assertIn("BTC 15m data", plan_text)


if __name__ == "__main__":
    unittest.main()
