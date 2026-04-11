from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.artifacts import persist_run_artifacts
from agents.data import compute_split_metadata
from agents.evaluation import build_walk_forward_folds, simulate_policy
from agents.risk import evaluate_risk


def build_test_config() -> dict:
    return {
        "asset": {"symbol": "BTC/USDT", "timeframe": "1m", "exchange": "binance", "fetch_limit": 1200},
        "dataset": {"source_mode": "snapshot"},
        "snapshot": {"path": "ignored.parquet", "auto_write": False},
        "runtime": {"max_cycles": 200, "warmup_bars": 120, "artifact_output_dir": "artifacts"},
        "splits": {"train_ratio": 0.6, "validation_ratio": 0.2, "minimum_oos_bars": 150},
        "simulation": {"initial_equity": 10000.0, "slippage_bps": 5, "signal_delay_bars": 1, "optuna_trials": 5, "trade_history_limit": 1000},
        "risk": {
            "max_abs_position": 1,
            "max_turnover_per_bar": 1,
            "max_drawdown_pause": 0.15,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.05,
            "block_high_volatility": True,
        },
        "walk_forward": {"train_bars": 240, "validation_bars": 60, "test_bars": 40, "step_bars": 20, "purge_bars": 5},
        "logging": {"level": "INFO"},
    }


class PipelineTests(unittest.TestCase):
    def test_split_metadata_keeps_oos_after_validation(self) -> None:
        config = build_test_config()
        split_metadata = compute_split_metadata(config, total_rows=1200)
        self.assertLess(split_metadata["train_end"], split_metadata["validation_end"])
        self.assertLess(split_metadata["validation_end"], split_metadata["oos_start"] + 1)
        self.assertGreaterEqual(split_metadata["oos_start"], split_metadata["validation_end"])
        self.assertGreater(split_metadata["oos_end"], split_metadata["oos_start"])

    def test_signal_delay_and_trade_costs_apply_to_next_bar(self) -> None:
        close = pl.Series("close", [100.0, 110.0, 121.0]).to_numpy()
        probs = pl.Series("prob", [0.9, 0.1, 0.1]).to_numpy()
        volatility_regime = pl.Series("regime", [0, 0, 0]).to_numpy()
        timestamps = pl.Series("timestamp", [1, 2, 3]).to_numpy()
        result = simulate_policy(
            close=close,
            probs=probs,
            volatility_regime=volatility_regime,
            strategy_params={"long_threshold": 0.6, "short_threshold": 0.4},
            risk_cfg={"max_abs_position": 1, "max_turnover_per_bar": 2, "block_high_volatility": False, "stop_loss_pct": 0.0, "take_profit_pct": 0.0},
            initial_equity=10000.0,
            slippage_bps=0.0,
            timestamp=timestamps,
        )
        self.assertEqual(len(result["returns"]), 2)
        self.assertAlmostEqual(result["returns"][0], 0.10, places=6)
        self.assertAlmostEqual(result["returns"][1], -0.10, places=6)
        self.assertEqual(result["trades"][0]["to_position"], 1)
        self.assertEqual(result["trades"][1]["to_position"], -1)

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

    def test_artifact_bundle_writes_expected_files(self) -> None:
        state = {
            "config": build_test_config(),
            "run_id": "test-run",
            "artifact_dir": "artifacts/test-run",
            "dataset_metadata": {"rows": 100, "source_mode": "snapshot"},
            "split_metadata": {"train_start": 0, "train_end": 60, "validation_start": 60, "validation_end": 80, "oos_start": 80, "oos_end": 99},
            "strategy_params": {"long_threshold": 0.6, "short_threshold": 0.4},
            "optimization_events": [{"split": "validation", "objective_name": "validation_sharpe", "objective_value": 1.23, "best_params": {"long_threshold": 0.6, "short_threshold": 0.4}, "affects_future_oos_only": True}],
            "feature_importances": [{"feature": "rsi_14", "importance": 10.0}],
            "trades": [{"action": "LONG"}],
            "trade_history_buffer": [{"action": "LONG"}],
            "equity_curve": [10050.0],
            "returns": [0.005],
            "performance": {
                "run_metrics": {"sharpe": 1.0, "max_drawdown": 0.01, "win_rate": 1.0, "total_return": 0.005, "trade_count": 1},
                "benchmark_metrics": {
                    "settings": {"train_bars": 240, "validation_bars": 60, "test_bars": 40, "purge_bars": 5, "signal_delay_bars": 1},
                    "expanding": {"mean_sharpe": 0.1, "mean_max_drawdown": 0.02, "mean_total_return": 0.03, "mean_win_rate": 0.6, "folds": [1]},
                    "rolling": {"mean_sharpe": 0.2, "mean_max_drawdown": 0.03, "mean_total_return": 0.04, "mean_win_rate": 0.7, "folds": [1]},
                },
            },
            "shap_rule": "Test rule",
            "cursor": 90,
            "paused": False,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = persist_run_artifacts(Path(tmp_dir), state)
            self.assertTrue((artifact_path / "report.md").exists())
            self.assertTrue((artifact_path / "benchmark_metrics.json").exists())
            report_text = (artifact_path / "report.md").read_text(encoding="utf-8")
            self.assertIn("Held-Out Simulation", report_text)
            benchmark_payload = json.loads((artifact_path / "benchmark_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("expanding", benchmark_payload)


if __name__ == "__main__":
    unittest.main()
