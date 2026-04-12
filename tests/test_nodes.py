from __future__ import annotations

import sys
import unittest
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.nodes import decision_node, evaluate_node, predict_node, risk_node


def _config() -> dict:
    return {
        "asset": {"symbol": "BTC/USDT", "timeframe": "1m"},
        "simulation": {"initial_equity": 10000.0, "slippage_bps": 0, "trade_history_limit": 1000},
        "risk": {
            "max_abs_position": 1,
            "max_turnover_per_bar": 1,
            "max_drawdown_pause": 0.5,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.03,
            "block_high_volatility": False,
        },
        "logging": {"decision_log_enabled": True},
        "reporting": {"decision_log_enabled": True},
    }


class NodeTests(unittest.TestCase):
    def test_decision_node_generates_expected_action(self) -> None:
        result = decision_node({"position": 0, "target_position": 1})
        self.assertEqual(result["last_action"], "LONG")

        result = decision_node({"position": 1, "target_position": 0})
        self.assertEqual(result["last_action"], "FLAT")

        result = decision_node({"position": -1, "target_position": -1})
        self.assertEqual(result["last_action"], "HOLD")

    def test_risk_node_produces_target_and_status(self) -> None:
        state = {
            "config": _config(),
            "position": 1,
            "entry_price": 100.0,
            "current_row": {"close": 97.0},
            "prediction": {"prob_up": 0.2, "regime": "normal"},
            "strategy_params": {"long_threshold": 0.6, "short_threshold": 0.4},
            "performance": {"run_metrics": {"max_drawdown": 0.0}},
        }
        result = risk_node(state)
        self.assertIn("target_position", result)
        self.assertIn("risk_status", result)
        self.assertTrue(result["risk_status"]["stop_triggered"])

    def test_predict_node_emits_prediction_payload(self) -> None:
        historical = pl.DataFrame(
            {
                "timestamp": [1_000, 2_000, 3_000],
                "close": [100.0, 101.0, 102.0],
                "volatility_regime": [0, 1, 0],
            }
        )
        result = predict_node(
            {
                "current_row": {"timestamp": 1_000, "close": 100.0, "volatility_regime": 1},
                "historical_data": historical,
                "cursor": 0,
                "lightgbm_model": None,
                "feature_columns": [],
            }
        )
        prediction = result["prediction"]
        self.assertIn("prob_up", prediction)
        self.assertEqual(prediction["regime"], "high_volatility")
        self.assertEqual(prediction["execution_timestamp"], 1_000)

    def test_evaluate_node_updates_metrics_and_decision_log(self) -> None:
        historical = pl.DataFrame(
            {
                "timestamp": [1_000, 2_000, 3_000],
                "close": [100.0, 101.0, 102.0],
            }
        )
        state = {
            "config": _config(),
            "historical_data": historical,
            "cursor": 0,
            "position": 0,
            "target_position": 1,
            "entry_price": None,
            "entry_cycle": None,
            "prediction": {
                "prob_up": 0.8,
                "regime": "normal",
                "signal_timestamp": 1_000,
                "execution_timestamp": 1_000,
                "source_model": "fallback",
            },
            "current_row": {"timestamp": 1_000, "close": 100.0, "rsi_14": 55.0},
            "last_action": "LONG",
            "strategy_params": {"long_threshold": 0.6, "short_threshold": 0.4},
            "risk_status": {"paused": False, "reasons": [], "stop_triggered": False, "take_profit_triggered": False, "blocked_high_volatility": False},
            "returns": [],
            "equity_curve": [],
            "trades": [],
            "trade_history_buffer": [],
            "completed_trades": [],
            "decision_log": [],
            "equity": 10000.0,
            "performance": {"run_metrics": {}, "benchmark_metrics": {}},
            "split_metadata": {"oos_end": 2},
        }
        result = evaluate_node(state)
        self.assertEqual(len(result["returns"]), 1)
        self.assertEqual(len(result["equity_curve"]), 1)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(len(result["decision_log"]), 1)
        self.assertEqual(result["position"], 1)


if __name__ == "__main__":
    unittest.main()
