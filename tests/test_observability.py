from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.artifacts import persist_run_artifacts
from agents.decision_audit import build_decision_audit_row
from agents.logging_utils import JsonlEventFormatter
from agents.metrics_utils import compute_trade_summary_statistics


def _config() -> dict:
    return {
        "asset": {"symbol": "BTC/USDT", "timeframe": "1m"},
        "simulation": {"initial_equity": 10000.0, "slippage_bps": 2},
        "logging": {"decision_log_enabled": True},
        "reporting": {"write_report_json": True, "persist_csv_exports": True},
    }


class ObservabilityTests(unittest.TestCase):
    def test_json_formatter_outputs_machine_readable_payload(self) -> None:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="ignored",
            args=(),
            exc_info=None,
        )
        record.event_payload = {
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "run_id": "abc",
            "event_type": "run_started",
            "stage": "run",
        }
        payload = json.loads(JsonlEventFormatter().format(record))
        self.assertEqual(payload["event_type"], "run_started")
        self.assertEqual(payload["run_id"], "abc")
        self.assertEqual(payload["stage"], "run")

    def test_trade_summary_statistics(self) -> None:
        summary = compute_trade_summary_statistics(
            [
                {"direction": 1, "pnl_pct": 0.02, "hold_bars": 3},
                {"direction": -1, "pnl_pct": -0.01, "hold_bars": 2},
                {"direction": 1, "pnl_pct": 0.01, "hold_bars": 4},
            ]
        )
        self.assertEqual(summary["trade_count"], 3)
        self.assertAlmostEqual(summary["win_rate"], 2 / 3, places=6)
        self.assertEqual(summary["max_consecutive_losses"], 1)
        self.assertEqual(summary["long_count"], 2)
        self.assertEqual(summary["short_count"], 1)

    def test_decision_audit_row_generation(self) -> None:
        state = {
            "config": _config(),
            "cursor": 10,
            "current_row": {"timestamp": 1700000000000, "close": 50000.0, "rsi_14": 55.0},
            "prediction": {"prob_up": 0.72, "regime": "normal", "source_model": "lightgbm_baseline"},
            "strategy_params": {"long_threshold": 0.6, "short_threshold": 0.4},
            "pre_risk_action": "LONG",
            "proposed_position": 1,
            "risk_status": {"paused": False, "blocked_high_volatility": False, "stop_triggered": False, "take_profit_triggered": False, "reasons": []},
            "last_action": "LONG",
            "position": 0,
            "target_position": 1,
        }
        row = build_decision_audit_row(
            state,
            strategy_return=0.001,
            realized_pnl_pct=None,
            unrealized_pnl_pct=0.0,
        )
        self.assertEqual(row["final_action"], "LONG")
        self.assertEqual(row["pre_risk_proposed_position"], 1)
        self.assertEqual(row["close_price"], 50000.0)
        self.assertIn("model_output", row)
        self.assertIn("selected_features", row)

    def test_partial_artifact_persistence_writes_failure_files(self) -> None:
        state = {
            "config": _config(),
            "run_id": "failed-run",
            "artifact_dir": "artifacts/failed-run",
            "dataset_metadata": {},
            "split_metadata": {},
            "strategy_params": {},
            "performance": {"run_metrics": {}, "benchmark_metrics": {}},
            "returns": [],
            "equity_curve": [],
            "trades": [],
            "trade_history_buffer": [],
            "completed_trades": [],
            "decision_log": [],
            "optimization_events": [],
            "event_log": [{"event_type": "run_failed"}],
            "error_info": {
                "error_type": "RuntimeError",
                "error_message": "boom",
                "traceback": "traceback lines",
                "failed_at_utc": "2026-01-01T00:00:00+00:00",
            },
            "last_state_snapshot": {"cursor": 12, "position": 1},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = persist_run_artifacts(Path(tmp_dir), state)
            self.assertTrue((artifact_path / "error_summary.json").exists())
            self.assertTrue((artifact_path / "traceback.txt").exists())
            self.assertTrue((artifact_path / "last_state_snapshot.json").exists())
            report_text = (artifact_path / "report.md").read_text(encoding="utf-8")
            self.assertIn("Failure / Fallbacks", report_text)
            self.assertIn("RuntimeError", report_text)

    def test_csv_exports_serialize_nested_trade_fields(self) -> None:
        state = {
            "config": _config(),
            "run_id": "nested-csv-run",
            "artifact_dir": "artifacts/nested-csv-run",
            "dataset_metadata": {},
            "split_metadata": {},
            "strategy_params": {},
            "optimization_events": [],
            "feature_importances": [],
            "trade_history_buffer": [],
            "completed_trades": [],
            "equity_curve": [10000.0],
            "returns": [0.0],
            "performance": {"run_metrics": {}, "benchmark_metrics": {}},
            "event_log": [],
            "decision_log": [],
            "shap_rule": "",
            "trades": [
                {
                    "cycle": 1,
                    "action": "LONG",
                    "risk_reasons": ["blocked_high_volatility", "max_drawdown_pause"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = persist_run_artifacts(Path(tmp_dir), state)
            trades_csv = (artifact_path / "trades.csv").read_text(encoding="utf-8")
            self.assertIn("risk_reasons", trades_csv)
            self.assertIn("blocked_high_volatility", trades_csv)
            self.assertIn("max_drawdown_pause", trades_csv)


if __name__ == "__main__":
    unittest.main()
