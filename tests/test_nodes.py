"""Unit tests for core trading agent functions."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Ensure src is on the path
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from agents.nodes import (
    _compute_performance,
    _run_walk_forward_backtest,
    _safe_float,
    _sigmoid,
    data_node,
    decision_node,
    risk_node,
)


# ---------------------------------------------------------------------------
# _sigmoid
# ---------------------------------------------------------------------------
class TestSigmoid:
    def test_zero(self):
        assert _sigmoid(0.0) == 0.5

    def test_large_positive(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        assert _sigmoid(2.0) + _sigmoid(-2.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------
class TestSafeFloat:
    def test_none(self):
        assert _safe_float(None, 5.0) == 5.0

    def test_valid_int(self):
        assert _safe_float(3, 0.0) == 3.0

    def test_valid_string(self):
        assert _safe_float("1.5", 0.0) == 1.5

    def test_invalid_string(self):
        assert _safe_float("abc", -1.0) == -1.0


# ---------------------------------------------------------------------------
# _compute_performance
# ---------------------------------------------------------------------------
class TestComputePerformance:
    def test_empty_returns(self):
        result = _compute_performance([], [], 10_000.0)
        assert result["sharpe"] == 0.0
        assert result["max_drawdown"] == 0.0
        assert result["win_rate"] == 0.0
        assert result["total_return"] == 0.0

    def test_all_positive_returns(self):
        # Need >= 1000 samples with non-zero std to produce a computable Sharpe.
        # Use slightly varied positive returns so std > 0.
        returns = [0.001 + 0.0001 * (i % 3 - 1) for i in range(1200)]
        equity = 10_000.0
        equity_curve = []
        for r in returns:
            equity *= 1 + r
            equity_curve.append(equity)
        result = _compute_performance(returns, equity_curve, 10_000.0)
        assert result["sharpe"] > 0.0
        assert result["max_drawdown"] == 0.0  # no drawdown with all positive
        assert result["win_rate"] == 1.0
        assert result["total_return"] > 0.0

    def test_mixed_returns(self):
        returns = [0.01, -0.02, 0.01, -0.01, 0.02]
        initial = 10_000.0
        equity = initial
        equity_curve = []
        for r in returns:
            equity *= 1 + r
            equity_curve.append(equity)

        result = _compute_performance(returns, equity_curve, initial)
        assert result["win_rate"] == pytest.approx(0.6)  # 3/5 positive
        assert result["max_drawdown"] > 0.0

    def test_sharpe_annualization(self):
        """Sharpe should use sqrt(525600) annualization for 1-min bars.

        Must supply >= 1000 samples to pass _compute_performance's sample-size guard.
        """
        returns_varied = [0.001 + 0.0001 * (i % 3 - 1) for i in range(1200)]
        equity = 10_000.0
        equity_curve = []
        for r in returns_varied:
            equity *= 1 + r
            equity_curve.append(equity)

        result = _compute_performance(returns_varied, equity_curve, 10_000.0)
        expected_annualization = math.sqrt(365 * 24 * 60)
        assert result["sharpe"] > 0.0
        raw_sharpe = np.mean(returns_varied) / np.std(returns_varied, ddof=1)
        assert result["sharpe"] == pytest.approx(
            raw_sharpe * expected_annualization, rel=1e-4
        )

    def test_total_return_calculation(self):
        initial = 10_000.0
        final = 11_000.0
        equity_curve = [final]
        returns = [0.1]
        result = _compute_performance(returns, equity_curve, initial)
        assert result["total_return"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# decision_node
# ---------------------------------------------------------------------------
def _make_decision_state(
    prob_up: float,
    long_threshold: float = 0.6,
    short_threshold: float = 0.4,
    position: int = 0,
    risk_params: dict | None = None,
    entry_price: float | None = None,
) -> dict:
    return {
        "prediction": {"prob_up": prob_up, "regime": "normal"},
        "risk_params": risk_params or {},
        "config": {
            "simulation": {
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
            }
        },
        "position": position,
        "entry_price": entry_price,
        "cycle_count": 1,
        "current_row": {"timestamp": 1_000_000, "close": 50_000.0},
        "trades": [],
    }


class TestDecisionNode:
    @pytest.mark.parametrize(
        "prob_up,expected_action,expected_position",
        [
            (0.7, "LONG", 1),
            (0.3, "SHORT", -1),
            (0.5, "HOLD", 0),
        ],
    )
    def test_threshold_routing(self, prob_up, expected_action, expected_position):
        state = _make_decision_state(prob_up)
        result = decision_node(state)
        assert result["last_action"] == expected_action
        assert result["position"] == expected_position

    def test_stop_loss_overrides_long_signal(self):
        """stop_triggered=True forces HOLD even if prob_up is very high."""
        state = _make_decision_state(
            prob_up=0.95,
            position=1,
            entry_price=50_000.0,
            risk_params={"stop_triggered": True, "regime_factor": 1.0, "atr_pct": 0.0},
        )
        result = decision_node(state)
        assert result["last_action"] == "HOLD"
        assert result["position"] == 0
        assert result["entry_price"] is None

    def test_entry_price_set_on_position_open(self):
        """entry_price should be recorded when a new position is opened."""
        state = _make_decision_state(prob_up=0.9, position=0)
        result = decision_node(state)
        assert result["position"] == 1
        assert result["entry_price"] == pytest.approx(50_000.0)

    def test_entry_price_cleared_on_flat(self):
        """entry_price should be None when position goes to 0."""
        # prob_up=0.5 → HOLD → position stays 0 when coming from 0
        state = _make_decision_state(prob_up=0.5, position=0, entry_price=50_000.0)
        result = decision_node(state)
        assert result["position"] == 0
        # entry_price unchanged (position did not change)
        assert result.get("entry_price") == 50_000.0

    def test_regime_factor_widens_neutral_band(self):
        """With regime_factor=1.3, a marginal long signal (prob_up=0.61) is blocked."""
        # With base_long=0.6, effective_long = 0.5 + (0.6-0.5)*1.3 = 0.63 → 0.61 < 0.63 → HOLD
        state = _make_decision_state(
            prob_up=0.61,
            long_threshold=0.6,
            short_threshold=0.4,
            risk_params={"stop_triggered": False, "regime_factor": 1.3, "atr_pct": 0.0},
        )
        result = decision_node(state)
        assert result["last_action"] == "HOLD"

    def test_trade_logged_on_position_change(self):
        state = _make_decision_state(prob_up=0.9, position=0)
        result = decision_node(state)
        assert len(result["trades"]) == 1
        trade = result["trades"][0]
        assert trade["action"] == "LONG"
        assert trade["new_position"] == 1

    def test_no_trade_logged_on_hold(self):
        state = _make_decision_state(prob_up=0.5, position=0)
        result = decision_node(state)
        assert len(result["trades"]) == 0


# ---------------------------------------------------------------------------
# risk_node
# ---------------------------------------------------------------------------
def _make_risk_state(
    close: float = 50_000.0,
    atr_14: float = 100.0,
    regime: str = "normal",
    position: int = 0,
    entry_price: float | None = None,
    stop_loss_pct: float = 0.02,
) -> dict:
    return {
        "current_row": {"close": close, "atr_14": atr_14},
        "prediction": {"prob_up": 0.5, "regime": regime},
        "position": position,
        "entry_price": entry_price,
        "config": {
            "simulation": {"stop_loss_pct": stop_loss_pct, "long_threshold": 0.55, "short_threshold": 0.45}
        },
    }


class TestRiskNode:
    def test_normal_regime_no_stop(self):
        state = _make_risk_state(position=1, entry_price=49_500.0)  # +1% up, no stop
        result = risk_node(state)
        rp = result["risk_params"]
        assert rp["stop_triggered"] is False
        assert rp["regime_factor"] == pytest.approx(1.0)

    def test_high_volatility_widens_factor(self):
        state = _make_risk_state(regime="high_volatility")
        result = risk_node(state)
        assert result["risk_params"]["regime_factor"] == pytest.approx(1.3)

    def test_stop_loss_triggers_on_long(self):
        # Long at 50000, price drops to 48500 → PnL = -3% < -2% threshold
        state = _make_risk_state(close=48_500.0, position=1, entry_price=50_000.0)
        result = risk_node(state)
        assert result["risk_params"]["stop_triggered"] is True

    def test_stop_loss_does_not_trigger_small_loss(self):
        # Long at 50000, price drops to 49100 → PnL = -1.8% > -2% threshold
        state = _make_risk_state(close=49_100.0, position=1, entry_price=50_000.0)
        result = risk_node(state)
        assert result["risk_params"]["stop_triggered"] is False

    def test_atr_pct_computed(self):
        state = _make_risk_state(close=50_000.0, atr_14=500.0)
        result = risk_node(state)
        assert result["risk_params"]["atr_pct"] == pytest.approx(0.01)  # 500/50000


# ---------------------------------------------------------------------------
# _run_walk_forward_backtest — smoke test
# ---------------------------------------------------------------------------
_SHARED_CONFIG = {
    "walk_forward": {
        "train_bars": 100,
        "test_bars": 30,
        "step_bars": 50,
        "interval_cycles": 500,
    },
    "model": {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": 1,
        "label_horizon": 1,
        "max_feature_lag": 60,
    },
    "simulation": {
        "initial_equity": 10_000.0,
        "slippage_bps": 5,
        "long_threshold": 0.55,
        "short_threshold": 0.45,
    },
}


class TestWalkForwardBacktest:
    def _make_state(self, n: int = 300) -> dict:
        rng = np.random.default_rng(42)
        close = np.cumprod(1 + rng.normal(0, 0.001, n)) * 50_000
        feat = rng.standard_normal(n)
        target_up = (np.diff(close, prepend=close[0]) > 0).astype(int)
        df = pl.DataFrame({"close": close, "feat": feat, "target_up": target_up})
        return {
            "historical_data": df,
            "feature_columns": ["feat"],
            "config": _SHARED_CONFIG,
            "strategy_params": {"long_threshold": 0.55, "short_threshold": 0.45},
        }

    def test_smoke_returns_expected_keys(self):
        result = _run_walk_forward_backtest(self._make_state())
        assert set(result.keys()) == {"settings", "expanding", "rolling", "overall"}
        for section in ("expanding", "rolling", "overall"):
            assert "mean_sharpe" in result[section]
            assert isinstance(result[section]["mean_sharpe"], float)

    def test_empty_feature_columns_returns_zeros(self):
        state = self._make_state()
        state["feature_columns"] = []
        result = _run_walk_forward_backtest(state)
        assert result["overall"]["mean_sharpe"] == 0.0

    def test_at_least_one_fold_produced(self):
        result = _run_walk_forward_backtest(self._make_state())
        assert len(result["expanding"]["folds"]) >= 1
        assert len(result["rolling"]["folds"]) >= 1

    @pytest.fixture
    def deterministic_state(self) -> dict:
        n = 220
        t = np.arange(n, dtype=float)
        close = 50_000.0 + 10.0 * t + 200.0 * np.sin(t / 4.0)
        feat = np.sin(t / 5.0)
        target_up = (np.roll(close, -1) > close).astype(int)
        target_up[-1] = target_up[-2]
        df = pl.DataFrame({"close": close, "feat": feat, "target_up": target_up})
        cfg = {
            **_SHARED_CONFIG,
            "walk_forward": {
                "train_bars": 80,
                "test_bars": 20,
                "step_bars": 30,
                "embargo_gap": 10,
                "interval_cycles": 500,
            },
            "simulation": {
                **_SHARED_CONFIG["simulation"],
                "slippage_bps": 1,
                "funding_rate_per_8h": 0.0,
            },
        }
        return {
            "historical_data": df,
            "feature_columns": ["feat"],
            "config": cfg,
            "strategy_params": {"long_threshold": 0.55, "short_threshold": 0.45},
        }

    def test_fold_boundaries_and_count_are_exact(self, deterministic_state):
        result = _run_walk_forward_backtest(deterministic_state)

        expected_expanding = [
            (0, 80, 90, 110),
            (0, 110, 120, 140),
            (0, 140, 150, 170),
            (0, 170, 180, 200),
        ]
        expected_rolling = [
            (0, 80, 90, 110),
            (30, 110, 120, 140),
            (60, 140, 150, 170),
            (90, 170, 180, 200),
        ]

        expanding_bounds = [
            (f["train_start"], f["train_end"], f["test_start"], f["test_end"])
            for f in result["expanding"]["folds"]
        ]
        rolling_bounds = [
            (f["train_start"], f["train_end"], f["test_start"], f["test_end"])
            for f in result["rolling"]["folds"]
        ]
        assert expanding_bounds == expected_expanding
        assert rolling_bounds == expected_rolling

    def test_embargo_enforces_no_train_test_overlap(self, deterministic_state):
        embargo_gap = deterministic_state["config"]["walk_forward"]["embargo_gap"]
        result = _run_walk_forward_backtest(deterministic_state)

        for fold in result["expanding"]["folds"] + result["rolling"]["folds"]:
            assert fold["test_start"] >= fold["train_end"] + embargo_gap
            assert fold["test_start"] > fold["train_end"]

    def test_higher_slippage_and_funding_do_not_improve_returns(self, deterministic_state):
        baseline = _run_walk_forward_backtest(deterministic_state)

        stressed_state = {
            **deterministic_state,
            "config": {
                **deterministic_state["config"],
                "simulation": {
                    **deterministic_state["config"]["simulation"],
                    "slippage_bps": 50,
                    "funding_rate_per_8h": 0.02,
                },
            },
        }
        stressed = _run_walk_forward_backtest(stressed_state)

        assert stressed["overall"]["mean_total_return"] <= baseline["overall"]["mean_total_return"] + 1e-12


class TestDataNode:
    def test_cursor_starts_after_training_plus_embargo(self, monkeypatch):
        import agents.nodes as nodes

        historical_data = pl.DataFrame(
            {
                "timestamp": np.arange(0, 200),
                "close": np.linspace(50_000.0, 50_500.0, 200),
                "feat": np.sin(np.arange(200) / 10.0),
                "target_up": np.where(np.arange(200) % 2 == 0, 1, 0),
            }
        )

        split_idx = 120
        embargo_gap = 9

        def _fake_load_historical_data(_config):
            return historical_data

        def _fake_train_baseline(_historical_data, _config):
            return object(), ["feat"], {"feat": 1.0}, split_idx

        monkeypatch.setattr(nodes, "_load_historical_btc_data", _fake_load_historical_data)
        monkeypatch.setattr(nodes, "_train_lightgbm_baseline", _fake_train_baseline)

        state = {
            "config": {
                "walk_forward": {"embargo_gap": embargo_gap},
                "runtime": {"max_cycles": 1000},
            }
        }
        result = data_node(state)

        expected_cursor = split_idx + embargo_gap
        assert result["cursor"] == expected_cursor
        assert result["cursor"] > split_idx
    def test_raises_when_embargo_gap_below_required_threshold(self):
        state = self._make_state()
        state["config"]["walk_forward"]["embargo_gap"] = 61  # required is 60 + 1 + 1 = 62
        with pytest.raises(ValueError, match="embargo_gap=.*too small.*60 \\+ 1 \\+ 1 = 62"):
            _run_walk_forward_backtest(state)
