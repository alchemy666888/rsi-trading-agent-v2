"""Unit tests for core trading agent functions."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src is on the path
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from agents.nodes import _compute_performance, _sigmoid, _safe_float


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
        returns = [0.01, 0.02, 0.01, 0.03, 0.01]
        equity_curve = [10_100.0, 10_302.0, 10_405.02, 10_717.17, 10_824.34]
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
        """Sharpe should use sqrt(525600) annualization for 1-min bars."""
        returns = [0.001] * 100
        equity = 10_000.0
        equity_curve = []
        for r in returns:
            equity *= 1 + r
            equity_curve.append(equity)

        result = _compute_performance(returns, equity_curve, 10_000.0)
        # With constant returns, std is 0 (ddof=1 with identical values still 0)
        # so sharpe would be 0 due to the guard clause... let's use slight variation
        returns_varied = [0.001 + 0.0001 * (i % 3 - 1) for i in range(100)]
        equity = 10_000.0
        equity_curve = []
        for r in returns_varied:
            equity *= 1 + r
            equity_curve.append(equity)

        result = _compute_performance(returns_varied, equity_curve, 10_000.0)
        expected_annualization = math.sqrt(365 * 24 * 60)
        # Sharpe should be positive (positive mean returns)
        assert result["sharpe"] > 0.0
        # Verify the annualization is in the right ballpark
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
