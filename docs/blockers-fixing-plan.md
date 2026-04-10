# Fixing Plan — “Blockers to Clear Before Go-Ahead”

**Date (UTC):** 2026-04-10  
**Scope:** unblock progression from internal prototyping to next project phases (RL / deployment-oriented work).

## Goal

Move the project from **No-go** to **Go-ahead** by resolving the four blockers:
1. Data volume/quality is insufficient for meaningful walk-forward validation.
2. Test suite is not green.
3. Readiness gates are not encoded/enforced.
4. Optimization can drift without robust constraints.

---

## Blocker 1 — Increase dataset size and ensure non-zero walk-forward folds

### Actions
1. Fetch real BTC/USDT 1m data for at least 6 months:
   - `uv run python scripts/fetch_real_data.py --months 6 --symbol BTC/USDT --out data/btc_1m_real.csv`
2. Update config defaults for production-like backtest runs:
   - `asset.local_cache_path: data/btc_1m_real.csv`
   - keep `fetch_limit` high enough for full file usage.
3. Add a hard warning (or fail-fast in strict mode) when walk-forward has zero folds.
4. Re-run full MVP and regenerate report.

### Acceptance criteria
- `data/btc_1m_real.csv` exists and contains sufficient rows for configured train/test/step/embargo settings.
- `reports/week*_report.md` shows **non-zero** walk-forward folds.
- Report metadata clearly indicates real data source + range.

### Estimated effort
- 0.5–1 day.

---

## Blocker 2 — Fix failing tests and restore green baseline

### Actions
1. Fix walk-forward tests to reflect current embargo constraints and config assumptions.
2. Fix `data_node` tests/mocks to match `_load_historical_btc_data` return signature and current behavior.
3. Add regression tests for:
   - zero-fold walk-forward handling,
   - minimum-data checks,
   - embargo-gap guardrails.
4. Run full test suite in CI-equivalent command.

### Acceptance criteria
- `uv run pytest -q` returns all passing.
- No failing tests in walk-forward/data-node areas.
- New tests cover previously failing edge cases.

### Estimated effort
- 0.5–1.5 days.

---

## Blocker 3 — Implement explicit readiness gates

### Actions
1. Define readiness thresholds in config (example):
   - `min_returns_for_sharpe: 1000`
   - `min_walk_forward_folds: 5`
   - `require_positive_wf_sharpe: true`
2. Add gate evaluation in report generation and/or run-end logic.
3. Print an explicit verdict in report:
   - `GO_AHEAD: true/false`
   - with reasons for failures.
4. Add tests for gate pass/fail scenarios.

### Acceptance criteria
- Report includes machine-readable and human-readable readiness verdict.
- Runs with insufficient evidence fail gates automatically.
- Gate logic covered by tests.

### Estimated effort
- 1 day.

---

## Blocker 4 — Bound optimization drift

### Actions
1. Constrain Optuna search space and add sanity checks on thresholds.
2. Track optimization stability metrics (e.g., threshold variance over last N events).
3. Add simple anti-degeneracy checks (e.g., reject parameter sets that suppress trading activity excessively).
4. Log optimization diagnostics in report.

### Acceptance criteria
- Thresholds stay within rational operating band.
- Optimization events include activity-aware diagnostics.
- No persistent “degenerate” threshold regime across runs.

### Estimated effort
- 1 day.

---

## Execution Order (Recommended)

1. **Data first** (Blocker 1)  
2. **Tests second** (Blocker 2)  
3. **Readiness gates third** (Blocker 3)  
4. **Optimization hardening fourth** (Blocker 4)

This order prevents tuning/feature work on unreliable evidence.

---

## Proposed 5-Day Sprint Plan

- **Day 1:** Real data ingestion + config defaults + baseline rerun.
- **Day 2:** Fix failing tests + add regression tests.
- **Day 3:** Implement readiness gates + report verdict.
- **Day 4:** Optuna drift controls + diagnostics.
- **Day 5:** End-to-end rerun, validate gates, finalize updated review and go/no-go decision.

---

## Exit Criteria for “Go-ahead”

Proceed to next phases only if all are true:
- Test suite is green.
- Walk-forward folds are non-zero and above minimum threshold.
- Returns sample size meets reliability threshold.
- Out-of-sample metrics pass configured readiness gates.
- Report outputs `GO_AHEAD: true` with no critical warnings.
