# Code Review: BTC Model-Based Research Prototype

## Review Scope & Methodology

This review evaluates the repository against its stated Week 1 research milestone, focusing on time-series correctness, leakage prevention, simulation integrity, artifact provenance, and phase-gate readiness. The review examines all source files in `src/agents/`, the test suite, configuration, and the entrypoint.

---

## Executive Summary

**Overall assessment: Strong research prototype with disciplined engineering.**

The codebase demonstrates an unusually high level of rigor for a research-stage trading prototype. Leakage prevention is treated as a first-class concern across training, calibration, walk-forward benchmarking, and held-out simulation. The artifact persistence layer is thorough, and the phase-gate system is well-designed to prevent premature advancement. The test suite covers the most critical invariants.

There are several areas where the implementation could be tightened, and a handful of latent risks that merit attention before expanding the scope of the project. None of these are showstoppers for the stated Week 1 milestone.

**Verdict: The codebase delivers on its stated goals. The pipeline runs end-to-end with appropriate safeguards, and the artifacts it produces are honest about what they represent.**

---

## 1. Time-Series Correctness

### 1.1 Label Construction — PASS

The target variable `target_up` is constructed as `close.shift(-1) > close` in `features.py` (line ~240). This is correct for a one-bar-ahead directional prediction.

Critically, `build_training_arrays_no_leakage()` in `modeling.py` **does not use** the pre-computed `target_up` column for training. Instead, it recomputes labels from within the training window:

```python
y_train = (close[1:] > close[:-1]).astype(int)
x_train = train_df.select(feature_cols).slice(0, train_df.height - 1).to_numpy()
```

This is an important design choice: it guarantees that no label depends on a close price from outside the training window. The terminal row of the training segment is consumed only for its close price (as a label source for the penultimate row), never as a feature row. This is correct.

**Test coverage:** `test_train_lightgbm_baseline_avoids_split_boundary_label_leakage` explicitly verifies that training arrays exclude the boundary row and that labels are computed internally. Good.

### 1.2 Validation Terminal Row Handling — PASS

`drop_terminal_supervised_row()` removes the last row of any validation or test segment before scoring. This prevents the terminal row's label from depending on a close price from the next split segment. Applied consistently in both `calibrate_thresholds()` and `run_walk_forward_backtest()`.

**Test coverage:** `test_drop_terminal_supervised_row_trims_last_bar`. Good.

### 1.3 Multi-Timeframe Feature Joins — PASS (with minor note)

`build_multi_timeframe_features()` in `features.py` uses:

1. `group_by_dynamic` with `closed="left"` to resample.
2. A `.shift(1)` on all feature columns before the join, so that at any base-timeframe bar `t`, only completed higher-timeframe buckets are visible.
3. `join_asof` with `strategy="backward"` for the temporal join.

This is the correct approach. The `.shift(1)` is the key safeguard — without it, partially-formed higher-timeframe candles would leak intra-bar information.

**Test coverage:** `test_multi_timeframe_future_spike_does_not_leak` introduces a price spike at bar 96 and verifies that all `tf_*` columns for bars before bar 92 are identical with and without the spike. This is a well-designed regression test.

**Minor note:** The test uses bar 92 as the safety boundary, which is conservative for the `4h` timeframe at 15m base (16 bars per bucket). This margin is adequate.

### 1.4 Embargo and Purge Gaps — PASS

`compute_split_metadata()` in `data.py` computes `required_embargo_gap = max(purge_bars, max_feature_lag + 1)` and places `oos_start` at `validation_end + required_embargo_gap`. At 15m base with `max_lag_bars=16` and `purge_bars=5`, this yields an embargo of 17 bars.

`data_node()` in `nodes.py` enforces this at runtime: if the cursor is below `validation_end + required_embargo_gap`, it is advanced forward. This is a belt-and-suspenders approach — correct.

**Test coverage:** `test_data_node_enforces_purge_and_embargo` and `test_split_metadata_records_feature_lag_and_embargo`. Both good.

**Question for the author:** The embargo gap accounts for `max_feature_lag` but not for multi-timeframe resampling lookback. At 15m base with `1d` MTF features, a daily candle requires 96 bars of history. The `join_asof` backward strategy plus the `.shift(1)` prevent forward leakage, but the *informational content* of the first few MTF-joined rows in the OOS segment will be identical to the last few in validation. This is not label leakage, but it is a mild form of feature overlap. For the current research milestone this is acceptable, but worth noting for future tightening.

---

## 2. Leakage Prevention

### 2.1 Train/Validation/OOS Isolation — PASS

The split logic is clean:
- Training: rows `[train_start, train_end)` — used only for `model.fit()`.
- Validation: rows `[validation_start, validation_end)` — used only for threshold calibration via Optuna.
- OOS: rows `[oos_start, oos_end)` — reserved for the LangGraph held-out simulation.

The embargo gap physically separates validation from OOS. No parameter search touches OOS data. Good.

### 2.2 Walk-Forward Fold Construction — PASS

`build_walk_forward_folds()` inserts a `purge_bars` gap between each fold's validation end and test start. Both expanding and rolling modes are supported. The test `test_walk_forward_folds_include_purge_gap` verifies this invariant for every fold.

`test_walk_forward_training_avoids_fold_boundary_label_leakage` mocks the LGBMClassifier and captures the training array sizes, verifying they match `train_end - train_start - 1` (the -1 from `build_training_arrays_no_leakage`). This is a good structural test.

### 2.3 Feature Column Exclusion — PASS

`get_model_feature_columns()` excludes `{"timestamp", "dt", "target_up"}`. The model never sees the target column or raw temporal identifiers. Correct.

### 2.4 Signal Delay Implementation — PASS (with nuance)

Signal delay is implemented in two parallel codepaths:

1. **`simulate_policy()`** (batch simulation for calibration/benchmark): Uses a `scheduled_actions` dict keyed by future bar index. Signals are placed at `idx + delay_bars` and executed when the loop reaches that bar.

2. **Runtime nodes** (`decision_node` + `evaluate_node`): Uses a `pending_signals` list. Signals are queued with `apply_cursor = cursor + delay_bars` and consumed when the cursor reaches the apply point.

**Test coverage:** `test_signal_delay_roundtrip` runs both codepaths on the same price/probability sequence and asserts bitwise equality of returns, equity curves, trade timestamps, and completed trades. This is excellent — it catches any divergence between the two implementations.

`test_signal_delay_shift_execution_timestamp_and_changes_return` verifies that delay=1 shifts execution timestamps by exactly one bar and produces different total returns than delay=0.

`test_stop_loss_flatten_is_delayed_by_signal_delay` and `test_take_profit_flatten_is_delayed_by_signal_delay` verify that risk flattening respects the delay. Good.

**Nuance:** In the runtime path, `risk_node` evaluates stop-loss and take-profit at *signal time* (the current cursor), but the resulting flat target is scheduled for `cursor + delay_bars`. This means a stop-loss can be triggered at bar `t` but not executed until bar `t+1`. This is documented in the development plan ("Risk checks are evaluated at signal time") and is consistent with the batch simulator. It is a design choice, not a bug.

---

## 3. Simulation Integrity

### 3.1 PnL Calculation — PASS

In `evaluate_node()`:

```python
market_return = (close_next - close_now) / close_now
strategy_return = (target_position * market_return) - transaction_cost
```

This is correct for a position-based return model. Transaction costs are proportional to the absolute position change, which is standard.

In `simulate_policy()`, the same formula is used. The funding rate is applied every `funding_interval_bars` steps and is deducted from the strategy return. `test_simulate_policy_applies_funding_cost` verifies this with a flat-price series where the only returns come from funding deductions.

### 3.2 Execution Timestamp Semantics — PASS

The execution timestamp convention is: `bar_timestamp == execution_timestamp`. This means the trade is recorded at the bar where the new position becomes active. This is explicitly documented, consistently implemented, and verified by:

- The readiness check `execution_semantics_consistent`, which scans all trade events for `bar_timestamp != execution_timestamp` misalignment.
- The assertion in `test_signal_delay_roundtrip` that checks `bar_timestamp == execution_timestamp` for both simulation and runtime trades.

### 3.3 Drawdown Pause Parity — PASS

`test_simulate_policy_drawdown_pause_matches_runtime` runs the batch simulator and the runtime node loop on the same declining price series with `max_drawdown_pause=0.05`. It verifies that both produce identical returns, equity curves, and trade lists, and that both report `paused=True`. This is a critical parity test.

### 3.4 Position Clamping and Turnover Limits — PASS

`clamp_position()` and `apply_turnover_limit()` in `risk.py` are simple and correct. The turnover limit caps the absolute position delta per bar.

### 3.5 Completed Trade Tracking — PASS

Both `simulate_policy()` and `evaluate_node()` maintain a `completed_trades` list. When a position changes from non-zero to something else (including reversal), a completed trade record is emitted with `pnl_pct`, `hold_bars`, `entry_price`, `exit_price`, and `close_reason`. The `entry_price` bookkeeping (set on entry, cleared on flat) is consistent between both codepaths.

---

## 4. Feature Engineering Quality

### 4.1 Indicator Coverage — Adequate for Research

The feature bank is intentionally broad: ~9 SMA/EMA/WMA periods, RSI, ROC, CCI, MFI, ADX, MACD (3 parameter sets), Bollinger Bands (3 widths), Stochastic variants, Aroon, OBV, ADOSC, Hilbert transforms, and 6 candlestick patterns. Plus lagged versions, rolling statistics, and multi-timeframe joins.

This is appropriate for a "does signal exist?" research question. The README correctly warns about overfitting risk with a wide feature bank on short data.

### 4.2 NaN/Null Handling — ACCEPTABLE

At the end of `build_features()`, all float columns are filled with `fill_nan(0.0).fill_null(0.0)`, and integer columns with `fill_null(0)`. This is a blunt instrument — it replaces genuinely missing indicator values (e.g., the first 200 bars for `sma_200`) with zero, which is semantically incorrect.

For the current research milestone, this is acceptable because:
1. The warmup period (`warmup_bars=240`) ensures the training split starts after most indicators have stabilized.
2. The model is tree-based (LightGBM), which handles arbitrary feature values reasonably well.

**Recommendation for future work:** Replace zero-fill with NaN-aware handling. LightGBM natively supports NaN features and will learn appropriate split points. This avoids injecting spurious zero-valued features into the early rows.

### 4.3 Volatility Regime — PASS

The volatility regime is computed by comparing short-horizon realized vol to its longer-term rolling mean, with window sizes adapted to the base timeframe. The implementation correctly selects the nearest available rolling window size for each timeframe.

### 4.4 ETH Proxy Features — ACCEPTABLE

When ETH data is available, the system computes BTC/ETH spread, return spread, and rolling correlations. The `rolling_corr` function is a naive O(n²) loop, which is fine for research but would need optimization for larger datasets.

---

## 5. Artifact Persistence & Provenance

### 5.1 Artifact Bundle Completeness — EXCELLENT

Each run produces a comprehensive artifact bundle:

- `report.md` and `report.json`: Human-readable and machine-readable experiment summaries
- `config.yaml`: Frozen run configuration
- `dataset_metadata.json`: Row counts, timestamp bounds, source mode, dataset hash, snapshot hash
- `split_metadata.json`: Train/validation/OOS boundaries
- `strategy_params.json`: Calibrated thresholds
- `optimization_events.json`: Full Optuna tuning provenance
- `benchmark_metrics.json` and `benchmark_sufficiency.json`: Walk-forward results and sufficiency flags
- `run_metrics.json`: Held-out simulation metrics
- `trades.json`, `completed_trades.json`, `equity_curve.json`, `returns.json`: Full trade and performance history
- `decision_log.jsonl` and `decision_summary.csv`: Per-bar decision audit trail
- `regression_ledger_entry.json`: Entry for the cross-run regression ledger
- `readiness.json`: Phase-gate evaluation results
- `shap_rule.txt`: SHAP interpretation

This is an unusually thorough artifact set for a Week 1 prototype. The cross-run regression ledger (`regression_ledger.json` at the artifact root) is a particularly good design choice — it enables tracking metric stability across runs without requiring external tooling.

### 5.2 Artifact Integrity Assertions — EXCELLENT

`persist_run_artifacts()` contains a battery of assertions (for non-error runs) that verify:

- No legacy KPI keys (`win_rate`, `trade_count`) are present in `run_metrics`
- `report.transition_count == run_metrics.transition_count == len(trades)`
- `report.completed_trade_count == run_metrics.completed_trade_count == len(completed_trades)`
- `report.bar_win_rate == run_metrics.bar_win_rate`
- `report.completed_trade_win_rate == run_metrics.completed_trade_win_rate`

These assertions catch metric pipeline inconsistencies at write time. The tests `test_artifact_persistence_asserts_trade_count_mismatch` and `test_artifact_persistence_rejects_legacy_kpi_schema` verify that these assertions fire correctly.

### 5.3 Error-Path Artifact Persistence — GOOD

On failure, the system still writes partial artifacts including `error_summary.json`, `traceback.txt`, and `last_state_snapshot.json`. The integrity assertions are skipped for error runs. This is correct — you always want to preserve diagnostic information from failed runs.

### 5.4 Data Hashing — GOOD

Three separate hashes are computed:
- `snapshot_hash`: SHA-256 of the snapshot file on disk (for reproducibility)
- `raw_data_hash`: SHA-256 of the canonical OHLCV+ETH DataFrame (for detecting data changes regardless of file format)
- `dataset_hash`: SHA-256 of the full metadata payload (including row counts, timestamps, etc.)

This layered approach enables precise provenance tracking.

---

## 6. Phase-Gate and Readiness System

### 6.1 Readiness Evaluation — WELL DESIGNED

`evaluate_readiness()` in `run_mvp.py` checks three categories:

**Engineering Validity:**
- Execution semantics consistency (bar_timestamp == execution_timestamp for all trades)
- KPI schema consistency (no legacy keys, counts match)

**Research Validity:**
- Triple-negative-Sharpe detection (validation, walk-forward, and held-out all negative)

**Evidence Sufficiency:**
- Snapshot-based input (exchange mode is not regression-eligible)
- Walk-forward sufficiency (minimum folds and test bars per mode)
- Benchmark sufficiency artifact availability
- Regression history availability
- Multi-window evidence sufficiency (minimum distinct historical windows)

All three categories must be green for the phase gate to open. This is conservative and appropriate.

### 6.2 Fine-Tuning Gate — CORRECTLY BLOCKED

The fine-tuning gate requires:
- All three readiness categories green
- Multi-window evidence sufficient (`min_historical_windows` distinct dataset windows in the regression ledger)

With `min_historical_windows=5`, this means at least 5 runs on distinct dataset windows must show acceptable results before any fine-tuning or advanced learning work begins. This is a good safeguard against premature optimization.

### 6.3 Test Coverage of Readiness — GOOD

- `test_evaluate_readiness_rejects_triple_negative_sharpe`
- `test_evaluate_readiness_blocks_exchange_mode`
- `test_evaluate_readiness_rejects_missing_regression_history`
- `test_evaluate_readiness_rejects_single_window_evidence`

These cover the most important failure modes.

---

## 7. Test Suite Assessment

### 7.1 Coverage Quality — GOOD

The test suite focuses on the highest-risk areas: leakage prevention, simulation parity, signal delay correctness, artifact integrity, and readiness logic. This is the right prioritization.

**Strengths:**
- The simulation parity tests (`test_simulate_policy_drawdown_pause_matches_runtime`, `test_signal_delay_roundtrip`) are excellent. They run both the batch and runtime codepaths on identical inputs and assert numerical equality.
- The leakage tests cover label boundaries, walk-forward folds, and multi-timeframe features.
- The artifact tests verify file existence, content correctness, and cross-run ledger behavior.

**Gaps:**
- No test covers the full `prepare_experiment()` → `graph.invoke()` → `persist_run_artifacts()` pipeline end-to-end with synthetic data. This would catch integration issues between modules.
- No test covers `load_historical_data()` with a real parquet/CSV fixture. The data loading path is only tested indirectly.
- No test covers edge cases in `compute_split_metadata()` when `total_rows` is very small (e.g., barely above `minimum_oos_bars + warmup_bars`).
- The `predict_node` test does not exercise the LightGBM model path (only the fallback).

### 7.2 Test Isolation — GOOD

Tests use `unittest.mock.patch` for external dependencies (LightGBM, Optuna). Artifact tests use `tempfile.TemporaryDirectory`. No tests require network access or external services.

---

## 8. Issues and Recommendations

### 8.1 HIGH Priority

**H1: `optimize_node` is a no-op.** The node emits a log event and returns an empty dict. This is documented ("currently a placeholder"), but it means the LangGraph loop has a dead node that adds overhead on every cycle. Consider removing it from the graph until it has functionality, or at minimum add a fast-path that skips event emission.

**H2: `decision_node` lacks config/split_metadata defaults.** The function accesses `state.get("config", {})` and `state.get("split_metadata", {})`, but the test `test_decision_node_generates_expected_action` passes a minimal state without these keys. This works because of Python's dict `.get()` defaults, but it means the test doesn't exercise the signal-delay logic at all. The test should be expanded to cover the delay path.

**H3: Walk-forward benchmark inside `prepare_experiment` runs before OOS simulation.** This is intentional (the benchmark provides context), but it means `prepare_experiment()` is expensive — it trains multiple LightGBM models across folds. For interactive development, consider adding a config flag to skip the benchmark.

### 8.2 MEDIUM Priority

**M1: Zero-fill for NaN features.** As noted in §4.2, `fill_nan(0.0)` replaces genuinely missing indicator values with zero. LightGBM handles NaN natively. Switching to NaN-aware features would improve model quality, especially for the early rows of each fold in walk-forward.

**M2: `simulate_policy` and runtime nodes share logic but not code.** The PnL calculation, trade recording, and completed-trade bookkeeping are implemented twice — once in `simulate_policy()` (batch) and once in `evaluate_node()` (runtime). The parity test catches divergence, but shared logic would be easier to maintain. Consider extracting a `step_simulation()` function used by both.

**M3: `rolling_corr` is O(n·w) per call.** The ETH correlation features use a naive sliding-window correlation with Python loops. For 120-bar windows on large datasets, this is slow. Consider `np.lib.stride_tricks` or pandas rolling correlation.

**M4: Feature column count is not fixed.** The model feature columns depend on which indicators produce non-null output, which varies with data length. This means the feature schema can differ between runs on different-length snapshots. Consider defining a fixed feature manifest.

**M5: `event_log` is accumulated in memory.** The `RunEventLogger` stores all events in `self.events` (a list). For long runs with verbose logging, this could consume significant memory. The JSONL file handler already persists events incrementally — consider making the in-memory list optional.

### 8.3 LOW Priority

**L1: Hardcoded fallback probability coefficients.** `fallback_probability()` in `modeling.py` uses hardcoded weights (1.35, 0.90, 0.65, 0.25) that are not documented or justified. This is only used when no LightGBM model is available, so it's low-risk, but the magic numbers should be commented.

**L2: `_get_git_commit_hash()` swallows all exceptions.** If git is not installed or the directory is not a git repo, this silently returns `"unknown"`. Consider logging a warning.

**L3: `test_development_plan_matches_default_timeframe` is a documentation test.** It checks that `docs/development-plan.md` contains "BTC 15m data". This is fragile and low-value. Consider removing or making it a linting check.

**L4: SHAP rule extraction lookback is capped at 300.** `derive_shap_rule()` uses `min(300, cursor + 1)` rows for the background segment. This is hardcoded and not configurable. For very long OOS runs, this may miss regime changes in SHAP attributions.

---

## 9. Configuration Review

The `config.yaml` is well-structured and covers all relevant parameters. A few observations:

- `max_lag_bars: 16` produces 16 lags × 5 base columns = 80 lag features. Combined with the full indicator bundle and MTF joins, the total feature count is likely 800–1200+. This is large relative to `train_ratio * 2000 ≈ 1200` training rows. The risk of overfitting is real, but this is acknowledged in the README.

- `optuna_trials: 20` is low for a 2D search space (long_threshold, short_threshold). Optuna's TPE sampler handles this reasonably well for such a simple space, but 50–100 trials would give more reliable calibration.

- `min_historical_windows: 5` in `readiness` is appropriately conservative for the fine-tuning gate.

- `signal_delay_bars: 1` is realistic for a 15m timeframe (execution at next bar's close after signal generation).

---

## 10. Alignment with Development Plan

| Plan Item | Status | Notes |
|-----------|--------|-------|
| Feature engineering over BTC 15m data | ✅ Complete | Broad indicator set with MTF joins |
| LightGBM directional baseline | ✅ Complete | Trained with leakage-safe label construction |
| Validation-set threshold calibration | ✅ Complete | Optuna-based, validation-only |
| Held-out OOS LangGraph simulation | ✅ Complete | Full node loop with signal delay |
| Purged walk-forward benchmark | ✅ Complete | Expanding + rolling, with sufficiency checks |
| Artifact persistence | ✅ Complete | Comprehensive bundle with integrity assertions |
| Simple simulation risk controls | ✅ Complete | Drawdown pause, stop-loss, take-profit, vol block |
| Regression ledger tracking | ✅ Complete | Cross-run append with dataset window keying |
| Readiness/phase-gate system | ✅ Complete | Three-category evaluation with fine-tuning gate |
| Not claiming self-improving loops | ✅ Correct | `optimize_node` is explicitly a no-op |
| Not claiming live/paper trading | ✅ Correct | No broker integration |

**The codebase is fully aligned with its Week 1 milestone.**

---

## 11. Summary of Findings

| Category | Rating | Key Observation |
|----------|--------|----------------|
| Leakage Prevention | **A** | Multi-layered: label isolation, terminal row dropping, embargo gaps, MTF shift, parity tests |
| Simulation Integrity | **A** | Dual-path implementation with bitwise parity verification |
| Artifact Provenance | **A** | Comprehensive, hash-backed, with integrity assertions |
| Phase-Gate Design | **A** | Conservative, multi-dimensional, correctly blocks premature advancement |
| Test Suite | **B+** | Covers critical invariants; lacks full integration test |
| Feature Engineering | **B** | Broad and correct, but zero-fill and no fixed schema |
| Code Maintainability | **B** | Some logic duplication between batch and runtime paths |
| Documentation | **A** | README, development plan, and inline comments are thorough and honest |

**Bottom line:** This is a well-engineered research prototype that takes correctness seriously. The most important invariants (no leakage, simulation parity, honest artifacts) are both implemented correctly and tested. The codebase is ready to serve as a baseline for the research questions it poses.