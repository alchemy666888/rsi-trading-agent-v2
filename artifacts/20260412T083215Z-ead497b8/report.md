# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T083215Z-ead497b8
- Generated (UTC): 2026-04-12T08:35:36.728680+00:00
- Report Detail Level: full
- Git Commit: 448e5354ab88e80a23b271517f93299f32951961
- Config Hash: b617b4658e86c9a3d53b1807d5d2e0cea9e82075a1db917fc62fd4fb012413b5
- Asset: BTC/USDT (15m)
- Dataset Source: snapshots/btcusdt_15m.parquet (snapshot)
- Snapshot Path: snapshots/btcusdt_15m.parquet
- Snapshot Hash: 791d41cb3045f28d538510831f691367c23f7137106c86aba710e76d0e4cd32b
- Raw Data Hash: d65b2063ad0c12e2b27a99cb769a37f0810491228a2618434eda6eb65c01ace1
- Benchmark Eligible Input: True
- Dataset Span (UTC): 2026-04-06T00:41:00+00:00 -> 2026-04-09T12:00:00+00:00
- Bars: total=5000, train=3000, validation=1000, oos=200

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -9.3156 |
| Max Drawdown | 1.32% |
| Total Return | -0.77% |
| Bar Win Rate | 47.50% |
| Transition Count | 9 |
| Completed Trade Win Rate | 50.00% |
| Completed Trade Count | 4 |
| Avg Completed Trade Return | 0.1955% |
| Profit Factor | 24.359176825221258 |
| Turnover | 0.0450 |
| Long Exposure | 0.50% |
| Short Exposure | 96.00% |
| Flat Exposure | 3.50% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.7048160934524403, "short_threshold": 0.4722700402430162}
- Objective: validation_sharpe_activity_adjusted
- Best Validation Score: -16.3075
- Best Validation Score (Adjusted): -16.3075
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): 48.9343
- Delta Total Return (Held-Out - Walk-Forward Mean): 1.65%
- Overfit Warnings:
  - Held-out and walk-forward Sharpe are both negative; strategy is not yet investment-ready.

## Risk Diagnostics

- Blocked Trades: 0
- Stop Loss Trigger Count: 0
- Take Profit Trigger Count: 0
- Drawdown Pause Activated Count: 0
- Drawdown Pause Cleared Count: 0
- Pause Durations (seconds): []
- Max Consecutive Losses: 1

## Trade Diagnostics

- Best Trade: {'direction': -1, 'open_cycle': 2, 'close_cycle': 32, 'open_timestamp': 1775677140000, 'close_timestamp': 1775678940000, 'entry_price': 92466.44243044061, 'exit_price': 91977.03439667303, 'pnl_pct': 0.005320980796759578, 'hold_bars': 30, 'close_reason': []}
- Worst Trade: {'direction': -1, 'open_cycle': 63, 'close_cycle': 71, 'open_timestamp': 1775680800000, 'close_timestamp': 1775681280000, 'entry_price': 91742.42094001302, 'exit_price': 91767.79001381506, 'pnl_pct': -0.0002764485643407255, 'hold_bars': 8, 'close_reason': []}
- Longest Hold (bars): 30
- Average Hold (bars): 16.00
- Long/Short Breakdown: {'long_trades': 1, 'short_trades': 3}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1014.
- Execution convention: signal observed at close(t), executed at close(t + delay), position earns returns from close(t + delay) to close(t + delay + 1).
- Risk checks (drawdown pause, volatility block, stop-loss, take-profit) are evaluated at signal time.
- Slippage assumption (bps): 5.00.
- Benchmark eligible input: True.
- Calibration diagnostics: degenerate_regime=False, transition_count=99, min_transition_count=12, one_side_exposure_max=0.9188.
- Readiness warning: Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.
- Readiness warning: Readiness hard blockers are present; do not advance to fine-tuning or next phase.
- Readiness warning: Fine-tuning gate remains closed until multi-window regression evidence meets minimum history requirements.
- Benchmark sufficiency artifact available: True.
- Regression history windows: 1/5.

## Phase Gate

- Phase Gate Decision: do_not_advance
- Fine-Tuning Gate Open: False
- Engineering Valid: True
- Research Valid: False
- Evidence Sufficient: False
- Execution Semantics Consistent: True
- KPI Schema Consistent: True
- Snapshot-Based Input: True
- Walk-Forward Sufficient: True
- Benchmark Sufficiency Artifact Available: True
- Historical Window Count: 1 (minimum 5)
- Hard Blocker [triple_negative_sharpe]: Validation, walk-forward, and held-out Sharpe are all negative.
- Hard Blocker [multi_window_history_insufficient]: Insufficient historical windows in persisted regression history.

## Failure / Fallbacks

- No failures captured.
