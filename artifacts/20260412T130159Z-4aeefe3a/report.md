# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T130159Z-4aeefe3a
- Generated (UTC): 2026-04-12T13:03:24.311250+00:00
- Report Detail Level: full
- Git Commit: cdfb31a732cc1942fd569f6301d872279eed604d
- Config Hash: 2f11a3cde0b88c9fd557e4fe834636ba1d80b768da386d980401243093083c7f
- Asset: BTC/USDT (15m)
- Dataset Source: snapshots/btcusdt_15m.parquet (snapshot)
- Snapshot Path: snapshots/btcusdt_15m.parquet
- Snapshot Hash: 791d41cb3045f28d538510831f691367c23f7137106c86aba710e76d0e4cd32b
- Raw Data Hash: 885759ae64aa49d634c19a35db1f25098d85af79ff12394a1309fa26b0c55ae2
- Benchmark Eligible Input: True
- Dataset Span (UTC): 2026-04-06T17:21:00+00:00 -> 2026-04-08T19:20:00+00:00
- Bars: total=3000, train=1800, validation=600, oos=200

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -55.6813 |
| Max Drawdown | 4.38% |
| Total Return | -4.22% |
| Bar Win Rate | 28.00% |
| Transition Count | 69 |
| Completed Trade Win Rate | 50.00% |
| Completed Trade Count | 34 |
| Avg Completed Trade Return | -0.0019% |
| Profit Factor | 0.9573820692217206 |
| Turnover | 0.3450 |
| Long Exposure | 5.50% |
| Short Exposure | 62.50% |
| Flat Exposure | 32.00% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.6582781044224901, "short_threshold": 0.4790120608837367}
- Objective: validation_sharpe_activity_adjusted
- Best Validation Score: -68.8530
- Best Validation Score (Adjusted): -68.8530
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): 1.4501
- Delta Total Return (Held-Out - Walk-Forward Mean): -1.86%
- Overfit Warnings:
  - Held-out and walk-forward Sharpe are both negative; strategy is not yet investment-ready.

## Risk Diagnostics

- Blocked Trades: 0
- Stop Loss Trigger Count: 0
- Take Profit Trigger Count: 0
- Drawdown Pause Activated Count: 0
- Drawdown Pause Cleared Count: 0
- Pause Durations (seconds): []
- Max Consecutive Losses: 4

## Trade Diagnostics

- Best Trade: {'direction': 1, 'open_cycle': 21, 'close_cycle': 24, 'open_timestamp': 1775642280000, 'close_timestamp': 1775642460000, 'entry_price': 90589.19771192869, 'exit_price': 90854.31439467313, 'pnl_pct': 0.002926581639319803, 'hold_bars': 3, 'close_reason': []}
- Worst Trade: {'direction': -1, 'open_cycle': 81, 'close_cycle': 90, 'open_timestamp': 1775645880000, 'close_timestamp': 1775646420000, 'entry_price': 91516.9936918559, 'exit_price': 91715.55697000238, 'pnl_pct': -0.002164990157683122, 'hold_bars': 9, 'close_reason': []}
- Longest Hold (bars): 12
- Average Hold (bars): 2.62
- Long/Short Breakdown: {'long_trades': 8, 'short_trades': 26}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1014.
- Execution convention: signal observed at close(t), executed at close(t + delay), position earns returns from close(t + delay) to close(t + delay + 1).
- Risk checks (drawdown pause, volatility block, stop-loss, take-profit) are evaluated at signal time.
- Slippage assumption (bps): 5.00.
- Benchmark eligible input: True.
- Calibration diagnostics: degenerate_regime=False, transition_count=278, min_transition_count=12, one_side_exposure_max=0.6020.
- Readiness warning: Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.
- Readiness warning: Readiness hard blockers are present; do not advance to fine-tuning or next phase.
- Benchmark sufficiency artifact available: True.
- Regression history windows: 5/5.

## Phase Gate

- Phase Gate Decision: do_not_advance
- Fine-Tuning Gate Open: False
- Engineering Valid: True
- Research Valid: False
- Evidence Sufficient: True
- Execution Semantics Consistent: True
- KPI Schema Consistent: True
- Snapshot-Based Input: True
- Walk-Forward Sufficient: True
- Benchmark Sufficiency Artifact Available: True
- Historical Window Count: 5 (minimum 5)
- Hard Blocker [triple_negative_sharpe]: Validation, walk-forward, and held-out Sharpe are all negative.

## Failure / Fallbacks

- No failures captured.
