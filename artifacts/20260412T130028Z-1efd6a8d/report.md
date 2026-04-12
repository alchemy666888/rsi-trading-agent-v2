# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T130028Z-1efd6a8d
- Generated (UTC): 2026-04-12T13:01:55.748778+00:00
- Report Detail Level: full
- Git Commit: cdfb31a732cc1942fd569f6301d872279eed604d
- Config Hash: 4051ea7c6c31bfb9e1dfedb5ca2afb16a2689582cc91542ac0159f2d09c8b4b2
- Asset: BTC/USDT (15m)
- Dataset Source: snapshots/btcusdt_15m.parquet (snapshot)
- Snapshot Path: snapshots/btcusdt_15m.parquet
- Snapshot Hash: 791d41cb3045f28d538510831f691367c23f7137106c86aba710e76d0e4cd32b
- Raw Data Hash: cbf1a53cbd8f8512c3b22c7a34ea56b82861e348030f6b6d4a60257d7f05adb6
- Benchmark Eligible Input: True
- Dataset Span (UTC): 2026-04-06T09:01:00+00:00 -> 2026-04-08T11:00:00+00:00
- Bars: total=3000, train=1800, validation=600, oos=200

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -73.3411 |
| Max Drawdown | 4.58% |
| Total Return | -4.56% |
| Bar Win Rate | 19.50% |
| Transition Count | 104 |
| Completed Trade Win Rate | 51.92% |
| Completed Trade Count | 52 |
| Avg Completed Trade Return | 0.0104% |
| Profit Factor | 1.2614254742527526 |
| Turnover | 0.5200 |
| Long Exposure | 25.50% |
| Short Exposure | 21.00% |
| Flat Exposure | 53.50% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.5202987678332734, "short_threshold": 0.34381166832770316}
- Objective: validation_sharpe_activity_adjusted
- Best Validation Score: -61.9297
- Best Validation Score (Adjusted): -61.9297
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): -19.1494
- Delta Total Return (Held-Out - Walk-Forward Mean): -2.31%
- Overfit Warnings:
  - Held-out and walk-forward Sharpe are both negative; strategy is not yet investment-ready.

## Risk Diagnostics

- Blocked Trades: 0
- Stop Loss Trigger Count: 0
- Take Profit Trigger Count: 0
- Drawdown Pause Activated Count: 0
- Drawdown Pause Cleared Count: 0
- Pause Durations (seconds): []
- Max Consecutive Losses: 6

## Trade Diagnostics

- Best Trade: {'direction': -1, 'open_cycle': 86, 'close_cycle': 90, 'open_timestamp': 1775616180000, 'close_timestamp': 1775616420000, 'entry_price': 90239.17469532574, 'exit_price': 89943.9427925633, 'pnl_pct': 0.003282398943120768, 'hold_bars': 4, 'close_reason': []}
- Worst Trade: {'direction': 1, 'open_cycle': 17, 'close_cycle': 19, 'open_timestamp': 1775612040000, 'close_timestamp': 1775612160000, 'entry_price': 89329.21075767519, 'exit_price': 89086.48932648372, 'pnl_pct': -0.0027171563381424013, 'hold_bars': 2, 'close_reason': []}
- Longest Hold (bars): 5
- Average Hold (bars): 1.79
- Long/Short Breakdown: {'long_trades': 27, 'short_trades': 25}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1014.
- Execution convention: signal observed at close(t), executed at close(t + delay), position earns returns from close(t + delay) to close(t + delay + 1).
- Risk checks (drawdown pause, volatility block, stop-loss, take-profit) are evaluated at signal time.
- Slippage assumption (bps): 5.00.
- Benchmark eligible input: True.
- Calibration diagnostics: degenerate_regime=False, transition_count=303, min_transition_count=12, one_side_exposure_max=0.3278.
- Readiness warning: Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.
- Readiness warning: Readiness hard blockers are present; do not advance to fine-tuning or next phase.
- Readiness warning: Fine-tuning gate remains closed until multi-window regression evidence meets minimum history requirements.
- Benchmark sufficiency artifact available: True.
- Regression history windows: 4/5.

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
- Historical Window Count: 4 (minimum 5)
- Hard Blocker [triple_negative_sharpe]: Validation, walk-forward, and held-out Sharpe are all negative.
- Hard Blocker [multi_window_history_insufficient]: Insufficient historical windows in persisted regression history.

## Failure / Fallbacks

- No failures captured.
