# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T033223Z-589b5591
- Generated (UTC): 2026-04-12T03:35:46.397137+00:00
- Report Detail Level: full
- Git Commit: a9d6baf554a5d66273bf5c34cca6f37889bbfba4
- Config Hash: 55a7a1ecea041cc720aaf6228f411f218f5932b8f99ea0830d4ce58c3a4e7657
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
| Sharpe | -8.8484 |
| Max Drawdown | 1.14% |
| Total Return | -0.72% |
| Bar Win Rate | 48.00% |
| Transition Count | 13 |
| Completed Trade Win Rate | 50.00% |
| Completed Trade Count | 6 |
| Avg Completed Trade Return | 0.0628% |
| Profit Factor | 1.7734482942614773 |
| Turnover | 0.0650 |
| Long Exposure | 0.00% |
| Short Exposure | 96.50% |
| Flat Exposure | 3.50% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.7492079232974844, "short_threshold": 0.47825003625662693}
- Objective: validation_sharpe
- Best Validation Score: -15.6493
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): 49.0299
- Delta Total Return (Held-Out - Walk-Forward Mean): 1.77%
- Overfit Warnings:
  - Held-out and walk-forward Sharpe are both negative; strategy is not yet investment-ready.

## Risk Diagnostics

- Blocked Trades: 0
- Stop Loss Trigger Count: 0
- Take Profit Trigger Count: 0
- Drawdown Pause Activated Count: 0
- Drawdown Pause Cleared Count: 0
- Pause Durations (seconds): []
- Max Consecutive Losses: 2

## Trade Diagnostics

- Best Trade: {'direction': -1, 'open_cycle': 2, 'close_cycle': 16, 'open_timestamp': 1775677140000, 'close_timestamp': 1775677980000, 'entry_price': 92466.44243044061, 'exit_price': 92108.12429602725, 'pnl_pct': 0.003890190329593146, 'hold_bars': 14, 'close_reason': []}
- Worst Trade: {'direction': -1, 'open_cycle': 63, 'close_cycle': 124, 'open_timestamp': 1775680800000, 'close_timestamp': 1775684460000, 'entry_price': 91742.42094001302, 'exit_price': 92093.75880082979, 'pnl_pct': -0.00381500185671213, 'hold_bars': 61, 'close_reason': []}
- Longest Hold (bars): 61
- Average Hold (bars): 19.50
- Long/Short Breakdown: {'long_trades': 0, 'short_trades': 6}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1014.
- Execution convention: signal observed at close(t), executed at close(t + delay), position earns returns from close(t + delay) to close(t + delay + 1).
- Slippage assumption (bps): 5.00.
- Benchmark eligible input: True.
- Readiness warning: Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.

## Phase Gate

- Phase Gate Decision: advance
- Fine-Tuning Gate Open: True
- Execution Semantics Consistent: True
- KPI Schema Consistent: True
- Snapshot-Based Input: True
- Walk-Forward Sufficient: True
- Hard Blocker: none

## Failure / Fallbacks

- No failures captured.
