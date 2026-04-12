# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T033112Z-606d272c
- Generated (UTC): 2026-04-12T03:31:12.506066+00:00
- Report Detail Level: full
- Git Commit: a9d6baf554a5d66273bf5c34cca6f37889bbfba4
- Config Hash: 55a7a1ecea041cc720aaf6228f411f218f5932b8f99ea0830d4ce58c3a4e7657
- Asset: BTC/USDT (15m)
- Dataset Source: None (None)
- Snapshot Path: None
- Snapshot Hash: None
- Raw Data Hash: None
- Benchmark Eligible Input: False
- Dataset Span (UTC): None -> None
- Bars: total=0, train=0, validation=0, oos=0

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | 0.0000 |
| Max Drawdown | 0.00% |
| Total Return | 0.00% |
| Bar Win Rate | 0.00% |
| Transition Count | 0 |
| Completed Trade Win Rate | 0.00% |
| Completed Trade Count | 0 |
| Avg Completed Trade Return | 0.0000% |
| Profit Factor | None |
| Turnover | 0.0000 |
| Long Exposure | 0.00% |
| Short Exposure | 0.00% |
| Flat Exposure | 0.00% |

## Calibration Summary

- Tuned Thresholds: {}
- Objective: n/a
- Best Validation Score: 0.0000
- Optimization Events Recorded: 0

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): 0.0000
- Delta Total Return (Held-Out - Walk-Forward Mean): 0.00%
- Overfit Warnings: none triggered.

## Risk Diagnostics

- Blocked Trades: 0
- Stop Loss Trigger Count: 0
- Take Profit Trigger Count: 0
- Drawdown Pause Activated Count: 0
- Drawdown Pause Cleared Count: 0
- Pause Durations (seconds): []
- Max Consecutive Losses: 0

## Trade Diagnostics

- Best Trade: None
- Worst Trade: None
- Longest Hold (bars): 0
- Average Hold (bars): 0.00
- Long/Short Breakdown: {'long_trades': 0, 'short_trades': 0}

## Data Quality / Readiness

- Missing value summary unavailable.
- Feature columns available for model: 0.
- Execution convention: signal observed at close(t), executed at close(t + delay), position earns returns from close(t + delay) to close(t + delay + 1).
- Slippage assumption (bps): 5.00.
- Benchmark eligible input: False.

## Phase Gate

- Phase Gate Decision: unknown
- Fine-Tuning Gate Open: False
- Execution Semantics Consistent: False
- KPI Schema Consistent: False
- Snapshot-Based Input: False
- Walk-Forward Sufficient: False
- Hard Blocker: none

## Failure / Fallbacks

- Error Type: FileNotFoundError
- Error Message: No such file or directory (os error 2): snapshots/btcusdt_15m.parquet
- Failed At (UTC): 2026-04-12T03:31:12.505488+00:00
