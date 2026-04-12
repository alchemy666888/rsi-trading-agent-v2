# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T013702Z-d73c98e2
- Generated (UTC): 2026-04-12T01:37:09.818624+00:00
- Report Detail Level: full
- Git Commit: 3634754e48891968590a6c869fa3fab301da2433
- Config Hash: e5400dd2aec9a3b182aef6a016e8379a55b6fbe8158af5ef499591fa551f23cc
- Asset: BTC/USDT (15m)
- Dataset Source: exchange:binance (exchange)
- Dataset Span (UTC): 2026-04-01T15:45:00+00:00 -> 2026-04-12T01:30:00+00:00
- Bars: total=1000, train=600, validation=200, oos=182

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -45.7752 |
| Max Drawdown | 5.21% |
| Total Return | -5.14% |
| Win Rate | 38.10% |
| Trade Count | 84 |
| Avg Trade Return | -0.0248% |
| Profit Factor | 0.7154593248446921 |
| Turnover | 0.4615 |
| Long Exposure | 10.99% |
| Short Exposure | 47.25% |
| Flat Exposure | 41.76% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.5656340171534544, "short_threshold": 0.3362535225465046}
- Objective: validation_sharpe
- Best Validation Score: -9.2479
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): -22.6706
- Delta Total Return (Held-Out - Walk-Forward Mean): -1.99%
- Overfit Warnings:
  - Held-out and walk-forward Sharpe are both negative; strategy is not yet investment-ready.

## Risk Diagnostics

- Blocked Trades: 0
- Stop Loss Trigger Count: 0
- Take Profit Trigger Count: 0
- Drawdown Pause Activated Count: 0
- Drawdown Pause Cleared Count: 0
- Pause Durations (seconds): []
- Max Consecutive Losses: 7

## Trade Diagnostics

- Best Trade: {'direction': -1, 'open_cycle': 45, 'close_cycle': 47, 'open_timestamp': 1775834100000, 'close_timestamp': 1775836800000, 'entry_price': 73026.4, 'exit_price': 72745.81, 'pnl_pct': 0.0038571293659386274, 'hold_bars': 2, 'close_reason': []}
- Worst Trade: {'direction': -1, 'open_cycle': 33, 'close_cycle': 44, 'open_timestamp': 1775823300000, 'close_timestamp': 1775834100000, 'entry_price': 72094.8, 'exit_price': 72909.0, 'pnl_pct': -0.011167345595193945, 'hold_bars': 11, 'close_reason': []}
- Longest Hold (bars): 11
- Average Hold (bars): 2.52
- Long/Short Breakdown: {'long_trades': 16, 'short_trades': 26}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1024.
- Assumed next-bar execution with slippage_bps=5.00.
- Readiness warning: Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.

## Failure / Fallbacks

- No failures captured.
