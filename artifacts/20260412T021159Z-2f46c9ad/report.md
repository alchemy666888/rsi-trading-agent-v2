# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T021159Z-2f46c9ad
- Generated (UTC): 2026-04-12T02:12:07.040824+00:00
- Report Detail Level: full
- Git Commit: 3634754e48891968590a6c869fa3fab301da2433
- Config Hash: 11da6aa8a7f7a02adb79ac62d7affdae985600885b394dc7377f201fdbb8decd
- Asset: BTC/USDT (15m)
- Dataset Source: exchange:binance (exchange)
- Dataset Span (UTC): 2026-04-01T16:15:00+00:00 -> 2026-04-12T02:00:00+00:00
- Bars: total=1000, train=600, validation=200, oos=182

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -45.7957 |
| Max Drawdown | 5.45% |
| Total Return | -5.45% |
| Win Rate | 46.15% |
| Trade Count | 79 |
| Avg Trade Return | -0.0363% |
| Profit Factor | 0.6779100271434596 |
| Turnover | 0.4341 |
| Long Exposure | 7.14% |
| Short Exposure | 56.04% |
| Flat Exposure | 36.81% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.6717344428216053, "short_threshold": 0.441164338964503}
- Objective: validation_sharpe
- Best Validation Score: -10.3155
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): -29.7021
- Delta Total Return (Held-Out - Walk-Forward Mean): -3.24%
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

- Best Trade: {'direction': -1, 'open_cycle': 43, 'close_cycle': 45, 'open_timestamp': 1775834100000, 'close_timestamp': 1775836800000, 'entry_price': 73026.4, 'exit_price': 72745.81, 'pnl_pct': 0.0038571293659386274, 'hold_bars': 2, 'close_reason': []}
- Worst Trade: {'direction': -1, 'open_cycle': 21, 'close_cycle': 42, 'open_timestamp': 1775814300000, 'close_timestamp': 1775834100000, 'entry_price': 71811.18, 'exit_price': 72909.0, 'pnl_pct': -0.015057400320948156, 'hold_bars': 21, 'close_reason': []}
- Longest Hold (bars): 21
- Average Hold (bars): 2.92
- Long/Short Breakdown: {'long_trades': 9, 'short_trades': 30}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1024.
- Assumed next-bar execution with slippage_bps=5.00.
- Readiness warning: Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.

## Failure / Fallbacks

- No failures captured.
