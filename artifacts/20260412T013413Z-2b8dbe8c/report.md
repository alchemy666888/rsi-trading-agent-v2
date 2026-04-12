# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T013413Z-2b8dbe8c
- Generated (UTC): 2026-04-12T01:34:21.055690+00:00
- Report Detail Level: full
- Git Commit: a1809699ec98ef749a6db87a7d15b19d5410d2fe
- Config Hash: e5400dd2aec9a3b182aef6a016e8379a55b6fbe8158af5ef499591fa551f23cc
- Asset: BTC/USDT (15m)
- Dataset Source: exchange:binance (exchange)
- Dataset Span (UTC): 2026-04-01T15:45:00+00:00 -> 2026-04-12T01:30:00+00:00
- Bars: total=1000, train=600, validation=200, oos=182

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -46.2075 |
| Max Drawdown | 5.42% |
| Total Return | -5.39% |
| Win Rate | 34.15% |
| Trade Count | 82 |
| Avg Trade Return | -0.0341% |
| Profit Factor | 0.6562432345128602 |
| Turnover | 0.4505 |
| Long Exposure | 6.59% |
| Short Exposure | 54.95% |
| Flat Exposure | 38.46% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.6379677424406822, "short_threshold": 0.38325588685959305}
- Objective: validation_sharpe
- Best Validation Score: -9.1686
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): -21.7454
- Delta Total Return (Held-Out - Walk-Forward Mean): -2.01%
- Overfit Warnings:
  - Held-out and walk-forward Sharpe are both negative; strategy is not yet investment-ready.

## Risk Diagnostics

- Blocked Trades: 0
- Stop Loss Trigger Count: 0
- Take Profit Trigger Count: 0
- Drawdown Pause Activated Count: 0
- Drawdown Pause Cleared Count: 0
- Pause Durations (seconds): []
- Max Consecutive Losses: 8

## Trade Diagnostics

- Best Trade: {'direction': -1, 'open_cycle': 45, 'close_cycle': 47, 'open_timestamp': 1775834100000, 'close_timestamp': 1775836800000, 'entry_price': 73026.4, 'exit_price': 72745.81, 'pnl_pct': 0.0038571293659386274, 'hold_bars': 2, 'close_reason': []}
- Worst Trade: {'direction': -1, 'open_cycle': 23, 'close_cycle': 44, 'open_timestamp': 1775814300000, 'close_timestamp': 1775834100000, 'entry_price': 71811.18, 'exit_price': 72909.0, 'pnl_pct': -0.015057400320948156, 'hold_bars': 21, 'close_reason': []}
- Longest Hold (bars): 21
- Average Hold (bars): 2.73
- Long/Short Breakdown: {'long_trades': 9, 'short_trades': 32}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1024.
- Assumed next-bar execution with slippage_bps=5.00.
- Readiness warning: Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.

## Failure / Fallbacks

- No failures captured.
