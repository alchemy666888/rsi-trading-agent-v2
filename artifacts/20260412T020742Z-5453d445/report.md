# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T020742Z-5453d445
- Generated (UTC): 2026-04-12T02:07:51.345017+00:00
- Report Detail Level: full
- Git Commit: 3634754e48891968590a6c869fa3fab301da2433
- Config Hash: e5400dd2aec9a3b182aef6a016e8379a55b6fbe8158af5ef499591fa551f23cc
- Asset: BTC/USDT (15m)
- Dataset Source: exchange:binance (exchange)
- Dataset Span (UTC): 2026-04-01T16:15:00+00:00 -> 2026-04-12T02:00:00+00:00
- Bars: total=1000, train=600, validation=200, oos=182

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -42.8152 |
| Max Drawdown | 6.95% |
| Total Return | -6.95% |
| Win Rate | 44.74% |
| Trade Count | 76 |
| Avg Trade Return | -0.0882% |
| Profit Factor | 0.4289589769648699 |
| Turnover | 0.4176 |
| Long Exposure | 11.54% |
| Short Exposure | 35.71% |
| Flat Exposure | 52.75% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.670503734361216, "short_threshold": 0.2520085319161853}
- Objective: validation_sharpe
- Best Validation Score: -10.7221
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): -30.2364
- Delta Total Return (Held-Out - Walk-Forward Mean): -5.30%
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

- Best Trade: {'direction': -1, 'open_cycle': 43, 'close_cycle': 45, 'open_timestamp': 1775834100000, 'close_timestamp': 1775836800000, 'entry_price': 73026.4, 'exit_price': 72745.81, 'pnl_pct': 0.0038571293659386274, 'hold_bars': 2, 'close_reason': []}
- Worst Trade: {'direction': 1, 'open_cycle': 180, 'close_cycle': 182, 'open_timestamp': 1775957400000, 'close_timestamp': 1775959200000, 'entry_price': 73017.61, 'exit_price': 71643.69, 'pnl_pct': -0.018816282811776497, 'hold_bars': 2, 'close_reason': []}
- Longest Hold (bars): 15
- Average Hold (bars): 2.26
- Long/Short Breakdown: {'long_trades': 15, 'short_trades': 23}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1024.
- Assumed next-bar execution with slippage_bps=5.00.
- Readiness warning: Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.

## Failure / Fallbacks

- No failures captured.
