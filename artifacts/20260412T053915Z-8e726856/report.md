# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T053915Z-8e726856
- Generated (UTC): 2026-04-12T05:42:39.912393+00:00
- Report Detail Level: full
- Git Commit: e6cf6a85a3271467c5b17a3f9fc3b6c9d9aae11d
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

- Tuned Thresholds: {"long_threshold": 0.5672417643420733, "short_threshold": 0.47506769344909433}
- Objective: validation_sharpe
- Best Validation Score: -18.3448
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): 48.0216
- Delta Total Return (Held-Out - Walk-Forward Mean): 1.62%
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
