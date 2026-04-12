# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T130458Z-318dc238
- Generated (UTC): 2026-04-12T13:06:23.960072+00:00
- Report Detail Level: full
- Git Commit: cdfb31a732cc1942fd569f6301d872279eed604d
- Config Hash: 0e3ad07c8e5e0a74e0be831779aabece9b1ebb473591b9551ccd34cdc94967c0
- Asset: BTC/USDT (15m)
- Dataset Source: snapshots/btcusdt_15m.parquet (snapshot)
- Snapshot Path: snapshots/btcusdt_15m.parquet
- Snapshot Hash: 791d41cb3045f28d538510831f691367c23f7137106c86aba710e76d0e4cd32b
- Raw Data Hash: 44adbd8168dd99ba7e2a8fa8bd4047f2f45209cc1497b7f7082719b5a47c874a
- Benchmark Eligible Input: True
- Dataset Span (UTC): 2026-04-07T10:01:00+00:00 -> 2026-04-09T12:00:00+00:00
- Bars: total=3000, train=1800, validation=600, oos=200

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -18.7850 |
| Max Drawdown | 1.92% |
| Total Return | -1.57% |
| Bar Win Rate | 45.00% |
| Transition Count | 41 |
| Completed Trade Win Rate | 70.00% |
| Completed Trade Count | 20 |
| Avg Completed Trade Return | 0.0310% |
| Profit Factor | 1.5560257654051501 |
| Turnover | 0.2050 |
| Long Exposure | 76.00% |
| Short Exposure | 10.00% |
| Flat Exposure | 14.00% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.5209073959509595, "short_threshold": 0.4768958000257443}
- Objective: validation_sharpe_activity_adjusted
- Best Validation Score: -27.8696
- Best Validation Score (Adjusted): -27.8696
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): 37.8113
- Delta Total Return (Held-Out - Walk-Forward Mean): 0.88%
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

- Best Trade: {'direction': 1, 'open_cycle': 41, 'close_cycle': 58, 'open_timestamp': 1775703480000, 'close_timestamp': 1775704500000, 'entry_price': 92804.59922769862, 'exit_price': 93187.62386297833, 'pnl_pct': 0.004127216091305508, 'hold_bars': 17, 'close_reason': []}
- Worst Trade: {'direction': 1, 'open_cycle': 7, 'close_cycle': 28, 'open_timestamp': 1775701440000, 'close_timestamp': 1775702700000, 'entry_price': 93107.27725119257, 'exit_price': 92708.03112087611, 'pnl_pct': -0.004288022828111915, 'hold_bars': 21, 'close_reason': []}
- Longest Hold (bars): 27
- Average Hold (bars): 7.60
- Long/Short Breakdown: {'long_trades': 20, 'short_trades': 0}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1014.
- Execution convention: signal observed at close(t), executed at close(t + delay), position earns returns from close(t + delay) to close(t + delay + 1).
- Risk checks (drawdown pause, volatility block, stop-loss, take-profit) are evaluated at signal time.
- Slippage assumption (bps): 5.00.
- Benchmark eligible input: True.
- Calibration diagnostics: degenerate_regime=False, transition_count=159, min_transition_count=12, one_side_exposure_max=0.5870.
- Readiness warning: Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.
- Readiness warning: Readiness hard blockers are present; do not advance to fine-tuning or next phase.
- Benchmark sufficiency artifact available: True.
- Regression history windows: 6/5.

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
- Historical Window Count: 6 (minimum 5)
- Hard Blocker [triple_negative_sharpe]: Validation, walk-forward, and held-out Sharpe are all negative.

## Failure / Fallbacks

- No failures captured.
