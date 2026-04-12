# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T125857Z-4b321b07
- Generated (UTC): 2026-04-12T13:00:24.416183+00:00
- Report Detail Level: full
- Git Commit: cdfb31a732cc1942fd569f6301d872279eed604d
- Config Hash: 8a8f0661bf24dd5c45b3daeca260d4f962102377aff2212898d032fdac16d052
- Asset: BTC/USDT (15m)
- Dataset Source: snapshots/btcusdt_15m.parquet (snapshot)
- Snapshot Path: snapshots/btcusdt_15m.parquet
- Snapshot Hash: 791d41cb3045f28d538510831f691367c23f7137106c86aba710e76d0e4cd32b
- Raw Data Hash: 7fad5e24d1e6641f2076f86eff9a86b5f3bbe9ff5fc230f30086f4973b53cd92
- Benchmark Eligible Input: True
- Dataset Span (UTC): 2026-04-06T00:41:00+00:00 -> 2026-04-08T02:40:00+00:00
- Bars: total=3000, train=1800, validation=600, oos=200

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -56.6204 |
| Max Drawdown | 4.54% |
| Total Return | -4.33% |
| Bar Win Rate | 27.50% |
| Transition Count | 90 |
| Completed Trade Win Rate | 53.33% |
| Completed Trade Count | 45 |
| Avg Completed Trade Return | 0.0019% |
| Profit Factor | 1.0334693568689572 |
| Turnover | 0.4500 |
| Long Exposure | 28.00% |
| Short Exposure | 35.50% |
| Flat Exposure | 36.50% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.524422892613006, "short_threshold": 0.4404468671829895}
- Objective: validation_sharpe_activity_adjusted
- Best Validation Score: -65.2867
- Best Validation Score (Adjusted): -65.2867
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): -0.9210
- Delta Total Return (Held-Out - Walk-Forward Mean): -1.95%
- Overfit Warnings:
  - Held-out and walk-forward Sharpe are both negative; strategy is not yet investment-ready.

## Risk Diagnostics

- Blocked Trades: 0
- Stop Loss Trigger Count: 0
- Take Profit Trigger Count: 0
- Drawdown Pause Activated Count: 0
- Drawdown Pause Cleared Count: 0
- Pause Durations (seconds): []
- Max Consecutive Losses: 3

## Trade Diagnostics

- Best Trade: {'direction': 1, 'open_cycle': 27, 'close_cycle': 29, 'open_timestamp': 1775582640000, 'close_timestamp': 1775582760000, 'entry_price': 86705.37123722174, 'exit_price': 86943.6301786324, 'pnl_pct': 0.0027479144372590802, 'hold_bars': 2, 'close_reason': []}
- Worst Trade: {'direction': 1, 'open_cycle': 2, 'close_cycle': 7, 'open_timestamp': 1775581140000, 'close_timestamp': 1775581440000, 'entry_price': 86718.70111882349, 'exit_price': 86497.8203406528, 'pnl_pct': -0.002547095093917817, 'hold_bars': 5, 'close_reason': []}
- Longest Hold (bars): 13
- Average Hold (bars): 2.82
- Long/Short Breakdown: {'long_trades': 20, 'short_trades': 25}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1014.
- Execution convention: signal observed at close(t), executed at close(t + delay), position earns returns from close(t + delay) to close(t + delay + 1).
- Risk checks (drawdown pause, volatility block, stop-loss, take-profit) are evaluated at signal time.
- Slippage assumption (bps): 5.00.
- Benchmark eligible input: True.
- Calibration diagnostics: degenerate_regime=False, transition_count=300, min_transition_count=12, one_side_exposure_max=0.4799.
- Readiness warning: Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.
- Readiness warning: Readiness hard blockers are present; do not advance to fine-tuning or next phase.
- Readiness warning: Fine-tuning gate remains closed until multi-window regression evidence meets minimum history requirements.
- Benchmark sufficiency artifact available: True.
- Regression history windows: 3/5.

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
- Historical Window Count: 3 (minimum 5)
- Hard Blocker [triple_negative_sharpe]: Validation, walk-forward, and held-out Sharpe are all negative.
- Hard Blocker [multi_window_history_insufficient]: Insufficient historical windows in persisted regression history.

## Failure / Fallbacks

- No failures captured.
