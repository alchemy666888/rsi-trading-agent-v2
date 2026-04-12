# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260412T130328Z-f9809f9c
- Generated (UTC): 2026-04-12T13:04:54.508142+00:00
- Report Detail Level: full
- Git Commit: cdfb31a732cc1942fd569f6301d872279eed604d
- Config Hash: 2572e05a444cd9252d9bbdad5b1349ff7e98b836246deb10a89baee97408692d
- Asset: BTC/USDT (15m)
- Dataset Source: snapshots/btcusdt_15m.parquet (snapshot)
- Snapshot Path: snapshots/btcusdt_15m.parquet
- Snapshot Hash: 791d41cb3045f28d538510831f691367c23f7137106c86aba710e76d0e4cd32b
- Raw Data Hash: 615fe6bb97e380d701b1cbb8150b092ac26930e9aad7f073c71988f94cff09bd
- Benchmark Eligible Input: True
- Dataset Span (UTC): 2026-04-07T01:41:00+00:00 -> 2026-04-09T03:40:00+00:00
- Bars: total=3000, train=1800, validation=600, oos=200

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -106.8819 |
| Max Drawdown | 6.06% |
| Total Return | -6.01% |
| Bar Win Rate | 16.00% |
| Transition Count | 98 |
| Completed Trade Win Rate | 40.82% |
| Completed Trade Count | 49 |
| Avg Completed Trade Return | -0.0263% |
| Profit Factor | 0.4963209307342861 |
| Turnover | 0.4900 |
| Long Exposure | 7.50% |
| Short Exposure | 47.50% |
| Flat Exposure | 45.00% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.7468698680429599, "short_threshold": 0.47900372015575876}
- Objective: validation_sharpe_activity_adjusted
- Best Validation Score: -48.7608
- Best Validation Score (Adjusted): -48.7608
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): -47.2652
- Delta Total Return (Held-Out - Walk-Forward Mean): -3.45%
- Overfit Warnings:
  - Held-out and walk-forward Sharpe are both negative; strategy is not yet investment-ready.

## Risk Diagnostics

- Blocked Trades: 0
- Stop Loss Trigger Count: 0
- Take Profit Trigger Count: 0
- Drawdown Pause Activated Count: 0
- Drawdown Pause Cleared Count: 0
- Pause Durations (seconds): []
- Max Consecutive Losses: 4

## Trade Diagnostics

- Best Trade: {'direction': -1, 'open_cycle': 102, 'close_cycle': 108, 'open_timestamp': 1775677140000, 'close_timestamp': 1775677500000, 'entry_price': 92466.44243044061, 'exit_price': 92242.29531064368, 'pnl_pct': 0.00242998202768141, 'hold_bars': 6, 'close_reason': []}
- Worst Trade: {'direction': -1, 'open_cycle': 15, 'close_cycle': 28, 'open_timestamp': 1775671920000, 'close_timestamp': 1775672700000, 'entry_price': 91782.82689200154, 'exit_price': 92256.49890340975, 'pnl_pct': -0.0051342942452664575, 'hold_bars': 13, 'close_reason': []}
- Longest Hold (bars): 13
- Average Hold (bars): 2.24
- Long/Short Breakdown: {'long_trades': 11, 'short_trades': 38}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1014.
- Execution convention: signal observed at close(t), executed at close(t + delay), position earns returns from close(t + delay) to close(t + delay + 1).
- Risk checks (drawdown pause, volatility block, stop-loss, take-profit) are evaluated at signal time.
- Slippage assumption (bps): 5.00.
- Benchmark eligible input: True.
- Calibration diagnostics: degenerate_regime=False, transition_count=209, min_transition_count=12, one_side_exposure_max=0.5886.
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
