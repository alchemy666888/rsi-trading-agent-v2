# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260414T071934Z-986f1dd7
- Generated (UTC): 2026-04-14T07:21:28.984231+00:00
- Report Detail Level: full
- Git Commit: e396ae673694d0e763d82fd5738eaa0801870bdf
- Config Hash: e24cdd89ea26b752b3a4554e263d856115e3656bd41557121af8b39eb07abecc
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
| Sharpe | -7.9580 |
| Max Drawdown | 1.32% |
| Total Return | -0.66% |
| Bar Win Rate | 48.00% |
| Transition Count | 7 |
| Completed Trade Win Rate | 66.67% |
| Completed Trade Count | 3 |
| Avg Completed Trade Return | 0.2651% |
| Profit Factor | 29.766192972964678 |
| Turnover | 0.0350 |
| Long Exposure | 0.00% |
| Short Exposure | 97.00% |
| Flat Exposure | 3.00% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.7457948834570064, "short_threshold": 0.4762792780794978}
- Objective: validation_sharpe_activity_adjusted
- Best Validation Score: -16.1193
- Best Validation Score (Adjusted): -16.1193
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): 48.6780
- Delta Total Return (Held-Out - Walk-Forward Mean): 1.71%
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
- Average Hold (bars): 21.67
- Long/Short Breakdown: {'long_trades': 0, 'short_trades': 3}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1014.
- Execution convention: signal observed at close(t), executed at close(t + delay), position earns returns from close(t + delay) to close(t + delay + 1).
- Risk checks (drawdown pause, volatility block, stop-loss, take-profit) are evaluated at signal time.
- Slippage assumption (bps): 5.00.
- Benchmark eligible input: True.
- Calibration diagnostics: degenerate_regime=False, transition_count=99, min_transition_count=12, one_side_exposure_max=0.9208.
- Readiness warning: Validation Sharpe, walk-forward mean Sharpe, and held-out Sharpe are all negative.
- Readiness warning: Readiness hard blockers are present; do not advance to fine-tuning or next phase.
- Benchmark sufficiency artifact available: True.
- Regression history windows: 7/5.

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
- Historical Window Count: 7 (minimum 5)
- Hard Blocker [triple_negative_sharpe]: Validation, walk-forward, and held-out Sharpe are all negative.
- Decision Rationale:
  - engineering_validity=green
  - research_validity=red (triple_negative_sharpe)
  - evidence_sufficiency=green
  - multi_window_evidence_sufficient=True
  - phase_gate_decision=do_not_advance
  - fine_tuning_gate_open=False

## Failure / Fallbacks

- No failures captured.
