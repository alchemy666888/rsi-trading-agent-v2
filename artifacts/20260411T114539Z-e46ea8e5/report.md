# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260411T114539Z-e46ea8e5
- Generated (UTC): 2026-04-11T11:45:47.595954+00:00
- Report Detail Level: full
- Git Commit: 510ef89d6097e9c5db50baa3acd72ba28bd6c851
- Config Hash: e5400dd2aec9a3b182aef6a016e8379a55b6fbe8158af5ef499591fa551f23cc
- Asset: BTC/USDT (15m)
- Dataset Source: exchange:binance (exchange)
- Dataset Span (UTC): 2026-04-01T02:00:00+00:00 -> 2026-04-11T11:45:00+00:00
- Bars: total=1000, train=600, validation=200, oos=199

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -31.2603 |
| Max Drawdown | 5.15% |
| Total Return | -4.91% |
| Win Rate | 55.81% |
| Trade Count | 43 |
| Avg Trade Return | -0.0127% |
| Profit Factor | 0.8607087849538304 |
| Turnover | 0.4372 |
| Long Exposure | 20.60% |
| Short Exposure | 56.28% |
| Flat Exposure | 23.12% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.5214300126882973, "short_threshold": 0.47943505959772875}
- Objective: validation_sharpe
- Best Validation Score: -20.8892
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): -4.8924
- Delta Total Return (Held-Out - Walk-Forward Mean): -0.20%
- Overfit Warnings: none triggered.

## Risk Diagnostics

- Blocked Trades: 0
- Stop Loss Trigger Count: 0
- Take Profit Trigger Count: 0
- Drawdown Pause Activated Count: 0
- Drawdown Pause Cleared Count: 0
- Pause Durations (seconds): []
- Max Consecutive Losses: 6

## Trade Diagnostics

- Best Trade: {'direction': 1, 'open_cycle': 17, 'close_cycle': 20, 'open_timestamp': 1775744100000, 'close_timestamp': 1775746800000, 'entry_price': 70788.01, 'exit_price': 71123.71, 'pnl_pct': 0.004742328538406504, 'hold_bars': 3, 'close_reason': []}
- Worst Trade: {'direction': -1, 'open_cycle': 97, 'close_cycle': 119, 'open_timestamp': 1775816100000, 'close_timestamp': 1775835900000, 'entry_price': 71810.39, 'exit_price': 72745.81, 'pnl_pct': -0.012858747466005238, 'hold_bars': 22, 'close_reason': []}
- Longest Hold (bars): 22
- Average Hold (bars): 3.44
- Long/Short Breakdown: {'long_trades': 19, 'short_trades': 24}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1024.
- Assumed next-bar execution with slippage_bps=5.00.

## Failure / Fallbacks

- No failures captured.
