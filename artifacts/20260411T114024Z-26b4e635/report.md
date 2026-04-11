# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260411T114024Z-26b4e635
- Generated (UTC): 2026-04-11T11:40:33.077878+00:00
- Report Detail Level: full
- Git Commit: 054eed91404a4cffff25a2a67611a3ac6de13a57
- Config Hash: e5400dd2aec9a3b182aef6a016e8379a55b6fbe8158af5ef499591fa551f23cc
- Asset: BTC/USDT (15m)
- Dataset Source: exchange:binance (exchange)
- Dataset Span (UTC): 2026-04-01T01:45:00+00:00 -> 2026-04-11T11:30:00+00:00
- Bars: total=1000, train=600, validation=200, oos=199

## Headline KPIs

| Metric | Value |
|---|---:|
| Sharpe | -33.6726 |
| Max Drawdown | 5.56% |
| Total Return | -5.10% |
| Win Rate | 54.90% |
| Trade Count | 51 |
| Avg Trade Return | 0.0008% |
| Profit Factor | 1.010611284059765 |
| Turnover | 0.5176 |
| Long Exposure | 15.08% |
| Short Exposure | 52.26% |
| Flat Exposure | 32.66% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.6541668567680966, "short_threshold": 0.44177835750722394}
- Objective: validation_sharpe
- Best Validation Score: -16.7262
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): -13.6651
- Delta Total Return (Held-Out - Walk-Forward Mean): -1.53%
- Overfit Warnings: none triggered.

## Risk Diagnostics

- Blocked Trades: 0
- Stop Loss Trigger Count: 0
- Take Profit Trigger Count: 0
- Drawdown Pause Activated Count: 0
- Drawdown Pause Cleared Count: 0
- Pause Durations (seconds): []
- Max Consecutive Losses: 3

## Trade Diagnostics

- Best Trade: {'direction': -1, 'open_cycle': 48, 'close_cycle': 53, 'open_timestamp': 1775771100000, 'close_timestamp': 1775775600000, 'entry_price': 72450.53, 'exit_price': 71946.01, 'pnl_pct': 0.007012480608723282, 'hold_bars': 5, 'close_reason': []}
- Worst Trade: {'direction': -1, 'open_cycle': 100, 'close_cycle': 120, 'open_timestamp': 1775817900000, 'close_timestamp': 1775835900000, 'entry_price': 71800.01, 'exit_price': 72745.81, 'pnl_pct': -0.013001436096456986, 'hold_bars': 20, 'close_reason': []}
- Longest Hold (bars): 20
- Average Hold (bars): 2.55
- Long/Short Breakdown: {'long_trades': 18, 'short_trades': 33}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1024.
- Assumed next-bar execution with slippage_bps=5.00.

## Failure / Fallbacks

- Error Type: ComputeError
- Error Message: CSV format does not support nested data
- Failed At (UTC): 2026-04-11T11:40:33.077061+00:00
