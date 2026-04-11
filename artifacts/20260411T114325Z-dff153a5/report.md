# BTC Model-Based Research Prototype Report

## Run Metadata

- Run ID: 20260411T114325Z-dff153a5
- Generated (UTC): 2026-04-11T11:43:34.059486+00:00
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
| Sharpe | -31.4320 |
| Max Drawdown | 5.28% |
| Total Return | -4.84% |
| Win Rate | 51.92% |
| Trade Count | 52 |
| Avg Trade Return | 0.0083% |
| Profit Factor | 1.107055281262776 |
| Turnover | 0.5276 |
| Long Exposure | 24.12% |
| Short Exposure | 46.73% |
| Flat Exposure | 29.15% |

## Calibration Summary

- Tuned Thresholds: {"long_threshold": 0.5270478734774753, "short_threshold": 0.42663430993257145}
- Objective: validation_sharpe
- Best Validation Score: -16.2024
- Optimization Events Recorded: 1

## Benchmark Comparison

- Delta Sharpe (Held-Out - Walk-Forward Mean): -12.4518
- Delta Total Return (Held-Out - Walk-Forward Mean): -1.44%
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
- Worst Trade: {'direction': -1, 'open_cycle': 101, 'close_cycle': 120, 'open_timestamp': 1775818800000, 'close_timestamp': 1775835900000, 'entry_price': 71910.09, 'exit_price': 72745.81, 'pnl_pct': -0.01148822179586706, 'hold_bars': 19, 'close_reason': []}
- Longest Hold (bars): 19
- Average Hold (bars): 2.63
- Long/Short Breakdown: {'long_trades': 22, 'short_trades': 30}

## Data Quality / Readiness

- Missing values filled during feature prep: 0.
- Feature columns available for model: 1024.
- Assumed next-bar execution with slippage_bps=5.00.

## Failure / Fallbacks

- No failures captured.
