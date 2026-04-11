# BTC Model-Based Research Prototype Report

- Generated (UTC): 2026-04-11T11:17:53.486439+00:00
- Asset: BTC/USDT
- Timeframe: 15m
- Dataset mode: exchange
- Run ID: 20260411T111746Z-d4fc2bbd

## Held-Out Simulation

- OOS Bars: 800..999
- Completed Cycles: 199
- Paused: False

| Metric | Value |
|---|---:|
| Sharpe Ratio | -32.0132 |
| Max Drawdown | 4.95% |
| Total Return | -4.82% |
| Win Rate | 29.15% |
| Trade Count | 103 |

## Walk-Forward Benchmark

- Train Bars: 500
- Validation Bars: 150
- Test Bars: 120
- Purge Bars: 5
- Signal Delay Bars: 1

| Mode | Mean Sharpe | Mean Max Drawdown | Mean Total Return | Mean Win Rate | Folds |
|---|---:|---:|---:|---:|---:|
| Expanding | -25.2529 | 4.59% | -4.59% | 30.25% | 1 |
| Rolling | -25.2529 | 4.59% | -4.59% | 30.25% | 1 |

## Calibration Rule

- If ret_1_lag_14 is elevated (current=-0.0008), the model estimate for long probability typically increases.

## Active Risk Rules

- Max abs position: 1
- Max turnover per bar: 1
- Max drawdown pause: 15.00%
- Block high volatility: False
- Stop loss pct: 3.00%
- Take profit pct: 5.00%

## Optimization Provenance

| Event | Split | Objective | Value | Long Threshold | Short Threshold | Future OOS Only |
|---:|---|---|---:|---:|---:|---|
| 1 | validation | validation_sharpe | -12.8995 | 0.5599 | 0.2916 | True |
