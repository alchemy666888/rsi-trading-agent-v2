# BTC Model-Based Research Prototype Report

- Generated (UTC): 2026-04-11T10:23:26.091114+00:00
- Asset: BTC/USDT
- Timeframe: 1m
- Dataset mode: exchange
- Run ID: 20260411T102317Z-9321f7c3

## Held-Out Simulation

- OOS Bars: 800..999
- Completed Cycles: 199
- Paused: False

| Metric | Value |
|---|---:|
| Sharpe Ratio | -511.7636 |
| Max Drawdown | 4.83% |
| Total Return | -4.83% |
| Win Rate | 24.12% |
| Trade Count | 95 |

## Walk-Forward Benchmark

- Train Bars: 500
- Validation Bars: 150
- Test Bars: 120
- Purge Bars: 5
- Signal Delay Bars: 1

| Mode | Mean Sharpe | Mean Max Drawdown | Mean Total Return | Mean Win Rate | Folds |
|---|---:|---:|---:|---:|---:|
| Expanding | -700.0685 | 3.34% | -3.30% | 15.97% | 1 |
| Rolling | -697.1323 | 3.58% | -3.58% | 10.92% | 1 |

## Calibration Rule

- If ret_1_lag_40 is elevated (current=-0.0002), the model estimate for long probability typically decreases.

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
| 1 | validation | validation_sharpe | -553.3943 | 0.5392 | 0.4126 | True |
