# BTC Model-Based Research Prototype Report

- Generated (UTC): 2026-04-11T11:05:11.047667+00:00
- Asset: BTC/USDT
- Timeframe: 1m
- Dataset mode: exchange
- Run ID: 20260411T110459Z-3af3cfcf

## Held-Out Simulation

- OOS Bars: 800..999
- Completed Cycles: 199
- Paused: False

| Metric | Value |
|---|---:|
| Sharpe Ratio | -552.4429 |
| Max Drawdown | 4.43% |
| Total Return | -4.43% |
| Win Rate | 12.06% |
| Trade Count | 88 |

## Walk-Forward Benchmark

- Train Bars: 500
- Validation Bars: 150
- Test Bars: 120
- Purge Bars: 5
- Signal Delay Bars: 1

| Mode | Mean Sharpe | Mean Max Drawdown | Mean Total Return | Mean Win Rate | Folds |
|---|---:|---:|---:|---:|---:|
| Expanding | -695.9291 | 3.26% | -3.26% | 15.13% | 1 |
| Rolling | -652.9606 | 3.06% | -3.06% | 14.29% | 1 |

## Calibration Rule

- If close_lag_11 is elevated (current=72815.6300), the model estimate for long probability typically decreases.

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
| 1 | validation | validation_sharpe | -538.5597 | 0.7489 | 0.2526 | True |
