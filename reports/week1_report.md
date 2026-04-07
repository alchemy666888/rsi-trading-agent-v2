# Week 1 Trading Agent Report (LightGBM + Walk-Forward)

- Generated (UTC): 2026-04-07T16:07:38.295034+00:00
- Asset: BTC/USDT
- Timeframe: 1m
- Completed Cycles: 100

## Performance

| Metric | Value |
|---|---:|
| Sharpe Ratio | 66.7568 |
| Max Drawdown | 0.33% |
| Total Return | 0.56% |
| Win Rate | 62.00% |

## SHAP Rule

- If ret_1_lag_39 is elevated (current=0.0007), the model estimate for long probability typically decreases.

## Walk-Forward Backtest

- Train Bars: 400
- Test Bars: 100
- Step Bars: 50

| Mode | Mean Sharpe | Mean Max Drawdown | Mean Total Return | Mean Accuracy | Folds |
|---|---:|---:|---:|---:|---:|
| Expanding | -300.7972 | 2.49% | -2.47% | 51.10% | 10 |
| Rolling | -425.1572 | 3.28% | -3.27% | 47.70% | 10 |

- Overall:
  - Mean Sharpe: -362.9772
  - Mean Max Drawdown: 2.89%
  - Mean Total Return: -2.87%
  - Mean Accuracy: 49.40%

## Top 10 LightGBM Feature Importances

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | volume_lag_17 | 143.8390 |
| 2 | ret_1_lag_16 | 127.4324 |
| 3 | volume_z_30 | 123.8729 |
| 4 | ret_1_lag_55 | 123.8566 |
| 5 | ret_1_lag_39 | 118.4368 |
| 6 | volume_lag_59 | 107.4924 |
| 7 | tf_5m_ht_leadsine | 99.0757 |
| 8 | ret_max_5 | 98.5707 |
| 9 | ret_1_lag_29 | 95.6459 |
| 10 | ret_1_lag_10 | 90.3125 |

## Optimization Events

| Cycle | Optuna Objective | Long Threshold | Short Threshold |
|---:|---:|---:|---:|
| 1 | -38.8868 | 0.5573 | 0.2991 |
| 2 | -47.9547 | 0.5612 | 0.4244 |
| 3 | -37.4295 | 0.5775 | 0.3812 |
| 4 | -35.3449 | 0.7146 | 0.4060 |
| 7 | -35.4306 | 0.6320 | 0.3211 |
| 9 | -37.8162 | 0.6954 | 0.3431 |
| 10 | -50.6122 | 0.6351 | 0.4040 |
| 11 | -42.1524 | 0.6165 | 0.3383 |
| 12 | -51.4311 | 0.6719 | 0.4647 |
| 13 | -49.4666 | 0.6031 | 0.4388 |
| 14 | -49.0421 | 0.5271 | 0.4322 |
| 15 | -37.6742 | 0.5309 | 0.3831 |
| 17 | -30.5601 | 0.5679 | 0.2853 |
| 20 | -5.1399 | 0.6093 | 0.4319 |
| 30 | 3.2213 | 0.5238 | 0.3717 |
| 40 | -15.4037 | 0.5737 | 0.3077 |
| 41 | -25.8785 | 0.6105 | 0.4556 |
| 42 | -35.5630 | 0.5903 | 0.4423 |
| 43 | -32.3353 | 0.6759 | 0.4162 |
| 44 | -20.9706 | 0.5343 | 0.3607 |
| 50 | -18.6737 | 0.5668 | 0.3371 |
| 60 | -11.4496 | 0.7192 | 0.4161 |
| 70 | 13.9440 | 0.6633 | 0.3686 |
| 80 | -1.3157 | 0.7051 | 0.3229 |
| 90 | 2.6819 | 0.6215 | 0.4608 |
| 100 | 29.2889 | 0.6013 | 0.3635 |
