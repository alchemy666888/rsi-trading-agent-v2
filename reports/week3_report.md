# Week 1 Trading Agent Report (LightGBM + Walk-Forward)

- Generated (UTC): 2026-04-10T02:02:00.810148+00:00
- Asset: BTC/USDT
- Timeframe: 1m
- Completed Cycles: 162

## Run Metadata

- Git Commit Hash: `3d9ccd2f69741436b592638839e1dd9b122453b7`
- Config Hash (SHA-256): `ad9e069d87e703ad61c4d577702bd6562f9fbf8d89b6a0d737692bff5bbabc21`
- Data Source: `exchange_pull`
- Data File Path: `binance:BTC/USDT:1m`
- Data Row Count: 1000
- First Timestamp (UTC): 2026-04-09T09:22:00+00:00
- Last Timestamp (UTC): 2026-04-10T02:01:00+00:00

### Model Hyperparameters

```json
{
  "colsample_bytree": 0.85,
  "label_horizon": 1,
  "learning_rate": 0.05,
  "max_depth": -1,
  "max_feature_lag": 60,
  "min_child_samples": 20,
  "min_train_rows": 400,
  "n_estimators": 200,
  "n_jobs": -1,
  "num_leaves": 63,
  "random_state": 42,
  "subsample": 0.85,
  "train_split_ratio": 0.7,
  "walk_forward_n_estimators": 120
}
```

## Out-of-Sample Performance (Walk-Forward)

| Metric | Value |
|---|---:|
| WF Mean Sharpe | 0.0000 |
| WF Mean Accuracy | 0.00% |

## Live Simulation Performance (post-training data only)

| Metric | Value |
|---|---:|
| Sharpe Ratio | 0.0000 |
| Max Drawdown | 1.99% |
| Total Return | -1.77% |
| Win Rate | 42.59% |

## SHAP Rule

- If volume_lag_11 is elevated (current=0.7787), the model estimate for long probability typically increases.

## Walk-Forward Backtest

> [!WARNING]
> Reliability warning: only 162 returns were observed. Sharpe and related risk-adjusted metrics are unstable below 1,000 returns.

- Train Bars: 800
- Test Bars: 100
- Step Bars: 200

| Mode | Mean Sharpe | Mean Max Drawdown | Mean Total Return | Mean Accuracy | Folds |
|---|---:|---:|---:|---:|---:|
| Expanding | 0.0000 | 0.00% | 0.00% | 0.00% | 0 |
| Rolling | 0.0000 | 0.00% | 0.00% | 0.00% | 0 |

- Overall:
  - Mean Sharpe: 0.0000
  - Mean Max Drawdown: 0.00%
  - Mean Total Return: 0.00%
  - Mean Accuracy: 0.00%

## Methodology

- Walk-Forward Train Bars: 800
- Walk-Forward Test Bars: 100
- Walk-Forward Step Bars: 200
- Walk-Forward Embargo Gap: 62
- Trading Costs: slippage=5 bps, funding_rate_per_8h=0.0001
- Decision Thresholds: long=0.8492, short=0.2364

## Top 10 LightGBM Feature Importances

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | volume_lag_3 | 148.2951 |
| 2 | rsi_14_lag_27 | 147.5627 |
| 3 | mfi_8 | 124.1724 |
| 4 | volume_lag_26 | 118.1980 |
| 5 | ret_1_lag_52 | 112.5560 |
| 6 | volume_lag_42 | 102.6770 |
| 7 | eth_vol_chg_1 | 93.2456 |
| 8 | volume_lag_43 | 92.3225 |
| 9 | volume_lag_20 | 89.3609 |
| 10 | volume_lag_11 | 80.2406 |

## SHAP Knowledge Base

*26 unique rules accumulated across sessions.*

1. If volume_lag_42 is elevated (current=17.0496), the model estimate for long probability typically increases.
2. If ret_1_lag_52 is elevated (current=0.0018), the model estimate for long probability typically decreases.
3. If ret_1_lag_40 is elevated (current=0.0013), the model estimate for long probability typically increases.
4. If volume_lag_42 is elevated (current=2.5781), the model estimate for long probability typically decreases.
5. If volume_lag_3 is elevated (current=1.4679), the model estimate for long probability typically increases.
6. If mfi_8 is elevated (current=95.0747), the model estimate for long probability typically increases.
7. If volume_lag_3 is elevated (current=1.9648), the model estimate for long probability typically increases.
8. If volume_lag_3 is elevated (current=1.8101), the model estimate for long probability typically increases.
9. If volume_lag_20 is elevated (current=84.1561), the model estimate for long probability typically increases.
10. If volume_lag_3 is elevated (current=2.3657), the model estimate for long probability typically increases.
11. If volume_lag_26 is elevated (current=2.6926), the model estimate for long probability typically decreases.
12. If volume_lag_23 is elevated (current=90.8113), the model estimate for long probability typically decreases.
13. If volume_lag_11 is elevated (current=2.3090), the model estimate for long probability typically increases.
14. If volume_lag_7 is elevated (current=2.9021), the model estimate for long probability typically decreases.
15. If eth_vol_chg_1 is elevated (current=-0.6048), the model estimate for long probability typically increases.
16. If volume_lag_20 is elevated (current=130.2940), the model estimate for long probability typically increases.
17. If volume_lag_3 is elevated (current=1.3040), the model estimate for long probability typically increases.
18. If volume_lag_26 is elevated (current=3.0266), the model estimate for long probability typically decreases.
19. If volume_lag_26 is elevated (current=5.9139), the model estimate for long probability typically decreases.
20. If volume_lag_11 is elevated (current=0.7787), the model estimate for long probability typically increases.

## Optimization Events

| Cycle | Optuna Objective | Long Threshold | Short Threshold |
|---:|---:|---:|---:|
| 10 | 67.2996 | 0.8077 | 0.4057 |
| 20 | 69.6513 | 0.8339 | 0.4534 |
| 30 | 53.3791 | 0.8303 | 0.4018 |
| 40 | 44.8180 | 0.8335 | 0.3594 |
| 50 | 48.2150 | 0.7992 | 0.4200 |
| 60 | 44.2961 | 0.8388 | 0.4397 |
| 70 | 45.7425 | 0.8097 | 0.4460 |
| 80 | 35.7284 | 0.8407 | 0.2474 |
| 90 | 27.7080 | 0.8246 | 0.3722 |
| 100 | 20.2909 | 0.8186 | 0.2245 |
| 110 | 10.5971 | 0.8342 | 0.2198 |
| 120 | 11.9992 | 0.8145 | 0.3833 |
| 130 | 10.6833 | 0.8122 | 0.2280 |
| 140 | 17.3574 | 0.8010 | 0.2281 |
| 150 | 19.6727 | 0.8214 | 0.2301 |
| 160 | 20.7643 | 0.8492 | 0.2364 |
