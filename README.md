# BTC Model-Based Research Prototype

This repository is a research-oriented BTC/USDT 15-minute trading prototype built around a small LangGraph execution loop, a LightGBM directional model, heavy feature engineering, threshold calibration on a validation split, and held-out simulation with artifact persistence.

It is intentionally positioned as a reproducible research baseline, not as a production trading system.

## Current Status

The codebase is aligned to a model-based research milestone:

- loads BTC/USDT OHLCV data from Binance through `ccxt`, or from a local snapshot
- builds a large technical-indicator feature set with lags and multi-timeframe joins
- trains a LightGBM binary classifier to predict next-bar direction
- calibrates long/short decision thresholds on a validation split with Optuna
- runs a held-out out-of-sample simulation through a LangGraph workflow
- computes a purged walk-forward benchmark for expanding and rolling windows
- applies simple simulation risk controls such as turnover caps, drawdown pauses, stop-loss, and take-profit
- writes a versioned artifact bundle for each run

What this repository does not do today:

- live trading
- paper trading with broker integration
- online/self-improving learning loops
- RL policy optimization
- LoRA or any other model fine-tuning
- RAG, memory systems, or human approval workflows

## Why This Exists

The project is useful as a compact experimentation harness for questions like:

- Does a large TA-style feature bank contain any short-horizon directional signal on recent BTC 1m data?
- How sensitive is simple threshold-based execution to validation tuning?
- Do held-out results resemble walk-forward benchmark behavior, or are they likely overfit?
- What artifacts should be preserved before claiming any strategy improvement?

In other words, this repo is better thought of as a structured research notebook in code form than as an autonomous trading agent.

## End-to-End Flow

The runtime graph is:

```text
data -> predict -> risk -> decision -> evaluate -> optimize
```

At a high level:

1. Historical data is loaded and transformed into a feature matrix.
2. The train split is used to fit a LightGBM baseline.
3. The validation split is used to calibrate decision thresholds only.
4. A purged walk-forward benchmark is run separately for context.
5. The LangGraph loop then steps forward only on held-out out-of-sample bars.
6. Metrics, trades, calibration metadata, and reports are persisted under `artifacts/<run_id>/`.

## Repository Layout

```text
.
├── config/config.yaml             # Runtime, split, model, simulation, and risk settings
├── data/                          # Sample data and SHAP rules
├── docs/                          # Development notes and milestone plans
├── reports/                       # Historical reports retained for reference
├── scripts/fetch_real_data.py     # Helper to build a larger local BTC snapshot
├── src/agents/
│   ├── artifacts.py               # Report generation and artifact persistence
│   ├── data.py                    # Exchange/snapshot loading and split computation
│   ├── evaluation.py              # Threshold calibration and walk-forward benchmarking
│   ├── features.py                # Feature engineering
│   ├── graph.py                   # LangGraph workflow wiring
│   ├── modeling.py                # LightGBM training, inference, SHAP rule derivation
│   ├── nodes.py                   # Runtime nodes used during held-out simulation
│   ├── risk.py                    # Risk guardrails
│   ├── setup.py                   # Experiment bootstrap
│   └── state.py                   # Typed runtime state
├── tests/                         # Regression and pipeline tests
└── run_mvp.py                     # Main entrypoint
```

## Installation

### Requirements

- Python 3.11+
- `uv`
- TA-Lib available for the Python package to link against

Install dependencies:

```bash
uv sync
```

If you also want test dependencies:

```bash
uv sync --group dev
```

## Quick Start

### Option 1: Run against a live exchange snapshot

This is the default mode in [`config/config.yaml`](/Users/antee/Documents/projects/rsi-trading-agent-v2/config/config.yaml).

```bash
uv run python run_mvp.py
```

In this mode the system:

- fetches recent BTC/USDT 1m candles from Binance via `ccxt`
- optionally enriches the dataset with ETH proxy columns when available
- builds features, trains the baseline, calibrates thresholds, and runs the held-out simulation

### Option 2: Run against a local snapshot for reproducibility

For stable local experiments, set:

```yaml
dataset:
  source_mode: "snapshot"

snapshot:
  path: "data/btc_1m_real.csv"
  auto_write: false
```

Then run:

```bash
uv run python run_mvp.py
```

Snapshot mode is the better default for repeatable benchmarks, debugging, and documentation examples.

## Building a Local Dataset Snapshot

The repository includes a helper script that fetches paginated BTC/USDT 15-minute bars from Binance:

```bash
uv run python scripts/fetch_real_data.py --months 6 --out data/btc_1m_real.csv
```

Example options:

- `--months`: approximate amount of history to fetch
- `--symbol`: trading pair, default `BTC/USDT`
- `--out`: destination CSV path

This script is useful when you want a larger and more stable local input file than the short exchange pull used by the default runtime configuration.

## Configuration Guide

The main configuration file is [`config/config.yaml`](/Users/antee/Documents/projects/rsi-trading-agent-v2/config/config.yaml).

### `asset`

Controls market source settings:

- `symbol`: exchange symbol, default `BTC/USDT`
- `timeframe`: candle interval, currently `1m`
- `exchange`: `ccxt` exchange id, currently `binance`
- `fetch_limit`: number of rows fetched in exchange mode

### `dataset` and `snapshot`

Controls where historical data comes from:

- `dataset.source_mode = "exchange"` loads fresh OHLCV through `ccxt`
- `dataset.source_mode = "snapshot"` reads a local `.csv` or `.parquet`
- `snapshot.auto_write = true` writes the fetched exchange dataset back to disk

### `runtime`

Controls run length and outputs:

- `max_cycles`: max number of held-out LangGraph steps to execute
- `warmup_bars`: minimum history required before splits become valid
- `artifact_output_dir`: root directory for versioned run artifacts

### `splits`

Defines the coarse train/validation/out-of-sample segmentation:

- `train_ratio`
- `validation_ratio`
- `minimum_oos_bars`

The held-out simulation only starts at `oos_start`, after train and validation have already been consumed.

### `simulation`

Controls execution assumptions:

- `initial_equity`
- `slippage_bps`
- `signal_delay_bars`
- `optuna_trials`
- `trade_history_limit`
- optional `funding_rate_per_8h`

The implementation applies signals to the next bar and deducts slippage when position changes occur.

### `risk`

Controls simple portfolio guardrails:

- `max_abs_position`
- `max_turnover_per_bar`
- `max_drawdown_pause`
- `stop_loss_pct`
- `take_profit_pct`
- `block_high_volatility`

These are simulation guardrails, not broker-enforced risk controls.

### `model`

Controls the LightGBM baseline:

- `n_estimators`
- `learning_rate`
- `num_leaves`
- `max_depth`
- `min_child_samples`
- `subsample`
- `colsample_bytree`
- `random_state`
- `n_jobs`

### `features`

Controls feature-engineering breadth:

- `include_multi_timeframe`
- `max_lag_bars`

This section can materially change both runtime cost and model behavior because the feature matrix is very wide.

### `walk_forward`

Controls benchmark fold generation:

- `train_bars`
- `validation_bars`
- `test_bars`
- `step_bars`
- `purge_bars`

The benchmark supports both expanding and rolling modes and inserts a purge gap between validation and test windows.

## Data and Features

Feature generation is intentionally broad and includes:

- raw OHLCV-derived returns and ranges
- many TA-Lib indicators such as SMA, EMA, RSI, ROC, ATR, NATR, CCI, MFI, ADX, MACD, Bollinger Bands, stochastic variants, Aroon, OBV, ADOSC, Hilbert transforms, and several candlestick patterns
- lagged versions of selected columns
- rolling means, standard deviations, extrema, z-scores, and realized volatility windows
- optional multi-timeframe joins from 5m, 15m, and 1h aggregates
- optional BTC/ETH relative features when ETH proxy data is successfully fetched
- a binary target: whether the next close is above the current close

This is a high-capacity feature set. It is good for experimentation, but it also raises the risk of overfitting, especially on short exchange pulls.

## Modeling and Calibration

The core modeling logic lives in [`src/agents/modeling.py`](/Users/antee/Documents/projects/rsi-trading-agent-v2/src/agents/modeling.py) and [`src/agents/evaluation.py`](/Users/antee/Documents/projects/rsi-trading-agent-v2/src/agents/evaluation.py).

The baseline workflow is:

1. Train a LightGBM binary classifier on the train split.
2. Score the validation split.
3. Use Optuna to search for `long_threshold` and `short_threshold` that maximize validation Sharpe in a simple simulator.
4. Freeze those thresholds for the future held-out run.
5. Derive a compact SHAP-based interpretation string for the latest eligible observation.

That validation-only threshold tuning is an important design choice: the held-out out-of-sample segment is reserved for forward-style simulation, not for parameter search.

## Held-Out Simulation Logic

The LangGraph loop is intentionally simple:

- `data_node` exposes the current held-out row
- `predict_node` generates a next-bar directional probability
- `risk_node` transforms that into a capped target position under active guardrails
- `decision_node` translates the target into `LONG`, `SHORT`, `FLAT`, or `HOLD`
- `evaluate_node` applies next-bar PnL, slippage, optional funding, and trade logging
- `optimize_node` is currently a placeholder and does not mutate strategy parameters during the run

This means the project is graph-based structurally, but not yet adaptive in the sense of performing online strategy updates.

## Walk-Forward Benchmark

The walk-forward benchmark is separate from the live-style held-out loop and exists to give a sanity-check baseline.

It:

- rebuilds train/validation/test folds before the held-out region
- supports both expanding and rolling windows
- inserts a purge gap to reduce leakage between validation and test segments
- retrains the LightGBM model per fold
- re-calibrates thresholds on each fold's validation segment
- evaluates fold-level Sharpe, drawdown, return, and win rate

This benchmark is one of the more important safeguards in the repo because it helps prevent over-interpreting a single held-out slice.

## Artifacts Produced Per Run

Every execution writes a versioned directory under `artifacts/<run_id>/`.

Typical files include:

- `report.md`: human-readable experiment summary
- `config.yaml`: frozen run config
- `dataset_metadata.json`: row count, timestamp bounds, source mode, source reference, dataset hash
- `split_metadata.json`: train/validation/oos boundaries
- `strategy_params.json`: chosen long/short thresholds
- `optimization_events.json`: tuning provenance
- `feature_importances.json`: LightGBM gain importances
- `benchmark_metrics.json`: walk-forward benchmark results
- `run_metrics.json`: held-out simulation metrics
- `trades.json`: position-change events
- `trade_history_buffer.json`: recent trade history with timestamps and realized returns
- `equity_curve.json`: simulated equity path
- `returns.json`: per-step strategy returns
- `shap_rule.txt`: one-line SHAP interpretation

Example artifact path already present in the repo:

- [`artifacts/20260411T102317Z-9321f7c3/report.md`](/Users/antee/Documents/projects/rsi-trading-agent-v2/artifacts/20260411T102317Z-9321f7c3/report.md)

## Reading the Results Carefully

One of the most important things about this repository is not overclaiming from noisy numbers.

A few practical cautions:

- Short 15-minute windows can produce unstable Sharpe values.
- A wide feature bank can easily overfit recent exchange pulls.
- Negative or extreme Sharpe in example artifacts should be treated as a research outcome, not as a bug by itself.
- Good held-out performance without decent walk-forward behavior should be viewed skeptically.
- Exchange mode is convenient, but snapshot mode is much better for reproducibility.

The current artifact examples in this repository should be interpreted primarily as evidence that the pipeline runs end-to-end and preserves provenance, not as evidence of alpha.

## Testing

Run tests with:

```bash
uv run pytest
```

There are also regression-oriented checks under [`tests/test_pipeline.py`](/Users/antee/Documents/projects/rsi-trading-agent-v2/tests/test_pipeline.py) covering split handling, signal timing, risk behavior, fold construction, and artifact persistence.

## Development Notes

Helpful references in the repository:

- [`docs/development-plan.md`](/Users/antee/Documents/projects/rsi-trading-agent-v2/docs/development-plan.md): milestone framing and near-term priorities
- [`reports/mvp_report.md`](/Users/antee/Documents/projects/rsi-trading-agent-v2/reports/mvp_report.md): archived historical report, not the current source of truth

## Known Limitations

- No realistic exchange execution modeling beyond simple slippage and optional funding.
- No fees model beyond the configured slippage proxy.
- No position sizing beyond capped integer exposure.
- No portfolio support beyond a single BTC/USDT stream.
- No online retraining during the held-out LangGraph run.
- No production hardening for secrets, retries, observability, or broker safety.
- Default exchange pulls are relatively short, which makes results noisy.

## Suggested Next Steps

If you want to push this prototype forward responsibly, the highest-value directions are:

1. Standardize snapshot-based benchmark datasets and keep them under versioned experiment control.
2. Add stronger regression tests around feature generation and artifact schemas.
3. Run feature ablations before trying larger model searches.
4. Expand benchmark coverage across multiple historical regimes.
5. Only consider more complex learning loops after reproducibility is stable.

## Minimal Commands Reference

Install:

```bash
uv sync
```

Run the prototype:

```bash
uv run python run_mvp.py
```

Fetch a larger local dataset:

```bash
uv run python scripts/fetch_real_data.py --months 6 --out data/btc_1m_real.csv
```

Run tests:

```bash
uv run pytest
```
