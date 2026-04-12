# Development Plan

## Current Stage

The repository is aligned to a **Week 1 research milestone**:
- feature engineering over BTC 15m data
- LightGBM directional baseline
- validation-set threshold calibration
- held-out out-of-sample LangGraph simulation
- purged walk-forward benchmark
- artifact persistence and simple simulation risk controls

This repo is intentionally **not** claiming:
- self-improving learning loops
- RL policy optimization
- LoRA or model fine-tuning
- RAG / vector memory
- live trading or paper trading

## Current Architecture

Runtime graph:

`data -> predict -> risk -> decision -> evaluate -> optimize`

Behavioral modules:
- `agents.data`: exchange/snapshot loading and split computation
- `agents.features`: feature engineering
- `agents.modeling`: LightGBM training, inference, SHAP summary
- `agents.risk`: simulation risk guardrails
- `agents.evaluation`: calibration and walk-forward benchmark
- `agents.artifacts`: report and artifact bundle persistence

## Current Validation Rules

- Train split is used only for model fitting.
- Validation split is used only for threshold calibration (with terminal supervised rows dropped for one-bar horizon integrity).
- Out-of-sample bars are reserved for the LangGraph run.
- Walk-forward uses purge gaps between validation and test windows.
- Signals are applied with a one-bar delay by default.
- Execution timestamp convention is explicit: execution occurs on the bar where the new position becomes active (`bar_timestamp == execution_timestamp`).
- Risk controls (drawdown pause, volatility block, stop-loss, take-profit) are signal-time rules with delayed action when `signal_delay_bars > 0`.

## Walk-Forward Sufficiency Contract

Readiness checks treat walk-forward evidence as sufficient only when:
- both expanding and rolling modes meet `min_folds_per_mode`
- both expanding and rolling modes meet `min_total_test_bars`
- explicit sufficiency metadata is persisted in artifacts

## Regression Tracking Contract

Every successful run appends a persisted regression ledger entry containing:
- run identifier, commit hash, and config hash
- dataset window span/hash metadata
- validation Sharpe, walk-forward mean Sharpe, held-out Sharpe
- max drawdown, total return, transition count, completed trade count
- calibrated thresholds

Fine-tuning and next-phase work remain blocked until the ledger demonstrates stable evidence across multiple historical windows.

## Near-Term Next Steps

1. Add offline benchmark fixtures and broader regression coverage.
2. Add feature ablations and stress-period benchmark presets.
3. Improve snapshot management and reproducibility metadata.
4. Consider model search only after benchmark stability is demonstrated.

## Fine-Tuning Gate

Future learning work should only start after:
- stable snapshot-based runs
- reproducible held-out and walk-forward metrics
- persisted artifacts across multiple historical windows
- clear benchmark regression tracking
- explicit readiness checks show engineering validity, research validity, and evidence sufficiency as green
