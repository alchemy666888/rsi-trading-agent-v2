# Development Plan

## Current Stage

The repository is aligned to a **Week 1 research milestone**:
- feature engineering over BTC 1m data
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
- Validation split is used only for threshold calibration.
- Out-of-sample bars are reserved for the LangGraph run.
- Walk-forward uses purge gaps between validation and test windows.
- Signals are applied with a one-bar delay.

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
