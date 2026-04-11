# BTC Model-Based Research Prototype

This repository is currently a **Week 1 research prototype**, not a self-improving trading agent.

What it does today:
- builds a LightGBM directional baseline on BTC 1m features
- calibrates thresholds on a validation split
- runs the LangGraph loop only on a held-out out-of-sample segment
- produces a purged walk-forward benchmark and an artifact bundle
- enforces simple simulation risk limits such as drawdown pause and turnover caps

What it does **not** do today:
- live trading or paper trading
- RL, LoRA, or model fine-tuning
- RAG, vector memory, or human-in-the-loop review

Run:

```bash
uv sync
uv run python run_mvp.py
```

For reproducible local runs, switch `dataset.source_mode` to `snapshot` and point `snapshot.path` at a saved CSV or Parquet snapshot.
