# Recurring Self-Improving Trading AI Agent  

**Version**: 1.0  
**Date**: 2026-04-06  
**Asset Focus**: BTC 
**Goal**: A recurring, stateful agent that continuously learns from its own trades and historical data, turning every cycle into measurable improvement.

---

## Table of Contents

- [1. AI Agent Design](#1-ai-agent-design)
  - [Core Components](#core-components)
  - [Feedback Mechanisms for Recurring Improvement](#feedback-mechanisms-for-recurring-improvement)
  - [State Management and Memory](#state-management-and-memory)
  - [Key Metrics and Self-Update Logic](#key-metrics-and-self-update-logic)
- [2. Vibe Coding Plan](#2-vibe-coding-plan)
  - [Recommended Tech Stack](#recommended-tech-stack)
  - [Minimum Viable Agent (First Weekend Build)](#minimum-viable-agent-first-weekend-build)
  - [Iterative Improvements (Next 4–6 Weeks)](#iterative-improvements-next-4-6-weeks)
  - [Injecting the Self-Improvement Loop (No Over-Engineering)](#injecting-the-self-improvement-loop-no-over-engineering)
  - [Testing & Validation Strategy](#testing--validation-strategy)

---

## 1. AI Agent Design

### Core Components

The agent is built as a **closed recurring loop** orchestrated by **LangGraph StateGraph**. Every cycle (1–5 min or event-driven) flows through these 9 tightly-coupled but loosely-coupled modules:

1. **Data Ingestion** – CCXT/Polygon/Kafka pulls real-time ticks + incremental historical data.
2. **Data Processing & Feature Engineering** – Polars + TA-Lib generates 200–500 features (RSI, MACD, Bollinger, regime labels, on-chain metrics for BTC).
3. **Market Analysis & Prediction** – Temporal Fusion Transformer / LightGBM ensemble outputs probability distribution (not point forecast) + regime detection.
4. **Risk Management** – Real-time VaR/CVaR, position limits, correlation exposure, dynamic stop-loss.
5. **Trading Decision** – RL Policy (PPO/SAC) + RAG from knowledge base decides Long/Short/Hold + Kelly/CVaR sizing.
6. **Order Execution** – CCXT Pro + TWAP/VWAP smart routing (simulation first, then live).
7. **Performance Evaluation** – VectorBT/QuantStats computes Sharpe, Sortino, Calmar, Profit Factor, Win Rate, attribution (SHAP).
8. **Self-Optimization & Learning** – LoRA fine-tuning, Optuna hyperparam tuning, Experience Replay Buffer.
9. **Memory & Knowledge Base** – Pinecone/Weaviate (vector) + Neo4j (causal graph) stores rules, patterns, failed trades.

**Central Orchestrator**: LangGraph `StateGraph` with cyclic edges. All modules communicate via a single `AgentState` TypedDict.

### Feedback Mechanisms for Recurring Improvement

- **Daily Recurring Pre-Learning**: Every 00:00 UTC (or every 500 real trades) the agent replays the latest 6–12 months + full historical dataset in a Walk-Forward batch job. LoRA adapters are fine-tuned on per-asset basis.
- **Online Learning**: After every trade, the most recent experience is pushed to the Replay Buffer. The prediction model receives incremental updates (no full retrain).
- **Performance-Based Reward Shaping**: RL reward = `Sharpe × (1 - MaxDrawdown_penalty) + WinRate_bonus - Slippage_penalty`. High-value experiences (Sharpe > 1.5 or large drawdown recovery) get higher sampling weight.
- **Regime-Aware Trigger**: When volatility regime changes (detected by prediction node), force a knowledge extraction pass (SHAP → new If-Then rules).

### State Management and Memory

- **Short-term State** (in-memory, per-cycle):  
  `current_price`, `features_df`, `prediction`, `position`, `portfolio_value`, `cycle_count`, `last_action`.
- **Long-term Memory** (persistent):  
  - Vector DB: embeddings of every trade + market pattern.  
  - Graph DB: causal relationships (e.g., “FundingRate < -0.01 → Long probability +82%”).  
  - Replay Buffer: 10k+ high-value episodes for RL.

RAG retrieval happens in the Decision node: “Show me the 5 most similar historical regimes before I decide.”

### Key Metrics and Self-Update Logic

The agent optimises these metrics **automatically**:

| Metric              | Target (BTC) | How the Agent Uses It for Self-Update |
|---------------------|--------------|---------------------------------------|
| Sharpe Ratio        | > 2.0        | Primary reward signal; triggers LoRA fine-tune if < 1.5 |
| Max Drawdown        | < 12%        | Penalty term in reward; forces risk-node parameter tightening |
| Win Rate            | > 55%        | Bonus in replay sampling |
| Profit Factor       | > 1.8        | Used in evolutionary strategy to evolve new rules |
| Calmar Ratio        | > 1.5        | Long-term fitness score for Optuna |
| Slippage %          | < 0.05%      | Direct feedback to Execution node (adjusts expected fill price) |

After each evaluation step, the Optimize node runs:
1. Compute delta from previous cycle.
2. If any metric worsens → increase historical replay weight.
3. Run Optuna on top 3 hyperparameters.
4. Save new LoRA adapter + updated rules to Knowledge Base.

---

## 2. Vibe Coding Plan

Fast, intuitive, low-friction development. Ship first, improve daily. No premature abstraction.

### Recommended Tech Stack

- **Language**: Python 3.11+
- **Orchestration**: LangGraph (stateful cyclic graphs + human-in-the-loop)
- **Data**: Polars (speed), ClickHouse/Parquet (storage), Feast (feature store)
- **ML**: PyTorch + PEFT/LoRA (fine-tuning), LightGBM (fast baseline), Stable-Baselines3 (RL)
- **Backtesting**: VectorBT (vectorised, blazing fast)
- **Broker**: CCXT Pro (simulation + live), Polygon.io (historical + real-time)
- **Knowledge**: Pinecone/Weaviate + Neo4j
- **Tuning**: Optuna
- **Infra**: Docker + Ray (for later scaling)

### Minimum Viable Agent (First Weekend Build)

**Goal**: A working LangGraph loop that ingests historical BTC 1m data, makes simulated trades, evaluates, and does one simple optimisation.

**Weekend Plan** (≈ 8–10 hours):

```bash
# 1. Setup
mkdir trading-agent && cd trading-agent
uv init
uv add langgraph polars ta-lib vectorbt optuna ccxt pyarrow pyyaml shap

# 2. Folder structure (copy from previous chat)
# src/agents/nodes.py, src/agents/graph.py, etc.
```

**Core files to build (in order)**:

1. `config/config.yaml` – API keys, asset, cycle_interval.
2. `src/agents/state.py` – `AgentState` TypedDict.
3. `src/agents/nodes.py` – stub functions for `data_node`, `predict_node`, `decision_node`, `evaluate_node`, `optimize_node`.
4. `src/agents/graph.py` – build `StateGraph`, add cyclic edge back to start.
5. `run_mvp.py` – compile graph and run 100 simulated cycles on historical data.

**Success criteria for weekend**:  
- Loop runs end-to-end.  
- Produces a Markdown report with Sharpe, drawdown, and one SHAP rule.  
- `cycle_count` increments and triggers a dummy optimisation every 10 cycles.

### Iterative Improvements (Next 4–6 Weeks)

| Week | Focus | Deliverable |
|------|------|-------------|
| 1    | Feature engineering + LightGBM baseline | Walk-forward backtest with 300 features |
| 2    | Risk node + SHAP rule extraction | Knowledge Base with 50+ BTC-specific rules |
| 3    | RL warm-up (PPO on replay buffer) | Policy replaces rule-based decision |
| 4    | LoRA fine-tuning pipeline (Dagster) | Daily recurring pre-learning job |
| 5    | Paper trading integration | Live simulation with real tick data |
| 6    | Full memory (Pinecone + Neo4j) + human-in-the-loop | Agent can be paused for review |

### Injecting the Self-Improvement Loop (No Over-Engineering)

- Use **one cyclic edge** in LangGraph: `Evaluate → Optimize → back to Data`.
- In `optimize_node`:  
  ```python
  if state["performance"]["sharpe"] < 1.5 or state["cycle_count"] % 10 == 0:
      run_optuna_tune(state)
      update_lora_adapter(state)
      append_to_replay_buffer(state)
  ```
- Trigger recurring pre-learning with a simple cron/Dagster job that calls the same graph with `historical_mode=True`.
- Keep the loop **stateless between cycles except for the persistent Knowledge Base**.

### Testing & Validation Strategy

1. **Walk-Forward + Purged K-Fold** (mandatory in every training run).
2. **Paper Trading** (3 months minimum) before any real capital.
3. **Live-with-Cap** (start with 0.5% of total capital, max 2% per trade).
4. **Daily Regression Suite**: run 30-day rolling backtest every morning; alert if Sharpe drops > 0.3.
5. **Black-Swan Stress Test**: replay 2022 bear market + 2025 hypothetical events.

**Final Guardrails**:
- Max global drawdown 15% → auto-pause + notify.
- All decisions must be explainable (SHAP values stored).
- Full audit log for every trade.

---

