# RSI Trading Agent v2 — Development Plan Review Report

**Reviewer:** Claude (automated code review)
**Date:** 2026-04-09
**Branch reviewed:** `claude/review-development-plan-9NZko`
**Development plan:** `docs/development-plan.md`

---

## Summary of Findings

The repository implements a LangGraph-orchestrated, LightGBM-driven BTC trading agent with 300+ engineered features, Optuna threshold tuning, SHAP rule extraction, and walk-forward validation. The infrastructure is well-conceived and demonstrates clear "vibe coding" discipline — iterative, expressive, and grounded in feedback loops (Markdown reports, SHAP rules, Optuna objectives).

**However**, two critical bugs were patched on April 9 (in-sample evaluation bias + embargo gap off-by-one) *after* both existing reports were generated. All reported metrics pre-date these fixes and must be treated as suspect. The walk-forward out-of-sample results — the only trustworthy signal — show no predictive edge (49.4% accuracy, mean Sharpe −362.98), which is the primary blocker for advancing to later phases.

---

## Phase Alignment Score

| Phase | Planned Deliverable | Actual Status | Score |
|---|---|---|:---:|
| **MVP (Weekend)** | 100-cycle LangGraph loop, LightGBM baseline, dummy optimization, Markdown report | Fully delivered: cyclic LangGraph graph, LightGBM trained per cycle, 15-trial Optuna tuning, auto-generated Markdown report | 🟢 |
| **Week 1: Feature Eng + LightGBM** | 300+ engineered features, multi-timeframe, walk-forward validation | 300+ TA-Lib + lag + rolling features across 1m/5m/15m/1h; expanding + rolling walk-forward in `_run_walk_forward_backtest()` | 🟢 |
| **Week 2: Risk Node + SHAP** | Dedicated risk management node, structured SHAP rule extraction | SHAP rule extraction implemented (`_derive_shap_rule()`), but **no dedicated risk node** — risk logic is embedded inside `decision_node` | 🟡 |
| **Week 3: RL Warm-up (PPO)** | PPO reinforcement learning agent integration | Not started. No RL/PPO code found anywhere in the codebase. | 🔴 |
| **Week 4: LoRA Fine-tuning** | LoRA adapter fine-tuning pipeline for LLM policy | Stub only. `update_lora_adapter()` in `nodes.py` is a no-op placeholder. | 🔴 |
| **Week 5: Paper Trading** | Live paper trading via CCXT Pro, real-time data feed | Not implemented. Agent iterates on cached historical data only; no live execution path. | 🔴 |
| **Week 6: Memory + HITL** | Pinecone vector DB, Neo4j graph DB, replay buffer, human-in-the-loop | All memory is in-memory per session. Replay buffer is a stub. No persistence layer or HITL mechanism. | 🔴 |

---

## Validity Assessment of Results/Reports

### Live Simulation Metrics (`week1_report.md`, generated April 7)

| Metric | Reported Value | Validity Verdict |
|---|---|---|
| Sharpe Ratio | 66.76 | **INVALID** — computed on ~100 bars; code itself warns < 1,000 bars is unreliable. Statistically meaningless. |
| Max Drawdown | 0.33% | Plausible but unreliable at this sample size. |
| Total Return | +0.56% | On 100 bars only; not meaningful. |
| Win Rate | 62% | Likely biased by the now-fixed in-sample evaluation bug. |

### Walk-Forward Out-of-Sample Metrics (`week1_report.md`, generated April 7)

| Metric | Reported Value | Validity Verdict |
|---|---|---|
| Mean Sharpe (overall) | −362.98 | **This is the real signal.** Model has no durable edge. |
| Mean Accuracy | 49.40% | Essentially random; LightGBM is not generalizing. |
| Mean Total Return | −2.87% | Strategy loses money on unseen data. |
| Mean Max Drawdown | 2.89% | Plausible. |

### Methodology Soundness

**Correct implementations:**
- RSI via TA-Lib at 9 periods (5/8/10/14/20/30/50/100/200) — standard multi-period approach
- Slippage: 5 bps applied on position changes — reasonable for BTC spot
- Walk-forward: expanding + rolling with configurable `train_bars`/`test_bars`/`step_bars` — correct design
- Sharpe annualization: `√525600` factor for 1-minute bars — correct
- Embargo gap: fixed in commit `bc73b50` (April 9) — critical for preventing train/test leakage

**Issues found:**

| # | Issue | Severity |
|---|---|---|
| 1 | **In-sample evaluation bias** (commit `3d83c9a`, April 9): live sim Sharpe of 66.76 was almost certainly computed partly on LightGBM training data. Both existing reports pre-date this fix. | Critical |
| 2 | **Embargo gap off-by-one** (commit `bc73b50`, April 9): walk-forward folds had leakage between train/test windows. Walk-forward results in `week1_report.md` are therefore also suspect. | Critical |
| 3 | **Sample size**: 100-cycle simulation on ~1,500 bars gives far too few out-of-sample observations for reliable Sharpe estimation. | High |
| 4 | **No funding rate costs**: perpetuals carry ±0.01%/8h funding. Ignoring this overstates directional returns. | Medium |
| 5 | **LoRA/replay buffer stubs**: called in `optimize_node` but do nothing — silent no-ops that give a false impression of a running self-improvement loop. | Medium |

---

## Go / No-Go Recommendation

### NO-GO — do not advance to Week 3+ (RL / LoRA / Paper Trading)

### Conditional GO — re-validate Week 1 completion first

The infrastructure is solid and the vibe-coding ethos is well-executed. But the two critical bug fixes applied on April 9 have not been reflected in fresh reports, and the walk-forward signal is negative. Building PPO, LoRA, or a paper trading layer on top of an unvalidated baseline is premature.

---

## Blockers and Required Actions

### P0 — Must fix before any further development

1. **Re-run full simulation post-fix**: Both reports were generated before the April 9 critical patches. Run `python run_mvp.py` and generate a fresh `reports/week2_report.md` to establish a clean baseline.

2. **Achieve positive walk-forward signal**: Out-of-sample accuracy of 49.4% and mean Sharpe of −362.98 indicate the model is fitting noise. The feature engineering must demonstrate a reproducible edge (accuracy consistently > 51–52%, positive mean Sharpe on ≥ 3 rolling folds) before adding architectural complexity.

### P1 — Address before Week 3

3. **Extract dedicated `risk_node`**: The development plan specifies a standalone risk management node with position sizing and regime-aware triggers. Currently this logic is folded into `decision_node`. Extract it into its own node in `graph.py` and add corresponding fields to `AgentState` in `state.py`. This keeps the graph semantics clean for future RL integration.

4. **Replace silent stubs with explicit TODOs**: `update_lora_adapter()` and `append_to_replay_buffer()` are called in `optimize_node` but do nothing. Add a clear `# TODO: Week 4` comment and a log line so it is obvious these are deferred, not broken.

### P2 — Address before Week 5

5. **Increase unit test coverage**: Tests cover only `sigmoid`, `safe_float`, and `compute_performance`. Add at minimum a smoke test for `_run_walk_forward_backtest()` (one fold, mocked data) and `decision_node` (known prob_up → expected action).

---

## Next-Phase Readiness Statement

**Week 1 is 80% complete.** The LangGraph loop, 300+ feature pipeline, LightGBM baseline, Optuna tuning, and SHAP extraction are all functional. However, the results are unvalidated due to pre-fix reports, and the walk-forward signal is currently negative — the most important indicator.

**Week 2 is partially started.** SHAP rule extraction works. The risk node needs to become a proper standalone graph node.

**Weeks 3–6 should not begin** until a fresh post-fix run confirms a positive walk-forward signal, even if modest.

---

## Critical Files Reference

| File | Purpose | Key Functions |
|---|---|---|
| `src/agents/nodes.py` | All core computation (1,113 lines) | `_run_walk_forward_backtest()`, `_run_optuna_tune()`, `_derive_shap_rule()`, `evaluate_node()`, `decision_node()` |
| `src/agents/graph.py` | LangGraph StateGraph definition | Where `risk_node` should be added |
| `src/agents/state.py` | AgentState TypedDict | Will need new fields for risk node |
| `config/config.yaml` | All tunable parameters | Thresholds, walk-forward windows, model params |
| `reports/week1_report.md` | Latest results (pre-fix, suspect) | Reference only |
| `tests/test_nodes.py` | Unit tests (limited coverage) | Extend with walk-forward + decision smoke tests |
| `run_mvp.py` | Entry point | Re-run to generate post-fix baseline report |

---

## Verification Checklist (Post-Fix)

- [ ] `pytest tests/` passes with zero failures
- [ ] `python run_mvp.py` completes 100 cycles without error
- [ ] Fresh report shows walk-forward out-of-sample accuracy > 51% on majority of folds
- [ ] Fresh report live sim Sharpe is computed strictly on held-out (post-training) bars
- [ ] No `NaN` or `inf` values in Sharpe computations in the report
- [ ] `risk_node` added as a named node in `graph.py` before Week 3 work begins
