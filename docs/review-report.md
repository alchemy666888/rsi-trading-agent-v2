# RSI Trading Agent v2 — Development Plan Review Report

**Reviewer:** Claude (automated code review)
**Date:** 2026-04-09
**Branch reviewed:** `claude/review-development-plan-N5ztI`
**Development plan:** `docs/development-plan.md`

---

## Summary of Findings

The repository implements a LangGraph-orchestrated, LightGBM-driven BTC trading agent with 300+ engineered features, Optuna threshold tuning, SHAP rule extraction, walk-forward validation, and a newly extracted standalone risk node. The infrastructure is well-conceived and demonstrates clear "vibe coding" discipline — iterative, expressive, grounded in feedback loops (Markdown reports, SHAP rules, Optuna events).

**Critical context on prior reports**: Two bugs were patched on April 9 (in-sample evaluation bias; walk-forward embargo gap off-by-one). All reports generated before April 9 (`mvp_report.md`, `week1_report.md`) contain invalid metrics and must be ignored. The sole trustworthy baseline is `reports/week2_report.md` (generated post-fix, April 9).

**Signal status**: The post-fix walk-forward out-of-sample results show no predictive edge (WF accuracy 48.71%, mean Sharpe −116.44). The most probable root cause is that the training dataset is 5,000 bars of synthetic Geometric Brownian Motion data — a pure random walk with no exploitable market structure. No ML model can learn a real edge from random-walk data.

**Changes applied in this review cycle**:
- `risk_node` extracted as a dedicated 6-node graph position (between `predict` and `decision`)
- `decision_node` updated to respect `risk_params` (stop-loss override, regime-aware threshold widening) and now tracks `entry_price`
- `update_lora_adapter` stub replaced with explicit `# TODO: Week 4` warning log
- `risk_params` field added to `AgentState`
- Test coverage expanded: `risk_node`, `decision_node` (parametrized thresholds + stop-loss + regime), `_run_walk_forward_backtest` (smoke)

---

## Phase Alignment Score

| Phase | Plan Deliverable | Actual Status | Score |
|---|---|---|:---:|
| **MVP (Weekend)** | Working LangGraph loop, 100-cycle sim, LightGBM baseline, dummy Optuna, Markdown report | Fully delivered. 6-node cyclic graph (`data→predict→risk→decision→evaluate→optimize`), LightGBM trained once, 15-trial Optuna every 10 cycles, auto-generated Markdown. | 🟢 |
| **Week 1: Feature Eng + LightGBM** | 300+ features, multi-timeframe, walk-forward backtest | 300+ TA-Lib + lag-60 + rolling-stat features across 1m/5m/15m/1h. Expanding + rolling walk-forward with 61-bar embargo gap. Matches spec exactly. | 🟢 |
| **Week 2: Risk Node + SHAP** | Dedicated risk node, 50+ SHAP rules in knowledge base | `risk_node` now a standalone graph node with stop-loss and regime-aware threshold widening. SHAP rule extraction works (`_derive_shap_rule()`). Only single latest rule stored — no structured rule KB yet. | 🟡 |
| **Week 3: RL Warm-up (PPO)** | PPO policy replacing rule-based decision | Not started. No RL/PPO code anywhere. | 🔴 |
| **Week 4: LoRA Fine-tuning** | Daily recurring LoRA adapter pipeline | `update_lora_adapter()` is an explicit `# TODO: Week 4` stub with a warning log. No LLM, no PEFT, no Dagster. | 🔴 |
| **Week 5: Paper Trading** | Live CCXT Pro, real-tick execution | Agent runs on cached CSV only. No live execution path exists. | 🔴 |
| **Week 6: Memory + HITL** | Pinecone/Weaviate, Neo4j, replay buffer, HITL | All state is in-process/session-scoped. No persistence layer or HITL mechanism. | 🔴 |

---

## Validity Assessment of Results/Reports

### Report Status

| Report | Generated | Verdict |
|---|---|---|
| `reports/mvp_report.md` | 2026-04-06 | **INVALID** — pre-dates both critical bug fixes |
| `reports/week1_report.md` | 2026-04-07 | **INVALID** — pre-dates both critical bug fixes |
| `reports/week2_report.md` | 2026-04-09 | **VALID** — generated post-fix; sole trustworthy baseline |

### Invalidated Metrics (`week1_report.md`, April 7)

| Metric | Reported Value | Issue |
|---|---|---|
| Live Sharpe | 66.76 | Computed on ~100 bars including training data (in-sample bias); code warns < 1,000 bars unreliable |
| Win Rate | 62% | Inflated by in-sample bias |
| WF Mean Sharpe | −362.98 | Suspect — embargo gap off-by-one caused train/test leakage |

### Post-Fix Metrics (`week2_report.md`, April 9) — Sole Trustworthy Baseline

| Metric | Value | Assessment |
|---|---|---|
| WF Mean Sharpe (overall) | **−116.44** | Deeply negative; no durable edge |
| WF Mean Accuracy | **48.71%** | Below 50%; model is worse than a coin flip |
| Live Sim Sharpe | **−73.49** | No edge on held-out bars |
| Live Total Return | **−9.08%** | Strategy is loss-making on unseen data |
| Win Rate | **46.55%** | More losing trades than winning |
| Max Drawdown | **10.24%** | Within plan's 12% cap, but on a losing strategy |

### Methodology Soundness

**Correctly implemented:**
- RSI via `talib.RSI()` at 9 periods (5/8/10/14/20/30/50/100/200) ✓
- Sharpe annualization: `√525,600` for 1-minute bars — mathematically correct ✓
- Slippage: 5 bps per position change via `vbt.Portfolio.from_signals(fees=...)` ✓
- Walk-forward: expanding + rolling with 61-bar embargo gap (= max_lag + 1) ✓
- Post-fix simulation cursor starts after `split_idx + embargo_gap`, preventing in-sample evaluation ✓

**Remaining issues:**

| # | Issue | Severity |
|---|---|---|
| 1 | **Synthetic training data**: `btc_1m_synthetic.csv` is GBM-generated with no market microstructure. No ML model can learn a real signal from a pure random walk — this is the primary explanation for zero edge. | **Critical** |
| 2 | **Optuna threshold drift**: `long_threshold` converges to ~0.71 across all 1,100 cycles; the model effectively never longs. The Optuna search space or objective needs re-examination once real data is in place. | High |
| 3 | **No funding rate costs**: Perpetuals carry ±0.01%/8h. Ignoring this overstates directional returns. | Medium |
| 4 | **Single SHAP rule stored**: Week 2 plan calls for a growing knowledge base of 50+ BTC-specific rules. Currently only the latest rule is retained in `AgentState`. | Medium |

---

## Go / No-Go Recommendation

### NO-GO — Do not advance to Week 3+ (RL / LoRA / Paper Trading)

The infrastructure scaffolding is solid and the vibe coding discipline is executed well. The core signal is absent because the training data is synthetic GBM. Advancing to RL, LoRA, or paper trading atop a strategy that loses 9% on unseen data would be premature.

---

## Blockers and Required Actions

### P0 — Must resolve before further development

1. **Replace synthetic data with real BTC 1m OHLCV data**
   - Switch `config.yaml → local_cache_path` to a real dataset (e.g., ~6 months Binance 1m ≈ 260,000 rows)
   - Set `fetch_limit: 260000` (or use a pre-downloaded Parquet/CSV)
   - Re-run `python run_mvp.py` to establish a valid `reports/week3_report.md` baseline

2. **Achieve positive walk-forward signal**
   - Gate for Week 3: WF accuracy > 51% and positive mean Sharpe on ≥ 3 consecutive rolling folds
   - If accuracy stays below 51% with real data, investigate feature leakage, label construction, and class imbalance before adding architectural complexity

### P1 — Address before Week 3

3. **Build a structured SHAP rule store**
   - Add a `shap_rules: list[str]` field to `AgentState` and accumulate rules across optimization events
   - Log unique rules to disk (JSON) so the knowledge base grows across sessions

4. **Investigate Optuna threshold convergence**
   - Long threshold consistently hits ~0.71, effectively blocking long entries
   - Expand or randomize the Optuna search space; add a constraint that `long_threshold < 0.65`

### P2 — Address before Week 5

5. **Expand unit tests further**
   - Add regression test that asserts `walk_forward_interval_cycles` config key triggers `_run_walk_forward_backtest` at the correct cycle counts
   - Add a test that verifies stop-loss fires correctly within a multi-step simulation loop

---

## Concrete Next-Phase Readiness Statement

| Phase | Status |
|---|---|
| **MVP** | Complete ✓ |
| **Week 1** | 90% complete — infrastructure done; results invalid due to synthetic data |
| **Week 2** | 70% complete — `risk_node` extracted and wired; SHAP extraction works; rule KB not yet persistent |
| **Week 3+** | **Should not begin** until real data produces a positive WF signal |

**The single highest-leverage action is replacing synthetic GBM training data with real BTC 1m market data.** All other improvements are secondary until a genuine predictive edge is confirmed on unseen real data.

---

## Graph Architecture (Post-Review)

```
START → data → predict → risk → decision → evaluate → optimize
                                                           ↓
                                                    (if done) → END
                                                    (else)    → data
```

## Critical Files Reference

| File | Key Functions |
|---|---|
| `src/agents/nodes.py` | `risk_node()`, `decision_node()`, `_run_walk_forward_backtest()`, `_run_optuna_tune()`, `_derive_shap_rule()`, `evaluate_node()`, `optimize_node()` |
| `src/agents/graph.py` | 6-node StateGraph — `risk` node now between `predict` and `decision` |
| `src/agents/state.py` | `AgentState` — includes `risk_params: dict[str, Any]` and `entry_price: float \| None` |
| `config/config.yaml` | Change `local_cache_path` to real data; bump `fetch_limit` to 260000+ |
| `reports/week2_report.md` | Valid post-fix baseline — WF Sharpe −116.44, Accuracy 48.71% |
| `tests/test_nodes.py` | Covers: `_sigmoid`, `_safe_float`, `_compute_performance`, `decision_node`, `risk_node`, `_run_walk_forward_backtest` |

---

## Verification Checklist (After Real Data Integration)

- [ ] `pytest tests/` — all tests pass (including new decision_node, risk_node, walk-forward smoke tests)
- [ ] `python run_mvp.py` — completes without NaN/inf in any metric
- [ ] Walk-forward accuracy > 51% on majority of rolling folds
- [ ] Walk-forward mean Sharpe is positive (even modestly)
- [ ] Live sim Sharpe computed strictly on post-training bars
- [ ] `update_lora_adapter` warning log visible in output (confirms stub is not silent)
- [ ] `risk_node` appears as a named node in `graph.py` and runs on every cycle
