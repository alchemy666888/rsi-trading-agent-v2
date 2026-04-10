# RSI Trading Agent v2 — Development Plan Review Report

**Reviewer:** Claude (automated code review)  
**Date:** 2026-04-10  
**Supersedes:** docs/review-report.md (2026-04-09)  
**Branch reviewed:** `main` (HEAD: `3d9ccd2`)  
**Development plan:** `docs/development-plan.md`

---

## 1. Executive Summary

The repository implements a LangGraph-orchestrated, LightGBM-driven BTC trading agent with 300+ engineered features, Optuna threshold tuning, SHAP rule extraction, walk-forward validation, and a standalone risk node. The infrastructure is well-conceived and demonstrates clear "vibe coding" discipline — iterative, expressive, grounded in feedback loops.

**24-hour delta from April 9 review:** Four PRs were merged addressing embargo gap validation, test coverage expansion, report metadata, and a real-data fetch script. These are genuine infrastructure improvements. However, the **P0 blocker remains unresolved**: the training data is still 5,000 bars of synthetic GBM (random walk) with no exploitable market structure. No new metrics have been generated.

**Bugs found in this review cycle:**
- 5 test failures (3 from outdated embargo_gap in fixtures, 1 from stale monkeypatch return type, 1 from misplaced test method) — **fixed**
- 2 blocks of dead/unreachable code in `_load_historical_btc_data` — **removed**

**Recommendation: NO-GO** for advancing to Week 3+ (RL / LoRA / Paper Trading). The single highest-leverage action is running `scripts/fetch_real_data.py` to obtain real BTC market data.

---

## 2. Changes Since Prior Review (April 9)

| PR | Commit | Change | Assessment |
|---|---|---|---|
| #5 | `a9a38c1` | Real data pipeline (`scripts/fetch_real_data.py`), Optuna widening, SHAP KB accumulation, funding rate, configurable `label_horizon`/`max_feature_lag` | Good infrastructure; not yet exercised |
| #6 | `c43fede` | Test coverage expansion (cursor embargo, walk-forward validation, slippage stress) | Strengthens correctness guarantees |
| #7 | `71047cb` | Report metadata (git hash, config hash, data source, timestamps) | Improves auditability and reproducibility |
| #8 | `d3f7a61` | Embargo gap validation via `_required_embargo_gap()` / `_validate_embargo_gap()` | Prevents future leakage bugs |

**New issues introduced by these PRs:**
1. Dead code at `nodes.py` lines 568–574 and 637–643 — unreachable `_build_features` calls after `return` statements (removed in this review)
2. Three walk-forward tests used `embargo_gap: 10` which no longer passes the new validation requiring ≥ 62 (fixed in this review)
3. `TestDataNode.test_cursor_starts_after_training_plus_embargo` — monkeypatch returned 1 value but `_load_historical_btc_data` now returns a 2-tuple (fixed in this review)
4. `TestDataNode.test_raises_when_embargo_gap_below_required_threshold` — called `self._make_state()` which does not exist on `TestDataNode` (fixed in this review)

---

## 3. Architecture Overview

```
START → data → predict → risk → decision → evaluate → optimize
                                                          ↓
                                                   (if done) → END
                                                   (else)    → data
```

| Component | File | Lines | Purpose |
|---|---|---|---|
| State | `src/agents/state.py` | 55 | `AgentState` TypedDict with 26 fields |
| Graph | `src/agents/graph.py` | 41 | 6-node LangGraph `StateGraph` with cyclic edge |
| Nodes | `src/agents/nodes.py` | ~1,310 | All trading logic: features, model, prediction, risk, decision, evaluation, optimization |
| Entry point | `run_mvp.py` | 359 | Config loading, graph invocation, report generation, SHAP persistence |
| Data fetcher | `scripts/fetch_real_data.py` | 129 | CCXT Binance paginated OHLCV download |
| Tests | `tests/test_nodes.py` | ~480 | 35 tests covering core functions |

---

## 4. "Vibe Coding" Alignment Assessment

The development plan explicitly calls for rapid prototyping, minimal/expressive code, clear feedback loops, and no over-engineering.

| Principle | Assessment |
|---|---|
| **Rapid prototyping** | **Strong.** MVP delivered in one weekend, Week 1/2 features added within 4 days. Commit history shows daily iterations. |
| **Minimal/expressive code** | **Good.** `graph.py` is 41 lines. State is a single TypedDict. `nodes.py` at ~1,310 lines is dense but well-organized. |
| **Clear feedback loops** | **Strong.** Markdown reports with walk-forward metrics, SHAP rules, optimization events, feature importances. Run metadata (git hash, config hash) added for reproducibility. |
| **No over-engineering** | **Good.** No premature abstraction — no database, no microservices, no Dagster yet. SHAP rules persist via a simple JSON file. Risk node is a clean extraction, not an over-built framework. |
| **Ship first, improve daily** | **Partially met.** Daily improvements visible in commit history, but the "ship" part requires real data that has not yet been produced. |

---

## 5. Phase Alignment Score

| Phase | Plan Deliverable | Actual Status | Score |
|---|---|---|:---:|
| **MVP (Weekend)** | Working LangGraph loop, 100-cycle sim, LightGBM baseline, dummy Optuna, Markdown report | Fully delivered. 6-node cyclic graph, LightGBM trained, Optuna every 10 cycles, auto-generated Markdown with run metadata. | 🟢 |
| **Week 1: Feature Eng + LightGBM** | 300+ features, multi-timeframe, walk-forward backtest | 95% complete. 300+ TA-Lib features across 1m/5m/15m/1h. Expanding + rolling walk-forward with validated 62-bar embargo gap. Missing: validated on real data. | 🟢 |
| **Week 2: Risk Node + SHAP** | Dedicated risk node, 50+ SHAP rules in knowledge base | 80% complete. `risk_node` is a standalone graph node with stop-loss and regime-aware threshold widening. SHAP rules now accumulate across sessions (capped at 100) and persist to disk. Not yet tested with enough real-data cycles to produce 50+ unique rules. | 🟡 |
| **Week 3: RL Warm-up (PPO)** | PPO policy replacing rule-based decision | Not started. No RL/PPO code anywhere. | 🔴 |
| **Week 4: LoRA Fine-tuning** | Daily recurring LoRA adapter pipeline | `update_lora_adapter()` is an explicit `# TODO: Week 4` stub with a warning log. No LLM, no PEFT, no Dagster. | 🔴 |
| **Week 5: Paper Trading** | Live CCXT Pro, real-tick execution | Agent runs on cached CSV only. No live execution path exists. | 🔴 |
| **Week 6: Memory + HITL** | Pinecone/Weaviate, Neo4j, replay buffer, HITL | All state is in-process/session-scoped. No persistence layer or HITL mechanism (aside from SHAP rules JSON). | 🔴 |

---

## 6. Validity Assessment of Results/Reports

### Report Status

| Report | Generated | Verdict |
|---|---|---|
| `reports/mvp_report.md` | 2026-04-06 | **INVALID** — pre-dates critical bug fixes (in-sample bias, embargo off-by-one) |
| `reports/week1_report.md` | 2026-04-07 | **INVALID** — pre-dates critical bug fixes |
| `reports/week2_report.md` | 2026-04-09 | **VALID** — generated post-fix; sole trustworthy baseline |
| `reports/week3_report.md` | Not generated | Config targets this path, but `btc_1m_real.csv` does not exist |

### Post-Fix Metrics (`week2_report.md`) — Sole Trustworthy Baseline

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
- RSI via `talib.RSI()` at 9 periods (5/8/10/14/20/30/50/100/200) — standard Wilder's smoothing ✓
- Sharpe annualization: `√525,600` for 1-minute bars — mathematically correct ✓
- Slippage: 5 bps per position change via `vbt.Portfolio.from_signals(fees=...)` ✓
- Walk-forward: expanding + rolling with 62-bar embargo gap (validated at runtime) ✓
- Post-fix simulation cursor starts after `split_idx + embargo_gap`, preventing in-sample evaluation ✓
- Funding rate: 0.01% per 8h (480 bars) modeled in `evaluate_node` ✓

**Remaining methodology concerns:**

| # | Issue | Severity |
|---|---|---|
| 1 | **Synthetic training data**: `btc_1m_synthetic.csv` is GBM-generated random walk. No ML model can learn a real signal from random data — this is the primary explanation for zero edge. | **Critical** |
| 2 | **Optuna threshold drift**: `long_threshold` converges to ~0.71 by cycle 50 and remains pinned for all 1,050 remaining cycles. The model effectively never enters long positions. Activity penalty was added in PR #5 but did not break this convergence on synthetic data. | High |
| 3 | **Baseline model trained without embargo**: `_train_lightgbm_baseline()` uses `train_split_ratio=0.7` with no gap between train and test. Walk-forward results (which have their own embargo) are unaffected, but live simulation uses this biased baseline. | Medium |
| 4 | **NaN zero-fill**: After dropping 250 warmup rows, remaining NaN/null values are filled with 0.0. Forward-fill or indicator-specific defaults would be safer. | Low |
| 5 | **Fixed position sizing**: All positions are 1 or -1; no Kelly or volatility-adjusted sizing. | Low |

---

## 7. Code Quality Assessment

**Strengths:**
- Clean node separation with single `AgentState` TypedDict communication
- Comprehensive 300+ feature bundle across 4 timeframes
- Walk-forward backtesting with both expanding and rolling modes
- Embargo gap validated at runtime with clear `ValueError` messages
- 35 tests covering: `_sigmoid`, `_safe_float`, `_compute_performance`, `decision_node` (parametrized), `risk_node`, walk-forward (smoke + boundary + embargo + stress), `data_node` (cursor + validation)
- Run metadata in reports (git hash, config hash) enables reproducibility
- SHAP rules persist across sessions via `data/shap_rules.json`

**Issues fixed in this review:**

| # | Issue | Fix |
|---|---|---|
| 1 | Dead code: unreachable `_build_features()` calls after `return` in both branches of `_load_historical_btc_data` | Removed (nodes.py lines 568–574, 637–643) |
| 2 | 3 walk-forward tests used `embargo_gap: 10` incompatible with validation requiring ≥ 62 | Updated fixture to `embargo_gap: 62`, `n=283` with recalculated boundaries |
| 3 | `test_cursor_starts_after_training_plus_embargo` — monkeypatch returned 1 value, function now returns 2-tuple | Updated fake to return `(data, metadata)`, added `model` config |
| 4 | `test_raises_when_embargo_gap_below_required_threshold` — called `self._make_state()` on wrong class | Built inline state instead of relying on missing method |

---

## 8. Go / No-Go Recommendation

### NO-GO — Do not advance to Week 3+ (RL / LoRA / Paper Trading)

The infrastructure scaffolding is solid and maturing. The vibe coding discipline is well executed. However, the **core signal is absent** because the training data is synthetic GBM. Advancing to RL, LoRA, or paper trading atop a strategy that loses 9% on unseen random-walk data would be premature.

**Status unchanged from April 9.** The fundamental blocker (synthetic data) persists. The 24-hour delta is positive in infrastructure quality (embargo validation, test coverage, report metadata, SHAP persistence, real-data fetcher script) but the fetcher has not been run and no new metrics exist.

---

## 9. Blockers and Required Actions

### P0 — Must resolve before further development

1. **Run `scripts/fetch_real_data.py` to obtain real BTC 1m OHLCV data**
   - `python scripts/fetch_real_data.py --months 6 --out data/btc_1m_real.csv`
   - Verify output has ~260,000 rows
   - This is the single highest-leverage action in the entire project

2. **Re-run `python run_mvp.py` on real data and generate `reports/week3_report.md`**
   - Gate criteria for Week 3: WF accuracy > 51% and positive mean Sharpe on ≥ 3 consecutive rolling folds
   - If accuracy stays below 51%, investigate: feature leakage, label construction, class balance, and try `label_horizon: 5`

### P1 — Address before Week 3

3. **Verify SHAP rule accumulation produces 50+ unique rules**
   - Run a full 1,100-cycle simulation on real data and check `data/shap_rules.json`
   - Code generates one rule per optimization event (every 10 cycles = 110 events), but deduplication may reduce count

4. **Investigate Optuna threshold convergence on real data**
   - If `long_threshold` still pins to ~0.71, redesign the Optuna objective or search space
   - Consider multi-objective optimization (Sharpe + activity)

5. **Add embargo to baseline model training**
   - `_train_lightgbm_baseline()` uses `train_split_ratio=0.7` with no embargo between train and test
   - Walk-forward results are unaffected, but live simulation uses this baseline

### P2 — Address before Week 5

6. **Expand test coverage**: Add regression test that the walk-forward interval trigger fires at correct cycle counts; add multi-step simulation stop-loss integration test

7. **Improve funding rate model**: Accrue continuously rather than discretely every 480 bars

8. **Add position sizing**: Replace fixed 1/-1 with Kelly or volatility-adjusted sizing

---

## 10. Concrete Next-Phase Readiness Statement

| Phase | Status |
|---|---|
| **MVP** | Complete ✓ |
| **Week 1** | 95% complete — infrastructure validated; results pending real data |
| **Week 2** | 80% complete — risk_node + SHAP KB infrastructure done; rules not yet generated at scale on real data |
| **Week 3+** | **Should not begin** until real data produces a positive walk-forward signal |

**The single highest-leverage action is replacing synthetic GBM training data with real BTC 1m market data.** All other improvements are secondary until a genuine predictive edge is confirmed on unseen real data.

---

## 11. Critical Files Reference

| File | Key Functions / Purpose |
|---|---|
| `src/agents/nodes.py` | `data_node()`, `predict_node()`, `risk_node()`, `decision_node()`, `evaluate_node()`, `optimize_node()`, `_build_features()`, `_run_walk_forward_backtest()`, `_run_optuna_tune()`, `_derive_shap_rule()`, `_validate_embargo_gap()` |
| `src/agents/graph.py` | 6-node StateGraph with cyclic edge |
| `src/agents/state.py` | `AgentState` TypedDict (26 fields) |
| `config/config.yaml` | Change `local_cache_path` to real data; `embargo_gap: 62`; `fetch_limit: 260000` |
| `run_mvp.py` | Entry point, report generation, SHAP rules persistence |
| `scripts/fetch_real_data.py` | CCXT Binance paginated data fetcher |
| `reports/week2_report.md` | Valid post-fix baseline — WF Sharpe −116.44, Accuracy 48.71% |
| `tests/test_nodes.py` | 35 tests covering core functions (all passing) |

---

## 12. Verification Checklist (After Real Data Integration)

- [ ] `scripts/fetch_real_data.py` completes and produces 260,000+ rows
- [ ] `data/btc_1m_real.csv` exists and matches config path
- [ ] `pytest tests/` — all 35 tests pass
- [ ] `python run_mvp.py` — completes without NaN/inf in any metric
- [ ] Walk-forward accuracy > 51% on majority of rolling folds
- [ ] Walk-forward mean Sharpe is positive (even modestly)
- [ ] `data/shap_rules.json` contains 20+ unique rules after full run
- [ ] `long_threshold` does not pin to a single value across all optimization cycles
- [ ] Live sim Sharpe computed strictly on post-training bars
- [ ] `update_lora_adapter` warning log visible in output
- [ ] `risk_node` appears as a named node in graph and runs on every cycle
