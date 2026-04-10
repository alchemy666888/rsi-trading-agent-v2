# RSI Trading Agent v2 — Review Against Vibe Coding Development Plan

**Reviewer:** Codex (GPT-5.3-Codex)
**Date (UTC):** 2026-04-10
**Plan reviewed:** `docs/development-plan.md`

## 1) Summary of findings

- The repo implements a functioning LangGraph cyclic trading agent (`data → predict → risk → decision → evaluate → optimize`) with clear module separation and practical reporting output. This aligns well with the MVP spirit of shipping a simple loop first.  
- The implementation has evolved beyond a pure “RSI bot” into a LightGBM + large TA feature pipeline. This is more capable, but it is no longer “minimal code” in the strict vibe-coding sense.
- Data ingestion is present in two modes: local cache CSV and exchange pull via CCXT; there is also a dedicated fetch script for real Binance 1m BTC data.
- Feedback loops are strong: structured markdown report output, optimization event logs, SHAP rule extraction, and runtime metadata hashing/commit stamping.
- Current reported strategy performance is not deployment-ready. Latest report (`reports/week3_report.md`, generated 2026-04-10) shows negative return and weak win rate, with Sharpe intentionally clamped to `0.0` because samples are below reliability threshold (<1000 returns).
- Validation coverage is partial: there are good unit tests around decision/risk/performance helpers, but the current test suite fails (`5 failed`) due to configuration/test drift and a data-node mocking mismatch.

---

## 2) Repository analysis (structure, key files, dependencies, outputs)

### Core structure

- `run_mvp.py`: entrypoint; builds graph, initializes state, runs loop, persists SHAP rules, writes report.
- `src/agents/graph.py`: defines the cyclic LangGraph workflow.
- `src/agents/state.py`: shared `AgentState` TypedDict schema.
- `src/agents/nodes.py`: nearly all trading logic (data load/feature engineering/model train/predict/risk/decision/evaluate/optimize/walk-forward).
- `config/config.yaml`: runtime/model/simulation/walk-forward settings.
- `scripts/fetch_real_data.py`: paginated Binance OHLCV fetch utility for real data.
- `reports/*.md`: generated run reports.
- `tests/test_nodes.py`: unit tests (core math + node behavior).

### Dependencies

`pyproject.toml` shows a pragmatic stack aligned with the plan: `langgraph`, `polars`, `ta-lib`, `vectorbt`, `optuna`, `ccxt`, `shap`, and `lightgbm`.

### Existing results

- `reports/mvp_report.md` (older MVP report, 2026-04-06).
- `reports/week2_report.md` (post-walk-forward enhancements, 2026-04-09).
- `reports/week3_report.md` (latest, 2026-04-10; exchange-pulled ~1000 rows, 162 cycles).

---

## 3) Plan-to-implementation comparison (phase-by-phase)

Legend: **🟢 Green / 🟡 Yellow / 🔴 Red**

| Phase | Plan intent | Current implementation assessment | Score |
|---|---|---|---|
| MVP weekend | End-to-end LangGraph loop with simulated trading + report | Fully present and runnable; loop architecture and report generation are in place. | 🟢 |
| Data ingestion | Historical BTC 1m ingestion, configurable source | Implemented via local CSV cache + CCXT pull; plus script to fetch months of real data. | 🟢 |
| Signal generation | RSI-led baseline, quick intuitive logic | RSI is present, but system now relies mainly on a high-dimensional LightGBM feature set. Powerful, but less “minimal.” | 🟡 |
| Backtesting/validation | Walk-forward + clean evaluation path | Walk-forward logic + embargo checks exist and are explicitly enforced; however recent run has 0 folds due to too-small dataset and tests currently fail. | 🟡 |
| Optimization loop | Simple recurring self-improvement | Optuna threshold tuning + SHAP rule extraction + replay append exist; LoRA hook is stubbed (warning only). | 🟡 |
| Deployment readiness | Step toward paper/live with guardrails | No live order execution module in graph yet; results are currently negative and statistically weak. | 🔴 |

### Vibe-coding spirit check

- **Rapid prototyping:** strong (single-file-heavy nodes, fast iteration, generated reports). ✅
- **Minimal but expressive code:** mixed (expressive yes, minimal no; `nodes.py` is large and complex). ⚠️
- **Clear feedback loops:** strong (logs + markdown + metadata + optimization history). ✅
- **Avoid over-engineering:** mixed (300+ features and broad TA surface may be beyond MVP necessity). ⚠️

---

## 4) Validity assessment of results/report

### Latest report reviewed

- Primary reference: `reports/week3_report.md` generated **2026-04-10T02:02:00Z**.
- Data provenance in report: `exchange_pull`, `binance:BTC/USDT:1m`, row count `1000`.

### Methodology soundness checks

**What is sound:**
- Target construction uses forward return shift (`target_up`) and model training split is separated from simulation cursor.
- Embargo-gap validation is explicit and strict (`embargo_gap >= max_feature_lag + label_horizon + 1`).
- Transaction costs are included (`slippage_bps` in both simulation and vectorbt folds).
- Funding rate cost hook exists and is periodically applied.
- Sharpe reliability guard prevents misleading annualized Sharpe with too-few samples.

**What limits result validity right now:**
1. **Insufficient data window for robust walk-forward in latest run** (1000 rows with current `train/test/step/embargo` settings yields no folds in report).  
2. **Sharpe is not informative in latest run** by design (`0.0` because only 162 returns < 1000 threshold).  
3. **Current strategy outcomes are negative** (total return -1.77%, win rate 42.59% in week3 report), so no demonstrated edge yet.  
4. **Test suite instability** (5 failures) reduces confidence in regression safety.

### Plausibility of reported metrics

- Given only 162 evaluated cycles and explicit Sharpe guard, `Sharpe = 0.0` is plausible and expected.
- Negative total return with sub-50% win rate is consistent with reported threshold settings and sparse sample.
- Reported WF metrics being all zeros in latest run is consistent with “0 folds” condition, not evidence of good performance.

### Missing critical validations

- No passing end-to-end regression test that asserts non-empty walk-forward folds on production-like data volume.
- No robust out-of-sample acceptance gate (e.g., minimum folds + minimum duration + confidence intervals).
- No cost sensitivity sweep (slippage/funding stress) surfaced in report.

---

## 5) Go / No-go recommendation

## **No-go** (for next deployment-oriented phase)

The implementation quality is promising and aligned with iterative vibe development, but current evidence is not reliable enough to proceed to paper/live deployment phases.

### Blockers to clear before go-ahead

1. Increase dataset size for standard runs (e.g., real BTC 1m multi-month data via `scripts/fetch_real_data.py`) and ensure walk-forward folds are non-zero.
2. Fix failing tests in `tests/test_nodes.py` so CI baseline is green.
3. Add an explicit readiness gate in code/report:
   - minimum returns >= 1000,
   - minimum walk-forward folds >= 5 (both rolling and/or expanding),
   - positive out-of-sample risk-adjusted performance after costs.
4. Keep optimization simple but bounded (avoid threshold drift without evidence of real edge).

---

## 6) Concrete next-phase readiness statement

**Readiness status: NOT READY for deployment phase (No-go).**

The project is **ready for the next internal iteration** (data scale-up + validation hardening), but **not ready for paper/live trading progression** until the blockers above are resolved and a reproducible out-of-sample edge is demonstrated.
