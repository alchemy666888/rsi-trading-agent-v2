You are acting as a senior quantitative trading systems reviewer and ML research auditor.

Your task is to review the implementation quality of the repository below against its stated development plan, with special focus on time-series correctness, leakage prevention, simulation integrity, artifact quality, and phase-gate readiness.

Repository:
https://github.com/alchemy666888/rsi-trading-agent-v2

Primary review objective:
Determine whether the implementation quality is good enough for the project to advance to the next phase, based on the stated development plan and the current artifact evidence.

You must review:
1. docs/development-plan.md
2. config/config.yaml
3. run_mvp.py
4. src/agents/data.py
5. src/agents/features.py
6. src/agents/modeling.py
7. src/agents/evaluation.py
8. src/agents/risk.py
9. src/agents/nodes.py
10. src/agents/graph.py
11. tests/test_pipeline.py
12. artifacts/<user will provide as run_id>/*
   especially:
   - report.md
   - benchmark_metrics.json
   - run_metrics.json
   - split_metadata.json
   - strategy_params.json
   - optimization_events.json if present
   - trades / returns / equity artifacts if present

Important review context:
- The repo is a research-oriented BTC/USDT 15-minute prototype, not a production trading system.
- The current development plan says the project is at a Week 1 research milestone:
  - feature engineering over BTC data
  - LightGBM directional baseline
  - validation-only threshold calibration
  - held-out out-of-sample LangGraph simulation
  - purged walk-forward benchmark
  - artifact persistence and simple simulation risk controls
- The plan explicitly says the repo is NOT yet claiming:
  - self-improving learning loops
  - RL policy optimization
  - LoRA or fine-tuning
  - RAG / vector memory
  - live trading or paper trading
- The “fine-tuning gate” says future learning work should only start after:
  - stable snapshot-based runs
  - reproducible held-out and walk-forward metrics
  - persisted artifacts across multiple historical windows
  - clear benchmark regression tracking

Current run to assess:
- Run ID: <user will provide as run_id>
- Asset/timeframe: BTC/USDT (15m)
- Dataset source: exchange:binance
- Dataset span: 2026-04-01T02:00:00Z to 2026-04-11T11:45:00Z
- Total bars: 1000
- Split bars: train=600, validation=200, OOS=199
- Held-out KPIs:
  - Sharpe: about -31.26
  - Max drawdown: about 5.15%
  - Total return: about -4.91%
  - Trade count: 43 in report.md headline section, but run_metrics.json shows trade_count=87, so check consistency carefully
- Thresholds:
  - long_threshold ≈ 0.52143
  - short_threshold ≈ 0.47944
- Validation best score is strongly negative
- Walk-forward overall mean Sharpe is also strongly negative (about -26.37)
- Overfit warning was not triggered in the artifact report even though both held-out and benchmark performance are poor

Critical instruction:
Do not give a superficial code review.
Trace all timeframe-sensitive logic end to end:
- base timeframe assumptions
- feature windows
- multi-timeframe aggregation logic
- target construction
- train / validation / OOS split logic
- purge-gap walk-forward logic
- prediction timestamp vs execution timestamp
- signal delay handling
- funding interval logic
- annualization logic for Sharpe
- stop-loss / take-profit timing
- turnover and slippage timing
- held-out simulation cursor advancement
- whether OOS bars are truly held out from fitting and threshold selection
- whether walk-forward fold construction is temporally valid
- whether any field in config is declared but not actually used

Specifically verify:
A. Development-plan compliance
- Is the implementation truly aligned with the current planned stage?
- Did the implementation accidentally drift into claims or architecture not justified by the plan?
- Are any required plan items missing or only partially implemented?

B. Time-series and leakage integrity
- Any lookahead leakage in feature engineering, target labeling, asof joins, resampling, threshold calibration, SHAP usage, walk-forward folds, or OOS execution?
- Any leakage from resampled higher-timeframe features due to group labeling or backward joins?
- Is the target “next close > current close” used in a causally valid way?
- Is validation used only for calibration and not contaminated by OOS?
- Is OOS simulation genuinely forward-style?

C. Signal timing and execution semantics
- The code/config mentions signal_delay_bars=1. Verify whether this is truly implemented everywhere or only documented.
- Check whether the evaluation path and the calibration simulator implement the same timing semantics.
- Check whether the LangGraph runtime behavior matches the batch simulation logic used for calibration and walk-forward.
- Identify any mismatch between stated “next-bar execution” and actual implementation.

D. Risk and simulation realism
- Are slippage, turnover limits, stop loss, take profit, drawdown pause, and volatility blocking implemented coherently?
- Are they enforced at the correct timestamp / bar boundary?
- Are there unrealistic assumptions that are acceptable for Week 1 research but must block progression to the next phase?
- Does the funding logic match the timeframe correctly?

E. Artifact and metrics integrity
- Are report.md, run_metrics.json, and other persisted artifacts internally consistent?
- Explain any inconsistencies such as trade_count mismatch between report headline and run_metrics/trade logs if present.
- Assess whether artifact persistence is sufficiently complete and reliable for research reproducibility.
- Assess whether current artifact quality is enough for regression tracking across future runs.

F. Test coverage quality
- Do tests actually protect the most failure-prone time-series and simulation semantics?
- What important failure modes are currently untested?
- Are there hidden gaps between what tests assert and what the runtime actually does?

G. Phase-gate decision
Decide whether the project should:
1. advance to the next phase now,
2. advance conditionally after a small set of fixes,
3. remain in the current phase until material research-quality issues are resolved.

Your answer must be structured exactly as follows:

# 1. Executive Verdict
Give a clear verdict:
- Advance
- Conditional Advance
- Do Not Advance

Then give a 5-10 sentence explanation.

# 2. What Stage the Repository Is Actually In
State the actual maturity level of the implementation, not the aspirational one.
Explain whether it is:
- prototype wiring only
- research baseline
- reproducible research baseline
- paper-trading candidate
- production candidate
or something similar.

# 3. Development Plan Compliance Review
Create a table with columns:
- Plan Item
- Implemented?
- Evidence
- Quality
- Risks / Gaps

Use the plan as the source of truth.

# 4. End-to-End Timeframe-Sensitive Logic Audit
Walk through the entire temporal chain:
data load -> feature build -> target label -> split construction -> model training -> validation calibration -> walk-forward folds -> OOS graph execution -> metrics/artifacts

For each stage:
- describe intended behavior
- describe actual behavior in code
- say whether it is temporally valid
- identify any discrepancy or hidden assumption

# 5. Implementation Quality Findings
Group findings into:
- Critical
- Major
- Minor
- Nice-to-have

Be concrete and file-level.
For each finding include:
- why it matters
- exact files/functions involved
- whether it blocks phase advancement

# 6. Evidence From Current Run Artifacts
Interpret the <user will provide as run_id> artifacts carefully.
Discuss:
- what the strongly negative validation, walk-forward, and held-out metrics imply
- whether the current implementation is behaving consistently even if performance is bad
- whether poor results look like “expected research outcome” vs “implementation defect” vs “possible metric/reporting inconsistency”

# 7. Hidden or Non-Obvious Issues
Call out subtle issues such as:
- config fields that are not actually enforced
- discrepancies between documentation and code
- metric/report mismatches
- misleading naming
- logic that is correct for 15m but fragile if timeframe changes
- places where future contributors could accidentally introduce leakage

# 8. Test Coverage Assessment
State:
- what is well covered
- what is weakly covered
- the 10 highest-value additional tests to add next

# 9. Phase-Gate Recommendation
Give a decision on whether it can move to the next phase.
Define the next phase explicitly.
Then provide:
- Must-fix before advancing
- Should-fix soon after
- Safe to defer

# 10. Priority Action Plan
Provide a prioritized action list with:
- Priority
- Action
- Why
- Estimated effort
- Blocks next phase? (yes/no)

# 11. Final Bottom Line
A concise final paragraph saying whether this codebase is good enough to move forward and under what conditions.

Review standards:
- Be strict.
- Prefer evidence over assumption.
- Do not confuse “pipeline runs” with “implementation is phase-ready.”
- Distinguish research underperformance from engineering defects.
- Treat time-series leakage and timing mismatches as highest-severity risks.
- If something is ambiguous, say so explicitly.
- Cite specific file names and functions throughout.