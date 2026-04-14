# One-Round Fix + Fine-Tune Plan (No Repeat Round)

**Date (UTC):** 2026-04-14  
**Intent:** execute a single tightly-scoped round that clears phase blockers and provides a deterministic go/no-go outcome for advancing to the next development phase.

---

## 1) Why this plan is one-round capable

This plan is designed to avoid iterative churn by enforcing:

1. **Frozen scope** (only correctness + readiness gating + evidence reliability).  
2. **Pre-declared acceptance contracts** for each workstream.  
3. **Single-pass execution order** where each step unblocks the next without branching.  
4. **Hard stop criteria** that prevent moving forward with ambiguous quality.

No RL, no LoRA, no new model family, and no architecture expansion are included in this round.

---

## 2) Full end-to-end trace baseline (what must be treated as source of truth)

Before changing behavior, lock the current baseline from these artifacts and modules:

- Design baseline: `docs/development-plan.md`, `README.md`, `docs/code-review-report.md`.  
- Runtime and gate behavior: `run_mvp.py`, `config/config.yaml`.  
- Temporal pipeline implementation: `src/agents/data.py`, `features.py`, `modeling.py`, `evaluation.py`, `risk.py`, `nodes.py`, `graph.py`.  
- Regression expectations: `tests/test_pipeline.py`, `tests/test_nodes.py`, `tests/test_observability.py`.  
- Representative evidence bundle to benchmark against: `artifacts/20260412T053915Z-8e726856/*`.

The current architecture remains:

`data -> predict -> risk -> decision -> evaluate -> optimize`

This plan keeps that architecture unchanged and upgrades reliability/readiness guarantees around it.

---

## 3) Single-round objective and definition of done

### Objective

Produce **one merged change set** that makes the repository phase-gate trustworthy, reproducible, and non-ambiguous for advancement decisions.

### Definition of done (all must pass)

1. **Engineering validity = green**
   - runtime execution semantics consistent
   - KPI schema consistency enforced
   - tests green
2. **Research validity = explicit and enforceable**
   - clearly blocks advancement on triple-negative validation/WF/OOS performance
   - clearly blocks on degenerate calibration regimes
3. **Evidence sufficiency = green only when actually sufficient**
   - snapshot-only benchmark eligibility respected
   - WF sufficiency contract respected
   - multi-window history threshold respected
4. **Artifact integrity = internally consistent**
   - report/readiness/run-metrics consistency checks pass
   - no conflicting “advance” language when blockers exist
5. **Go/no-go output is deterministic**
   - same config + same snapshot window produces same gate verdict semantics.

If any item fails, this round is considered failed and does **not** advance.

---

## 4) Workstream plan (design → implementation → execution/testing)

## Workstream A — Gate semantics hardening (highest priority)

### Design intent
Make readiness outputs impossible to misinterpret by humans or downstream automation.

### Implementation tasks
1. Normalize readiness schema to include explicit booleans for each gate family and final decision rationale.
2. Enforce invariant: if `research_validity.green == false`, `phase_gate_decision` cannot be `advance`.
3. Enforce invariant: if any `hard_blockers` exist, `fine_tuning_gate_open` must be `false`.
4. Ensure `report.md` and `readiness.json` derive from the same readiness payload path (no divergent formatting logic).
5. Add explicit blocker-to-message mapping in artifacts for auditability.

### Test contract
- Unit tests for all truth-table branches of readiness decision.
- Regression tests verifying “triple negative => do_not_advance”.
- Regression tests verifying “hard blockers => fine_tuning_gate_open=false”.

### Acceptance
- No artifact can claim “advance” when research/evidence blockers are present.

---

## Workstream B — Artifact consistency and provenance lock

### Design intent
Prevent decision drift caused by inconsistent metrics across report surfaces.

### Implementation tasks
1. Add/strengthen cross-artifact consistency assertions among:
   - `run_metrics.json`
   - `report.json`
   - `report.md`
   - `readiness.json`
   - trade/completed-trade counts and exposures
2. Fail fast in persistence when schema-level contradictions are detected.
3. Record consistency-check results in persisted metadata for audit trace.

### Test contract
- Tests that intentionally inject mismatched counts and confirm persistence rejects them.
- Tests that validate no legacy KPI fields survive in final artifacts.

### Acceptance
- Artifact persistence must be atomic: either coherent bundle written or failure with explicit diagnostics.

---

## Workstream C — Temporal semantics parity lock (batch vs graph)

### Design intent
Guarantee identical timing semantics across calibration simulator, walk-forward simulator, and runtime graph path.

### Implementation tasks
1. Re-validate parity for:
   - signal delay
   - execution timestamp semantics
   - stop-loss / take-profit delayed flattening
   - slippage and turnover placement
   - funding cadence by timeframe bars
2. Add a parity-check helper fixture used by both runtime and simulation tests.
3. Introduce strict assertions for “same inputs -> equivalent returns/trades/equity outputs” across both paths.

### Test contract
- Expand parity tests to include risk-triggered closes and direction reversals under delay.
- Add edge cases around final-bar behavior and delayed queued actions.

### Acceptance
- Zero tolerated divergences between batch and runtime semantics in covered scenarios.

---

## Workstream D — Evidence sufficiency and reproducibility contract

### Design intent
Make advancement depend on reproducible evidence, not single-run luck.

### Implementation tasks
1. Ensure snapshot-only enforcement for regression-eligible runs remains strict.
2. Validate walk-forward sufficiency thresholds against configured minimum folds/bars per mode.
3. Ensure multi-window evidence uses robust window identity (snapshot/raw/hash/timestamps).
4. Add deterministic runbook for generating N-window evidence from row slices.

### Test contract
- Tests for insufficient folds, insufficient bars, and insufficient window diversity.
- Tests for exchange-mode rejection from readiness gate.

### Acceptance
- Advancement remains blocked until all evidence sufficiency checks pass by configuration.

---

## 5) Execution schedule (single PR, single validation pass)

### Phase 0 — Pre-flight lock (same day)
- Freeze config used for validation run(s).
- Record current failing/working baseline from tests and one reference artifact run.

### Phase 1 — Implement A + B
- Gate semantics + artifact consistency changes.
- Run targeted tests immediately.

### Phase 2 — Implement C + D
- Temporal parity and evidence sufficiency tightening.
- Run full test suite.

### Phase 3 — One full validation execution
- Run a single canonical snapshot-based end-to-end execution.
- Verify readiness + report + metrics coherence.
- Compare against acceptance checklist.

### Phase 4 — Decision
- If all acceptance criteria pass: mark **Conditional Advance -> Advance**.
- If any criterion fails: mark **Do Not Advance**, carry failure report forward; do not open next-phase work.

---

## 6) Command-level validation checklist (must be run in this round)

1. Static + unit regression:
   - `uv run pytest -q`
2. Focused timing/gate tests:
   - `uv run pytest -q tests/test_pipeline.py tests/test_nodes.py tests/test_observability.py`
3. End-to-end run:
   - `uv run python run_mvp.py --config config/config.yaml`
4. Artifact consistency spot-check:
   - compare `artifacts/<run_id>/report.md`, `report.json`, `run_metrics.json`, `readiness.json`, `trades.json`, `completed_trades.json`.

Round is complete only if all commands succeed and artifact checks pass.

---

## 7) Risk controls to prevent second tuning round

1. **Scope guard:** no new strategy logic, no new features, no model-family swaps.
2. **Time guard:** no exploratory parameter search outside existing config bounds.
3. **Quality guard:** failing any acceptance criterion blocks merge.
4. **Decision guard:** advancement decision must be tied to persisted readiness evidence.

---

## 8) Exact handoff output required at end of this one round

Produce a single final validation memo containing:

1. Gate verdict (`advance` or `do_not_advance`) with exact UTC timestamp.
2. Test summary (pass/fail counts + command outputs).
3. Artifact consistency statement (counts/metrics alignment confirmed).
4. Evidence sufficiency statement (folds/bars/windows thresholds met or not met).
5. Explicit next-phase recommendation:
   - If pass: begin next development phase.
   - If fail: remain in current phase with blocker list (no additional tuning round implied).

---

## 9) Expected outcome from this plan

After one disciplined round, the team should have a **final, non-ambiguous phase decision** supported by coherent artifacts and passing temporal correctness tests.

That satisfies the requirement to avoid repeated fix/fine-tune loops while still preserving research and engineering integrity.
