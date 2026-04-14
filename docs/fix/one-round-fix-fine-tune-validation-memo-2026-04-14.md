# One-Round Fix & Fine-Tune — Final Validation Memo

Authoritative plan: `docs/fix/one-round-fix-fine-tune-plan-2026-04-14.md`
Execution branch: `claude/one-round-fix-fine-tune-wJjSA`
Run ID under evaluation: `20260414T071934Z-986f1dd7`
Memo author: Claude (automated one-round execution)

---

## 1. Gate Verdict

- **Phase gate decision:** `do_not_advance`
- **Fine-tuning gate:** `closed` (`fine_tuning_gate_open = false`)
- **Round status:** `completed` (all Section 6 commands succeeded; all cross-artifact consistency checks passed)
- **UTC timestamp of gate evaluation (from `readiness.json` → `consistency_checks.json.generated_at_utc`):** `2026-04-14T07:21:28Z`

Rationale (from `readiness.decision_rationale`, verbatim):

1. `engineering_validity=green`
2. `research_validity=red (triple_negative_sharpe)`
3. `evidence_sufficiency=green`
4. `multi_window_evidence_sufficient=True`
5. `phase_gate_decision=do_not_advance`
6. `fine_tuning_gate_open=False`

Hard blocker driving the `do_not_advance` verdict:

- `triple_negative_sharpe` — "Validation, walk-forward, and held-out Sharpe are all negative." (validation Sharpe = -16.1193, walk-forward mean Sharpe = -56.6361, held-out Sharpe = -7.9580).

This matches the plan's Section 5 hard blocker catalogue; no hard blocker was suppressed or overridden.

---

## 2. Test Summary

Per plan Section 6 command checklist, executed from repository root on branch `claude/one-round-fix-fine-tune-wJjSA`:

| # | Command | Result |
|---|---------|--------|
| 1 | `uv run python -m pytest tests -q` | **57 passed, 0 failed, 0 errors** |
| 2 | Focused parity & gate subset (`tests/test_pipeline.py::test_parity_direction_reversal_under_delay`, `test_parity_stop_loss_close_under_delay`, `test_parity_take_profit_close_under_delay`, `test_parity_final_bar_no_execution_leak`, `test_evaluate_readiness_triple_green_produces_advance`, `test_evaluate_readiness_hard_blocker_closes_fine_tuning_gate`, `test_evaluate_readiness_rejects_insufficient_walk_forward_evidence`, `test_regression_windows_script_generates_deterministic_starts`, `test_artifact_consistency_check_rejects_mismatched_readiness`, `test_artifact_consistency_check_persists_metadata_on_success`) | **10 passed, 0 failed** |
| 3 | `uv run python run_mvp.py --config config/config.yaml` (end-to-end MVP run) | **succeeded** (Run ID `20260414T071934Z-986f1dd7`; process exit 0; readiness warning emitted by design) |
| 4 | Artifact consistency spot-check across `report.md`, `report.json`, `run_metrics.json`, `readiness.json`, `trades.json`, `completed_trades.json`, `consistency_checks.json` | **passed** (see Section 3) |

New tests added in this round (net +10 over pre-round baseline of 47):

- `tests/test_pipeline.py::test_parity_direction_reversal_under_delay`
- `tests/test_pipeline.py::test_parity_stop_loss_close_under_delay`
- `tests/test_pipeline.py::test_parity_take_profit_close_under_delay`
- `tests/test_pipeline.py::test_parity_final_bar_no_execution_leak`
- `tests/test_pipeline.py::test_evaluate_readiness_triple_green_produces_advance`
- `tests/test_pipeline.py::test_evaluate_readiness_hard_blocker_closes_fine_tuning_gate`
- `tests/test_pipeline.py::test_evaluate_readiness_rejects_insufficient_walk_forward_evidence`
- `tests/test_pipeline.py::test_regression_windows_script_generates_deterministic_starts`
- `tests/test_pipeline.py::test_artifact_consistency_check_rejects_mismatched_readiness`
- `tests/test_pipeline.py::test_artifact_consistency_check_persists_metadata_on_success`

---

## 3. Artifact Consistency Statement

All persisted files for Run ID `20260414T071934Z-986f1dd7` are cross-coherent per the atomic-persistence contract introduced by Workstream B. The check is programmatically enforced inside `persist_run_artifacts` by `_run_artifact_consistency_checks`; its output is serialised to `consistency_checks.json` and is reproduced below verbatim:

```json
{
  "generated_at_utc": "2026-04-14T07:21:28.984286+00:00",
  "kpi_schema_consistent": true,
  "kpi_counts_consistent": true,
  "readiness_checks_ok": true,
  "readiness_advance_conflict": false,
  "readiness_fine_tune_conflict": false,
  "trade_rows_count": 7,
  "completed_trade_rows_count": 3,
  "run_transition_count": 7,
  "run_completed_trade_count": 3,
  "report_transition_count": 7,
  "report_completed_trade_count": 3,
  "hard_blocker_count": 1,
  "errors": [],
  "overall_ok": true
}
```

Manual cross-file verification confirms the same invariants:

| Invariant | `run_metrics.json` | `report.json` (`headline_kpis`) | `trades.json` / `completed_trades.json` | Status |
|-----------|--------------------|-------------------------------|-----------------------------------------|--------|
| Transition count | 7 | 7 | 7 rows in `trades.json` | **coherent** |
| Completed trade count | 3 | 3 | 3 rows in `completed_trades.json` | **coherent** |
| Sharpe | -7.9580 | -7.9580 | — | **coherent** |
| Max drawdown | 0.01322 | 0.01322 | — | **coherent** |
| Bar win rate | 0.48 | 0.48 | — | **coherent** |
| Completed trade win rate | 0.6667 | 0.6667 | — | **coherent** |

Phase-gate invariants (Workstream A) hold across `readiness.json` and `report.md`:

- `phase_gate_decision = do_not_advance` (both files)
- `fine_tuning_gate_open = false` (both files)
- `hard_blockers` length = 1 (both files); `hard_blocker_message_map` present (readiness) and rendered (report Phase Gate section)
- `decision_rationale` rendered identically in both files
- No case where `research_validity.green = false` but `phase_gate_decision = "advance"` (invariant asserted at runtime)
- No case where `hard_blockers` non-empty but `fine_tuning_gate_open = true` (invariant asserted at runtime)

Conclusion: artifact-consistency contract satisfied in full.

---

## 4. Evidence Sufficiency Statement

Per Workstream D the run must demonstrate snapshot-based, multi-window, reproducible evidence. The run produced:

- `dataset_metadata.json` — confirms snapshot-based input (`snapshot_based = true` on readiness payload).
- `regression_ledger_entry.json` + `regression_ledger_snapshot.json` — confirm regression tracking is wired and fresh.
- `readiness.json.evidence_sufficiency`:
  - `snapshot_based = true`
  - `walk_forward_sufficient = true`
  - `benchmark_sufficiency_artifact_available = true`
  - `regression_tracking_available = true`
  - `regression_history_entry_count = 10`
  - `historical_window_count = 7` (minimum threshold 5)
  - `multi_window_evidence_sufficient = true`
  - `evidence_blockers = []`
  - `green = true`
- `scripts/run_regression_windows.py::_compute_window_starts` is deterministic and validated against an invalid-window guard (new test `test_regression_windows_script_generates_deterministic_starts`).
- README multi-window regression runbook (lines 414-418) verified present and consistent with the script's CLI.

Conclusion: evidence sufficiency is **green** and the round's evidence package is reproducible from the committed config + snapshot pins.

---

## 5. Workstream Acceptance Summary

| Workstream | Title | Primary change | Acceptance |
|-----------|-------|----------------|------------|
| A | Gate semantics hardening | `run_mvp.evaluate_readiness` now asserts invariants and emits `blocker_message_map` + `decision_rationale`; report renders rationale | **met** (3 gate tests passing, invariants asserted at runtime, rationale visible in `report.md` and `readiness.json`) |
| B | Artifact consistency and provenance lock | `persist_run_artifacts` calls `_run_artifact_consistency_checks` atomically before any write; `consistency_checks.json` persisted | **met** (2 artifact-consistency tests passing, `consistency_checks.json` present, coherence verified) |
| C | Temporal semantics parity lock | Runtime-vs-batch parity fixtures driving stop/take-profit/reversal/final-bar scenarios under `signal_delay_bars` | **met** (4 parity tests passing) |
| D | Evidence sufficiency & reproducibility contract | Deterministic multi-window regression runbook test; evidence blocks in readiness; README runbook preserved | **met** (1 runbook test passing, `multi_window_evidence_sufficient = true`, regression ledger present) |

All four acceptance contracts defined in the plan are satisfied.

---

## 6. Explicit Next-Phase Recommendation

**Do NOT advance to fine-tuning / next phase.**

Justification: the run's `phase_gate_decision` is `do_not_advance` because of a hard research blocker (`triple_negative_sharpe`). The plan is explicit that a hard blocker closes the fine-tuning gate and that the round's purpose is gate enforcement, not research-result remediation. The engineering / artifact / temporal-parity / evidence contracts are all green; the research result is not. Therefore:

1. Treat this round as **complete and successful at the contract level** — all one-round fix-and-fine-tune deliverables landed, were tested, and produced a deterministic, consistent artifact bundle.
2. Do **not** open a fine-tuning phase on this model/config combination. The negative Sharpe across validation, walk-forward, and held-out confirms the baseline is not profitable and fine-tuning would amount to overfitting on noise.
3. The next recommended action (out of scope for this round, but for the record) is a research-level review: feature set revision, label re-definition, or regime restriction — not parameter tuning — with a fresh snapshot and a new round of this same gate.

---

## 7. Hash of Inputs / Pointers

- Plan: `docs/fix/one-round-fix-fine-tune-plan-2026-04-14.md`
- Run artifacts: `artifacts/20260414T071934Z-986f1dd7/`
- Test file: `tests/test_pipeline.py`
- Modified sources: `run_mvp.py`, `src/agents/artifacts.py`
- New tests added: see Section 2.
- Branch: `claude/one-round-fix-fine-tune-wJjSA`

End of memo.
