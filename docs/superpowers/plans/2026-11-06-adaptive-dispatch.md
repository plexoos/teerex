# Adaptive dispatch Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement the tasks in this session.

**Goal:** A mathematical, trainable, reproducible marimo demonstration of Bayesian CPU/GPU allocation.

**Architecture:** Keep fitting, Bayesian updates, and experiment orchestration in a focused Python module. Reuse the existing phase semantics; verify a lightweight makespan evaluator against `teerex.replay`. The notebook supplies the narrative, controls, diagnostic figures, and decision inspection.

**Tech Stack:** Existing numpy, scipy, pandas, matplotlib, marimo; no GPU or new dependencies.

**Spec:** docs/superpowers/specs/2026-11-06-adaptive-dispatch-design.md

## Global Constraints

- Train on the current event CSV, never precomputed artifacts from the older study.
- Completion time plus CPU allocation cost is the approved objective.
- No references to the external example in the delivered notebook.
- Final test data cannot select policies, hyperparameters, or seeds.
- All changes remain reviewable in the requested checkout; no publishing. The user subsequently requested a commit on a new branch for testing.

## Review Focus

- An unknown source condition fails clearly, rather than silently extrapolating.
- Nonpositive/nonfinite measurements and duplicate event identities fail ingestion.
- Shared parameter uncertainty is sampled once per rollout, not per event.
- Contention and CPU speed cannot be separated at one fixed CPU allocation.
- Report actual paired outcomes even if Bayesian planning ties or loses.

### Task 1: Model and simulator

**Files:** Create `teerex/adaptive.py`, `tests/test_adaptive.py`.
**Interfaces:** `load_events`, `fit_source`, `MachineBelief`, `batch_makespan`, `Action`.

- [x] Write failing tests for run-held-out fitting, Gaussian updates, no contention evidence at one worker, and simulator equivalence.
- [x] Run `.venv/bin/python -m pytest tests/test_adaptive.py -q`; expect missing implementation failures.
- [x] Implement source fitting, conditional Gaussian mixture updates, and fast batch replay.
- [x] Run focused tests; expect all passing.

### Task 2: Paired experiments

**Files:** Extend `teerex/adaptive.py` and `tests/test_adaptive.py`.
**Interfaces:** `run_suite` returns metrics, decision histories, and paired summaries; `source_diagnostics` returns held-out predictions and coverage.

- [x] Test matched worlds, determinism, pre-update decisions, and hindsight cost bound.
- [x] Implement frozen validation baseline and distinct measured/transfer/stress regimes.
- [x] Run focused tests and the default experiment; retain losses and ties.

### Task 3: Notebook and verification

**Files:** Create `notebooks/bayesian_dispatch.py`; update README entry.

- [x] Write problem, mathematics, data audit, alternatives, training, online decision inspector, paired evaluation, limitations, and next experiment design.
- [x] Run `.venv/bin/marimo check notebooks/bayesian_dispatch.py` and execute HTML export.
- [x] Run full pytest suite and independent review; fix substantive findings.
- [x] Deliver notebook path, measured findings, and verification limits.

## Execution record

- Source audit: 14,000 current rows, seven conditions, 28 runs, one GPU UUID, one CPU-thread setting; saved legacy manifest reports 22,000 rows.
- Ruling: use rolling finite batches with barriers to avoid unobserved arrival and partial-completion assumptions; more general scheduling is a next extension.

- Task 1 complete: source fit, persistent condition/run offsets, Gaussian-mixture machine updates, and async batch evaluator; analytic and replay checks pass.
- Task 2 complete: paired measured/transfer/stress experiments with five policies, frozen validation baseline, overhead and paired intervals.
- Task 3 complete: notebook executes and exports; mathematical rendering and performance figures visually checked.
- Independent review: simulator and posterior moments verified; new-run predictive variation corrected and one-worker identifiability wording clarified. Regression tests cover both.
- Final focused checks: 14 passed. Full suite: 218 passed; three unchanged legacy snapshot tests fail (event count 22k vs 14k, CSV checksum, frozen streams containing absent muon events).
- No raw measurements, saved legacy policies, dependencies, or existing notebooks changed.
