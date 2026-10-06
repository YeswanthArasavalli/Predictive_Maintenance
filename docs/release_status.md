# Release Status

## Current status

**FINAL / FROZEN.** The scientific/modeling project is complete. The authoritative
release state is the **Phase 2.7A freeze** (see
`reports/PHASE_2_7A_FREEZE_RELEASE_REPORT.md`). This document is the single
pointer to the current status so readers do not have to reconstruct it from
historical reports.

## Historical phases (retained for provenance)

| Phase | Meaning | Status | Report |
|---|---|---|---|
| Phase 0 | Data audit, leakage audit, target/horizon design | APPROVED | `reports/PHASE_0_DATA_AUDIT.md`, `PHASE_0_FINAL_GATE.md` |
| Phase 1 | Leakage-safe FD004 preprocessing, frozen split | APPROVED | `reports/PHASE_1_DATA_PIPELINE.md`, `PHASE_1_FINAL_GATE.md` |
| Phase 2 | Classical baselines (RUL + failure-risk) | APPROVED / CLOSED | `reports/PHASE_2_BASELINE_RESULTS.md` |
| Phase 2.5 | Causal temporal-feature baselines | APPROVED / CLOSED | `reports/PHASE_2_5_TEMPORAL_BASELINES.md` |
| Phase 2.6 | Decision & cost-sensitive validation | APPROVED / CLOSED | `reports/PHASE_2_6_DECISION_COST_VALIDATION.md` |
| Phase 2.7 | Finalization & productionization audit | CONDITIONAL PASS | `reports/PHASE_2_7_FINALIZATION_AUDIT.md` |
| Phase 2.7A | Freeze-fix + final release gate | **PASS — FROZEN** | `reports/PHASE_2_7A_FREEZE_RELEASE_REPORT.md` |

## Current vs historical — how to read the reports

- **Current authoritative state:** Phase 2.7A. It supersedes the Phase 2.7
  audit's conditional findings (the P1/P2 fixes are done).
- **The Phase 2.7 audit report** is a *point-in-time* record: it describes the
  pre-fix state (e.g. a `deep-learning` keyword, missing `pyarrow`/`joblib` in
  `environment.json`, zero git commits). Those are **resolved/superseded** —
  the report is kept as an audit trail, **not** rewritten.
- **Historical reports** (Phases 0–2.6) are preserved unchanged for scientific
  provenance. Do not treat their prose as the current task list.

## What changed in this release-hardening pass (post-freeze)

This pass performed **public-repository hardening only** — no scientific change:

- Removed dead `CMAppssDataset` alias; removed placeholder `dev@example.com`
  from `pyproject.toml`.
- Corrected the `LICENSE` dataset note (the repo does **not** redistribute the
  NASA dataset) and two wrong checksum-path references in `data/` READMEs.
- Added the full `docs/` set and a recruiter-ready `README.md`.
- Re-ran the test suite (**147 passed**) and all four gates (**exit 0**) against
  the locked `.venv`.

No model, metric, split, feature, target, threshold, or conclusion changed. The
frozen scientific artifacts remain byte-identical.
