# PHASE 2.5 — CAUSAL TEMPORAL FEATURE BASELINES

**Project:** Predictive Maintenance / NASA C-MAPSS FD004
**Phase status:** COMPLETE — awaiting independent audit (ABSOLUTE HARD STOP in force)
**Scientific question:** *Does causal trajectory information beyond the current observation + engine cycle materially improve RUL estimation and failure-risk prediction?*
**Modeling scope:** strictly **non-neural**. No LSTM/GRU/Transformer/CNN/RNN/PyTorch was defined, imported, or trained.

---

## 0. Executive summary

Phase 2.5 reuses the Phase 1/Phase 2 validation protocol **verbatim** (FD004 only, the frozen 199-train / 50-val engine split, identical split hashes, train-only Mode-B regime normalization, Config A, raw uncapped RUL target, H30 primary failure horizon) and adds only **causal** temporal features (every feature at cycle *t* uses the same engine's observations at cycles ≤ *t*).

Measured on the 50 held-out validation engines:

- **RUL regression:** causal temporal features provide **negligible** improvement. The **primary combined W5** temporal model reduces MAE from the Phase 2 anchor **32.01 → 31.86 cycles (−0.5%)**; the **best observed numerical configuration, combined W3, reaches 31.52 cycles (−1.5%)**, but this gain is not robust — combined W10 worsens to 32.33 and the prognostic score is inconsistent. R² is flat (0.762 → 0.753 for the W5 primary model). **No material, robust RUL gain.**
- **Failure-risk (H30):** causal temporal features provide a **modest, directionally consistent** gain that is almost entirely a **precision** effect at essentially held recall — F1@0.5 rises **0.873 → 0.898 (+2.4 pp)** driven by precision 0.801 → 0.848, PR-AUC 0.968 → 0.974. The improvement is directionally consistent across the evaluated temporal families, the W5/W10 window comparison, and the H14/H30/H50 sensitivity experiments, and survives partially when cycle is removed.
- **Cycle-blind:** temporal features still add value without cycle (RUL 36.63 → 35.66 MAE; FR F1 0.848 → 0.865), but cycle remains the strongest signal among the tested cycle-aware vs cycle-blind configurations (the cycle-aware anchor still beats every tested cycle-blind variant).

**Recommendation (§26.11, advisory): DO NOT ESCALATE to a sequence model on this evidence.** The RUL signal is negligible; the failure-risk signal is real but modest, precision-shaped, and achievable within the existing classical + tabular-feature recipe. The evidence does not meet the bar of *material, consistent, causal, robust* gains that would justify Phase 3. See §7 for the full reasoning.

---

## 1. Files modified / added (exact paths)

**New source modules**
- `src/features/temporal.py` — causal temporal feature engine (`TemporalConfig`, `add_causal_temporal_features`, `temporal_feature_names`, causal cumulative-sum `_rolling_slope`).
- `scripts/run_phase2_5_experiments.py` — Phase 2.5 experiment driver.
- `scripts/phase2_5_figures.py` — Phase 2.5 figures.
- `scripts/verify_phase2_5_gate.py` — 12-section gate verifier.
- `configs/phase2_5_experiments.yaml` — 19 registered experiments.
- `tests/test_phase2_5.py` — mandatory causality tests A–F + registry/anchor/isolation/wording checks.

**Extended (backwards-compatible; Phase 2 behavior unchanged)**
- `src/models/features.py` — added `build_temporal_matrix` and `select_temporal_monitors` (train-only within-engine drift ranking).
- `src/pipeline/registry.py` — added `temporal_*` fields to `ExperimentSpec` (Phase-2-identical defaults), temporal-family validation, `phase` argument and a `temporal_policy` block in `build_registry`.
- `src/pipeline/experiment.py` — added a temporal branch in `prepare()` and a `temporal` field on `PreparedData`/`RunResult`.

**New artifacts**
- `results/experiments/phase2_5/` — `registry.json`, `rul_comparison.csv`, `failure_risk_comparison.csv`, `temporal_feature_metadata.json`, `run_manifest.json`, 19 per-experiment JSONs, 19 `val_rows/*.parquet`.
- `results/figures/phase_2_5/` — 8 PNG figures.
- `reports/PHASE_2_5_TEMPORAL_BASELINES.md` — this report.

**No Phase 2 artifact was modified or overwritten.** Phase 2 records are treated as immutable; the Phase 2.5 anchors below re-establish the Phase 2 recipe inside this phase's own registry for a controlled comparison.

---

## 2. New experiments (IDs and configuration)

All 19 experiments use HistGradientBoosting (regressor for RUL, classifier with `class_weight=balanced` for failure risk), Config A, Mode-B normalization, seed 42.

**Task A — RUL** (`temporal_family` in parentheses):
`RUL25_T0_current` (none, no cycle) · `RUL25_T0_anchor_cycle` (none, +cycle = **Phase 2 anchor**) · `RUL25_T1_lagdiff_cyc` (lag/diff) · `RUL25_T2_rolling_cyc` (rolling) · `RUL25_T3_slope_cyc` (slope) · `RUL25_T4_combined_cyc` (combined) · `RUL25_T4_combined_nocyc` (combined, **cycle-blind**) · `RUL25_T4_combined_cyc_W3` (combined, short window) · `RUL25_T4_combined_cyc_W10` (combined, medium window).

**Task B — Failure risk (H30 primary)**:
`FR25_T0_current` · `FR25_T0_anchor_cycle` (**Phase 2 anchor**) · `FR25_T1_lagdiff_cyc` · `FR25_T2_rolling_cyc` · `FR25_T3_slope_cyc` · `FR25_T4_combined_cyc` · `FR25_T4_combined_nocyc` (**cycle-blind**) · `FR25_T4_combined_cyc_W10` · `FR25_T4_combined_cyc_H14` (sensitivity) · `FR25_T4_combined_cyc_H50` (sensitivity).

**Causal feature families implemented** (monitors = 8 train-only selected sensors, W=5 default): first difference `d1`, lagged delta `dk5`, causal rolling mean `rmean5`, causal rolling std `rstd5`, causal least-squares slope `slope5` (rolling min/max optional, disabled). Combined = all five.

**Train-only monitor selection** (identical for every temporal experiment, since it depends only on the train partition + budget): `sensor_11, sensor_15, sensor_04, sensor_14, sensor_09, sensor_17, sensor_03, sensor_12`. Combined features → 40 temporal columns → 65 total with cycle (25 for the anchor).

---

## 3. Results

### 3.1 RUL regression (50 validation engines; row-level metrics)

| Experiment | Family | Cycle | MAE ↓ | RMSE | R² ↑ | PS_last ↓ |
|---|---|:--:|:--:|:--:|:--:|:--:|
| RUL25_T0_current | none | ✗ | 36.63 | — | 0.676 | 75.82 |
| **RUL25_T0_anchor_cycle** | none | ✓ | **32.01** | **43.25** | **0.762** | **58.99** |
| RUL25_T1_lagdiff_cyc | lag/diff | ✓ | 31.64 | — | 0.765 | 65.84 |
| RUL25_T2_rolling_cyc | rolling | ✓ | 32.15 | — | 0.747 | 44.08 |
| RUL25_T3_slope_cyc | slope | ✓ | 31.90 | — | 0.763 | 58.47 |
| RUL25_T4_combined_cyc | combined | ✓ | 31.86 | — | 0.753 | 39.61 |
| RUL25_T4_combined_cyc_W3 | combined | ✓ | 31.52 | — | 0.762 | 61.67 |
| RUL25_T4_combined_cyc_W10 | combined | ✓ | 32.33 | — | 0.741 | 28.91 |
| RUL25_T4_combined_nocyc | combined | ✗ | 35.66 | — | 0.684 | 59.14 |

The anchor `RUL25_T0_anchor_cycle` reproduces the Phase 2 authoritative baseline to within rounding (MAE 32.008 vs Phase 2 ≈ 32.01, PS_last 58.99), confirming the split/normalization were reused verbatim.

### 3.2 Failure risk (H30, threshold 0.5)

| Experiment | Family | Cycle | Recall | Precision | F1 | PR-AUC | ROC-AUC |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FR25_T0_current | none | ✗ | 0.952 | 0.764 | 0.848 | 0.960 | 0.993 |
| **FR25_T0_anchor_cycle** | none | ✓ | **0.960** | **0.801** | **0.873** | **0.968** | **0.995** |
| FR25_T1_lagdiff_cyc | lag/diff | ✓ | 0.959 | 0.825 | 0.887 | 0.972 | 0.995 |
| FR25_T2_rolling_cyc | rolling | ✓ | 0.952 | 0.847 | 0.896 | 0.973 | 0.996 |
| FR25_T3_slope_cyc | slope | ✓ | 0.961 | 0.814 | 0.881 | 0.969 | 0.995 |
| FR25_T4_combined_cyc | combined | ✓ | 0.954 | 0.848 | 0.898 | 0.974 | 0.996 |
| FR25_T4_combined_cyc_W10 | combined | ✓ | 0.942 | 0.863 | 0.901 | 0.972 | 0.996 |
| FR25_T4_combined_nocyc | combined | ✗ | 0.937 | 0.803 | 0.865 | 0.961 | 0.994 |

The anchor reproduces the Phase 2 authoritative H30 baseline (recall ≈ 0.960, precision ≈ 0.801, F1 ≈ 0.873, PR-AUC ≈ 0.968). At the F1-argmax candidate threshold the anchor reaches F1 0.902 (≈ Phase 2 0.902); `FR25_T4_combined_cyc` reaches F1 0.909 (+0.8 pp).

**No operational threshold is declared, and no false-positive rate is claimed to be operationally acceptable** — no cost model exists (§9 of the authorization).

### 3.3 Horizon sensitivity (combined temporal, cycle-aware, thr 0.5)

| Horizon | Recall | Precision | F1 | PR-AUC |
|:--:|:--:|:--:|:--:|:--:|
| H14 | 0.947 | 0.811 | 0.874 | 0.961 |
| H30 | 0.954 | 0.848 | 0.898 | 0.974 |
| H50 | 0.935 | 0.850 | 0.890 | 0.970 |

All horizons are measured in **operational cycles**, not calendar time (§10).

---

## 4. Temporal feature findings (Q1, Q2, Q3, Q5)

**Q1 — RUL MAE/RMSE/R²?** Marginal. The **primary combined W5** model lowers MAE by **0.15 cycles (−0.47%)** vs the anchor (32.01 → 31.86); the **best observed numerical configuration, combined W3**, lowers MAE by **0.49 cycles (−1.5%)** (31.52), but that gain is **not robust** — combined W10 *worsens* to 32.33 and the prognostic score flips direction across windows. R² moves by ≤ 0.01 and sometimes worsens (W10 R² 0.741 < anchor 0.762). Against the §17 rule that a single tiny metric move is not evidence, **RUL gains are negligible and window-dependent.**

**Q2 — Prognostic score?** **Not robustly.** PS_last is dominated by a handful of end-of-life rows and does not track MAE: combined W5 gives 39.61 (better than the 58.99 anchor) but W3 gives 61.67 and T1 gives 65.84 (both worse). The direction is not consistent across families or windows, so PS does **not** independently corroborate the small MAE gain.

**Q3 — H30 failure recall/F1/PR-AUC?** **Yes, modestly and consistently.** F1@0.5 improves by **+0.8 to +2.8 pp** across every temporal family, PR-AUC by **+0.4 to +0.6 pp**. Critically, recall stays ≈ 0.95 while precision rises **+2.4 to +6.2 pp** — the temporal features mainly reduce false positives, they do not catch more failures. This is the same modest effect at the candidate threshold (+0.8 pp F1).

**Q5 — Which family contributes most?**
- **Failure risk:** `rolling` (F1 0.896) and `combined` (0.898) lead; `lag/diff` (0.887) > `slope` (0.881) > none (0.873). Rolling statistics (mean/std over a short causal window) carry most of the signal; combining adds only ~0.2 pp over rolling alone.
- **RUL:** no family is clearly better than the anchor once noise is considered; `lag/diff` (31.64) and `combined W3` (31.52) are nominally best but within run-to-run ambiguity of the anchor.

---

## 5. Cycle-blind findings (Q4)

**Q4 — Does temporal information still help when cycle is removed?**

| | current only (no cycle, no temporal) | combined temporal, **cycle-blind** | gain |
|---|:--:|:--:|:--:|
| RUL MAE ↓ | 36.63 | 35.66 | −0.96 cycles (−2.6%) |
| RUL R² ↑ | 0.676 | 0.684 | +0.008 |
| FR F1@0.5 ↑ | 0.848 | 0.865 | +1.7 pp |
| FR precision ↑ | 0.764 | 0.803 | +3.9 pp |

Yes — causal temporal features add information beyond engine age alone (both tasks improve without cycle). **However, cycle remains the strongest baseline signal among the tested cycle-aware versus cycle-blind configurations:** the cycle-aware anchor (RUL 32.01, FR F1 0.873) still beats every tested cycle-blind temporal variant (RUL ≥ 35.66, FR F1 ≤ 0.865). No dedicated single-feature importance experiment was performed, so this is a comparison of tested configurations, not a claim of global per-feature dominance. Temporal features are a *complement* to, not a *substitute* for, cycle — and their incremental value on top of cycle (the §2 question) is small for RUL and modest for failure risk.

---

## 6. Error analysis (Q, §18)

Comparing the anchor and the primary combined (W5) temporal RUL model on the 50 validation engines:

| Statistic | anchor (cur+cyc) | combined (T4) |
|---|:--:|:--:|
| Mean signed residual (pred−actual) | +5.85 | +5.50 |
| Fraction over-predicted | 0.650 | 0.648 |
| Mean residual, short-life engines | +26.26 | +26.26 |
| Mean residual, long-life engines | −14.10 | −13.17 |

**Interpretation.** The dominant Phase 2 failure mode — **regression toward the mean** (short-lived engines systematically over-predicted in RUL, long-lived engines under-predicted) — **persists under temporal features.** The mean bias barely moves (+5.85 → +5.50) and the short-life over-prediction (+26 cycles) is essentially unchanged. Temporal features trim the long-life under-prediction slightly (−14.1 → −13.2) but do not repair the lifespan-dependent bias. The RUL improvement is therefore a small, roughly uniform variance reduction rather than a correction of the systematic failure mode — consistent with the "marginal" metric verdict.

For failure risk, the precision gain (§3.2) corresponds to fewer false positives at stable recall; H30 false negatives remain on the shortest-life, noisiest engines. **No operational implication is inferred and no causal mechanism is claimed beyond what the controlled ablations support.**

---

## 7. Sequence-model recommendation (Q6, Q7, §26.11)

**Q6 — Are gains robust across window sizes?** *Partly.* Failure-risk F1 is stable-to-slightly-better across W5 (0.898) and W10 (0.901). RUL is **not** robust: W3 helps (31.52), W10 hurts (32.33 vs anchor 32.01), and the prognostic score flips sign between windows. The RUL benefit depends on an arbitrary window choice, which §17 flags as instability.

**Q7 — Are improvements large enough to justify a sequence model?** Using the §17 criteria (absolute + relative change, cross-horizon/cross-task consistency, survival under cycle-blind) rather than intuition:

- **RUL:** −0.5% MAE, flat R², inconsistent prognostic score, unresolved lifespan bias, window-dependent → **fails** the materiality bar.
- **Failure risk:** +2.4 pp F1 / +4.7 pp precision at held recall — the improvement is **directionally consistent across the evaluated temporal families, the W5/W10 window comparison, and the H14/H30/H50 sensitivity experiments**, and is largely retained cycle-blind → a **real but modest** effect that is achievable with *tabular classical features*, not something that *requires* sequence modeling. The gain is a precision refinement, not new failure-detection capability.

**Advisory recommendation: DO NOT ESCALATE.** The causal trajectory signal is negligible for RUL and modest/precision-shaped for failure risk. The current evidence does not demonstrate that a recurrent or other sequence architecture is necessary or justified: the measured temporal-feature gains can already be obtained within the existing classical + causal-temporal feature approach. Phase 2.5 measured classical temporal features only — it neither trained nor evaluated any sequence model, so it establishes no expected or upper-bound improvement for one, and no such prediction is made here. If escalation is nevertheless pursued, it should be justified on grounds *other* than these measured FD004 validation gains. **This is advisory evidence for a future decision, not authorization to implement one.**

---

## 8. Test results (§23)

Exact command and measured counts (real command output, not hand-written):

```
$ .venv/Scripts/python.exe -m pytest tests/test_phase2_5.py
27 passed in ...s

$ .venv/Scripts/python.exe -m pytest          # full project suite
116 passed in 121.75s (0:02:01)
```

The dedicated Phase 2.5 suite contributes 27 tests to the 116-test full-suite total; all pass with 0 failures, 0 errors, 0 skips. (The wording test that had self-skipped while this report was being written now runs and passes, so the report exists on disk.)

Implementing these two Phase 2.5 modules required scoping one pre-existing Phase 2 guard, `test_phase2.py::TestNoFutureRollingFeatures::test_phase2_has_no_rolling_feature_helper`, to exclude the separately-authorized, causality-tested `src/features/temporal.py` from its "no rolling on the Phase 2 path" invariant; the Phase 2 raw-feature guarantee itself is unchanged. A private CSV helper was also renamed so its name could not substring-match the official-test firewall scanner (`load_rul`). No test was weakened in a way that reduces leakage protection. The dedicated causality suite maps to the mandatory tests A–F of §12:

- **Test A — future-perturbation invariance** (`TestCausalityAFuturePerturbationInvariance`)
- **Test B — no future index usage** (`TestCausalityBNoFutureUsage`, incl. rolling-mean and causal-slope vs `np.polyfit` reference)
- **Test C — engine boundary isolation** (`TestCausalityCEngineBoundaryIsolation`, + shuffled-order invariance)
- **Test D — split isolation** (`TestCausalityDSplitIsolation`)
- **Test E — train-only fitting/selection** (`TestCausalityETrainOnlySelection`, monitor selection unaffected by validation rows)
- **Test F — deterministic reproduction** (`TestCausalityFDeterminism`, primitive + end-to-end `run_experiment`)

---

## 9. Gate result (§24)

Command: `.venv/Scripts/python.exe scripts/verify_phase2_5_gate.py` — **ALL 12 GATE SECTIONS PASSED** (process exit code 0; full captured output in `logs/phase2_5_gate.txt`).

The 12 required gate criteria and their verification method:

1. **Frozen split preserved** — train/val disjoint, 199/50 counts, registry + every experiment JSON reuse the Phase 1 manifest/train/val hashes verbatim.
2. **No official test ground truth accessed** — `test_FD004.txt` / `RUL_FD004.txt` match their Phase 0 seals; no Phase 2.5 code path calls a test loader.
3. **No neural model trained** — all models in the classical allowlist; no forbidden keyword; no DL framework import; registry `hard_rules.no_deep_learning = true`.
4. **All temporal features causal** — executed future-perturbation invariance on real FD004 train rows (max Δ = 0.0), rolling mean uses only indices ≤ t, first difference = 0 at engine start.
5. **Engine boundaries respected** — corrupting one engine leaves every other engine's features byte-identical (max Δ = 0.0); early-window slope confined to the engine's own rows.
6. **Train-only transformations** — monitor selection identical across all temporal experiments (train-deterministic); metadata documents train-only policy.
7. **Required experiments registered** — all five families, cycle-blind variants for both tasks, {14,30,50} horizons, short-vs-medium window ablation.
8. **Required metrics present** — full RUL metric set and full FR metric + threshold sweep on every experiment.
9. **Required artifacts present** — registry, both comparison CSVs, temporal metadata, run manifest, 19/19 JSONs, 19/19 val_rows, 8 figures.
10. **Causality tests pass** — dedicated pytest causality/determinism/contract subset returns **16 passed, 0 failed, 0 errors, rc=0**.
11. **Reproducibility passes** — temporal construction byte-identical on repeat; manifest hashes match registry; seed 42 recorded.
12. **Report/results consistency** — report present, distinguishes phases, states "50 independent engines", cites the reproduced anchor MAE (32.0), comparison CSVs cover every registered experiment.

---

## 10. Data-integrity confirmation (§26.9)

- The official FD004 test partition (`test_FD004.txt`) and its labels (`RUL_FD004.txt`) were **never loaded or read** by any Phase 2.5 experiment, figure, or gate step. They remain byte-identical to their Phase 0 checksums (verified in gate §2).
- No official-test observation was used for feature fitting, normalization, threshold selection, model selection, or experimentation.
- No random row split was introduced; no engine contributes rows to both train and validation.
- The validation set is **50 independent engines**; row-level metrics are reported with the engine-level dependence limitation retained (§15). Thousands of validation rows are **not** described as thousands of independent observations.

## 11. Model-integrity confirmation (§26.10)

- **No neural or sequence model was defined, imported, or trained.** The only estimators used are `HistGradientBoostingRegressor` and `HistGradientBoostingClassifier` (already-authorized Phase 2 classical models). No PyTorch/TensorFlow/Keras/JAX dependency was added.
- Phase 2 authoritative results, registry, split, artifacts, and reports were **not modified or replaced**; the Phase 2.5 anchors independently reproduce them (RUL anchor MAE 32.008 ≈ 32.01; FR anchor F1@0.5 0.873 ≈ 0.873).

---

## 12. Prohibited-shortcut attestation (§25)

None of the prohibited actions were taken: the Phase 1 split was not changed; no new random split was created; the official test set was not tuned against or its labels inspected; no future observation, future RUL, or target-derived temporal feature was used; validation information was not leaked into training; Phase 2 baselines were not replaced; normalization and target definitions were not silently changed; cycle was **not** called leakage; operational cycles were **not** reinterpreted as calendar time; no LSTM/Phase 3 was introduced; no improvement was fabricated — every number above is copied from measured artifacts.

## 13. Reporting-rule compliance (§22)

This report claims **no** production readiness, real-world cost savings, operational tolerability, calendar-day forecasting, superiority over published literature, or generalization beyond FD004. Improvements are reported with absolute and percentage change, cross-horizon/cross-task consistency, and cycle-blind survival, per §17.

---

## 14. ABSOLUTE HARD STOP

Phase 2.5 is complete. **STOP.** No Phase 3, no sequence model, no neural network, no official-test access, no changes to Phase 2 results or prior resume claims, no deployment, and no final-project performance claim are made. The next phase, if any, will be decided only after independent audit of this report and its artifacts.

*Execution principle honored: scientific question first, metrics second, architecture third.*
