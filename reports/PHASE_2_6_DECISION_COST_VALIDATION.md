# PHASE 2.6 — DECISION & COST-SENSITIVE VALIDATION

## Final Report — Predictive Maintenance / NASA C-MAPSS FD004

> **Scope.** Phase 2.6 is a validation-only, decision-analysis checkpoint. It reuses
> the sealed Phase 2 / Phase 2.5 validation predictions; it retrains no model, adds
> no architecture, and never reads the official FD004 test partition. Its single
> question is whether the *measured predictive* improvements translate into
> *decision-level* value under transparent cost assumptions.

> **Cost caveat (applies to every number in this report).** There is no validated
> real-world cost model for this dataset. Every cost below is an **illustrative /
> scenario assumption, not observed real-world financial data.** Costs are expressed
> in unitless "scores", normalized so that one false positive costs `C_FP = 1` unit.
> Only the ratio `C_FN / C_FP` is meaningful. The previously circulated directional
> estimate is retired and is deliberately **not** reused or repriced here.

---

## 1. Files modified

New Phase 2.6 files only. No Phase 2 / 2.5 file or historical artifact was altered.

| Path | Role |
| --- | --- |
| `src/business/__init__.py` | Business-layer package marker |
| `src/business/decision.py` | Pure, deterministic cost/decision/engine/bootstrap core (imports only numpy/pandas/stdlib) |
| `configs/phase2_6_decision.yaml` | Cost ratios, threshold grid, model IDs, bootstrap seed |
| `scripts/run_phase2_6_decision_analysis.py` | Driver: reads sealed val predictions, writes all decision artifacts |
| `scripts/phase2_6_figures.py` | 8 required figures |
| `scripts/verify_phase2_6_gate.py` | 12-section gate verifier |
| `tests/test_phase2_6.py` | Mandatory Tests A–I + wording audit |

Generated artifacts (read-only inputs; outputs isolated under `results/decisions/`):

- `results/decisions/phase2_6/threshold_cost.csv`
- `results/decisions/phase2_6/cost_ratio_sensitivity.csv`
- `results/decisions/phase2_6/decision_cost_comparison.csv`
- `results/decisions/phase2_6/engine_level_analysis.csv`
- `results/decisions/phase2_6/false_negative_analysis.json`
- `results/decisions/phase2_6/alert_fatigue.json`
- `results/decisions/phase2_6/rul_decision_analysis.json`
- `results/decisions/phase2_6/bootstrap.csv`
- `results/decisions/phase2_6/analysis_registry.json`
- `results/decisions/phase2_6/run_manifest.json`
- `results/figures/phase_2_6/fig1..fig8.png`

Decision analyses are registered **separately** from predictive experiments
(`results/decisions/…`, not `results/experiments/…`), per §19.

## 2. Models evaluated

Only existing Phase 2 / Phase 2.5 models — none refit.

| Role | Model ID | Source |
| --- | --- | --- |
| Model A — anchor | `FR25_T0_anchor_cycle` | Phase 2 H30 anchor (cycle features) |
| Model B — temporal | `FR25_T4_combined_cyc` | Phase 2.5 combined causal-temporal (W5) |
| Temporal H14 | `FR25_T4_combined_cyc_H14` | Phase 2.5, horizon 14 |
| Temporal H50 | `FR25_T4_combined_cyc_H50` | Phase 2.5, horizon 50 |
| RUL anchor / temporal | `RUL25_T0_anchor_cycle` / `RUL25_T4_combined_cyc` | Phase 2 / 2.5 |

**Documented limitation (no silent model switch).** The Phase 2 anchor exists **only at
H30**. Robustness at H14/H50 therefore uses the temporal model alone (a within-model
sensitivity), and the cross-model cost comparison is made **only** at the shared H30
horizon. No comparison mixes horizons or invents a matched anchor.

## 3. Cost assumptions

- Framework: `Expected Cost = FN × C_FN + FP × C_FP` (the §4 primary framework). TP/TN
  are free; no operational benefit is invented.
- Normalization: `C_FP = 1` illustrative unit; `C_FN = ratio`.
- Cost-ratio sensitivity grid: **1, 2, 5, 10, 20, 50, 100** (§5).
- Threshold grid: **0.1 … 0.9** in steps of 0.1 (§6).
- Every figure/label carries: *"Illustrative scenario assumptions — not observed
  financial costs."* No threshold is declared operationally optimal.

## 4. Threshold analysis (measured, H30)

Confusion accounting is exact: for every (threshold × ratio) row,
`TP + FP + TN + FN = 12310` (full validation row count), and stored tables reproduce
from the sealed predictions with max abs delta `0.00e+00`. Selected precision/recall at
the fixed decision threshold 0.5 (from the same predictions used everywhere else):

| Model | Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| anchor `FR25_T0_anchor_cycle` | 0.5 | 0.801 | 0.960 | 0.873 |
| temporal `FR25_T4_combined_cyc` | 0.5 | 0.848 | 0.954 | 0.898 |

Values are the authoritative `default_0.5` measurements already recorded in the Phase 2.5
failure-risk artifact (`results/experiments/phase2_5/failure_risk_comparison.csv`); no
model was rerun or refit to produce them.

The temporal gain is **precision-oriented** (unchanged conclusion from Phase 2.5):
raising the operating point trades a little recall for materially fewer false alarms —
the mechanism that drives every cost result below. Full per-threshold detail is in
`threshold_cost.csv` and Figures 1, 2, 4, 5, 6.

## 5. Cost-ratio sensitivity — cross-model, H30 (measured)

For each ratio, the *best* threshold per model (minimum expected cost; ties toward the
lower threshold) is reported. Values are illustrative units.

| C_FN:C_FP | anchor cost | anchor thr | temporal cost | temporal thr | cheaper | Δ units | Δ % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1:1  | 309  | 0.8 | 293  | 0.8 | **temporal** | 16  | +5.2% |
| 2:1  | 456  | 0.8 | 386  | 0.6 | **temporal** | 70  | +15.4% |
| 5:1  | 662  | 0.4 | 591  | 0.4 | **temporal** | 71  | +10.7% |
| 10:1 | 844  | 0.3 | 715  | 0.2 | **temporal** | 129 | +15.3% |
| 20:1 | 916  | 0.1 | 831  | 0.1 | **temporal** | 85  | +9.3% |
| 50:1 | 1096 | 0.1 | 1131 | 0.1 | **anchor**   | −35 | −3.2% |
| 100:1| 1396 | 0.1 | 1631 | 0.1 | **anchor**   | −235| −16.8% |

**Reading.** The temporal model has lower expected cost across the low-to-moderate
ratio band (1–20). The advantage **reverses at the extreme ratios (50, 100)**: when a
missed failure is assumed enormously more expensive than a false alarm, the anchor's
higher recall wins and the temporal model's higher precision no longer pays. The
temporal advantage is therefore **real but not robust across the full cost grid.**

## 6. Engine-level analysis (50 validation engines)

Rows within an engine are temporally correlated and are **not** treated as independent
events. Engine aggregates at the fixed threshold 0.5 (from `engine_level_analysis.csv`,
`false_negative_analysis.json`, `alert_fatigue.json`).

| Quantity (thr 0.5, H30) | anchor | temporal |
| --- | --- | --- |
| Engines ever detected | 50 | 50 |
| Engines ever missed (≥1 FN) | 27 | **25** |
| Engines with ≥1 false positive | 47 | **38** |
| FP rows | 369 | **265** |
| FP rate of negatives | 3.43% | **2.46%** |
| FN rows | **62** | 72 |
| Positive prediction runs (total) | 204 | **112** |
| False-alarm runs (total) | 128 | **43** |
| Mean false-alarm runs / engine | 2.56 | **0.86** |

**False negatives (§10).** Temporal misses slightly *more* rows (72 vs 62) but touches
*fewer* engines (25 vs 27). The raw RUL of missed rows sits almost entirely in the
[~14–18, 30] band — i.e. positives just above the H30 horizon cutoff. Temporal features
**redistribute** borderline false negatives rather than eliminating them; they do not
rescue the near-boundary cases.

**False positives (§11).** Temporal consistently reduces false alarms: fewer FP rows,
lower FP rate, fewer engines affected. This is not labeled "acceptable" — with no
validated intervention cost, the absolute FP level is a decision input, not a verdict.

**Alert fatigue (§13).** On raw predictions (no suppression policy invented), the
temporal model cuts repeated false-alarm runs from 128 to 43 (mean per engine 2.56 →
0.86). This is the clearest, most decision-relevant benefit observed in Phase 2.6.

**Limitation.** Engine-level "earliest detection" and alert persistence are descriptive
aggregates at one fixed threshold; a production detection rule (dwell time, hysteresis,
confirmation count) is deliberately **not** defined, because it would require an
arbitrary policy this phase is not authorized to assume.

## 7. RUL decision analysis (measured — not forced into the binary cost model)

Per §9, RUL is analyzed as a prognostic-quality question, not a per-row cost.

| Metric (validation) | anchor | temporal | direction |
| --- | --- | --- | --- |
| MAE (cycles) | 32.01 | 31.86 | tiny improvement (−0.15, −0.47%) |
| RMSE | 43.25 | 44.02 | **worse** for temporal |
| Worst abs error p95 | 91.16 | 94.07 | **worse** |
| Worst abs error p99 | 124.16 | 134.84 | **worse** |
| Worst abs error max | 171.31 | 182.97 | **worse** |
| Mean signed residual | +5.85 | +5.50 | bias persists |
| Short-life mean signed residual | +30.52 | +28.40 | bias persists (over-optimistic on short-lived engines) |
| Long-life mean signed residual | −8.85 | −8.15 | bias persists |
| End-of-life MAE (raw RUL ≤ 10; 550 rows) | 6.72 | 4.87 | **improvement** |

**Reading.** The headline MAE improvement (−0.47%) is immaterial and is offset by
**worse** central (RMSE) and tail (p95/p99/max) error. The lifespan-dependent bias from
Phase 2.5 persists — the model still systematically over-predicts remaining life for
short-lived engines and under-predicts for long-lived ones. The one genuine gain is a
better end-of-life estimate (MAE 6.72 → 4.87), which matters for the final maintenance
window but does **not** change the overall RUL conclusion. No operational saving is
claimed from this difference.

## 8. Uncertainty analysis — engine-level bootstrap (measured)

Because engines are the independent units, the bootstrap resamples the **50 validation
engines with replacement** (not rows). Deterministic seed 42, 2000 resamples, at the
fixed threshold 0.5 (separate from the best-threshold view in §5). Reported difference is
`temporal − anchor` cost (negative ⇒ temporal cheaper), with a 2.5 / 97.5 percentile CI.

| C_FN:C_FP | mean diff | CI low | CI high | P(temporal cheaper) |
| --- | --- | --- | --- | --- |
| 1:1  | −94.18 | −143.00 | −47.98  | **1.000** |
| 2:1  | −84.46 | −142.00 | −29.00  | **0.999** |
| 5:1  | −55.30 | −151.03 | +43.02  | 0.875 |
| 10:1 | −6.70  | −176.05 | +171.02 | 0.536 |
| 20:1 | +90.51 | −233.05 | +431.05 | 0.295 |
| 50:1 | +382.12| −423.30 | +1225.05| 0.183 |
| 100:1| +868.15| −761.28 | +2525.10| 0.149 |

**Reading.** At a *fixed* operating point the temporal advantage is statistically clear
only for the lowest ratios (1, 2) where the CI excludes zero entirely. By ratio 5–10 the
CI spans zero (P ≈ 0.88 → 0.54), and by 20+ the point estimate favors the anchor. This
**tempers** the best-threshold comparison in §5: the decision advantage is not robust to
sampling uncertainty over independent engines and does not survive aggressive
miss-cost assumptions. It is reported as an interval, not a significance claim.

## 9. Tests

`tests/test_phase2_6.py` implements the mandatory Tests A–I plus a wording audit.
Exact result: **31 passed, 0 failed, 0 skipped.**

- A cost-calculation correctness · B confusion-matrix accounting (matches sklearn, sums
  to 12310) · C threshold determinism · D cost-ratio determinism · E no test-set access
  (official files verified unchanged since seal) · F engine-level aggregation
  correctness · G bootstrap reproducibility (same seed ⇒ same CI) · H no neural imports
  (import allowlist) · I Phase 2/2.5 artifacts unchanged (SHA-256 + F1≈0.8977) · plus
  report never claims financial savings.

No existing leakage test was weakened.

## 10. Gate

`scripts/verify_phase2_6_gate.py` — **all 12 sections PASS.**

1. No retraining (inputs are stored val predictions; no `.fit`, no `run_experiment`) ✅
2. Official-test firewall (checksums sealed; no test load in Phase 2.6 code) ✅
3. No neural / sequence imports ✅
4. Cost-math correctness ✅
5. Confusion accounting (TP+FP+TN+FN = rows; reproduces stored table) ✅
6. Threshold / cost-ratio determinism ✅
7. Engine-level aggregation (per-engine sums = global counts; 50 engines) ✅
8. Bootstrap reproducibility (reseed = stored CI) ✅
9. Required artifacts + ≥8 figures ✅
10. Illustrative labeling & prohibited-wording scan ✅
11. Phase 2 / 2.5 history unchanged (hashes; F1 = 0.8977) ✅
12. Report / results consistency ✅

## 11. Test-data integrity

The official FD004 test partition (`test_FD004.txt`, `RUL_FD004.txt`) was **never read**
in Phase 2.6 and remains byte-identical to its Phase 0 seal (verified by checksum in
SECTION 2 and Test E). The raw-RUL reference for the FN and RUL analyses is the sealed
`data/processed/FD004_val.parquet` — the 50-engine validation partition only. This phase
is validation-only.

## 12. Final decision

**A. Does temporal failure-risk modeling have decision value?**
*Advantage only under certain cost ratios.* Under illustrative `C_FN:C_FP` between 1 and
20 the temporal model yields lower validation expected cost, and it clearly reduces
false-alarm rows and repeated false-alarm runs. The advantage is **not robust**: it
reverses at extreme miss-cost ratios (50, 100) in the best-threshold view, and the
engine-level bootstrap shows a CI excluding zero only at the lowest ratios. There is no
universal, ratio-independent decision win.

**B. Does temporal RUL modeling have meaningful decision value?**
*No / insufficient evidence.* The MAE change is immaterial (−0.47%) and is accompanied
by worse RMSE and worse tail error; lifespan bias persists. The one real gain (better
end-of-life MAE) does not change any operational conclusion. No decision value is
claimed.

**C. Is the temporal feature layer justified?**
*Conditionally, as a maintained option — not as a universal upgrade.* Its defensible
value is precision-side: fewer false alarms and a strong reduction in repeated
false-alarm (alert-fatigue) runs, with modest cost benefit in the low-to-moderate ratio
band. It does not degrade recall materially and it improves end-of-life RUL. It is
justified for alert-quality/false-alarm-reduction use cases but does **not** clear a bar
that warrants further model escalation.

**D. Is a sequence model justified?**
*No.* Applying the §24 strict standard, the residual limitations observed here —
borderline false negatives concentrated just above the horizon cutoff, the persistent
lifespan-dependent RUL bias, and the non-robust decision advantage — are **not** shown
to be problems that causal-temporal tabular features cannot address, and none of the
evidence demonstrates an unresolved predictive/decision gap that requires sequence
modeling. The original LSTM idea is not, by itself, justification.

### Phase 2.6 outcome

> **DO NOT proceed to Phase 3. DO NOT escalate to a sequence model.**

The classical + causal-temporal solution demonstrates a modest, precision-oriented,
cost-dependent decision benefit. The measured engine-level differences are real, but the
decision advantage is not robust across cost ratios and does not establish an unresolved
decision problem requiring neural escalation. This is a valid, successful outcome of a
rigorous checkpoint.

**HARD STOP.** No Phase 3, no LSTM/neural implementation, no test-set access, no
modifications to Phase 2/2.5 results or resume claims. Awaiting independent audit and
explicit authorization.
