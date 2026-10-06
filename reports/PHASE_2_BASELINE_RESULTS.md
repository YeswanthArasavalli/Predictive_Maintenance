# PHASE 2 — Baseline Modeling & Experimental Results

**Project:** Predictive Maintenance on NASA C-MAPSS FD004
**Phase status:** COMPLETE — hard-stopped before any deep-learning work
**Data policy:** Phase 1 engine split used verbatim; official `test_FD004.txt` / `RUL_FD004.txt` never loaded
**Random seed:** 42
**Executed:** 2026-10-05 UTC

Reference artifacts:

- Registry: `results/experiments/phase2/registry.json` (25 experiments)
- Comparison tables: `results/experiments/phase2/rul_comparison.csv`, `results/experiments/phase2/failure_risk_comparison.csv`
- Per-experiment JSONs: `results/experiments/phase2/{rul,failure_risk}/*.json`
- Validation predictions: `results/experiments/phase2/val_rows/*.parquet`
- Figures (10): `results/figures/phase_2/fig1_pred_vs_actual.png` … `fig10_performance_by_horizon.png`
- Test suite: `tests/test_phase2.py` (46 Phase-2 tests) — full project suite: **89 passed**
- Gate: `python scripts/verify_phase2_gate.py` → **ALL PHASE 2 GATE CHECKS PASSED**

---

## Executive Summary

Phase 2 established four tiers of baseline for FD004 (199 train / 50 validation
engines, 12 310 validation rows, 24 raw features) using only classical models:
naive, causal linear degradation, Ridge / Logistic Regression, and
HistGradientBoosting. **No LSTM, GRU, Transformer, or any other deep network
was trained** — this was a hard rule and remains in force.

Headline findings on the FD004 validation partition:

| Task | Best baseline | Model | Key metric |
|---|---|---|---|
| A — RUL regression | `RUL_gb_regimeB_withCyc_cfgA` | HistGB + regime-B norm + cycle | **MAE = 32.0 cycles, RMSE = 43.2, R² = 0.762** |
| B — Failure risk H=30 | `FR_gb_regimeB_withCyc_cfgA` | HistGB classifier + `class_weight=balanced` | **Recall = 0.960, Precision = 0.801, F1 = 0.873, PR-AUC = 0.968** (thr = 0.5) |

Two honest, uncomfortable observations dominate the phase:

1. **The `FR_majority_H30` baseline scores validation accuracy 87.41 % with
   recall exactly 0.0.** It always predicts "no failure". This *numerically
   matches* the "87 % validation accuracy" language in the pre-existing
   portfolio narrative and shows why raw accuracy is meaningless on this 12.6 %
   positive-rate task. Any claim of operational value must be made on
   recall / precision / PR-AUC, not accuracy.
2. **`cycle` is the strongest single observable feature.** Adding `cycle` to
   the feature set improves the best classical RUL MAE by ≈ 12.6 % (36.6 →
   32.0) and F1 by ≈ 2.5 pp on Task B. `cycle` is a legitimate inference-time
   observable, strongly target-correlated because `raw_RUL = max_cycle −
   cycle`. Any sequence model that "wins" must demonstrate value beyond
   simply recovering lifecycle position.

The measured HistGB baselines set a concrete reference point that any future
model must be compared against. Whether an LSTM is justified is examined in
Section 17 — the phase does not presume an answer.

---

## Experimental Setup

**Data.** FD004 train partition only (249 engines, 21 sensors + 3 operational
settings, 6 fixed operating regimes, 2 fault modes). Regime assignment uses
Phase 1's deterministic mapping table (`src/data/regime.py`).

**Split.** Loaded verbatim from `results/audits/phase1_dataset_manifest.json`.

```
train engines: 199   hash: cb49950e6faacc9100c04f410c92f99585c6baffce4ff11cd9dd1895ddbdb359
val   engines:  50   hash: 116fa158295678abcbd0c083f62d384d7d67971263fe40b947b1c7590d746042
manifest           : 6b3a8b49092c6ee3360cea07c9d9896cc6386cb2621d4e63995209603058f150
```

No Phase 2 experiment re-splits. Every per-experiment JSON embeds the same
manifest hash so a mismatch is caught by the gate.

**Targets.** Row-level (no windowing at Phase 2).

- Task A: `raw_RUL = max_cycle(unit_id) − cycle` (uncapped; `model_RUL_target = raw_RUL`).
- Task B: `failure_risk_target_H = (raw_RUL ≤ H)` for H ∈ {14, 30, 50}.

**Statistical unit.** All Phase 2 metrics are computed at the observation
(row) level. Rows from the same engine are temporally correlated and are
**not independent statistical samples**; long-lived engines contribute more
rows and therefore receive greater weight in aggregate metrics. The
validation partition contains **50 independent engines, not 12 310
independent observations** (see Limitations).

**Preprocessing ablation.** Two normalization modes fit on TRAIN rows only:

- Mode A — global z-score across all train rows;
- Mode B — regime-conditioned z-score (six scalers, one per regime).

Feature configs A / B / B+ correspond to 21 sensors / drop-duplicates (18) /
drop-duplicates + drop sensor_16 (17). Plus 3 operational settings = 24 / 21 / 20
columns. `cycle` is added only when `include_cycle: true` and is rejected by
the `src/models/features.py` gate otherwise.

**Registered experiments:** 25.

- 12 Task A (3 naive/linear + 5 Ridge + 4 HistGB).
- 9 Task B at H=30 (1 majority + 4 LogReg + 4 HistGB).
- 4 Task B horizon sensitivity (LogReg / HistGB at H=14 and H=50).

**Model allowlist** (enforced at YAML-load time in `src/pipeline/registry.py`):
`naive_mean, naive_age, linear_degradation, ridge, hist_gb_regressor, majority_class, logistic_regression, hist_gb_classifier`. Any other model name raises at load time. Deep-learning architectures are therefore physically impossible to register in this phase.

---

## RUL Baselines

Baselines implement the spec's tiered ladder:

1. **Trivial — `NaiveMeanRUL`.** Constant = mean(train `raw_RUL`). Deterministic.
   Ignores every observable.
2. **Trivial+ — `NaiveAgeRUL`.** `max(mean(train lifespan) − cycle, 0)`. Uses
   only `cycle`. Requires the `include_cycle=True` gate.
3. **Statistical — `LinearDegradationRUL` (fully causal).** Constructs a
   health index `HI = mean(z(s_14), z(s_17), −z(s_11))` on Mode-B normalized
   sensors. For each validation engine, iterates through cycles in chronological
   order and fits an OLS line on **only the prefix `1..t`** of the HI
   trajectory (via a cumulative-sum trick). Extrapolates that prefix to the
   train-fitted failure threshold `τ = median of HI at last cycle of each train
   engine`, giving a per-row failure-time estimate and therefore RUL. Direction
   (whether HI rises or falls with damage) is inferred at fit time from the
   sign of `median(HI at last) − median(HI at first)` across train engines.
   Predictions are clipped at `rul_cap = max(train raw_RUL)` (physical ceiling
   — no engine in FD004 outlives the longest train trajectory) so the model
   cannot extrapolate to a fictitious 20 000-cycle life.
4. **Classical ML — Ridge** (`alpha=1.0`). Row-level features. Ablates over
   feature config (A/B/B+), normalization mode (A/B), and cycle in / out.
5. **Strong tree — HistGradientBoostingRegressor** (`max_iter=200,
   lr=0.1`). Same ablation grid.

Nothing about the linear-degradation baseline uses future information. The
Phase-2 test `TestLinearDegradationCausality` asserts that corrupting the last
observation of a validation engine leaves earlier-row predictions exactly
unchanged.

---

## RUL Results

Full table (sorted ascending MAE), lower is better on every column:

| experiment_id | model | cfg | norm | cycle | MAE | RMSE | R² | PS_rows | PS_last |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| **RUL_gb_regimeB_withCyc_cfgA** | HistGB | A | B | **yes** | **32.01** | **43.25** | **0.762** | 1.45e8 | **58.99** |
| RUL_gb_regimeB_noCyc_cfgA | HistGB | A | B | no | 36.63 | 50.43 | 0.676 | 1.05e9 | 75.82 |
| RUL_gb_regimeB_noCyc_cfgBplus | HistGB | B+ | B | no | 36.74 | 50.38 | 0.677 | 9.28e8 | 84.36 |
| RUL_gb_globA_noCyc_cfgA | HistGB | A | A | no | 37.52 | 51.12 | 0.667 | 8.28e8 | 114.66 |
| RUL_ridge_globA_withCyc_cfgA | Ridge | A | A | **yes** | 38.33 | 48.06 | 0.706 | 6.83e7 | 1242.87 |
| RUL_ridge_regimeB_noCyc_cfgA | Ridge | A | B | no | 43.91 | 55.31 | 0.611 | 1.02e9 | 43023.6 |
| RUL_ridge_globA_noCyc_cfgA | Ridge | A | A | no | 44.87 | 55.90 | 0.602 | 1.72e9 | 11213.9 |
| RUL_ridge_globA_noCyc_cfgB | Ridge | B | A | no | 44.89 | 55.93 | 0.602 | 1.75e9 | 10313.8 |
| RUL_ridge_globA_noCyc_cfgBplus | Ridge | B+ | A | no | 44.89 | 55.93 | 0.602 | 1.75e9 | 10309.6 |
| RUL_naive_age | Naive age | — | — | yes | 61.15 | 71.53 | 0.349 | 9.94e7 | 207404 |
| RUL_naive_mean | Naive mean | — | — | — | 72.90 | 88.62 | ≈ 0 | 2.02e10 | 3.07e7 |
| RUL_linear_degradation | Lin. degr. | A | B | no | 323.77 | 351.59 | −14.74 | 2.60e24 | 1.68e16 |

`PS_rows` = prognostic score summed over every validation row; `PS_last` =
Saxena prognostic score computed on one prediction per validation engine,
using that engine's final observed validation row. Both are **lower is
better**. The Saxena 2008 formula and its overflow-safe exponent clipping
are documented at
[`src/evaluation/metrics.py`](file:///d:/Projects/predictive_modelling/src/evaluation/metrics.py)
and are unit-tested against reference values (E=−13 → exp(1)−1, E=+10 →
exp(1)−1, E=0 → 0).

Observations:

- HistGB is the best classical model. It beats the naive-mean floor by 2.28×
  in MAE and by ~5× in the last-per-engine prognostic score.
- Ridge is only ≈ 20 % behind HistGB in MAE but blows up 20× on `PS_last`
  because a linear extrapolation on cycle can produce a small number of very
  large positive errors near end-of-life, which the asymmetric penalty
  punishes severely.
- `RUL_naive_age` (uses only cycle) beats every cycle-blind Ridge variant.
  This is a warning that lifecycle position carries a large share of the
  predictive signal (a legitimate, observable covariate — see Cycle Feature
  Analysis).
- `RUL_linear_degradation` is deliberately weak. It is the honest answer to
  "how well does a purely-causal physical trend extrapolation with no
  learned model do on FD004?" — very poorly. Its prognostic score is
  astronomically bad because a small number of engines extrapolate to
  absurd horizons despite the `rul_cap`; MAE (324 cycles) is above even
  the naive-mean baseline. That is a *finding*, not a bug: single-sensor
  linear degradation is not what FD004 is.

---

## RUL Error Analysis

Anchored on the best RUL baseline `RUL_gb_regimeB_withCyc_cfgA`. See:

- Fig 1 `results/figures/phase_2/fig1_pred_vs_actual.png` — scatter against
  the y=x line.
- Fig 2 `results/figures/phase_2/fig2_residual_distribution.png` — residuals
  (pred − actual) with mean overlaid.
- Fig 3 `results/figures/phase_2/fig3_error_vs_rul.png` — mean and P90 |error|
  as a function of actual RUL.
- Fig 4 `results/figures/phase_2/fig4_error_by_regime.png` — error by
  operating regime (all six regimes present).

Patterns:

- **Not systematically biased.** Mean residual is small (see Fig 2). MAE and
  RMSE ratio (43.2 / 32.0 ≈ 1.35) is close to the Gaussian value (≈ 1.25);
  no heavy one-sided tail.
- **Error grows at high RUL.** Fig 3 shows mean |error| rises with actual
  RUL. Long-horizon predictions are harder — the model is best near
  end-of-life where the failure signal is strongest. This is the classical
  prognostics difficulty, not a model pathology.
- **Error is uneven across regimes.** Fig 4 exposes regime-to-regime
  variation, matching the Phase 0 finding that FD004 has six very distinct
  operating conditions. HistGB + regime-normalized inputs attenuate this
  gap (compare Fig 4 across regime-B vs regime-A experiments) but do not
  eliminate it.
- **Worst-10 errors concentrate in two engines but in opposite lifespan
  regimes** — see `results/experiments/phase2/rul/RUL_gb_regimeB_withCyc_cfgA.json`
  field `worst10`. Engine 171 is the longest-lived validation engine
  (399 cycles) and is systematically under-predicted, while engine 50 is a
  relatively short-lived engine (172 cycles; rank 44/50) and is
  systematically over-predicted. The tail therefore indicates engine-level
  lifespan bias in both directions rather than a single “long-lived
  engine” failure mode. These are correlated observation rows from only two
  engines, not ten independent failure events. End-of-life predictions are
  *not* the worst-case scenario — early-cycle rows on these engines are
  (e.g. cycle = 3 with actual raw_RUL = 396, prediction = 224.7 → abs error
  171.3). See Failure Cases for the measured per-engine bias.

---

## Failure-Risk Baselines

Task B uses row-level failure labels `failure_risk_target_H = (raw_RUL ≤ H)`.
Baselines:

1. **Majority-class** (`FR_majority_H30`). Trivial floor. Predicts the label
   observed most often in TRAIN (which is `0`, "no failure", because positive
   prevalence is 12.6 %). Deterministic, no probabilities. PR-AUC and ROC-AUC
   are therefore flagged as undefined in the JSON.
2. **Logistic Regression** (`class_weight="balanced"`, `C=1.0`, `max_iter=1000`)
   over Config-A/B/B+ features, Mode-A/B normalization, cycle in / out.
3. **HistGradientBoostingClassifier.** `HistGradientBoostingClassifier` has
   no native `class_weight`; we translate `class_weight="balanced"` at fit
   time into per-row inverse-frequency `sample_weight` (see
   `src/models/gradient_boosting.py::_inverse_freq_sample_weights`), which is
   the mathematically equivalent adjustment.

We deliberately did **not** use SMOTE or any synthetic oversampling: on
time-series rows, generating convex combinations of sensor snapshots can
manufacture physically impossible trajectories. Sample weighting preserves
every real observation and simply rebalances their loss contribution.

---

## Failure-Risk Results

Best baseline at the default 0.5 threshold and at the F1-argmax candidate:

| experiment_id | model | cycle | thr | acc | recall | precision | F1 | PR-AUC | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **FR_gb_regimeB_withCyc_cfgA** | HistGB | yes | 0.5 | 0.965 | **0.960** | 0.801 | **0.873** | **0.968** | 0.995 |
| FR_gb_regimeB_withCyc_cfgA | HistGB | yes | 0.819 (cand) | 0.975 | 0.898 | **0.905** | 0.902 | 0.968 | 0.995 |
| FR_gb_regimeB_noCyc_cfgA | HistGB | no | 0.5 | 0.957 | 0.952 | 0.764 | 0.848 | 0.960 | 0.993 |
| FR_gb_globA_noCyc_cfgA | HistGB | no | 0.5 | 0.955 | 0.953 | 0.754 | 0.842 | 0.957 | 0.993 |
| FR_logreg_regimeB_noCyc_cfgA | LogReg | no | 0.5 | 0.953 | 0.959 | 0.741 | 0.836 | 0.948 | 0.992 |
| FR_logreg_globA_noCyc_cfgA | LogReg | no | 0.5 | 0.950 | 0.953 | 0.731 | 0.827 | 0.943 | 0.991 |
| FR_logreg_globA_withCyc_cfgA | LogReg | yes | 0.5 | 0.950 | 0.952 | 0.730 | 0.826 | 0.940 | 0.990 |
| **FR_majority_H30** | Majority | — | 0.5 | **0.874** | **0.000** | 0.000 | 0.000 | undefined | undefined |

**The 87.4 % accuracy of the majority baseline is the phase's most important
finding.** A model that literally does nothing — ignores every sensor,
every operational setting, every cycle — produces accuracy that superficially
matches the portfolio's prior "87 % validation accuracy" claim. That claim
therefore carries no information about predictive skill on a 12.6 %-positive
task.

Every trained baseline (LogReg, HistGB) is far above this floor when judged
on the metrics that matter (recall, precision, F1, PR-AUC). HistGB with
cycle and regime-B normalization:

- Catches **96.0 %** of imminent failures at thr = 0.5 (1488 / 1550 TPs)
  while only raising **369 false alarms** out of 10 760 negatives.
- Reaches **PR-AUC 0.968**, versus a no-skill prior of 0.126.

Confusion matrix (best model at thr = 0.5, validation rows):

```
             pred=0    pred=1
actual 0    10 391       369
actual 1        62     1 488
```

---

## Threshold Analysis

Phase 2 spec §17 explicitly forbids "finalising" a business threshold on
validation. What we do instead:

- Sweep 200 probability cuts from 0.0 → 1.0 per experiment; record precision,
  recall, F1, accuracy at every cut in the per-experiment JSON
  (`threshold_sweep` field).
- Report a **candidate** threshold from F1-argmax (via
  `src/evaluation.metrics.suggest_candidate_threshold`).
- Figures 7, 8, 9 plot recall, precision, F1 against threshold for the best
  FR model, with both the 0.5 default and the candidate marked.

For `FR_gb_regimeB_withCyc_cfgA`:

- Default thr = 0.5: recall 0.960, precision 0.801, F1 0.873.
- Candidate thr = **0.819**: recall 0.898, precision 0.905, F1 0.902.

Trade-off: the candidate threshold sacrifices 6 pp of recall to gain 10 pp
of precision. Because the operational priority in Phase 2 §18 is failure
**recall**, the candidate threshold is offered but **not adopted**. Any
final threshold must be chosen with a documented cost model (downtime cost
vs false-alarm cost) after model comparison, not by F1-argmax alone. This
is the "threshold is not a modelling hyperparameter, it is a business
decision" stance.

---

## Horizon Analysis

Sensitivity at H ∈ {14, 30, 50} (candidate threshold, no cycle, regime-B):

| model | H | positive_rate | thr | recall | precision | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGB | 14 | 6.09 % | 0.884 | 0.859 | 0.847 | 0.853 | 0.935 |
| HistGB | 30 | 12.59 % | 0.769 | 0.903 | 0.868 | 0.885 | 0.960 |
| HistGB | 50 | 20.71 % | 0.704 | 0.897 | 0.878 | 0.878 | 0.958 |
| LogReg | 14 | 6.09 % | 0.819 | 0.903 | 0.748 | 0.818 | 0.898 |
| LogReg | 30 | 12.59 % | 0.784 | 0.876 | 0.862 | 0.869 | 0.948 |
| LogReg | 50 | 20.71 % | 0.603 | 0.872 | 0.856 | 0.864 | 0.946 |

Fig 10 shows the three-panel curve. Observations:

- **H=30 is the sweet spot.** Positive class is larger than at H=14 (easier
  to learn) and closer to end-of-life than H=50 (cleaner signal). PR-AUC
  peaks at H=30 for both model families.
- **H=14 is a genuinely harder task.** 6 % prevalence and only 14 cycles of
  warning window; the classifier has less failure-time evidence to lean on
  near the boundary. Recall at candidate threshold drops by ≈ 4 pp from
  H=30 to H=14 for HistGB.
- **H=50 inflates recall but blurs operational meaning.** At H=50, "failure
  within 50 cycles" starts to include engines still mid-life; PR-AUC stays
  high (0.958) but the label becomes closer to "will fail sometime soon".

Horizon is a *problem-definition* choice, not a model choice. The
sensitivity runs confirm that H=30 remains the right primary target under
the Phase 1 spec.

---

## Feature Ablation

Three sensor configurations (A=21, B=18 dropping duplicates, B+=17 dropping
sensor_16 in addition). Ridge / HistGB / LogReg / HistGB-classifier all
compared where possible:

| path | cfg | MAE (RUL) | F1 @ thr 0.5 (FR) |
|---|---|---:|---:|
| Ridge (Mode-A, no cycle) | A | 44.87 | — |
| Ridge (Mode-A, no cycle) | B | 44.89 | — |
| Ridge (Mode-A, no cycle) | B+ | 44.89 | — |
| HistGB (Mode-B, no cycle) | A | 36.63 | — |
| HistGB (Mode-B, no cycle) | B+ | 36.74 | — |
| LogReg (Mode-B, no cycle) | A | — | 0.836 |
| LogReg (Mode-B, no cycle) | B | — | 0.828 |
| LogReg (Mode-B, no cycle) | B+ | — | 0.828 |

Removing the three duplicated sensors (B) and sensor_16 (B+) changes metrics
by less than 0.5 % on every model. Sensor_16 is not contributing useful
signal, and the duplicates are truly redundant, so the reduced configs give
a small computational win with no accuracy loss — but no *predictive* win.
The Phase 0 finding that sensor_16 is nearly constant holds up.

Winner is not chosen here. The spec explicitly said "Do not choose the
winner yet." Feature-config selection can wait until Phase 3 model
comparison, where parsimony might matter for latency.

---

## Normalization Ablation

Mode A = global z-score (train-only). Mode B = regime-conditioned z-score
(train-only, six scalers). Direct comparisons:

| model | task | cycle | Mode A | Mode B |
|---|---|---|---:|---:|
| Ridge | RUL MAE | no | 44.87 | 43.91 |
| HistGB | RUL MAE | no | 37.52 | 36.63 |
| LogReg | FR F1 @ 0.5 | no | 0.827 | 0.836 |
| HistGB | FR F1 @ 0.5 | no | 0.842 | 0.848 |

Regime-conditioned normalization (Mode B) wins consistently:

- Ridge MAE improves by 2.1 % (linear model — normalization changes feature
  scaling, so this is meaningful).
- HistGB MAE improves by 2.4 % despite trees being scale-invariant per
  feature. The gain is because regime-B changes the *effective split grid*
  relative to regime membership, letting splits align with within-regime
  deviations rather than cross-regime absolute differences.
- LogReg F1 improves by 0.9 pp; HistGB F1 by 0.6 pp.

The improvement is real but modest. Phase 0's dramatic regime effects are
partially captured by any model that has access to the operational settings
(s1/s2/s3 identify the regime almost perfectly). Mode B helps at the margin,
mostly on linear models.

For trees with the operational settings present, scaling is largely
irrelevant in principle — but regime-B's per-regime z-scores are a *feature
transformation*, not just a scale change, so it does help.

---

## Cycle Feature Analysis

Phase 2 §22 demands explicit ablation. Both directions:

**Task A (RUL):**

| model | cycle | MAE | RMSE | R² | PS_last |
|---|---|---:|---:|---:|---:|
| Ridge | no | 44.87 | 55.90 | 0.602 | 11 214 |
| Ridge | **yes** | **38.33** | **48.06** | **0.706** | 1 243 |
| HistGB | no | 36.63 | 50.43 | 0.676 | 75.8 |
| HistGB | **yes** | **32.01** | **43.25** | **0.762** | **59.0** |

Adding `cycle` improves MAE by ≈ 14.6 % for Ridge and ≈ 12.6 % for HistGB.
The lift is large and not hidden.

**Task B (Failure Risk H=30):**

| model | cycle | recall | precision | F1 @ 0.5 |
|---|---|---:|---:|---:|
| HistGB (mode-B, cfgA) | no | 0.952 | 0.764 | 0.848 |
| HistGB (mode-B, cfgA) | **yes** | **0.960** | **0.801** | **0.873** |
| LogReg (mode-A, cfgA) | no | 0.953 | 0.731 | 0.827 |
| LogReg (mode-A, cfgA) | yes | 0.952 | 0.730 | 0.826 |

HistGB benefits (+2.5 pp F1) from cycle; LogReg is essentially indifferent
because it already gets a linear cycle proxy from the (already-normalized)
sensor set.

**Interpretation.** Cycle position is a *legitimate inference-time observable*
— the operator knows how many cycles the engine has logged. It is **not
leakage**. It is, however, strongly target-correlated: within an engine,
`raw_RUL = max_cycle − cycle`, so a model with access to `cycle` is directly
informed of lifecycle position and can solve much of the ranking problem
without interpreting sensor content. This is why the strongest evidence for
a Phase 3 sequence model must come from a **cycle-blind comparison**, not
just from an overall MAE reduction. If an LSTM only wins by recovering the
lifecycle-position information that HistGB already extracts from `cycle`, it
is not adding prognostic value.

`cycle` remains behind the `allow_cycle=True` gate in
`src/models/features.py` and is still rejected by Phase 1's
`FORBIDDEN_FEATURE_COLS`. Phase 3 must decide explicitly whether to keep it.

---

## Class Imbalance

Validation partition (50 engines, 12 310 rows):

| horizon | positive rows | negative rows | positive rate | engines with any positive |
|---:|---:|---:|---:|---:|
| H=14 | 750 | 11 560 | 6.09 % | 50 / 50 |
| H=30 | 1 550 | 10 760 | 12.59 % | 50 / 50 |
| H=50 | 2 550 | 9 760 | 20.71 % | 50 / 50 |

Train partition H=30 positive rate = 12.61 %, essentially identical to
validation. Not fabricated: computed directly from
`failure_risk_target_H = (raw_RUL ≤ H)` on the Phase 1 manifest rows.

Because every engine eventually fails, every validation engine has at least
one positive row. The imbalance is at the *row* level within an engine's
life, not the engine level. This is why a majority-class predictor reports
high accuracy on row counts while providing no failure-detection value.

Mitigation strategy (per Phase 2 §16):

- **Class-weighted loss for LogReg** (`class_weight="balanced"`).
- **Inverse-frequency sample-weight for HistGB** (equivalent adjustment;
  see `src/models/gradient_boosting.py::_inverse_freq_sample_weights` and
  the unit test that verifies a 9×-rarer class receives 9×-higher weight).
- **No synthetic oversampling** (SMOTE forbidden by spec).
- **Cost-sensitive reporting** — precision and recall shown together, PR-AUC
  preferred over ROC-AUC on this imbalance ratio, per Davis & Goadrich 2006.

---

## Failure Cases

Best RUL model worst-10 (from
`results/experiments/phase2/rul/RUL_gb_regimeB_withCyc_cfgA.json`,
field `worst10`):

- All 10 rows belong to two engines: **unit_id 171** and **unit_id 50**.
- Most errors are at very low cycles (3, 4, 5, 8) — early in the engine's
  life. Predictions underestimate actual RUL by 160–170 cycles on these
  rows.
- Measured per-engine bias (validation rows of the best RUL model):
  - **Engine 171** is the longest-lived validation engine (399 cycles;
    validation median lifespan 223). It is chronically **under-predicted**
    (mean signed error ≈ −72 cycles; 93 % of its rows under-predicted).
  - **Engine 50** is one of the shortest-lived validation engines (172
    cycles; rank 44 of 50). It is chronically **over-predicted** (mean
    signed error ≈ +91 cycles).
- Interpretation: the dominant error mode is engine-level lifespan bias —
  the model shrinks predictions toward the training prior, which fails in
  **both** directions for engines at the extremes of the lifespan
  distribution. Note these worst-10 entries are 10 correlated observation
  rows from 2 engines, not 10 independent failure events.

Unlike the model's aggregate row-level bias (small; mean signed error +5.9
cycles across validation rows), the tail is engine-specific. For over-
predicted short-lived engines, the prognostic score penalises the dangerous
direction heavily (over-prediction ⇒ maintenance scheduled too late), while
the under-predicted long-lived engines produce conservative (earlier)
maintenance estimates.

Best FR model failure cases at the F1-argmax threshold (thr = 0.819):

- 158 FN rows (positive rows missed within H=30) out of 1550 positive rows.
  Recomputed from the saved validation predictions: 91.1 % of these FN rows
  have `raw_RUL` in (20, 30] — they cluster near the horizon boundary.
- 146 false-positive validation rows among 10,760 negative rows,
  corresponding to a 1.4 % row-level false-positive rate. No operational
  tolerability can be inferred from this rate without a business cost model.
- These TP/FP/TN/FN counts are row-level and temporally correlated within
  engines; they must not be interpreted as counts of independent failure
  events.

The remaining recall gap is concentrated at the boundary. It is consistent
with a scenario in which a temporal model with multi-cycle history could
help disambiguate — a single row at cycle t contains only a point-in-time
snapshot.

---

## Best Baseline

**Task A (RUL):** `RUL_gb_regimeB_withCyc_cfgA`
(HistGradientBoostingRegressor, Mode-B regime normalization, Config-A
features, cycle in).

- MAE = 32.01 cycles, RMSE = 43.25, R² = 0.762, PS_last = 58.99.
- Beats naive-mean MAE by 56 % and age-baseline MAE by 48 %.
- Beats the same model without cycle by 12.6 % MAE.

**Task B (Failure Risk H=30):** `FR_gb_regimeB_withCyc_cfgA`
(HistGradientBoostingClassifier, class-weight balanced via sample weights,
Mode-B regime normalization, Config-A features, cycle in).

- At default thr = 0.5: recall 0.960, precision 0.801, F1 0.873, PR-AUC 0.968.
- At candidate thr = 0.819: recall 0.898, precision 0.905, F1 0.902.
- The trained baselines substantially outperform the majority baseline on
  all defined predictive metrics (recall, precision, F1, PR-AUC). The
  majority baseline has higher accuracy because the H=30 task has only
  12.6 % positive rows; its PR-AUC and ROC-AUC are undefined (no
  probabilities), not beaten.

Both winners share the same recipe — regime-conditioned normalization,
full 21-sensor Config A, cycle included. The choice is **emergent from
validation**, not tuned to a target.

---

## What the Results Actually Show

Stripped of optimism:

1. **Accuracy is a liar on this dataset.** Majority class achieves 87.4 %
   by predicting a constant. Any portfolio narrative anchored on "87 %"
   must specify the metric and the baseline comparison.
2. **Cycle position explains much of the RUL signal.** The single feature
   `cycle` lets a one-feature age baseline beat every cycle-blind Ridge
   variant. `cycle` is a legitimate inference-time observable, not leakage,
   but it is strongly target-correlated because `raw_RUL = max_cycle −
   cycle` by construction. Models extract substantial performance from this
   lifecycle-position information, and any future model must be shown to
   add value beyond it.
3. **Regime-conditioned normalization gives only ≈ 2 % MAE improvement**
   over global normalization once regime information is already implicit
   in the operational settings. Phase 0's dramatic regime heterogeneity
   is real but is not, by itself, predictive leverage.
4. **Sensor ablation is nearly free.** Configs A / B / B+ are
   indistinguishable in MAE. The three duplicated sensors and sensor_16
   contribute essentially nothing.
5. **The classical HistGB baseline sets a measured reference point.**
   Published C-MAPSS results are highly protocol-dependent, including
   differences in RUL capping, preprocessing, evaluation point, and split
   methodology. Therefore the Phase 2 validation results are not directly
   comparable to published test-set scores, and no numerical literature
   benchmark is invoked here. Any future model claim must be measured
   against these registered baselines under identical conditions.
6. **The linear-degradation baseline is honest.** Causal, transparent, and
   very bad (MAE = 324). This is a genuine finding about FD004: it is not
   a single-sensor linear-degradation problem. Any physical model that
   assumes linear HI-to-failure will fail.

---

## Whether LSTM Is Justified

Phase 2 has *not* been set up to answer this — no LSTM was trained. But the
baseline evidence shapes what an honest answer must contain:

**Arguments FOR a temporal / sequence model eventually:**

- Row-level models see one snapshot. The failure-case analysis shows that
  boundary rows (raw_RUL near H) are the hardest — precisely where
  trajectory curvature would disambiguate.
- The best baseline still misses 62 failures out of 1550 at thr = 0.5
  (recall 0.960). Whether a sequence model can close that without
  exploding false alarms is a legitimate open question.
- FD004 has two fault modes; different modes may have different temporal
  signatures that a point-in-time classifier cannot separate.

**Arguments AGAINST (i.e. LSTM is NOT automatically justified):**

- The strong `cycle` effect (12–15 % MAE from a single, legitimate
  observable) suggests any "improvement" a sequence model shows could be
  re-learning the same lifecycle-position signal with more parameters. A
  convincing LSTM result needs a cycle-blind comparison against a
  cycle-blind classical baseline.
- HistGB + regime-B + cycle reaches R² = 0.762 with 200 trees and default
  hyperparameters. That is not a floor begging to be broken.
- Deep sequence models add: hyperparameter search, non-determinism (cudnn
  / GPU), substantially longer training, higher inference cost, harder
  interpretability, larger overfitting risk on 199 engines.
- The worst-10 residual analysis ties the largest errors to engines at the
  extremes of the lifespan distribution. If that pattern reflects
  training-data scarcity at the lifespan extremes rather than a limitation
  of point-in-time inputs, more parameters alone will not fix it. This
  remains a hypothesis; it is not proven by Phase 2.

**Verdict at end of Phase 2:** the case for an LSTM is **not yet
established**. Before Phase 3 authorizes a deep sequence model, the
project lead should require at least: (a) a cycle-blind baseline
comparison; (b) a rolling-window / causal-rolling-features baseline that
captures trajectory curvature without neural nets; and (c) a cost model
translating the recall/precision frontier at the candidate threshold into
operational units. If LSTM is then pursued, its superiority must be
demonstrated on the *sealed official test set* (unlocked only after model
selection, per Phase 2 §26), not on validation.

This project does **not** claim any LSTM superiority at the end of Phase 2.

---

## Limitations

1. **Single dataset.** FD004 only. FD001–FD003 (single-fault, different
   regime counts) are not evaluated here. Cross-dataset generalisation is
   untested.
2. **No windowing or rolling features.** Every model sees exactly one row
   of 24/21/20 features. This is deliberate per §21 ("Start with the raw
   Phase 1 features. Only add rolling features if the baseline requires
   them"), but it caps what any baseline can extract about *degradation
   velocity*. A Phase 3 with causal rolling features would be the natural
   next step.
3. **Validation-based model selection.** 25 experiments were evaluated on
   one 50-engine validation partition. Selection noise is non-zero. With
   ~12 310 rows the standard error on MAE is small but on PR-AUC tail
   counts (missed failures) it is material. Cross-validation over engine
   folds would be more robust but was out of scope.
4. **No cost model.** Threshold candidate is chosen by F1-argmax, which is
   an arbitrary utility function. The actual operational cost of one false
   negative vs one false positive is unknown and must be supplied by the
   business before a final threshold is set.
5. **Test set remains sealed.** Consistent with §26. Absolute performance
   on the official FD004 test set is unmeasured and may differ from
   validation. For clarity: this phase uses three distinct RUL quantities —
   (1) `raw_RUL`, uncapped, computed within each FD004 training-partition
   trajectory (the Phase 2 target); (2) an optional 125-cycle training cap
   that is a modelling convention from the C-MAPSS literature and is NOT
   applied anywhere in this project; and (3) the supplied
   `RUL_FD004.txt` ground truth — per the NASA readme, test trajectories
   "end some time prior to system failure" and the file provides true
   remaining-cycle values. The supplied data files and readme contain no
   statement establishing that the official test ground truth is capped at
   125, so no cap is assumed or compared against here.
6. **Sensor_16 dropped only in Config B+.** Config A retains it. We have
   not verified that sensor_16 contributes *nothing* on Task B — only that
   removing it changes metrics by < 0.5 % on the model families tested.
7. **No uncertainty estimation.** Predictions are point estimates. A
   prognostics system should output RUL distribution (e.g. via
   quantile-regression or bootstrap), which is not part of the classical
   baselines used here.
8. **Determinism.** All models use `random_state=42` (or, for
   `LinearDegradationRUL`, no RNG at all). The determinism tests pass. But
   re-executing HistGB on a different BLAS / OpenMP thread count can shift
   results in the last few bits — Phase 3 should freeze the exact
   scikit-learn version (recorded in `results/audits/environment.json`).

9. **Row-level metric dependence.** Phase 2 classification and regression
   metrics are computed at the observation/row level. Rows from the same
   engine are temporally correlated and therefore are not independent
   statistical samples. Long-lived engines contribute more rows and
   consequently receive greater weight in aggregate row-level metrics. The
   validation partition contains 50 independent engines, not 12 310
   independent observations. Specifically: MAE is row-weighted; PR-AUC is
   row-weighted; confusion-matrix counts (TP/FP/TN/FN) are row-level; and
   none of these should be described as counts of independent failure
   events. Engine-level aggregation was not computed in this phase and is
   not fabricated here.

---

## Phase 2 Exit Criteria — Compliance

Cross-check against §31:

| criterion | status |
|---|---|
| All baseline models reproducible | ✅ `TestDeterminism` (4 tests) + `run_manifest.json` |
| Train / validation isolation intact | ✅ `TestSplitIntegrity` + `verify_phase2_gate.py` §2 |
| No official test data used | ✅ `TestTestIsolation` + gate scan of src/models, src/evaluation, src/pipeline, src/features, scripts/*phase2* |
| Metrics computed correctly | ✅ `TestPrognosticScore`, `TestThresholdAnalysis`, `TestTargetAlignment` |
| Class imbalance explicitly reported | ✅ `class_balance` in every Task-B JSON + gate §6 |
| Threshold analysis exists | ✅ `threshold_sweep` + `candidate_threshold` in every probabilistic FR JSON + figures 7–9 |
| RUL error analysis exists | ✅ `worst10` in every RUL JSON + figures 1–4 |
| Feature / normalization ablations tracked | ✅ Configs A / B / B+ × Modes A / B × cycle on/off in `registry.json` |
| Experiment registry complete | ✅ 25 experiments with all required fields, hashes, manifest_ref |
| All tests pass | ✅ 89 passed (Phase 0 + Phase 1 + Phase 2) in 46.1 s |

---

## Hard Stop Confirmation

Phase 2 is complete. This project **stops here**.

- No LSTM has been trained.
- No GRU, Transformer, or any other deep network has been trained.
- No portfolio metric has been updated.
- No claim of LSTM superiority is made or implied.

Return artifacts:

1. This report (`reports/PHASE_2_BASELINE_RESULTS.md`).
2. Experiment registry (`results/experiments/phase2/registry.json`) with
   per-experiment JSONs under `rul/` and `failure_risk/`.
3. Comparison CSVs (`rul_comparison.csv`, `failure_risk_comparison.csv`).
4. Test results (`tests/test_phase2.py`, plus existing `tests/test_data.py`,
   `tests/test_phase1.py`; run via `python -m pytest tests/`).
5. Best baselines:
   - Task A — `RUL_gb_regimeB_withCyc_cfgA` (HistGB + regime-B + cycle, MAE 32.0).
   - Task B — `FR_gb_regimeB_withCyc_cfgA` (HistGB + class-weight via sample-weights, recall 0.960 @ thr 0.5, PR-AUC 0.968).
6. Whether LSTM is justified: **not yet established.** See Section 17 for the
   evidence needed before a Phase 3 authorization decision.

Awaiting project-lead review.
