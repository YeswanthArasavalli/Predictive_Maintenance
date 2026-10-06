# Phase 0 — Data Audit Report

**Project:** Predictive Maintenance — Multivariate Failure-Risk & Remaining Useful
Life (RUL) Forecasting
**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001–FD004)
**Phase scope:** Repository initialization, dataset preservation & integrity,
data-quality / sensor / operating-condition / degradation / RUL / leakage /
feasibility analysis. **No models were trained** (LSTM, GBM, ARIMA, or otherwise).
**Machine-readable evidence:** `results/audits/*.json`, `results/tables/*.csv`,
figures in `results/figures/phase_0/`. Companion: `reports/LEAKAGE_AUDIT.md`.

> **Final-gate note:** the ten scientific gate conditions raised at review are
> resolved in `reports/PHASE_0_FINAL_GATE.md` (RUL clipping, normalization claim
> re-audit, window/validation design, ARIMA, portfolio claims) and
> `reports/FD004_DISCREPANCY.md` (train/test count discrepancy). Where this report
> and the final-gate report differ, the **final-gate report supersedes**.

---

## 1. Executive summary

- The supplied C-MAPSS bundle is **clean and internally consistent**: 0 nulls,
  0 infinite values, 0 duplicate rows, 0 duplicate (unit, cycle) pairs, and every
  engine's cycles are consecutive `1..n` across all eight train/test files.
- **All 13 raw files pass SHA-256 integrity verification** against
  `data/.checksums.txt` (`results/audits/integrity.json`, `all_match = true`).
- Two **documentation defects in the pre-existing scaffolding were found and
  fixed** (see §3, §4): a truncated/corrupted checksum for `test_FD002.txt` in
  `data/raw/README.md`, and an incorrect "26 sensors" schema claim (the dataset
  has **21 sensors / 26 columns**).
- One **readme-vs-data discrepancy** was confirmed: `readme.txt` states FD004 has
  248 train / 249 test engines, but the files actually contain **249 train / 248
  test**. The FD004 test count matches `RUL_FD004.txt` (248) exactly, so
  scoring alignment is sound. Reported, not "corrected".
- **FD002/FD004 have six dominant operating regimes** that account for the large
  majority of sensor variance (median per-sensor between-regime share ≈ 99.97%,
  20/21 sensors >95%); the degradation signal is only clearly visible *within* a
  regime. This makes **operating-condition adjustment likely beneficial and
  possibly necessary — to be validated in Phase 1**, not a proven requirement
  (see `reports/PHASE_0_FINAL_GATE.md` §4).
- Recommended Phase-1 primary dataset **FD004**, target **RUL regression (Task A)
  first**, failure-risk horizon **H = 30 operational cycles** as the primary
  candidate, and **engine-level forward-chaining validation**.
- **Verdict: PASS WITH CONDITIONS** (§ PHASE 0 VERDICT).

---

## 2. Dataset provenance

- **Name:** NASA C-MAPSS Turbofan Engine Degradation Simulation.
- **Source:** NASA Ames / Penn State via the PHM Society data challenge;
  reference Saxena, Goebel, Simon & Eklund (2008), PHM08, Denver CO.
- **Delivered as:** 13 `.txt` files + `Damage Propagation Modeling.pdf` (not a
  `.zip`); the "source artifact" is this exact file bundle. Recorded under
  `data/raw/CMAPSSData_v1.0/` as an **unmodified copy** — no alternate version
  was downloaded or substituted.
- **Immutability:** no code writes to `data/raw`; all derived artifacts go to
  `data/interim`, `data/processed`, `results/`, `reports/`.
- **Full SHA-256 checksums** (complete, untruncated) are in
  `data/.checksums.txt` and `data/raw/README.md`.

---

## 3. Dataset inventory (§7)

Source: `results/audits/inventory.json`, `results/tables/dataset_inventory.csv`.

### Training

| Dataset | Engines | Rows | Min cyc/eng | Max cyc/eng | Median | Mean |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 100 | 20,631 | 128 | 362 | 199.0 | 206.3 |
| FD002 | 260 | 53,759 | 128 | 378 | 199.0 | 206.8 |
| FD003 | 100 | 24,720 | 145 | 525 | 220.5 | 247.2 |
| FD004 | **249** | 61,249 | 128 | 543 | 234.0 | 246.0 |

### Test

| Dataset | Engines | Rows | Min cyc/eng | Max cyc/eng | Median | Mean |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 100 | 13,096 | 31 | 303 | 133.5 | 131.0 |
| FD002 | 259 | 33,991 | 21 | 367 | 132.0 | 131.2 |
| FD003 | 100 | 16,596 | 38 | 475 | 148.0 | 166.0 |
| FD004 | **248** | 41,214 | 19 | 486 | 153.5 | 166.2 |

### Test RUL (final-cycle ground truth)

| Dataset | Test engines | Min RUL | Max RUL | Mean RUL | Median RUL |
|---|---:|---:|---:|---:|---:|
| FD001 | 100 | 7 | 145 | 75.5 | 86.0 |
| FD002 | 259 | 6 | 194 | 81.2 | 80.0 |
| FD003 | 100 | 6 | 145 | 75.3 | 77.5 |
| FD004 | 248 | 6 | 195 | 86.6 | 88.0 |

**Count alignment (§7 check):** every test file's engine count equals its RUL
file's line count (100/259/100/248). ✅ No test↔RUL mismatch.

**Readme↔data discrepancy (investigated):** `readme.txt` claims FD004 = 248
train / 249 test; observed = **249 train / 248 test**. The readme appears to have
the two numbers swapped. This is a **source-documentation defect, not a data
corruption** (checksums match). It does **not** affect test/RUL alignment.
FD001/FD002/FD003 match the readme exactly. Flagged per §21 (dataset contradicts
the supplied documentation → report the actual result).

---

## 4. Data-quality findings (§8)

Source: `results/audits/data_quality.json`.

| Check | Result (all 8 train/test files) |
|---|---|
| Null values (numeric cols) | **0** |
| NaN | 0 |
| Infinite values | 0 |
| Duplicate full rows | 0 |
| Duplicate (unit_id, cycle) | 0 |
| Invalid cycle ordering | none — cycles strictly increasing per engine |
| Missing cycles | none — cycles consecutive `1..n` for every engine |
| Impossible negative sensors | none — no sensor column contains negative values |

**Negative operational settings (expected, not a defect):** `operational_setting_1`
and `_2` legitimately take negative values in the single-condition datasets
(FD001/FD003); these are condition-deviation variables (altitude/Mach offsets),
not physically-impossible sensor readings. No filtering applied.

**Constant / near-constant / low-variance sensors:**
- FD001 & FD003 (single sea-level condition): `sensor_01`, `sensor_18`,
  `sensor_19` are **exactly constant**; `sensor_10` constant in FD001 train;
  `sensor_05`, `sensor_06`, `sensor_16` near-constant / very low variance.
- FD002 & FD004 (six conditions): no sensor is perfectly constant (condition
  switching adds spread), but `sensor_16` and `sensor_19` remain near-constant
  and `sensor_16` is low-variance.
- **No observations were deleted.** These are §9 removal *candidates only*.

**Scaffolding defects corrected in this phase:**
1. `data/raw/README.md` recorded a **truncated/incorrect** SHA-256 for
   `test_FD002.txt` (62 hex chars, tail copied from FD003). Replaced with the
   verified full hash `de7b5bf7…8d9b4e02` (matches `data/.checksums.txt`).
2. `configs/project.yaml` declared `n_sensors: 26`; corrected to **21** (26 total
   columns). `data/raw/README.md` schema listing corrected from `sensor_26` to
   `sensor_21` with an explanatory note about the readme wording quirk.
3. `src/data/loader.py` pointed at a non-existent `data/raw/CMAPSSData/`; now
   resolves the real `data/raw/CMAPSSData_v1.0/` from config.

---

## 5. Sensor analysis (§9)

Source: `results/tables/sensor_summary_<FD>.csv`,
`results/tables/sensor_correlation_<FD>.csv`,
`results/audits/sensor_analysis.json`. Per-sensor mean, std, min, max, median,
q01/q25/q75/q99, variance, missingness and unique-value counts are tabulated for
all 21 sensors in every dataset.

**FD004 highlights:**
- Highest raw variance: `sensor_09` (~1.14e5), `sensor_07`/`sensor_18`/`sensor_08`
  (~2.1e4). These are high-magnitude thermodynamic/rotational channels.
- **Perfectly redundant pairs (Pearson = 1.00):** `sensor_07 ≡ sensor_12`,
  `sensor_08 ≡ sensor_18`, `sensor_13 ≡ sensor_19`. `sensor_20 ≡ sensor_21`
  (r = 0.9999). Retaining both of each pair adds no information.
- Near-constant candidate: `sensor_16` (std < 0.1).
- **No sensor is dropped in Phase 0.** Recommended future-removal candidates:
  the exact-duplicate members and `sensor_16` (see §Verdict).

See **Fig 3** (variance distribution) and **Fig 4** (correlation matrix).

---

## 6. Operating-condition analysis (§10)

Source: `results/audits/operating_conditions.json`. Regimes are detected by
rounding the three operational settings and counting distinct combinations.

- **FD002 and FD004 both resolve to exactly 6 regimes** (train and test),
  well-populated (each ≥ ~8k rows in FD004 train). Representative regime keys
  (setting1, setting2, setting3): `(42,0.84,100)`, `(0,0,100)`, `(10,0.25,100)`,
  `(35,0.84,100)`, `(25,0.62,60)`, `(20,0.7,100)`.
- **Sensor variance is dominated by *between*-regime differences.** Re-audited
  across **all 21 sensors** (FD004 train): median between-regime variance fraction
  **0.9997**, mean 0.9924, **min 0.9084** (`sensor_16`), **20/21 sensors >95%**;
  FD002 is nearly identical (median 0.9998, min 0.8854). The per-sensor ratio is
  invariant to sensor rescaling, and the pooled figure is ~0.99 whether
  variance-weighted (0.9972) or standardized (0.9919), so it is **not materially
  distorted by sensor scale** and is **stable across FD002/FD004**.
- **Consequence:** raw (pooled) sensor↔cycle correlation is dominated by regime,
  not degradation — FD004 pooled |r| with cycle is ≤ ~0.10, whereas single-condition
  FD001 reaches |r| ≈ 0.6 for the same channels.

**Recommendation (not implemented in Phase 0, per §10):** operating-condition
adjustment is a **design option to validate in Phase 1**, not a proven
requirement. A high between-regime variance share shows regime is a strong
*confounder* of raw levels; it does **not** by itself prove that regime-conditioned
normalization improves predictive performance (the small within-regime residual is
where the degradation signal lives). **Primary candidate: regime-conditioned,
train-only-fitted normalization (Option B), with global train-only normalization
(Option A) as the ablation to measure the actual benefit.** Regime definitions must
be a deterministic function of the operational settings (or a train-fit clusterer)
to avoid leakage. See **Fig 6**, **Fig 7**, and `reports/PHASE_0_FINAL_GATE.md` §4.

---

## 7. Degradation analysis (§11)

Source: `results/audits/degradation_rul.json`, **Fig 5**.

- Degradation is **not uniform across sensors** and **not globally monotonic**.
  In FD001 (single condition) the clearest run-to-failure indicators are
  `sensor_11` (r=+0.63), `sensor_04` (+0.62), `sensor_12` (−0.61), `sensor_07`
  (−0.60), `sensor_15` (+0.59), `sensor_21` (−0.59), `sensor_17` (+0.57).
- Trajectories are **gradual with accelerating end-of-life components** and
  noticeable per-engine offsets (initial wear/manufacturing variation), consistent
  with the C-MAPSS damage-propagation model.
- Degradation is **regime-dependent in FD002/FD004**: raw multi-sensor plots are
  dominated by condition switching; the true trend is recoverable only after
  per-regime normalization.
- **Not every sensor is a degradation indicator** — flat/low-variance channels
  (`sensor_01/05/06/10/16/18/19`) carry little trend.

---

## 8. RUL analysis (§12)

Source: `src/data/loader.derive_train_rul`, `results/audits/degradation_rul.json`.

- Training RUL defined as `max_cycle(engine) − current_cycle`.
- **Verified:** RUL decreases by exactly 1 per cycle within each engine; every
  engine terminates at RUL = 0 at its final observed cycle
  (`every_engine_ends_at_zero_rul = true` for all four datasets).
- FD004 train RUL range 0–542 (mean ≈ 133); FD001 0–361 (mean ≈ 108).
- Test RUL (supplied final-cycle values) associated 1:1 with each test engine
  (`check_test_rul_alignment` passes for all datasets; unit-tested).
- **No future information is used as a feature** — the engine's `max_cycle` is
  used only to build the *target*; enforced by
  `test_rul_uses_only_own_engine_history`. See `reports/LEAKAGE_AUDIT.md`.
- **No model was trained.** See **Fig 2** (RUL distribution).

---

## 9. Candidate failure-risk horizons (§15)

Formulation: `P(failure within H future operational cycles)`, label = 1 iff
`RUL ≤ H`. Source: `results/audits/target_design.json`, **Fig 8**. Values are
**operational cycles, NOT calendar days** (§2/§21).

FD004 (primary):

| H (cycles) | Positives | Negatives | Positive rate | Imbalance (1:N) | Pos windows/engine |
|---:|---:|---:|---:|---:|---:|
| 14 | 3,735 | 57,514 | 6.1% | 1 : 15.4 | 15 |
| 30 | 7,719 | 53,530 | 12.6% | 1 : 6.9 | 31 |
| 50 | 12,699 | 48,550 | 20.7% | 1 : 3.8 | 51 |

- Every engine contributes positives (min engine life 128 ≫ 50), so no engine is
  all-negative. Positive windows per engine = H+1 (the trailing H+1 cycles).
- **Labels overlap heavily** (adjacent rows share the same label) → any windowed
  evaluation must be de-duplicated at engine level to avoid optimistic bias.
- H=14 is workable but strongly imbalanced (~15:1); H=50 eases imbalance but
  dilutes the "imminent failure" meaning. **H=30 is the balanced primary
  candidate** (~7:1, still operationally meaningful).
- FD001–FD003 show the same monotone pattern (positive rate grows with H),
  confirming the recommendation generalizes.

**Recommended Phase-1 candidate formulation:** `failure within 30 operational
cycles`, with 14 and 50 retained as sensitivity analyses. Class imbalance to be
handled by weighting / threshold tuning (not row shuffling — see leakage audit).

---

## 10. Leakage audit summary (§13)

Full detail in `reports/LEAKAGE_AUDIT.md`. Phase 0 trains nothing and fits no
transforms, so it introduces **no leakage**. The dominant Phase-1 risks are:
row-level splitting (L1), window bleed across split boundaries (L2), fitting
scalers on val/test (L3), encoding engine lifespan as a feature (L5), and
non-causal whole-series normalization (L9). All are mitigated by the mandatory
engine-level, train-only-fit strategy; regime-conditioning is the recommended
normalization option (validated in Phase 1). The RUL/feature code
is leakage-safe and covered by unit tests.

---

## 11. Recommended split strategy (§14)

**Do not** use a random row-level `train_test_split` (violates temporal order and
engine independence). Options considered:

| Strategy | Pros | Cons |
|---|---|---|
| A. Engine-level split | Preserves within-engine temporal integrity; no cross-engine leakage | Few engines → variance; must stratify by regime/life |
| B. Temporal split (cut by cycle) | Mirrors deployment | Engines have different lifespans; awkward alignment |
| C. Rolling / forward-chaining (engine-level) | Order-preserving, uses all engines, realistic | More compute; needs careful window control |
| D. Nested CV | Unbiased hyperparameter selection | Costly; overkill for Phase 1 baselines |

**Primary recommendation: engine-level stratified split (A) for the train/val
partition, evaluated with rolling/forward-chaining (C) across engine folds, with
the official test set held out for a single final pass.** Stratify engine folds
on operating regime (FD002/FD004) and coarse lifespan so each fold is
representative. Nested CV (D) reserved for final model selection only.

---

## 12. Baseline suitability (§16)

Assessed **without training**:

**RUL (Task A):**
- *Naive* (predict mean/last RUL, or linear extrapolation of the health index):
  essential floor; must be beaten.
- *Linear / ridge regression* on regime-normalized last-cycle features: strong,
  cheap, interpretable baseline. **Recommended first model.**
- *Gradient boosting (e.g. LightGBM/XGBoost)* on engineered window features:
  typically the C-MAPSS SOTA-style tabular baseline. **Recommended.**
- *LSTM* on sequences: justified only if it beats GBM on the *same* engine-level
  split; do **not** presume superiority (§21).

**Failure-risk (Task B):**
- *Logistic regression* on regime-normalized features: interpretable probability
  baseline; good calibration reference.
- *Gradient boosting classifier*: primary tabular baseline; handle imbalance via
  class weights / thresholding, not shuffling.
- *LSTM*: sequence model, evaluate against GBM on identical splits.

**ARIMA — recommendation: EXCLUDE from the core comparison.** ARIMA models a
*single univariate series* with (near-)stationary, calendar-equally-spaced
observations and no exogenous regime handling. C-MAPSS is **multi-sensor,
engine-independent, non-stationary, and regime-switching**; RUL is a
*terminal-event countdown*, not a smooth forecastable level. Fitting ARIMA per
engine to a degrading, condition-driven series is not methodologically sound and
would not be a fair "statistical baseline". **More appropriate statistical
baseline:** a per-engine **linear degradation-trend / health-index extrapolation**
(fit a monotonic trend on the regime-normalized indicator, extrapolate to the
failure threshold), plus the naive/mean predictor. If a time-series method is
required for completeness, use **state-space / Kalman-style trend tracking**
rather than ARIMA. ARIMA should appear only as a documented, honestly-labeled
weak reference, never as a headline model.

---

## 13. Dataset limitations

- **Simulation data**, not real engine run-time; no calendar-time mapping exists.
  "1 cycle = 1 day" is **not** assumed and must never be claimed (§21).
- **FD004 readme count discrepancy** (249 vs 248) — documented above.
- **Redundant/flat sensors** inflate naive feature importance; duplicates present.
- **Severe regime confounding** in FD002/FD004 — raw statistics are misleading.
- **Heavy label overlap** in the failure-risk formulation at larger H.
- **No cost/business ground truth** in the data; any savings figure would be
  fabricated and is therefore out of scope for Phase 0.

---

## 14. Recommended Phase 1 design

1. **Primary dataset FD004**; FD001 as the clean single-condition reference;
   FD002/FD003 for ablation.
2. **Regime-conditioned, train-only-fitted normalization** (per-regime z-score).
3. **Engine-level stratified split + forward-chaining CV**; official test held out.
4. **Task A (RUL) first** with baselines: naive → linear/ridge → GBM; add LSTM
   only as a challenger on identical splits.
5. **Task B (failure-risk)** with **H = 30 cycles** primary (14/50 sensitivity),
   logistic + GBM, imbalance via weights/threshold.
6. **Metrics:** Task A — RMSE + the C-MAPSS asymmetric scoring function; Task B —
   ROC-AUC, PR-AUC, calibration. Report honestly; no superiority assumed.
7. Enforce the `LEAKAGE_AUDIT.md` Phase-1 checklist as code + tests.

---

## 15. Exact unresolved questions

1. Should the LSTM be retained at all if GBM matches it on tabular features, or
   kept only as a sequence-modeling demonstration?
2. Final failure-risk horizon: is ~7:1 imbalance at H=30 acceptable, or do we
   prefer H=50 (~4:1) and re-tune thresholds? (Requires Phase-1 experiments.)
3. Regime detection: is the rounding-based 6-regime scheme robust enough, or do
   we need clustering (e.g. KMeans on settings) for exact regime assignment?
4. RUL truncation threshold (the common "cap RUL at 125" convention): adopt or
   report uncapped? Decision deferred to Phase 1 with justification.
5. Whether to formally drop the exact-duplicate sensors now or keep them for the
   ablation table.

---

## 16. Reproducibility information

Source: `results/audits/environment.json`, `requirements.txt`, `pyproject.toml`.

- **Python:** 3.14.6 (CPython), **OS:** Windows 11 (26H2), AMD64.
- **Packages:** numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1,
  matplotlib 3.11.2, PyYAML 6.0.3, pytest 9.1.1.
- **Seed policy:** global seed **42**; `numpy.random.default_rng(42)` for any
  sampling; explicit `random_state` for every learner/splitter in later phases;
  all stochastic outputs record their seed.
- **Pipeline (deterministic, from immutable raw data):**
  ```bash
  pip install -r requirements.txt
  python scripts/run_phase0_audit.py     # integrity + all audit JSON/tables
  python scripts/phase0_figures.py        # 8 figures -> results/figures/phase_0
  python scripts/record_env.py            # environment.json
  python -m pytest                        # 11 tests
  ```
- **Integrity:** `python -c "from src.data import integrity; print(integrity.all_match())"`
  → `True`.

---

## PHASE 0 VERDICT

**PASS WITH CONDITIONS**

Phase 0 objectives are met with verified, reproducible evidence: repository
structure exists, raw data is preserved and checksum-verified (all match), and
the data-quality / sensor / operating-condition / degradation / RUL / leakage /
feasibility analyses are complete and machine-readable. No models were trained.
The "conditions" are the Phase-1 controls below (train-only-fitted normalization,
with regime-conditioning as the primary option and global as its ablation;
mandatory engine-level temporal splits; the leakage checklist)
and the honest documentation of the FD004 readme discrepancy. Full resolution of
the ten scientific gate conditions is in `reports/PHASE_0_FINAL_GATE.md`.

### Recommended primary dataset
**FD004** (six operating conditions, two fault modes — the hardest, most
realistic case). FD001 as the clean single-condition reference; FD002/FD003 for
ablation.

### Recommended target
**Task A — RUL regression** first (well-defined, standard scoring), then
**Task B — failure-risk classification** as the business-facing task.

### Recommended horizon
**H = 30 operational cycles** (≈1:7 imbalance on FD004), with **14** and **50**
retained as sensitivity analyses. Explicitly **cycles, not days**.

### Recommended validation strategy
**Engine-level stratified split with forward-chaining (rolling) CV**, stratified
on operating regime and coarse engine lifespan; official test set used once.

### Recommended baseline models
- RUL: naive → **linear/ridge** → **gradient boosting**; LSTM as a challenger only.
- Failure-risk: **logistic regression** → **gradient boosting**; LSTM as challenger.
- **Exclude ARIMA** as a core baseline (methodologically unsuited); use a
  per-engine linear-degradation / health-index trend as the statistical baseline.

### Sensors recommended for further investigation
Trend-informative (from FD001 single-condition analysis): **sensor_02, sensor_03,
sensor_04, sensor_07, sensor_08, sensor_09, sensor_11, sensor_12, sensor_14,
sensor_15, sensor_17, sensor_20, sensor_21**.
Removal candidates (recommend, do not drop yet): **sensor_16** (near-constant),
and one member of each exact duplicate pair **sensor_07/sensor_12**,
**sensor_08/sensor_18**, **sensor_13/sensor_19**.

### Known limitations
Simulation-only (no calendar mapping); FD004 readme count discrepancy (249 vs
248); strong regime confounding of raw statistics; redundant/flat sensors; heavy
label overlap in failure-risk windows; no business-cost ground truth.

### Critical risks
1. **Leakage** via row-level splitting / non-causal normalization / lifespan
   encoding (mitigated by `LEAKAGE_AUDIT.md` checklist — must be enforced in code).
2. **Misleading pooled statistics** if regimes are not separated (would understate
   or fabricate degradation signal).
3. **Over-claiming** LSTM or ARIMA superiority without identical-split evidence.
4. **Imbalanced failure-risk** labels producing inflated accuracy if evaluated
   naively (must use PR-AUC / calibration, engine-level de-duplication).

### Exact next step
Await project-lead review. If approved, **Phase 1 begins by implementing
regime-conditioned, train-only-fitted normalization and an engine-level
forward-chaining splitter (with leakage unit tests), then the naive + linear RUL
baselines on FD004** — no deep models until the tabular baselines and split
integrity are established.

---

*END OF PHASE 0. No model training performed. Awaiting project-lead review.*
