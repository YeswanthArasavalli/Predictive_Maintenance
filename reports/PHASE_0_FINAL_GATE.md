# PHASE 0 — FINAL SCIENTIFIC GATE

**Project:** Predictive Maintenance — Multivariate Failure-Risk & RUL Forecasting
**Dataset:** NASA C-MAPSS (primary FD004; ablations FD001–FD003)
**Status entering gate:** PASS WITH CONDITIONS
**Models trained:** none. **Raw data modified:** no.
**Evidence:** `results/audits/gate_*.json`, `results/audits/*.json`,
`reports/FD004_DISCREPANCY.md`, `reports/LEAKAGE_AUDIT.md`.

This document resolves the ten gate conditions. Every quantitative claim is
computed by `scripts/phase0_gate_analysis.py` (deterministic, from the immutable
raw files) and stored under `results/audits/`.

---

## 1. Dataset decision

- **Primary dataset: FD004** — six operating regimes, two fault modes (HPC + Fan
  degradation), largest row count (61,249 train / 41,214 test). Hardest, most
  deployment-realistic case.
- **Reference / clean case: FD001** — single condition, single fault mode; used to
  expose degradation trends without regime confounding.
- **Ablations: FD002** (six conditions, one fault mode) and **FD003** (one
  condition, two fault modes) to separate the effect of *operating variability*
  from *fault-mode variability*.
- All engine counts, splits and scoring are derived from **parsed file contents at
  runtime**, never from `readme.txt` (see §2).

## 2. FD004 discrepancy decision

See `reports/FD004_DISCREPANCY.md` for the full investigation. Decision:

- **Authoritative:** file contents — `train_FD004.txt` = **249** engines,
  `test_FD004.txt` = **248** engines, `RUL_FD004.txt` = **248** values (1:1 with
  test). All SHA-256 verified.
- **Documentation:** `readme.txt` "248 train / 249 test" is a **transposed
  documentary error**, retained verbatim but never used to drive code.
- **No correction to the data is justified.** The files are internally consistent
  (train engines all reach RUL=0; test engines truncated; test↔RUL counts match).
  Only documentation was updated. Raw files remain untouched.

## 3. RUL target decision (Condition 2)

We separate two quantities and **never conflate them**:

- **`raw_RUL`** = `max_cycle(engine) − current_cycle` — the *physical* remaining
  cycles to failure. This is the only quantity that may be called "true RUL".
- **`model_RUL_target`** = an optional **clipped** view of `raw_RUL`, i.e.
  `min(raw_RUL, cap)`. A clipped value is a **training convenience**, not the
  physical RUL, and must always be labelled as such in metrics and prose.

### Observed raw-RUL distribution (train)

| Dataset | max | mean | median | q90 | q99 | % rows RUL>125 | % rows RUL>100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FD001 | 361 | 107.8 | 103 | 198 | 287 | 38.9% | 51.0% |
| FD002 | 377 | 108.2 | 103 | 200 | 288 | 39.1% | 51.2% |
| FD003 | 524 | 138.1 | 123 | 273 | 441 | 49.0% | 59.1% |
| **FD004** | **542** | **133.3** | **122** | **257** | **376** | **48.8%** | **58.9%** |

### Effect of candidate caps (FD004)

| Cap | % obs clipped | # obs clipped | engines w/ ≥1 clipped | target std (raw 89.78) | mass piled at cap |
|---|---:|---:|---:|---:|---:|
| none | 0% | 0 | — | 89.78 | — |
| 125 | **48.78%** | 29,875 | 249 / 249 | 40.67 | 49.18% |
| 100 | **58.94%** | 36,100 | 249 / 249 | 30.89 | 59.35% |

### Analysis per candidate

**No cap (raw_RUL):**
- *Target distribution:* heavy right tail (0–542), std 89.8; far-from-failure
  observations are a minority but carry large values.
- *Early-life:* fully represented; early cycles have large, informative RUL.
- *Advantages:* honest; metric equals physical RUL; no artificial ceiling.
- *Disadvantages:* early-life region (RUL≫H) has almost no degradation signal,
  so a model spends capacity fitting an easy, near-linear "healthy" regime; RMSE
  is dominated by the long-early tail; the C-MAPSS scoring function already
  penalises early over/under-estimation asymmetrically, so the tail matters.
- *LSTM relevance:* the network must model a wide output range; gradients from
  the flat early region are weak/uninformative.
- *Metric implication:* RMSE/MAE reflect true life-error but are inflated by the
  early tail where all models are equally "uninformative".

**Cap = 125 (the literature default):**
- *Target distribution:* ~49% of FD004 rows collapse onto the value 125 (a large
  point mass at the cap); std halves (89.8→40.7).
- *Early-life:* every observation with >125 cycles remaining becomes
  indistinguishable (label 125) — early-life ordering is **destroyed**.
- *Advantages:* focuses learning on the prognostically useful region; reduces
  tail dominance; matches most published C-MAPSS RMSE numbers (comparability);
  125 ≈ the practical maintenance lead time.
- *Disadvantages:* ~half the training rows are clipped — a strong, arbitrary
  prior; **must never be reported as physical RUL**; not justified by *this*
  dataset's own structure (median raw RUL is 122, so the cap sits at the median,
  not in a "healthy plateau").
- *LSTM relevance:* helps — the model concentrates on the informative descent;
  but the flat cap segment can cause the net to over-predict 125 near the cap.
- *Metric implication:* RMSE becomes RMSE-*wrt-clipped-target*; must be reported
  as such and is **not comparable** to uncapped RMSE.

**Cap = 100:**
- More aggressive: ~59% clipped, std 30.9, 59% mass at the cap. Discards even
  more early-life ordering; only marginally more focused than 125. Not preferred.

### Decision

- **Primary reported target = `raw_RUL` (no cap).** This keeps the headline
  metric honest and physical.
- **A clipped `model_RUL_target` (cap = 125) is permitted only as a training
  option / ablation**, always labelled "clipped RUL (cap=125)", never described
  as the true/physical RUL, and its metrics reported separately.
- The cap value is **not fixed now**: it will be chosen in Phase 1 by comparing
  capped vs uncapped training on the **validation** folds (not the test set),
  because 125 is arbitrary relative to FD004's own median (122).
- **Hard rule (§21):** the project must never describe a clipped value as the
  physical/true RUL.

## 4. Normalization decision

### 4.1 Variance-decomposition claim — audit & revision (Condition 3)

**Original claim (Phase 0):** "99.7–99.99% of sensor variance is between regimes
→ regime-conditioned normalization is **mandatory**." This was too strong.

**How it was computed (methodology, explicitly):**
- Regimes = distinct rounded `(setting1, setting2, setting3)` combinations; FD002
  and FD004 each resolve to exactly **6** regimes.
- For a single sensor *s*: `total_var = Var(all values)`; `between_var =
  Σ_g n_g (mean_g − grand_mean)² / (N − 1)` (one-way ANOVA between-group sum of
  squares, normalised by N−1); reported quantity = `between_var / total_var`.
- Sensors included: **all 21** in this re-audit (the original figure quoted only
  three hand-picked sensors — 02, 17, 21).
- Values were **not** standardized before decomposition. This is fine for the
  *per-sensor* ratio because between/total is **invariant to affine rescaling of
  that sensor**. Scale only matters when *pooling across* sensors.

**Re-audited result (all 21 sensors, FD004 train):** median between-fraction
**0.9997**, mean **0.9924**, **min 0.9084** (`sensor_16`), **20/21 sensors
>0.95**. FD002 is nearly identical (median 0.9998, min 0.8854). **Stable across
datasets.** ✓

**Scale-distortion check:** pooling sensors, the aggregate between-fraction is
0.9972 (raw, variance-weighted) vs 0.9919–0.9924 (standardized, equal-weight).
The two agree closely, so the headline is **not materially distorted by sensor
scale differences** — but the raw pooling is mildly inflated by the few
huge-variance sensors, so the standardized figure is the one to quote.

**Why the original conclusion overreached:** a high *between-regime share of
total variance* means operating regime is a **strong confounder** of raw sensor
levels. It does **not**, by itself, prove that regime-conditioned normalization
improves the *predictive* task. The small within-regime residual variance is
exactly where the degradation signal lives (confirmed in §7 of the Phase-0
report: single-condition FD001 shows |corr(sensor, cycle)| ≈ 0.6). Whether
removing the between-regime component helps or hurts generalisation is an
**empirical, modelling question**, not a Phase-0 fact.

**Revised conclusion (no "mandatory"):**
- Operating-condition adjustment **may be beneficial** (removes a dominant
  nuisance factor; makes sensor scales comparable across regimes).
- It **may be necessary** for models that pool all regimes with a single global
  scale, because raw pooling lets regime dominate the degradation signal.
- It **can introduce leakage if implemented incorrectly** — if regime
  definitions or per-regime transform parameters are fit on validation/test data,
  or if the regime label is derived from anything other than the operational
  settings that are available at prediction time.
- Therefore: **treat regime-conditioning as a design option to be validated in
  Phase 1**, not a proven requirement.

### 4.2 Normalization options (Condition 4) — conceptual, not implemented

**Option A — Global training-only normalization.**
Fit one scaler (z-score) on all training rows; transform val/test.
- *Leakage risk:* low for the scale itself (train-fit), **but** a single global
  scale ignores regime → regime dominates features (the confounding above).
- *Deployment feasibility:* highest (one transform, no regime inference).
- *Info at prediction time:* only needs the sensor values.
- *Complexity:* lowest.
- *Pros/cons:* simple, robust; statistically mixes six regimes in FD002/FD004,
  degrading the usable signal.

**Option B — Operating-regime-conditioned training-only normalization.**
Define regimes from operational settings; fit a **separate scaler per regime on
training rows of that regime**; at predict time route each row to its regime's
scaler.
- *Leakage risk:* **material** — regime *definitions* and *per-regime parameters*
  must be learned from **train only**. Mitigation: regimes are **deterministic
  functions of the operational settings** (rounding / fixed boundaries), NOT
  unsupervised clustering on the full dataset. If clustering were used, the
  clusterer must be fit on train and applied (`.predict`) to val/test — never
  fit on held-out data.
- *Deployment feasibility:* good — regime is observable from live settings.
- *Info at prediction time:* operational settings (available), no future data.
- *Complexity:* medium.
- *Pros/cons:* removes the dominant regime confound; risk of small per-regime
  samples and of leakage if the routing is fit incorrectly.

**Option C — Operating-setting-aware normalization.**
Model each sensor as a function of the (continuous) operational settings — e.g.
regress sensor on settings and keep the **residual**, or standardise within
setting bins — then feed residuals.
- *Leakage risk:* medium; the setting→sensor model must be train-fit only.
- *Deployment feasibility:* medium (need the fitted setting-response at runtime).
- *Info at prediction time:* operational settings (available).
- *Complexity:* highest.
- *Pros/cons:* smoothest handling of regimes and any within-regime setting drift;
  more moving parts and more places to leak.

**Recommendation:**
- **Primary: Option B (regime-conditioned, train-only)** with regimes defined as
  a **deterministic function of the operational settings** (the 6 fixed setting
  clusters), so no unsupervised fitting on held-out data is possible.
- **Ablation: Option A (global, train-only)** to measure how much regime
  conditioning actually buys. Option C is deferred as a stretch ablation.
- Phase-1 must include a leakage test proving regime parameters are train-fit and
  that val/test rows are only ever *transformed*, never *fit*.

## 5. Failure-risk horizon decision (Condition 5)

Label: `y_t = 1 if raw_RUL_t ≤ H else 0`. All values are **operational cycles,
not days**. Row-level counts (each observation is a candidate prediction point;
the count of *usable* windows additionally depends on lookback W — see §6).

**FD004 (primary): 61,249 rows, 249 engines.**

| Metric | H=14 | H=30 | H=50 |
|---|---:|---:|---:|
| Candidate windows (row-level) | 61,249 | 61,249 | 61,249 |
| Positive windows | 3,735 | 7,719 | 12,699 |
| Negative windows | 57,514 | 53,530 | 48,550 |
| Positive % | 6.10% | 12.60% | 20.73% |
| Positive engines | 249 | 249 | 249 |
| Negative engines | 0 | 0 | 0 |
| Positive windows / engine (min/max/mean) | 15/15/15 | 31/31/31 | 51/51/51 |
| Max consecutive positive windows | 15 | 31 | 51 |
| Avg consecutive positive run | 15.0 | 31.0 | 51.0 |
| Earliest positive label (cycles before failure) | 14 | 30 | 50 |
| Label-overlap ratio (run/(H+1)) | 1.0 | 1.0 | 1.0 |
| Every trajectory yields positives | yes | yes | yes |

**Interpretation of the structure:**
- Because every FD004 engine lives ≥ 128 cycles ≫ 50, **every** engine contributes
  exactly **H+1** positive windows, and they are the **contiguous tail** of the
  trajectory (RUL = H, H−1, …, 0). Hence `label_overlap_ratio = 1.0`: positives
  are perfectly autocorrelated within an engine.
- **Consequence:** naive row-level accuracy is meaningless (a "always negative"
  predictor already scores 93.9% / 87.4% / 79.3% at H=14/30/50). Evaluation must
  use **PR-AUC / ROC-AUC + calibration**, and metrics must be **aggregated per
  engine** (or window-strided) so the H+1 near-duplicate positives are not
  counted as independent evidence.

**Operational meaning of each horizon:**
- **H=14:** "will fail within ~2 weeks of *cycles*" — high-actionability, very
  imbalanced (1:15.4). Tight maintenance-lead alerting.
- **H=30:** "fail within ~1 month of cycles" — balanced-ish (1:6.9), still
  operationally meaningful. Good default.
- **H=50:** "fail within ~50 cycles" — least imbalanced (1:3.8) but blurs
  "imminent" failure and dilutes the positive class with earlier, less-degraded
  states.

**Decision:**
- **Primary candidate horizon: H = 30 operational cycles.**
- **Sensitivity horizons: H = 14 and H = 50** (report all three).
- **The final horizon is explicitly provisional** — it may change after Phase-1
  baselines reveal the achievable precision/recall trade-off at each H. No horizon
  is "approved" as a result claim yet.

## 6. Window definition (Condition 6)

For engine *e* with observations ordered by cycle, let **t** index the current
(observation) cycle. Parameters: lookback **W** (input length), horizon **H**
(failure-risk window).

**Input (strictly causal, past-inclusive):**

```
X_t^(e) = [ x_(t-W+1)^(e), x_(t-W+2)^(e), …, x_t^(e) ]      (W rows, engine e only)
```

where each `x_k` is the feature vector at cycle k (regime-normalized sensors;
operational settings are allowed as they are known at time t).

**Targets:**
- **RUL (Task A):** `y_t^(e) = raw_RUL_t^(e) = max_cycle(e) − t` (regression), or
  its clipped variant `min(raw_RUL, cap)` (see §3).
- **Failure-risk (Task B):** `y_t^(e) = 1 if raw_RUL_t^(e) ≤ H else 0`.

**Valid-window set for engine e:** `{ t : t ≥ W }` (a full lookback exists) within
that engine only. Windows never start before cycle W and never cross the engine's
end.

**Explicit boundary guarantees (to be enforced in code + unit tests in Phase 1):**
1. **No future observation enters `X_t`** — the window ends at `t` and uses only
   cycles `≤ t`. (Enforced by construction + an assertion that max cycle index in
   `X_t` equals `t`.)
2. **The target uses only future/end-of-life information** — `raw_RUL_t` depends
   on `max_cycle(e)`, which is legitimately *unknown at time t* and is therefore
   **target-only**, never a feature. The engine's `max_cycle` must never appear in
   any `x_k`. (Guarded by `test_rul_uses_only_own_engine_history`.)
3. **Windows do not cross engine boundaries** — every `X_t` is built within a
   single `unit_id`. (Assertion: all rows in a window share one `unit_id`.)
4. **Windows do not cross train/val/test boundaries** — the split is made at the
   **engine** level first, then windows are generated per engine inside each
   partition; a window can never contain engines from different partitions.
   (Enforced by the engine-level splitter in §7.)
5. **Test RUL (`RUL_FD00X.txt`) is used only by the final evaluator**, never in
   window/feature construction.

No training or window materialisation is performed in Phase 0.

## 7. Validation design (Condition 7)

**Audit of "engine-level stratified split + forward-chaining".**

Stratification variable appropriateness:
- **Operating regime:** appropriate to *balance representation* (each fold should
  contain engines from all 6 regimes) because regime strongly shifts features.
  It is a **known, leakage-free** attribute (derived from settings).
- **Trajectory length / lifespan:** **use with caution.** An engine's total
  lifespan is only knowable *after* it fails; for **test** engines it is **not
  known** (they are truncated). Stratifying on lifespan can (a) leak end-of-life
  information into the split design and (b) be **impossible to compute for the
  hold-out set**. Use only a *coarse, train-only* life band for stratification of
  **training folds**, never for the test partition, and never as a feature.
- **Fault mode:** FD001/FD002 are single-fault; FD003/FD004 are two-fault, but
  **the fault mode of an individual engine is not labelled in the data**. It can
  only be *inferred*, so stratifying on it would inject a noisy, derived quantity.
  **Do not stratify on fault mode.**

**Key distinction:** each engine has a single dominant degradation trajectory, so
engine-level stratification preserves representativeness **without** manufacturing
temporal information — *provided* the stratification key is leakage-free (regime:
yes; lifespan: train-only & coarse; fault mode: no).

**Options compared:**

| Method | Handles engine independence | Handles temporal order | Notes |
|---|---|---|---|
| GroupShuffleSplit (engine-level) | ✅ (groups=engines) | ❌ (random group draw) | Good for a single train/val engine split; ignores ordering. |
| GroupKFold (groups=engines) | ✅ | ❌ | K engine-disjoint folds; no leakage across engines; not order-preserving. |
| Stratified **group** split | ✅ | ❌ | Adds regime balance across engine folds; recommended for fold construction. |
| Forward-chaining / rolling (engine-level) | ✅ | ✅ | Repeated expanding-window train, next block val; mirrors deployment. |

**Recommended exact design:**
1. **Hold out the official test set (FD004 test + `RUL_FD004.txt`) untouched**
   until a single final evaluation.
2. On **training engines only**, build **engine-grouped, regime-stratified folds**
   (GroupKFold / StratifiedGroupKFold with `groups=unit_id`).
3. Evaluate with **forward-chaining over engine folds** (expanding training
   engines, next disjoint engine block as validation) so the reported trend is
   order-preserving and deployment-realistic.
4. All scalers/regime parameters are **fit inside each training fold only**.
5. Hyperparameter selection uses these validation folds; the test set is used
   exactly once.

This combines engine independence, regime representativeness, and temporal
realism without using lifespan or fault-mode as stratification keys.

## 8. ARIMA decision (Condition 8)

**RUL prediction — ARIMA inappropriate.** ARIMA models a **single univariate,
roughly stationary, equally-spaced** series and forecasts its *future level*.
C-MAPSS RUL is (a) **multivariate** (21 sensors + 3 settings), (b)
**non-stationary by construction** (monotone-ish degradation + regime switching),
(c) defined per-engine with **no calendar equidistance**, and (d) a
**terminal-event countdown** whose "answer" is the distance to an absorbing
failure state — not a smooth extrapolatable level. Fitting ARIMA per engine to a
regime-switching, noise-contaminated degradation series violates its assumptions
and cannot ingest the multivariate signal. It would be a weak, misleading
"baseline" included only for name-recognition.

**Failure-risk prediction — ARIMA inapplicable.** Task B is a **conditional
binary classification** (`P(fail within H)`), not a continuous forecast; ARIMA has
no natural formulation for it at all.

**Decision: EXCLUDE ARIMA from the core baseline comparison.** It may appear only
as a clearly-labelled weak reference in an appendix, never as a headline model,
and never as something the proposed models "must beat" for credibility.

**Better statistical / time-series baselines (to implement in Phase 1):**
- **Naive / constant:** predict the training mean RUL (or per-cycle last value).
- **Per-engine linear degradation trend:** fit a least-squares line to a
  regime-normalized health indicator over the observed history and extrapolate to
  the failure threshold → RUL = cycles until threshold crossing. Simple,
  interpretable, leakage-free (uses only past cycles of the same engine).
- **Health-index trend:** build a scalar health index (e.g. first PCA component /
  Mahalanobis distance from the healthy operating manifold, fit on train), then
  trend-extrapolate as above.
- **Exponential degradation model:** fit `h(t) = a + b·exp(κ t)` to the health
  index for accelerating end-of-life behaviour; threshold-crossing gives RUL.
These respect the multivariate, non-stationary, per-engine structure that ARIMA
does not.

## 9. Portfolio claim audit (Condition 9)

Status vocabulary: SUPPORTED / REQUIRES EXPERIMENT / NOT SUPPORTED / REQUIRES
REWORDING. No metric is retained merely because it was previously published.

| Original claim | Status | Reason | Future replacement |
|---|---|---|---|
| "LSTM forecasting model" | REQUIRES EXPERIMENT | No model trained yet; LSTM superiority is unproven and must be measured against GBM/linear on identical engine-level splits. | Report the empirically best model after Phase 1; keep LSTM only if it wins. |
| "Machine failure prediction" | SUPPORTED (as a task statement) | The dataset genuinely supports failure-risk & RUL tasks; this is a scope claim, not a result. | Keep, phrased as the problem addressed, not an achievement. |
| "Two weeks ahead" | REQUIRES REWORDING | Cycles ≠ days; no calendar mapping exists (§21). "Two weeks" is unjustified. | "Up to H operational cycles ahead" with H reported (e.g. H=30 cycles, provisional). |
| "87% accuracy" | NOT SUPPORTED | No experiment run; accuracy on a ~1:7 imbalanced, heavily-overlapping task is also the wrong metric. | Replace with PR-AUC / ROC-AUC + calibration, engine-aggregated, computed in Phase 1. |
| "Outperformed ARIMA" | NOT SUPPORTED | No models trained; ARIMA also judged methodologically unsuitable (§8). | Compare against chosen statistical baselines (linear/health-index trend), report actual deltas. |
| "Outperformed GBM" | NOT SUPPORTED | No models trained; GBM is a strong baseline that must actually be beaten. | Report LSTM-vs-GBM comparison on identical splits once measured. |
| "Higher failure recall" | REQUIRES EXPERIMENT | Recall claim needs a trained classifier + threshold + engine-level eval. | Report precision/recall/F1/PR-AUC at a justified operating point. |
| "~$1.15M/year" | NOT SUPPORTED | No cost ground truth in C-MAPSS; any savings figure is fabricated (§21). | Either drop, or present as an *illustrative* model with all cost assumptions stated as hypothetical, never as an achieved saving. |

**Rule enforced:** until Phase 1 produces measured, leakage-controlled results,
the portfolio may state only the *problem and method*, not any performance number.

## 10. Leakage controls (summary)

Full register in `reports/LEAKAGE_AUDIT.md`. Binding Phase-1 controls:
- Engine-level splits only; no row shuffling across the temporal axis.
- All transforms (scalers, regime parameters, health-index/PCA) **fit on training
  folds only**; val/test only ever transformed.
- Windows causal, single-engine, never crossing partition boundaries.
- `max_cycle`/lifespan is **target-only**, never a feature (unit-tested).
- Test RUL used only by the final evaluator, once.
- Imbalance handled by weights/thresholds, never by cross-engine/cross-time
  resampling.

---

## PHASE 0 FINAL STATUS

**APPROVED FOR PHASE 1**

All ten conditions are resolved with computed, reproducible evidence; raw data is
untouched and checksum-verified; no models were trained. Phase 1 may proceed
**only** with: FD004 primary (counts from file contents), `raw_RUL` as the honest
target (cap=125 as a labelled ablation only), regime-conditioned train-only
normalization (Option B) with global (Option A) as ablation, failure-risk horizon
H=30 primary with H=14/50 sensitivity (provisional), engine-grouped
regime-stratified forward-chaining validation, linear/health-index-trend
statistical baselines (ARIMA excluded), and the portfolio claims rewritten per §9.

**HARD STOP OBSERVED: Phase 1 is NOT started.** No feature engineering, no
baseline training, no LSTM. Awaiting project-lead authorization.
