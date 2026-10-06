# Leakage Audit — Phase 0 (mandatory, §13)

**Project:** Predictive Maintenance — Multivariate Failure-Risk & RUL Forecasting
**Primary dataset:** NASA C-MAPSS FD004 (secondary: FD001–FD003)
**Scope:** Identify and mitigate information-leakage risks *before* any model is
trained (Phase 0 trains nothing). Every item below is grounded in the observed
data structure documented in `reports/PHASE_0_DATA_AUDIT.md`.

---

## 0. What "leakage" means here

C-MAPSS is a **run-to-failure, engine-independent, temporal** dataset. Each
engine is one degradation trajectory; the target (RUL, or failure-within-H) is
derived from the engine's *own* final cycle. Leakage = any use of information
that would **not be available at prediction time** for a real in-service engine:
future cycles of the same engine, the engine's eventual lifespan, other
engines' statistics fitted on held-out data, or the test RUL vector.

The two tasks:
- **Task A (RUL):** regress remaining cycles to failure.
- **Task B (Failure-risk):** classify `failure within H operational cycles`.

Both share the same leakage surface.

---

## 1. Risk register

| # | Leakage risk | Why it applies to C-MAPSS | Severity | Mitigation (Phase 1) |
|---|---|---|---|---|
| L1 | **Random row-level split** | Rows of one engine are temporally ordered; a random split puts future rows of an engine in train and past rows in val, so the model sees the answer. | Critical | Split at **engine level**; never shuffle rows across the temporal axis. |
| L2 | **Overlapping windows crossing split boundaries** | Sliding-window sequence features (LSTM) can straddle the train/val or val/test engine boundary or a truncated trajectory. | Critical | Build windows **within each engine only**; assign whole engines to a split; windows never span two engines. |
| L3 | **Fitting scalers on validation/test** | Global StandardScaler/MinMax over all rows leaks held-out distribution (esp. harmful for FD004's 6 regimes). | High | **Fit every scaler on train only** (per operating regime), then `transform` val/test. |
| L4 | **Using future cycles to build features** | Rolling stats that include the current/future row, or `shift(-k)`, reveal the future. | High | Only causal windows (`rolling(...).mean()` over past rows, `shift(+k)`). |
| L5 | **Engine final lifespan in historical features** | `derive_train_rul` uses `max_cycle(engine)`; if that scalar is reused as a *feature*, RUL is directly encoded. | Critical | The engine max-cycle is used **only to build the target**, never as an input feature. Enforced by test `test_rul_uses_only_own_engine_history`. |
| L6 | **Test RUL used during feature generation** | `RUL_FD00X.txt` is ground-truth for scoring only. | Critical | Test RUL loaded **only** by the evaluator, never by the feature pipeline. |
| L7 | **Unit overlap between train/test** | If an engine id appears in both splits, identity leaks. | Medium | Verified: train/test are disjoint fleets (unit ids restart per file); splits are engine-disjoint by construction. |
| L8 | **Target leakage via engineered variables** | A feature like `cycles_since_start / total_life` embeds lifespan. | High | Ban any engineered feature that references end-of-trajectory; feature review gate in Phase 1. |
| L9 | **Normalization using future information** | Per-engine min-max normalization over the whole trajectory uses the future endpoint. | High | Normalize **per regime** using train-fitted stats; per-engine detrending must be causal (expanding window), not whole-series. |
| L10 | **Random shuffling before temporal split** | `train_test_split(shuffle=True)` destroys ordering and mixes engines. | Critical | No global shuffle; engine-level, order-preserving assignment. |
| L11 | **Cross-validation on the test fleet** | Repeatedly tuning on the official test set (with its RUL) turns it into a validation set. | High | Model selection on **train-derived validation folds only**; official test touched once, at the end. |
| L12 | **Class-imbalance resampling across the temporal axis** | SMOTE/oversampling that mixes rows from different engines or across time leaks neighbors. | Medium | Resample **within engine and within split**, after the split, on train only. |

---

## 2. Data-structure facts that make these risks real

- **Trajectory lengths vary widely.** FD004 train engines span 128–543 cycles;
  test engines are truncated (19–486 cycles) and end *before* failure. A naive
  row split or whole-series normalization would exploit the unseen tail.
- **Six operating regimes dominate FD002/FD004 variance.** For sensors 02/17/21,
  **~99.7–99.99%** of variance is *between* regimes, not within (see
  `results/audits/operating_conditions.json`). Pooling regimes for scaling or
  correlation both leaks condition identity and masks degradation.
- **Degradation signal is only visible within a regime.** Pooled sensor↔cycle
  correlation in FD004 is weak (max |r| ≈ 0.10) whereas in single-condition
  FD001 the same sensors reach |r| ≈ 0.6 (`results/audits/degradation_rul.json`).
  Any "feature" that improves by referencing the regime label must be built from
  the *known* operational settings (available at prediction time), never from
  the target.
- **Redundant sensors.** FD004 has exact duplicates `sensor_07≡sensor_12`,
  `sensor_08≡sensor_18`, `sensor_13≡sensor_19` (Pearson = 1.0). These are not a
  leakage risk per se, but inflate correlation-based feature selection.

---

## 3. Guards already implemented in Phase 0

- `src/data/loader.derive_train_rul` computes RUL strictly from each engine's
  own `max(cycle)` — no cross-engine or test information.
- `tests/test_data.py::test_rul_uses_only_own_engine_history` asserts that
  recomputing an engine's RUL in isolation equals its RUL in the full frame
  (i.e., the calculation cannot depend on other engines).
- `check_test_rul_alignment` guarantees the test fleet ↔ RUL vector is 1:1, so
  scoring cannot silently mis-associate ground truth.
- Raw data is immutable and checksum-verified, so no accidental "feature" can be
  baked into the source files.

---

## 4. Phase 1 leakage-prevention checklist (to be enforced in code)

1. Splitter operates on **unique engine ids**; a set of engine ids → train, a
   disjoint set → validation; official test untouched.
2. All scalers/encoders are `fit` on train folds only; `transform` elsewhere;
   normalization is **conditioned on operating regime** for FD002/FD004.
3. Window/sequence construction is per-engine and causal; assert no window
   contains rows from two engines or crosses a split boundary.
4. Unit test asserts the feature matrix contains no column derived from
   `max_cycle`, total lifespan, or test RUL.
5. Cross-validation = **rolling / forward-chaining at engine level** (see split
   recommendation), not KFold on rows.
6. Final reported metrics come from the official test set in a single pass.

---

## 5. Verdict on leakage posture

Phase 0 introduces **no** leakage (no models, no fitted transforms). The
parsing/target code is leakage-safe and unit-tested. The dominant Phase-1 risks
are **L1/L2/L3/L5/L9** (row splitting, window bleed, scaler fitting, lifespan
encoding, non-causal normalization); these are mitigated by the mandatory
engine-level, regime-conditioned, train-only-fit strategy above.
