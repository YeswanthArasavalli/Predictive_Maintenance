# PHASE 1 — FINAL GATE

**Project:** Predictive Maintenance — Multivariate Failure-Risk & RUL Forecasting
**Dataset:** NASA C-MAPSS FD004 (249 engines, 61,249 train rows)
**Phase 1 Scope:** Leakage-safe data contract, engine splitting, preprocessing.
**Models trained:** NONE. **Raw data modified:** NO.
**Test results:** 43 passed (11 Phase 0 + 32 Phase 1), 0 failed, 0 warnings.

---

## 1. Acceptance Matrix

| Requirement | Evidence | Status |
|---|---|---|
| Raw data immutable | SHA-256 checksums verified (`test_integrity_checksums_match`); pipeline never writes to `data/raw/` | **PASS** |
| Engine disjointness | `train_ids ∩ val_ids = ∅` (verified programmatically); 199 + 50 = 249 | **PASS** |
| Official test isolation | AST-level + grep audit: `run_phase1_pipeline.py` only calls `load_dataset(fd, "train")`; test data never loaded | **PASS** |
| Regime mapping deterministic | Fixed 6-entry lookup table; no learning; no data-dependent parameters | **PASS** |
| Train-only normalization | Per-regime scaler means match manual train-only computation; transforming val data leaves scaler unchanged | **PASS** |
| Causal preprocessing | Corrupting cycle 543 of engine 118 leaves cycle-30 window identical | **PASS** |
| RUL correctness | `raw_RUL == max_cycle - cycle` for all 199 train engines, 0 errors | **PASS** |
| H30 correctness | `failure_risk_target_H30 == (raw_RUL <= 30)` for all rows; boundary RUL=31→0, RUL=30→1 verified | **PASS** |
| Causal windows | `target_cycle == start_cycle + W - 1` for all 43,168 windows; 0 violations | **PASS** |
| No cross-engine windows | Structural guarantee: `groupby(UNIT_ID)` before windowing | **PASS** |
| No target features | `FORBIDDEN_FEATURE_COLS ∩ feature_cols = ∅`; runtime guard raises on violation | **PASS** |
| Reproducibility | Re-run with same seed: identical split, identical scaler params, identical window arrays | **PASS** |
| Artifact integrity | Loaded scalers produce identical transforms to in-memory scalers | **PASS** |
| Tests | 43/43 pass; 0 fail; 0 skip | **PASS** |

---

## 2. Split Design — Verified Details

**Algorithm:**
1. Load FD004 train (249 engines, 61,249 rows).
2. Assign `operating_regime` per row via deterministic rounding.
3. Extract one per-engine label (`.first()` of regime column) for stratification.
4. Shuffle engines with `np.random.RandomState(42)`.
5. Apply `StratifiedGroupKFold(n_splits=5)` — first fold = validation.
6. Train = 199 engines, Val = 50 engines.

**Properties:**
- `train_ids ∩ val_ids = ∅`: **Confirmed**
- `train_ids ∪ val_ids = {1..249}`: **Confirmed**
- Deterministic with seed 42: **Confirmed** (identical on re-run)
- SHA-256 of sorted train list: `cb49950e...`
- SHA-256 of sorted val list: `116fa158...`

**Val engine IDs (50):**
```
[1, 6, 12, 18, 22, 27, 37, 40, 42, 47, 50, 51, 60, 68, 73, 79, 81, 87, 94, 101,
 103, 111, 115, 116, 120, 126, 129, 138, 144, 150, 155, 159, 162, 164, 171,
 176, 181, 190, 193, 199, 200, 203, 209, 217, 226, 229, 234, 235, 242, 248]
```

**Stratification note:**
In FD004, each engine traverses ALL 6 regimes during its lifecycle (engines are
not grouped by regime). Therefore the per-engine stratification label is
effectively uniform, and `StratifiedGroupKFold` degenerates to a balanced group
split. This is **not a defect** — it is an inherent property of FD004's data
structure. The regime balance is guaranteed at the ROW level by the physics of
the dataset.

**Leakage confirmations:**
- RUL: NOT used for stratification (regime label only)
- Future degradation: NOT used (regime is from settings only)
- Validation performance: NOT used to create the split
- Official test data: NOT involved in any way

---

## 3. Normalization Audit

### Mode B (Regime-conditioned — Primary)

| Property | Answer |
|---|---|
| Fitting rows | Train-engine rows filtered by regime. Verified: mean matches manual computation per regime. |
| Can val rows affect scaler? | **No.** Transform only. Verified empirically. |
| Can test rows affect scaler? | **No.** Test data is never loaded. |
| Can val engine affect train scaler? | **No.** Scaler is fit before val data is processed. |
| Parameters fixed after fitting? | **Yes.** StandardScaler.mean_ and .scale_ are set once during .fit() |
| Uses future observations from same engine? | **No.** Scaler uses cross-sectional statistics from all train engines' rows at a given regime — this is a deployment-time constant, not per-engine future data. |
| Operational settings normalized? | **Yes** (columns 0-2 of 24) |
| Sensors normalized? | **Yes** (columns 3-23 of 24) |
| Targets normalized? | **No** |
| Identifiers normalized? | **No** |

### Causality Decision (Documented)

The regime-conditioned scaler parameters are computed from the **cross-sectional
distribution of all training engines** at each regime. At prediction time, these
parameters are fixed constants. A new observation at cycle `t` is transformed
using the same train-fitted parameters regardless of whether later observations
of the same engine exist. This is the "deployment-known constants learned
historically" approach explicitly preferred by the Phase 1 spec.

### Empty Regime Handling

If a regime has zero training observations, it will not appear in
`regime_scaler.scalers`. At transform time:
- If `fallback="error"`: raises ValueError.
- If `fallback="global"`: uses the global scaler (fit on all train rows).
- Current config: `fallback="global"` (safe for FD004 where all 6 regimes are populated).

---

## 4. Causality Test — Controlled Experiment

**Procedure:**
1. Selected engine 118 (543 cycles — the largest in the training set).
2. Generated windows on the original scaled data.
3. Corrupted `sensor_01` at cycle 543 (last observation) to value 999.0.
4. Regenerated all windows on the corrupted data.
5. Compared the FIRST window (target_cycle=30, start_cycle=1).

**Result:**
```
features_before(cycle 30) == features_after(cycle 30)  → TRUE (identical)
Window at corrupted cycle 543 DID change: TRUE
```

**Verdict:** Preprocessing is causal. Changing a future observation (cycle 543)
cannot alter feature values at an earlier prediction time (cycle 30). The
windowed features at time t use only observations from cycles ≤ t.

---

## 5. Windowing Audit — Sample Windows

| Window | unit_id | start_cycle | target_cycle | W | raw_RUL | H30 |
|--------|---------|-------------|--------------|---|---------|-----|
| #1527 | 9 | 178 | 207 | 30 | 127 | 0 |
| #12321 | 71 | 152 | 181 | 30 | 131 | 0 |
| #28151 | 165 | 43 | 72 | 30 | 183 | 0 |
| #34208 | 197 | 138 | 167 | 30 | 3 | 1 |
| #34287 | 198 | 76 | 105 | 30 | 59 | 0 |

**Mathematical proof of causality:**
For every window: `start_cycle + W - 1 = target_cycle`. Verified: 0 violations
across all 43,168 windows.

**Alignment proof:**
The target `raw_RUL` at time t = `max_cycle(engine) - target_cycle`. The last
row of X corresponds to cycle `target_cycle`. Confirmed by the `generate_windows`
implementation: features[i-W+1 : i+1] paired with rul_values[i].

---

## 6. Early-Cycle Handling — Clarification

The pipeline produces TWO levels of processed data:

1. **Row-level Parquet** (`FD004_train.parquet`): Contains ALL 48,939 training
   rows (cycles 1 through max). This is the full normalized dataset with targets.
   No rows are removed.

2. **Window-level NPZ** (`FD004_train_windows.npz`): Contains 43,168 windows.
   Only observations at cycle >= W (30) can form a complete window.
   Discarded: 29 per engine × 199 engines = 5,771 observations (12% of total).

The "first 29 cycles discarded" refers ONLY to the window dataset. Raw processed
observations are preserved in Parquet. Models consuming row-level features (GBM,
linear) can use all rows; sequence models consuming windows use the window NPZ.

---

## 7. RUL Audit

- `raw_RUL = max_cycle(engine) - cycle`: **Correct for all 199 engines, 0 errors.**
- `model_RUL_target == raw_RUL` (cap=None): **Confirmed.** Max value = 542 (> 125, so no accidental cap exists).
- Test RUL (`RUL_FD004.txt`): **Never loaded.** Not used in preprocessing.
- RUL in features: **Not present** in `feature_cols`. Guard raises on violation.

---

## 8. Failure-Risk Target Audit

**Boundary cases:**

| RUL | Label | Correct? |
|-----|-------|----------|
| 31 | 0 | Yes |
| 30 | 1 | Yes |
| 29 | 1 | Yes |
| 15 | 1 | Yes |
| 1 | 1 | Yes |
| 0 | 1 | Yes |
| 45 | 0 | Yes |
| 50 | 0 | Yes |
| 100 | 0 | Yes |
| 200 | 0 | Yes |

Full vectorized check: `(raw_RUL <= 30).astype(int8) == failure_risk_target_H30`
→ **TRUE for all 48,939 rows.**

No off-by-one error: RUL=30 is labeled positive (within 30 cycles inclusive).

---

## 9. Feature Count Audit

```
X.shape = (43168, 30, 24)
n_features = 24 = 3 operational settings + 21 sensors
```

No hidden columns. Exact ordered list:
```
[0]  operational_setting_1
[1]  operational_setting_2
[2]  operational_setting_3
[3]  sensor_01
[4]  sensor_02
...
[23] sensor_21
```

No targets, no identifiers, no derived columns in the feature tensor.

---

## 10. Regime Assignment Audit

**Source:** C-MAPSS FD004 is documented as having "six flight conditions." Each
engine operates under all six conditions during its lifecycle (mission profile).
The regime is determined by the instantaneous operational settings, which follow
a known experimental design.

**Rounding rationale:** Operational settings have small random perturbations
(typically <0.01 for setting1, <0.005 for setting2). Rounding to 1/2/0 decimals
collapses noise while preserving the 6 distinct condition clusters established
by the C-MAPSS simulation design.

**Behavior at boundaries:** All observed values in FD004 (train and test) round
to exactly these 6 keys with no ambiguity. No transition values were observed.

---

## 11. Sensor Configuration Audit

- Config A (DEFAULT): 21 sensors. No removal.
- Config B: removes sensor_12, sensor_18, sensor_19 (exact duplicates). 18 sensors.
- Config B+: also removes sensor_16. 17 sensors.
- Phase 1 uses Config A. No sensor is deleted without modeling evidence.
- No validation performance is used for sensor selection.

---

## 12. Official Test Isolation — Confirmed

**Method:** AST inspection + regex search of all Phase 1 source files.

**Result:** The Phase 1 pipeline (`run_phase1_pipeline.py`) calls only:
```python
load_dataset("FD004", "train")
```
No call to `load_dataset(..., "test")` or `load_rul(...)` exists in any Phase 1
source file. References to "test_FD004" and "RUL_FD004" exist ONLY in
docstrings/comments documenting the isolation policy.

**Verdict: PASS — official test data is completely isolated.**

---

## 13. Artifact Integrity

| Check | Result |
|-------|--------|
| Saved scaler mean == in-memory scaler mean | PASS |
| Saved scaler transform == in-memory transform | PASS |
| Manifest contains all required fields | PASS |
| Manifest reports `official_test_status: UNTOUCHED` | PASS |
| Manifest seed: 42 | PASS |
| Normalization mode: B, fit_scope: train_rows_per_regime | PASS |

---

## 14. Reproducibility Test

Re-ran the full pipeline with seed=42:
- Split: identical engine lists.
- Scaler: identical means and scales.
- Windows: identical X, y_rul, y_failure arrays (bit-for-bit exact).

**Verdict: PASS**

---

## 15. Test Quality Assessment

### Strengths (tests that would catch real bugs)

| Test | Would Catch |
|------|-------------|
| `test_no_engine_overlap` | Row-level split accidentally assigned |
| `test_window_causal` | Off-by-one in window boundary |
| `test_no_target_in_features` | Developer accidentally adds RUL to features |
| `test_scaler_fit_train_only` | Fitting scaler on combined train+val data |
| `test_changing_future_obs_does_not_alter_past_window` | Per-engine normalization that leaks future |
| `test_rul_correctness` | Incorrect max_cycle computation |
| `test_h30_labels` | Off-by-one in horizon comparison |

### Weaknesses identified

| Test | Issue |
|------|-------|
| `test_window_single_engine` | Only checks metadata type, not actual data |
| `test_regime_representation` | Passes trivially since ALL engines have ALL regimes |

**Assessment:** The weak tests are not harmful. The window_single_engine test is
structurally guaranteed by the implementation (`groupby(UNIT_ID)`). The
regime_representation test passes trivially for FD004 but would catch issues if
switching to FD002 (where engines DO have single regimes).

**No additional tests are required at this stage.** The 32 tests provide
sufficient leakage coverage for the Phase 1 contract.

---

## 16. Code Quality Findings

| Issue | Severity | Status |
|-------|----------|--------|
| Unused imports in regime.py, splitter.py, normalization.py, manifest.py, artifacts.py | Low | **FIXED** |
| Incorrect comment in splitter.py ("all rows share regime") | Medium | **FIXED** |
| `verify_phase1_gate.py` uses joblib.load | Informational | Trusted provenance (own artifacts) |
| No platform-specific path hardcoding | Good | Uses pathlib throughout |
| No silent exception handling | Good | All errors raised explicitly |
| No mutable global state | Good | All state in function scope |

---

## 17. Known Findings / Limitations

1. **FD004 engines span all 6 regimes.** The regime-stratified engine split is
   effectively a balanced group split. This is a data property, not a defect.
   The mechanism generalizes correctly to FD002 (single-regime engines).

2. **Regime transitions within a window.** After per-regime normalization, a
   window containing a regime transition will show a "baseline shift" in the
   normalized features. This is the intended behavior (removing regime effects).
   Models must learn across these transitions.

3. **Warm-up exclusion is window-only.** All 48,939 rows are preserved in the
   Parquet output. Only the sequence-model windows require W >= 30 observations.

---

## PHASE 1 STATUS

**APPROVED WITH CONDITIONS**

Conditions for Phase 2 entry:
1. The regime stratification finding (section 2) must be documented in Phase 2
   planning — the "regime-stratified" split is functionally a "balanced group
   split" for FD004.
2. The Phase 2 test suite must not assume single-regime engines when validating
   FD004 data paths.

All leakage controls are implemented, tested, and verified. The preprocessing
infrastructure is safe for model consumption.

---

**HARD STOP OBSERVED. Phase 2 is NOT started. No models trained. No baselines
prepared. Awaiting project-lead authorization.**
