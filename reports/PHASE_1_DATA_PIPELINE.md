# PHASE 1 — DATA PIPELINE DOCUMENTATION

**Project:** Predictive Maintenance — Multivariate Failure-Risk & RUL Forecasting
**Primary Dataset:** NASA C-MAPSS FD004
**Phase 1 Scope:** Leakage-safe data contract and preprocessing infrastructure.
**Models trained:** NONE. **Raw data modified:** NO.

---

## 1. Data Contract

See `reports/DATA_CONTRACT.md` for the full formal specification.

Key invariants:
- `raw_RUL`, `model_RUL_target`, and `failure_risk_target_H30` are TARGET columns and MUST NEVER enter the model input feature tensor.
- `unit_id` and `cycle` are IDENTIFIER columns used for structural operations only (splitting, windowing, metadata). They are never features.
- Operational settings and sensor values are the only inference-available inputs.

---

## 2. Split Design

**Strategy:** Engine-grouped, regime-stratified splitting.

**Rationale:**
- C-MAPSS engines are independent trajectories with no temporal ordering between them. Engine ID 200 is NOT "later" than engine 100.
- Classical "forward-chaining" (rolling-origin validation) requires a meaningful temporal sequence that does not exist across independent engines.
- Fabricating temporal ordering from arbitrary integer IDs would be methodologically unsound.
- Therefore, we implement the **most defensible mechanism**: engine-grouped splitting stratified on the operating regime to ensure representativeness.

**Implementation:**
- `sklearn.model_selection.StratifiedGroupKFold` (sklearn >= 1.3)
- Groups: `unit_id` (ensures no engine crosses the boundary)
- Stratification: `operating_regime` (ensures regime balance)
- n_splits = 5 (gives ~20% validation)
- Engines shuffled with fixed seed (42) before splitting to break file-order correlation

**Result:**
- Training engines: 199 (approximately 80%)
- Validation engines: 50 (approximately 20%)
- Official test set (248 engines): UNTOUCHED throughout all development

---

## 3. Regime Assignment

**Method:** Deterministic rounding of operational settings.

```
regime_key = f"{round(setting1, 1)}|{round(setting2, 2)}|{round(setting3, 0)}"
```

The six FD004 regimes are:

| Regime ID | Key | Description |
|-----------|-----|-------------|
| 0 | 0.0\|0.0\|100.0 | Sea level, idle |
| 1 | 10.0\|0.25\|100.0 | Low altitude, cruise |
| 2 | 20.0\|0.7\|100.0 | Mid altitude, high Mach |
| 3 | 25.0\|0.62\|60.0 | High altitude, descent |
| 4 | 35.0\|0.84\|100.0 | High altitude, cruise |
| 5 | 42.0\|0.84\|100.0 | Very high altitude, max cruise |

**Properties:**
- No learned parameters
- No clustering or fitting on any data split
- Same function works at inference time
- Unseen regime keys raise explicit errors
- Phase 0 audit verified this produces exactly 6 regimes in both train and test

---

## 4. Sensor Configuration

Default: **Config A** (all 21 sensors).

Alternative Config B removes exact duplicates identified in Phase 0:
- sensor_07 = sensor_12 (Pearson r = 1.0)
- sensor_08 = sensor_18 (Pearson r = 1.0)
- sensor_13 = sensor_19 (Pearson r = 1.0)

Phase 1 does NOT assert Config B is superior. Both are available for later ablation.

---

## 5. Normalization Modes

### Mode A — Global (Ablation)

Single `StandardScaler` (z-score) fit on ALL training-engine rows.
- Simple, robust
- Ignores the dominant regime effect on sensor distributions
- Available as a baseline ablation

### Mode B — Regime-Conditioned (Primary)

Per-regime `StandardScaler` (6 total), each fit only on training-engine rows belonging to that regime. At transform time, each row is routed to its regime's scaler.

- Removes the between-regime variance dominance
- Leakage-free: regimes are defined by settings (not learned from data), scalers fit on train only
- Fallback: if a row belongs to an unseen regime, falls back to a global scaler (also fit on train only)

### Causality Decision

Scaler parameters are **deployment-known constants** learned from the training partition. They are NOT computed from the future trajectory of the same engine. This is the only approach that can exist before prediction time: at deployment, the normalization parameters are fixed and available; new observations are transformed using these constants without any "peek-ahead."

---

## 6. RUL Targets

### `raw_RUL` (Primary)

```
raw_RUL_t = max_cycle(unit_id) - cycle_t
```

- Physical remaining cycles to failure
- Uses only the engine's own final observed cycle (available post-run-to-failure)
- No clipping in the primary experiment

### `model_RUL_target`

```
model_RUL_target = raw_RUL  (primary, no cap)
model_RUL_target = min(raw_RUL, 125)  (ablation only, always labelled "clipped")
```

- `raw_RUL` is never overwritten by a clipped value
- Clipped targets are never reported as physical RUL

---

## 7. Failure-Risk Targets

```
failure_risk_target_H30 = 1 if raw_RUL <= 30, else 0
```

- **Primary horizon:** H = 30 operational cycles
- **Definition:** "failure within the next 30 operational cycles" (NOT "within 30 days")
- **Support:** H=14 and H=50 via configuration for sensitivity analysis
- **Imbalance:** ~12.6% positive at H=30 in FD004

---

## 8. Windowing

**Parameters:**
- Lookback W: 30 (configurable)
- Valid prediction timepoints: cycle >= W within each engine
- Warm-up: first W-1 observations per engine are discarded (documented)

**Window structure:**
```
X_t = [x_{t-W+1}, x_{t-W+2}, ..., x_t]   shape: (W, n_features)
y_t = target value at cycle t
```

**Guarantees:**
- Windows NEVER cross engine boundaries
- No future observation (cycle > t) enters X
- Target columns NEVER enter X
- unit_id and cycle NEVER enter X
- All windows use the same feature ordering (deterministic)

**Output:**
- Train windows: 43,168 samples, shape (30, 24)
- Val windows: 10,860 samples, shape (30, 24)

---

## 9. Leakage Controls

All controls from `reports/LEAKAGE_AUDIT.md` are implemented and tested. Summary:

| Risk | Control | Test |
|------|---------|------|
| L1: Random row split | Engine-level split via StratifiedGroupKFold | `test_no_engine_overlap` |
| L2: Window crossing boundaries | Windows built within one engine only | `test_window_single_engine` |
| L3: Scaler on val/test | All scalers fit on train only | `test_scaler_fit_train_only` |
| L5: Lifespan in features | max_cycle never enters X | `test_no_target_in_features` |
| L9: Non-causal normalization | Scaler uses train partition, not per-engine future | `test_changing_future_obs_does_not_alter_past_window` |

Full evidence in `reports/PHASE_1_LEAKAGE_VERIFICATION.md`.

---

## 10. Preprocessing Artifacts

Location: `results/preprocessing/`

| File | Content |
|------|---------|
| `global_scaler.joblib` | Serialized sklearn StandardScaler (Mode A) |
| `regime_scalers.joblib` | Dict of 6 per-regime StandardScalers (Mode B) |
| `regime_mapping.json` | Fixed regime key -> ID mapping |
| `feature_schema.json` | Ordered list of feature columns |
| `sensor_config.json` | Active sensor configuration |
| `preprocessing_metadata.json` | Normalization mode, scope, seeds, parameters |

All artifacts are generated from training data only.

---

## 11. Tests

**Phase 1 test suite:** `tests/test_phase1.py` — 32 tests covering:
- Split integrity (5 tests)
- Regime assignment (4 tests)
- Target correctness (6 tests)
- Window generation and causality (6 tests)
- Normalization train-only enforcement (3 tests)
- Schema validation (4 tests)
- Data contract (4 tests)

**Phase 0 test suite:** `tests/test_data.py` — 11 tests, all still passing.

**Total: 43 tests, all passing.**

---

## 12. Reproducibility

- **Seed:** 42 (set globally; used for engine shuffle and split)
- **Python:** 3.14.6
- **NumPy/Pandas/sklearn:** versions recorded in manifest
- **Platform:** Windows (recorded in manifest)
- **Git commit:** recorded in manifest if available
- **Deterministic pipeline:** same inputs -> same outputs, verified by `test_deterministic_split`

---

## 13. Known Limitations

1. **Forward-chaining not implemented.** Engine IDs carry no temporal semantics; the concept of "later engines" is not applicable. Engine-grouped stratified splitting is used instead.
2. **Warm-up data discarded.** The first W-1=29 cycles of each engine are not represented in the window dataset. This is documented and inherent to the causal window design.
3. **No fault-mode stratification.** Individual engine fault modes (HPC vs Fan) are not labelled in the data. Stratification on this axis is impossible without inference.
4. **Official test untouched.** The test set (248 engines + RUL_FD004.txt) is never loaded by this pipeline. All development uses only the train_FD004.txt data partitioned into train/val.
5. **Single split (not K-fold).** Phase 1 produces one train/val split. Future phases may add K-fold validation for more robust model selection.

---

## 14. Interface for Future Models

All future models (Phase 2 onward) MUST consume data through this interface:

```python
# Load processed data
X_train = np.load("data/processed/FD004_train_windows.npz")["X"]  # (N, W, F)
y_rul_train = np.load("data/processed/FD004_train_windows.npz")["y_rul"]
y_fail_train = np.load("data/processed/FD004_train_windows.npz")["y_failure"]

# Load metadata
meta = json.loads(Path("data/processed/FD004_metadata.json").read_text())
feature_cols = meta["feature_cols"]  # ordered feature names

# Load scaler for any additional transformation
scaler = joblib.load("results/preprocessing/regime_scalers.joblib")

# Validate schema
from src.data.contract import validate_feature_columns
validate_feature_columns(feature_cols)
```

**Rules:**
- Never access `data/raw/` for preprocessing.
- Never fit new transformations on validation data.
- The official test set is loaded only by the final evaluator after all model selection is complete.
- RUL values reported must distinguish raw_RUL from any clipped model_RUL_target.
