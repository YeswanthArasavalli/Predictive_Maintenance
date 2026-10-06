# DATA CONTRACT — Phase 1

**Project:** Predictive Maintenance — Multivariate Failure-Risk & RUL Forecasting
**Primary Dataset:** NASA C-MAPSS FD004
**Status:** BINDING — all downstream code MUST conform to this contract.

---

## 1. Identifier Columns

| Column | Type | Role | Source |
|--------|------|------|--------|
| `unit_id` | int64 | Engine identifier | Raw file (column 1) |
| `cycle` | int64 | Operational cycle within engine's life | Raw file (column 2) |

These columns are used for data organization, splitting, and window construction. They are **never** inputs to any model feature tensor.

---

## 2. Operational Settings

| Column | Type | Role | Inference Available |
|--------|------|------|---------------------|
| `operational_setting_1` | float64 | Flight-level/ambient temperature | Yes |
| `operational_setting_2` | float64 | Mach number | Yes |
| `operational_setting_3` | float64 | Throttle/resolved level | Yes |

These are recorded sensor inputs describing the operating condition. They are known at prediction time and are permitted as model features. They are also used (via deterministic rounding) to assign the `operating_regime`.

---

## 3. Sensor Columns

`sensor_01` through `sensor_21` (21 columns, float64).

All are physically measurable turbofan engine parameters (temperatures, pressures, flows, speeds). Available at inference time.

### Sensor Configurations

| Config | Sensors | Count | Notes |
|--------|---------|-------|-------|
| **A** (default) | All sensor_01 through sensor_21 | 21 | No assumptions about redundancy |
| **B** | A minus {sensor_12, sensor_18, sensor_19} | 18 | Removes exact duplicates (Pearson r=1.0 with sensor_07, sensor_08, sensor_13) |
| **B+** | B minus {sensor_16} | 17 | Additionally drops near-constant sensor_16 |

Phase 1 defaults to **Config A**. No sensor deletion is mandated without modeling evidence.

---

## 4. Derived Columns

| Column | Type | Computation | Target/Feature |
|--------|------|-------------|----------------|
| `operating_regime` | int (0-5) | Deterministic rounding of operational settings | **May enter X** (leakage-free, inference-available) |
| `raw_RUL` | int64 | `max_cycle(unit_id) - cycle` | **TARGET ONLY** |
| `model_RUL_target` | int64 | `raw_RUL` (primary) or `min(raw_RUL, cap)` (ablation) | **TARGET ONLY** |
| `failure_risk_target_H30` | int8 | `1 if raw_RUL <= 30 else 0` | **TARGET ONLY** |

---

## 5. Target Columns (MUST NEVER be model input features)

- `raw_RUL` — physical remaining useful life in operational cycles.
- `model_RUL_target` — possibly clipped version of raw_RUL for training.
- `failure_risk_target_H30` — binary imminent-failure label at H=30.

The engine's `max_cycle(unit_id)` value is **target-generation metadata** and is implicitly encoded in raw_RUL. It MUST NEVER appear as a feature.

---

## 6. Forbidden Feature Columns

The following columns are **explicitly forbidden** from entering any model feature tensor `X`:

```
raw_RUL, model_RUL_target, failure_risk_target_H30, unit_id, cycle
```

This is enforced programmatically by `src.data.contract.validate_feature_columns()` and tested by `test_no_target_in_features`.

---

## 7. Inference-Time Availability

At prediction time for a real in-service engine, the following information is available:

- Current operational settings (settings 1-3)
- Current and past sensor readings (up to the present cycle t)
- The derived operating regime (computed from settings)
- The sequence of past observations within the lookback window

**NOT available at inference time:**
- The engine's eventual failure cycle (max_cycle)
- Any future sensor readings
- The raw_RUL or any target value

---

## 8. Feature Column Set (for model input)

Default (Config A, with operational settings):

```
[operational_setting_1, operational_setting_2, operational_setting_3,
 sensor_01, sensor_02, ..., sensor_21]
```

Total: **24 columns** per timestep. Window shape: `(W, 24)`.

---

## 9. Normalization Scope

| Mode | Fit Scope | Transform Scope |
|------|-----------|-----------------|
| A (Global) | All training-engine rows | Train, Val, Test (transform only) |
| B (Regime-conditioned) | Training-engine rows per regime | Train, Val, Test (routed to per-regime scaler) |

No scaler is EVER fit on validation or test data.

---

## 10. Contract Enforcement

- `validate_feature_columns()` raises `ValueError` on violation.
- `generate_windows()` checks forbidden columns before construction.
- Unit tests verify the contract at every boundary.
- Data checksums verify raw files remain immutable.
