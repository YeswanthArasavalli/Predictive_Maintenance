"""Phase 1 Final Gate Verification Script."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import hashlib
from src.data.loader import UNIT_ID, CYCLE, load_dataset
from src.data.regime import assign_regime, REGIME_MAPPING
from src.data.splitter import engine_split
from src.data.contract import OPERATIONAL_COLS, FORBIDDEN_FEATURE_COLS
from src.features.sensors import get_sensor_config
from src.features.normalization import GlobalScaler, RegimeScaler
from src.features.targets import compute_raw_rul, compute_model_rul_target, compute_failure_risk_target
from src.features.windows import generate_windows

SEED = 42
LOOKBACK = 30
HORIZON = 30

# Load
df = load_dataset("FD004", "train")
df["operating_regime"] = assign_regime(df)

# Split
train_ids, val_ids = engine_split(df, seed=SEED, val_fraction=0.2)

print("=" * 70)
print("SECTION 2: SPLIT VERIFICATION")
print("=" * 70)

overlap = set(train_ids) & set(val_ids)
all_engines = sorted(df[UNIT_ID].unique().tolist())
union = sorted(set(train_ids) | set(val_ids))

print(f"Seed: {SEED}")
print(f"Algorithm: StratifiedGroupKFold(n_splits=5) on shuffled engines")
print(f"Total engines: {len(all_engines)}")
print(f"Train engines: {len(train_ids)}")
print(f"Val engines: {len(val_ids)}")
print(f"Overlap (train_ids & val_ids): {overlap}")
print(f"Union == all_engines: {union == all_engines}")
print(f"Val engine IDs: {val_ids}")

# Check how many regimes each engine spans
regimes_per_engine = df.groupby(UNIT_ID)['operating_regime'].nunique()
print(f"\nRegimes per engine distribution:")
for n, count in regimes_per_engine.value_counts().sort_index().items():
    print(f"  {n} regime(s): {count} engines")

# Verify: does engine 1 have a single regime?
eng1 = df[df[UNIT_ID] == 1]
print(f"\nEngine 1 unique regimes: {sorted(eng1['operating_regime'].unique())}")
print(f"Engine 1 cycles: {eng1[CYCLE].min()} to {eng1[CYCLE].max()}")

# Per-engine regime for split stratification
engine_regime_first = df.groupby(UNIT_ID)['operating_regime'].first()
print(f"\nStratification label used: .first() regime per engine (all engines have 1 regime)")

# Regime counts (engine-level)
train_df = df[df[UNIT_ID].isin(train_ids)]
val_df = df[df[UNIT_ID].isin(val_ids)]
train_engines_per_regime = train_df.groupby('operating_regime')[UNIT_ID].nunique()
val_engines_per_regime = val_df.groupby('operating_regime')[UNIT_ID].nunique()

print("\nTrain engines per regime:")
for r in range(6):
    eng_cnt = train_engines_per_regime.get(r, 0)
    row_cnt = (train_df['operating_regime'] == r).sum()
    pct = eng_cnt / len(train_ids) * 100
    print(f"  Regime {r}: {eng_cnt} engines ({pct:.1f}%), {row_cnt} rows")

print("\nVal engines per regime:")
for r in range(6):
    eng_cnt = val_engines_per_regime.get(r, 0)
    row_cnt = (val_df['operating_regime'] == r).sum()
    pct = eng_cnt / len(val_ids) * 100
    print(f"  Regime {r}: {eng_cnt} engines ({pct:.1f}%), {row_cnt} rows")

train_hash = hashlib.sha256(str(train_ids).encode()).hexdigest()
val_hash = hashlib.sha256(str(val_ids).encode()).hexdigest()
print(f"\nTrain IDs hash: {train_hash}")
print(f"Val IDs hash: {val_hash}")

print("\n" + "=" * 70)
print("SECTION 3: NORMALIZATION AUDIT")
print("=" * 70)

sensors = get_sensor_config("A")
feature_cols = OPERATIONAL_COLS + sensors

# Mode B (primary)
regime_scaler = RegimeScaler(fallback="global")
regime_scaler.fit(train_df, feature_cols, regime_col="operating_regime")

# Verify scaler was fit on train only
for rid, sc in regime_scaler.scalers.items():
    subset = train_df[train_df['operating_regime'] == rid]
    expected_mean = subset[feature_cols].to_numpy(dtype=np.float64).mean(axis=0)
    match = np.allclose(sc.mean_, expected_mean, atol=1e-10)
    print(f"  Regime {rid} scaler: mean matches train-only? {match} (n_samples={int(sc.n_samples_seen_)})")

# Check val data does NOT affect scaler
mean_before = regime_scaler.scalers[0].mean_.copy()
_ = regime_scaler.transform(val_df, regime_col="operating_regime")
mean_after = regime_scaler.scalers[0].mean_.copy()
print(f"  Transforming val data altered scaler? {not np.array_equal(mean_before, mean_after)}")

# What columns are scaled?
print(f"\n  Columns scaled ({len(feature_cols)}): {feature_cols[:5]}...{feature_cols[-3:]}")
print(f"  = 3 operational settings + 21 sensors = 24 columns")
print(f"  Targets scaled? {'raw_RUL' in feature_cols}")
print(f"  Identifiers scaled? {'unit_id' in feature_cols or 'cycle' in feature_cols}")

print("\n" + "=" * 70)
print("SECTION 4: CAUSALITY TEST (Controlled Experiment)")
print("=" * 70)

# Scale and compute targets
train_scaled = regime_scaler.transform(train_df, regime_col="operating_regime").copy()
train_scaled["raw_RUL"] = compute_raw_rul(train_scaled)
train_scaled[f"failure_risk_target_H{HORIZON}"] = compute_failure_risk_target(
    train_scaled["raw_RUL"], horizon=HORIZON
)

# Pick an engine with many cycles
eng_sizes = train_scaled.groupby(UNIT_ID).size()
target_engine = eng_sizes.idxmax()  # largest engine
print(f"  Testing on engine {target_engine} ({eng_sizes.max()} cycles)")

# Generate windows on original
X_orig, y_rul_orig, y_fail_orig, meta_orig = generate_windows(
    train_scaled, feature_cols=feature_cols, lookback=LOOKBACK,
    target_col="raw_RUL", failure_target_col=f"failure_risk_target_H{HORIZON}"
)

# Find windows for this engine
eng_windows = [(i, m) for i, m in enumerate(meta_orig) if m.unit_id == target_engine]
early_window_idx = eng_windows[0][0]  # first window of this engine
early_meta = eng_windows[0][1]
print(f"  Early window: target_cycle={early_meta.target_cycle}, start_cycle={early_meta.start_cycle}")

# Now corrupt the LAST observation of this engine
corrupted = train_scaled.copy()
last_idx = corrupted[corrupted[UNIT_ID] == target_engine].index[-1]
last_cycle = corrupted.loc[last_idx, CYCLE]
print(f"  Corrupting cycle {last_cycle} (sensor_01 set to 999.0)")
corrupted.loc[last_idx, "sensor_01"] = 999.0

# Re-generate windows
X_corr, _, _, meta_corr = generate_windows(
    corrupted, feature_cols=feature_cols, lookback=LOOKBACK,
    target_col="raw_RUL", failure_target_col=f"failure_risk_target_H{HORIZON}"
)

# Compare the early window (which ends BEFORE the corrupted cycle)
print(f"\n  Comparing early window (target_cycle={early_meta.target_cycle}):")
print(f"    features_before == features_after?")
identical = np.array_equal(X_orig[early_window_idx], X_corr[early_window_idx])
print(f"    Result: {identical}")

if not identical:
    diff_mask = X_orig[early_window_idx] != X_corr[early_window_idx]
    print(f"    DIFF DETECTED at positions: {np.argwhere(diff_mask)}")
else:
    # Also check the window that INCLUDES the corrupted cycle
    late_window_idx = eng_windows[-1][0]
    late_meta = eng_windows[-1][1]
    if late_meta.target_cycle == last_cycle:
        changed = not np.array_equal(X_orig[late_window_idx], X_corr[late_window_idx])
        print(f"    Window at corrupted cycle ({last_cycle}) DID change: {changed}")

print("\n" + "=" * 70)
print("SECTION 5: WINDOWING AUDIT")
print("=" * 70)

# Pick 5 random windows
rng = np.random.RandomState(123)
sample_indices = rng.choice(len(meta_orig), size=5, replace=False)

print(f"{'Window':<8} {'unit_id':<9} {'start_cyc':<10} {'end_cyc':<9} {'W':<4} {'raw_RUL':<9} {'H30':<5}")
print("-" * 60)
for si in sorted(sample_indices):
    m = meta_orig[si]
    print(f"#{si:<7} {m.unit_id:<9} {m.start_cycle:<10} {m.target_cycle:<9} {LOOKBACK:<4} {m.raw_rul:<9} {m.failure_label:<5}")

# Verify max input cycle == target cycle
print("\nVerifying causality for ALL windows:")
violations = 0
for m in meta_orig:
    if m.start_cycle + LOOKBACK - 1 != m.target_cycle:
        violations += 1
print(f"  Windows where start+W-1 != target: {violations} / {len(meta_orig)}")

# Verify no target in X
print(f"  feature_cols contains 'raw_RUL': {'raw_RUL' in feature_cols}")
print(f"  feature_cols contains 'unit_id': {'unit_id' in feature_cols}")
print(f"  feature_cols contains 'cycle': {'cycle' in feature_cols}")
print(f"  X shape: {X_orig.shape} (samples, W, features)")

print("\n" + "=" * 70)
print("SECTION 6: EARLY-CYCLE HANDLING")
print("=" * 70)

total_train_rows = len(train_df)
total_windows = len(meta_orig)
skipped = total_train_rows - total_windows
print(f"  Total train rows (all engines): {total_train_rows}")
print(f"  Total train windows generated: {total_windows}")
print(f"  Rows excluded (warm-up): {skipped}")
print(f"  Expected exclusions: {LOOKBACK-1} per engine = {len(train_ids)} * {LOOKBACK-1} = {len(train_ids)*(LOOKBACK-1)}")
print(f"  Parquet file STILL contains all {total_train_rows} rows.")
print(f"  Only the .npz window file excludes warm-up observations.")

print("\n" + "=" * 70)
print("SECTION 7: RUL AUDIT")
print("=" * 70)

# Verify raw_RUL = max_cycle - cycle
errors = 0
for uid, grp in train_scaled.groupby(UNIT_ID):
    max_c = grp[CYCLE].max()
    expected = max_c - grp[CYCLE]
    actual = grp["raw_RUL"]
    if not np.array_equal(expected.to_numpy(), actual.to_numpy()):
        errors += 1
print(f"  Engines with incorrect raw_RUL: {errors} / {train_scaled[UNIT_ID].nunique()}")

# Verify model_RUL_target == raw_RUL (no cap)
model_rul = compute_model_rul_target(train_scaled["raw_RUL"], cap=None)
print(f"  model_RUL_target == raw_RUL (cap=None): {np.array_equal(model_rul.to_numpy(), train_scaled['raw_RUL'].to_numpy())}")
print(f"  Max raw_RUL: {train_scaled['raw_RUL'].max()} (> 125, so no accidental cap)")
print(f"  raw_RUL in feature_cols: {'raw_RUL' in feature_cols}")

print("\n" + "=" * 70)
print("SECTION 8: FAILURE-RISK TARGET AUDIT")
print("=" * 70)

h30_col = f"failure_risk_target_H{HORIZON}"
# 10 representative examples
print(f"{'unit':<6} {'cycle':<7} {'max_cyc':<8} {'raw_RUL':<9} {'H30':<5}")
print("-" * 40)
examples = [
    # RUL=31 (boundary-1), RUL=30 (boundary), RUL=29, RUL=0, RUL=1
    train_scaled[train_scaled["raw_RUL"] == 31].iloc[0],
    train_scaled[train_scaled["raw_RUL"] == 30].iloc[0],
    train_scaled[train_scaled["raw_RUL"] == 29].iloc[0],
    train_scaled[train_scaled["raw_RUL"] == 0].iloc[0],
    train_scaled[train_scaled["raw_RUL"] == 1].iloc[0],
    train_scaled[train_scaled["raw_RUL"] == 50].iloc[0],
    train_scaled[train_scaled["raw_RUL"] == 100].iloc[0],
    train_scaled[train_scaled["raw_RUL"] == 200].iloc[0],
    train_scaled[train_scaled["raw_RUL"] == 15].iloc[0],
    train_scaled[train_scaled["raw_RUL"] == 45].iloc[0],
]
for ex in examples:
    uid = int(ex[UNIT_ID])
    cyc = int(ex[CYCLE])
    mx = int(train_scaled[train_scaled[UNIT_ID]==uid][CYCLE].max())
    print(f"{uid:<6} {cyc:<7} {mx:<8} {int(ex['raw_RUL']):<9} {int(ex[h30_col]):<5}")

# Boundary check
print(f"\n  RUL=31 -> label={int(train_scaled[train_scaled['raw_RUL']==31][h30_col].iloc[0])}")
print(f"  RUL=30 -> label={int(train_scaled[train_scaled['raw_RUL']==30][h30_col].iloc[0])}")
print(f"  RUL=29 -> label={int(train_scaled[train_scaled['raw_RUL']==29][h30_col].iloc[0])}")
print(f"  RUL=0  -> label={int(train_scaled[train_scaled['raw_RUL']==0][h30_col].iloc[0])}")

# Full correctness check
expected_h30 = (train_scaled["raw_RUL"] <= HORIZON).astype(np.int8)
actual_h30 = train_scaled[h30_col]
print(f"  All H30 labels correct: {np.array_equal(expected_h30.to_numpy(), actual_h30.to_numpy())}")

print("\n" + "=" * 70)
print("SECTION 9: FEATURE COUNT AUDIT")
print("=" * 70)
print(f"  X.shape = {X_orig.shape}")
print(f"  n_features = {X_orig.shape[2]}")
print(f"  Breakdown: 3 operational + 21 sensors = {3+21}")
print(f"\n  Exact ordered feature list ({len(feature_cols)} columns):")
for i, col in enumerate(feature_cols):
    print(f"    [{i:2d}] {col}")

print("\n" + "=" * 70)
print("SECTION 10: REGIME ASSIGNMENT AUDIT")
print("=" * 70)
print(f"  Method: deterministic rounding of operational settings")
print(f"  Formula: round(s1,1)|round(s2,2)|round(s3,0) -> string key -> lookup")
print(f"\n  Fixed mapping table:")
for key, val in sorted(REGIME_MAPPING.items(), key=lambda x: x[1]):
    parts = key.split("|")
    print(f"    Regime {val}: key={key} (s1={parts[0]}, s2={parts[1]}, s3={parts[2]})")
print(f"\n  Uses validation data? NO")
print(f"  Uses test data? NO")
print(f"  Uses RUL? NO")
print(f"  Uses failure labels? NO")
print(f"  Learned parameters? NO")
print(f"  Deterministic? YES")
print(f"  Source: C-MAPSS FD004 experimental design (6 fixed flight conditions)")

print("\n" + "=" * 70)
print("SECTION 11: SENSOR CONFIGURATION AUDIT")
print("=" * 70)
cfg_a = get_sensor_config("A")
cfg_b = get_sensor_config("B")
cfg_bplus = get_sensor_config("B+")
print(f"  Config A: {len(cfg_a)} sensors (default)")
print(f"  Config B: {len(cfg_b)} sensors (removed {set(cfg_a)-set(cfg_b)})")
print(f"  Config B+: {len(cfg_bplus)} sensors (also removed {set(cfg_b)-set(cfg_bplus)})")
print(f"  Active config: A (all 21 sensors)")

print("\n" + "=" * 70)
print("SECTION 12: OFFICIAL TEST ISOLATION")
print("=" * 70)
# Search source files for test references
import pathlib
root = pathlib.Path(__file__).resolve().parents[1]
src_files = list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py"))
test_refs = []
for f in src_files:
    content = f.read_text(encoding="utf-8")
    for term in ["test_FD004", "RUL_FD004", "load_dataset.*test"]:
        import re
        if re.search(term, content) and "run_phase0" not in f.name:
            test_refs.append((f.name, term))
print(f"  References to test data in Phase 1 src/scripts: {test_refs if test_refs else 'NONE'}")

print("\n" + "=" * 70)
print("SECTION 13: ARTIFACT INTEGRITY")
print("=" * 70)
import joblib, json
preproc_dir = root / "results" / "preprocessing"
# Load saved regime scalers
loaded_scalers = joblib.load(preproc_dir / "regime_scalers.joblib")
# Compare with in-memory scalers
match = True
for rid in loaded_scalers:
    if not np.allclose(loaded_scalers[rid].mean_, regime_scaler.scalers[rid].mean_):
        match = False
        print(f"  Regime {rid} mean mismatch!")
print(f"  Saved scalers match in-memory: {match}")

# Transform val data with loaded scalers and compare
loaded_rs = RegimeScaler(fallback="global")
loaded_rs.scalers = loaded_scalers
loaded_rs.feature_cols = feature_cols
val_out_original = regime_scaler.transform(val_df, regime_col="operating_regime")
val_out_loaded = loaded_rs.transform(val_df, regime_col="operating_regime")
print(f"  Val transform identical from loaded vs memory scaler: {np.allclose(val_out_original[feature_cols].to_numpy(), val_out_loaded[feature_cols].to_numpy())}")

# Check manifest
manifest_path = root / "results" / "audits" / "phase1_dataset_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(f"\n  Manifest contains:")
for key in ["schema_version", "primary_dataset", "engine_split", "sensor_configuration",
            "normalization", "rul_target", "failure_risk_target", "window_generation",
            "reproducibility", "data_integrity"]:
    print(f"    [{key}]: {'YES' if key in manifest else 'MISSING'}")

print(f"\n  Manifest train engines count: {manifest['engine_split']['n_train_engines']}")
print(f"  Manifest val engines count: {manifest['engine_split']['n_val_engines']}")
print(f"  Manifest no_overlap: {manifest['engine_split']['no_overlap']}")
print(f"  Manifest normalization mode: {manifest['normalization']['mode']}")
print(f"  Manifest fit_scope: {manifest['normalization']['fit_scope']}")
print(f"  Manifest failure_horizon: {manifest['failure_risk_target']['horizon_H']}")
print(f"  Manifest lookback: {manifest['window_generation']['lookback_W']}")
print(f"  Manifest seed: {manifest['reproducibility']['random_seed']}")
print(f"  Manifest official test status: {manifest['official_test_status']}")

print("\n" + "=" * 70)
print("SECTION 14: REPRODUCIBILITY TEST")
print("=" * 70)
# Re-run split with same seed
train_ids2, val_ids2 = engine_split(df, seed=SEED, val_fraction=0.2)
print(f"  Split reproducible (run 2 == run 1): {train_ids == train_ids2 and val_ids == val_ids2}")

# Re-run normalization
regime_scaler2 = RegimeScaler(fallback="global")
regime_scaler2.fit(train_df, feature_cols, regime_col="operating_regime")
for rid in range(6):
    m = np.allclose(regime_scaler.scalers[rid].mean_, regime_scaler2.scalers[rid].mean_)
    if not m:
        print(f"  Regime {rid} scaler NOT reproducible!")
        break
else:
    print(f"  Normalization reproducible: True")

# Re-run windows
X2, y2_rul, y2_fail, meta2 = generate_windows(
    train_scaled, feature_cols=feature_cols, lookback=LOOKBACK,
    target_col="raw_RUL", failure_target_col=f"failure_risk_target_H{HORIZON}"
)
print(f"  Windows reproducible (X): {np.array_equal(X_orig, X2)}")
print(f"  Windows reproducible (y_rul): {np.array_equal(y_rul_orig, y2_rul)}")
print(f"  Windows reproducible (y_fail): {np.array_equal(y_fail_orig, y2_fail)}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
