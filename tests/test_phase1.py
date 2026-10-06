"""Phase 1 tests: leakage controls, split correctness, window causality,
target generation, normalization train-only enforcement, schema determinism.

All Phase 0 tests in test_data.py must continue to pass alongside these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import CYCLE, UNIT_ID, load_dataset
from src.data.contract import (
    FORBIDDEN_FEATURE_COLS,
    OPERATIONAL_COLS,
    SENSOR_COLS,
    TARGET_COLS,
    feature_columns,
    validate_feature_columns,
)
from src.data.regime import REGIME_MAPPING, assign_regime, assign_regime_safe
from src.data.splitter import engine_split
from src.features.sensors import get_sensor_config, ALL_SENSORS
from src.features.normalization import GlobalScaler, RegimeScaler
from src.features.targets import (
    compute_raw_rul,
    compute_model_rul_target,
    compute_failure_risk_target,
)
from src.features.windows import generate_windows


SEED = 42
LOOKBACK = 30
FAILURE_HORIZON = 30


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fd004_train() -> pd.DataFrame:
    df = load_dataset("FD004", "train")
    df["operating_regime"] = assign_regime(df)
    return df


@pytest.fixture(scope="module")
def split_sets(fd004_train) -> tuple[list[int], list[int]]:
    return engine_split(fd004_train, seed=SEED, val_fraction=0.2)


@pytest.fixture(scope="module")
def train_df(fd004_train, split_sets) -> pd.DataFrame:
    train_ids, _ = split_sets
    return fd004_train[fd004_train[UNIT_ID].isin(train_ids)].copy()


@pytest.fixture(scope="module")
def val_df(fd004_train, split_sets) -> pd.DataFrame:
    _, val_ids = split_sets
    return fd004_train[fd004_train[UNIT_ID].isin(val_ids)].copy()


@pytest.fixture(scope="module")
def scaled_train(train_df) -> pd.DataFrame:
    sensors = get_sensor_config("A")
    cols = OPERATIONAL_COLS + sensors
    scaler = RegimeScaler(fallback="global")
    scaler.fit(train_df, cols, regime_col="operating_regime")
    df = scaler.transform(train_df, regime_col="operating_regime")
    df["raw_RUL"] = compute_raw_rul(df)
    df["model_RUL_target"] = compute_model_rul_target(df["raw_RUL"], cap=None)
    df[f"failure_risk_target_H{FAILURE_HORIZON}"] = compute_failure_risk_target(
        df["raw_RUL"], horizon=FAILURE_HORIZON
    )
    return df


@pytest.fixture(scope="module")
def scaled_val(val_df, train_df) -> pd.DataFrame:
    sensors = get_sensor_config("A")
    cols = OPERATIONAL_COLS + sensors
    scaler = RegimeScaler(fallback="global")
    scaler.fit(train_df, cols, regime_col="operating_regime")
    df = scaler.transform(val_df, regime_col="operating_regime")
    df["raw_RUL"] = compute_raw_rul(df)
    df["model_RUL_target"] = compute_model_rul_target(df["raw_RUL"], cap=None)
    df[f"failure_risk_target_H{FAILURE_HORIZON}"] = compute_failure_risk_target(
        df["raw_RUL"], horizon=FAILURE_HORIZON
    )
    return df


# ===========================================================================
# SPLIT TESTS
# ===========================================================================


class TestSplit:
    def test_no_engine_overlap(self, split_sets):
        """Train and validation engine sets must be completely disjoint."""
        train_ids, val_ids = split_sets
        overlap = set(train_ids) & set(val_ids)
        assert overlap == set(), f"Engine overlap detected: {overlap}"

    def test_all_engines_assigned(self, fd004_train, split_sets):
        """Every engine from the source data is assigned to exactly one partition."""
        train_ids, val_ids = split_sets
        all_ids = set(fd004_train[UNIT_ID].unique())
        assigned = set(train_ids) | set(val_ids)
        assert all_ids == assigned

    def test_deterministic_split(self, fd004_train):
        """Same seed produces identical split on re-run."""
        a1, b1 = engine_split(fd004_train, seed=SEED, val_fraction=0.2)
        a2, b2 = engine_split(fd004_train, seed=SEED, val_fraction=0.2)
        assert a1 == a2
        assert b1 == b2

    def test_regime_representation(self, split_sets, fd004_train):
        """Both train and val partitions contain all 6 regimes."""
        train_ids, val_ids = split_sets
        train_regimes = set(
            fd004_train[fd004_train[UNIT_ID].isin(train_ids)]["operating_regime"].unique()
        )
        val_regimes = set(
            fd004_train[fd004_train[UNIT_ID].isin(val_ids)]["operating_regime"].unique()
        )
        assert train_regimes == set(range(6)), f"Train missing regimes: {set(range(6)) - train_regimes}"
        assert val_regimes == set(range(6)), f"Val missing regimes: {set(range(6)) - val_regimes}"

    def test_expected_engine_counts(self, split_sets):
        """Approximate 80/20 split for 249 engines."""
        train_ids, val_ids = split_sets
        assert len(train_ids) + len(val_ids) == 249
        # StratifiedGroupKFold with n_splits=5 gives ~50 in val
        assert 40 <= len(val_ids) <= 60


# ===========================================================================
# REGIME TESTS
# ===========================================================================


class TestRegime:
    def test_regime_values_in_0_to_5(self, fd004_train):
        regimes = fd004_train["operating_regime"].unique()
        assert set(regimes) == {0, 1, 2, 3, 4, 5}

    def test_regime_is_deterministic(self, fd004_train):
        """Calling assign_regime twice yields the same result."""
        r1 = assign_regime(fd004_train)
        r2 = assign_regime(fd004_train)
        pd.testing.assert_series_equal(r1, r2)

    def test_unseen_regime_raises(self):
        """An invalid operational setting must trigger ValueError."""
        fake = pd.DataFrame({
            "operational_setting_1": [99.9],
            "operational_setting_2": [0.55],
            "operational_setting_3": [80.0],
        })
        with pytest.raises(ValueError, match="Unseen regime"):
            assign_regime(fake)

    def test_assign_regime_safe_returns_minus1(self):
        fake = pd.DataFrame({
            "operational_setting_1": [99.9],
            "operational_setting_2": [0.55],
            "operational_setting_3": [80.0],
        })
        result = assign_regime_safe(fake)
        assert result.iloc[0] == -1


# ===========================================================================
# TARGET TESTS
# ===========================================================================


class TestTargets:
    def test_rul_correctness(self, scaled_train):
        """raw_RUL == max_cycle(engine) - cycle for every row."""
        for uid, grp in scaled_train.groupby(UNIT_ID):
            max_c = grp[CYCLE].max()
            expected = max_c - grp[CYCLE]
            np.testing.assert_array_equal(grp["raw_RUL"].to_numpy(), expected.to_numpy())

    def test_rul_ends_at_zero(self, scaled_train):
        """Every engine's last observation has raw_RUL == 0."""
        last_rows = scaled_train.groupby(UNIT_ID).tail(1)
        assert (last_rows["raw_RUL"] == 0).all()

    def test_model_rul_target_no_cap(self, scaled_train):
        """With cap=None, model_RUL_target == raw_RUL."""
        pd.testing.assert_series_equal(
            scaled_train["model_RUL_target"],
            scaled_train["raw_RUL"],
            check_names=False,
        )

    def test_model_rul_target_with_cap(self, fd004_train):
        raw = compute_raw_rul(fd004_train)
        clipped = compute_model_rul_target(raw, cap=125)
        assert clipped.max() <= 125
        # Rows where raw <= 125 are unchanged
        mask = raw <= 125
        np.testing.assert_array_equal(
            clipped[mask].to_numpy(), raw[mask].to_numpy()
        )

    def test_h30_labels(self, scaled_train):
        """failure_risk_target_H30 == 1 iff raw_RUL <= 30."""
        expected = (scaled_train["raw_RUL"] <= 30).astype(np.int8)
        actual = scaled_train["failure_risk_target_H30"]
        np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())

    def test_h14_h50_labels(self, fd004_train):
        raw = compute_raw_rul(fd004_train)
        for h in [14, 50]:
            label = compute_failure_risk_target(raw, horizon=h)
            expected = (raw <= h).astype(np.int8)
            np.testing.assert_array_equal(label.to_numpy(), expected.to_numpy())


# ===========================================================================
# WINDOW TESTS
# ===========================================================================


class TestWindows:
    def test_window_shape(self, scaled_train):
        """X has correct shape (n_samples, W, n_features)."""
        feature_cols = OPERATIONAL_COLS + get_sensor_config("A")
        X, y_rul, y_fail, meta = generate_windows(
            scaled_train,
            feature_cols=feature_cols,
            lookback=LOOKBACK,
            target_col="raw_RUL",
            failure_target_col=f"failure_risk_target_H{FAILURE_HORIZON}",
        )
        assert X.shape[1] == LOOKBACK
        assert X.shape[2] == len(feature_cols)
        assert len(y_rul) == X.shape[0]
        assert len(y_fail) == X.shape[0]

    def test_window_single_engine(self, scaled_train):
        """Each window belongs to exactly one unit_id."""
        feature_cols = OPERATIONAL_COLS + get_sensor_config("A")
        X, y_rul, y_fail, meta = generate_windows(
            scaled_train,
            feature_cols=feature_cols,
            lookback=LOOKBACK,
            target_col="raw_RUL",
            failure_target_col=f"failure_risk_target_H{FAILURE_HORIZON}",
        )
        # All metadata entries have a single unit_id (structural guarantee)
        for m in meta:
            assert isinstance(m.unit_id, int)

    def test_window_causal(self, scaled_train):
        """Max cycle in window == target_cycle (the 'current' observation)."""
        feature_cols = OPERATIONAL_COLS + get_sensor_config("A")
        _, _, _, meta = generate_windows(
            scaled_train,
            feature_cols=feature_cols,
            lookback=LOOKBACK,
            target_col="raw_RUL",
            failure_target_col=f"failure_risk_target_H{FAILURE_HORIZON}",
        )
        for m in meta:
            assert m.start_cycle >= 1
            assert m.target_cycle == m.start_cycle + LOOKBACK - 1

    def test_no_target_in_features(self, scaled_train):
        """Target columns must not appear in the feature tensor."""
        feature_cols = OPERATIONAL_COLS + get_sensor_config("A")
        # Verify none of the forbidden columns are in feature_cols
        violations = set(feature_cols) & set(FORBIDDEN_FEATURE_COLS)
        assert violations == set()
        # And generate_windows raises if we try to include a target
        with pytest.raises(ValueError, match="forbidden columns"):
            generate_windows(
                scaled_train,
                feature_cols=feature_cols + ["raw_RUL"],
                lookback=LOOKBACK,
            )

    def test_engines_shorter_than_W_skipped(self):
        """An engine with < W observations produces no windows."""
        fake = pd.DataFrame({
            UNIT_ID: [1, 1, 2],
            CYCLE: [1, 2, 1],
            "operating_regime": [0, 0, 1],
            "raw_RUL": [10, 5, 3],
            "failure_risk_target_H30": [0, 0, 0],
        })
        for s in OPERATIONAL_COLS:
            fake[s] = 0.0
        for s in ALL_SENSORS:
            fake[s] = 1.0
        feature_cols = OPERATIONAL_COLS + ALL_SENSORS
        X, _, _, meta = generate_windows(
            fake, feature_cols=feature_cols, lookback=30,
            target_col="raw_RUL", failure_target_col="failure_risk_target_H30",
        )
        assert X.shape[0] == 0  # no engine has 30 cycles

    def test_changing_future_obs_does_not_alter_past_window(self, scaled_train):
        """Test 8: causality — modifying a future observation cannot change
        the feature values in an earlier window."""
        feature_cols = OPERATIONAL_COLS + get_sensor_config("A")

        # Generate windows on original data
        X_orig, _, _, meta_orig = generate_windows(
            scaled_train,
            feature_cols=feature_cols,
            lookback=LOOKBACK,
            target_col="raw_RUL",
            failure_target_col=f"failure_risk_target_H{FAILURE_HORIZON}",
        )

        # Corrupt one engine's LAST observation only
        corrupted = scaled_train.copy()
        # Pick a specific engine
        one_engine_id = meta_orig[0].unit_id
        engine_mask = corrupted[UNIT_ID] == one_engine_id
        last_idx = corrupted[engine_mask].index[-1]
        for c in feature_cols:
            corrupted.loc[last_idx, c] = 999.0  # extreme corruption

        X_corr, _, _, meta_corr = generate_windows(
            corrupted,
            feature_cols=feature_cols,
            lookback=LOOKBACK,
            target_col="raw_RUL",
            failure_target_col=f"failure_risk_target_H{FAILURE_HORIZON}",
        )

        # Find a window that ends BEFORE the corrupted cycle
        # The corrupted row is the last cycle of the engine.
        # Any window whose target_cycle < that cycle must be identical.
        corrupted_cycle = corrupted.loc[last_idx, CYCLE]
        mismatches = 0
        for i, m in enumerate(meta_orig):
            if m.unit_id == one_engine_id and m.target_cycle < corrupted_cycle:
                # This window should NOT contain the corrupted row
                np.testing.assert_array_almost_equal(
                    X_orig[i], X_corr[i],
                    err_msg=f"Window {i} at cycle {m.target_cycle} was altered by future corruption!"
                )
                mismatches += 1
        # At least some windows were checked
        assert mismatches > 0, "No windows found to check causality"


# ===========================================================================
# SCALING / NORMALIZATION TESTS
# ===========================================================================


class TestNormalization:
    def test_scaler_fit_train_only(self, train_df, val_df):
        """Global scaler params are derived from train rows only."""
        sensors = get_sensor_config("A")
        cols = OPERATIONAL_COLS + sensors
        scaler = GlobalScaler()
        scaler.fit(train_df, cols)

        # Manual computation
        X_train = train_df[cols].to_numpy(dtype=np.float64)
        expected_mean = X_train.mean(axis=0)
        np.testing.assert_array_almost_equal(scaler.scaler.mean_, expected_mean)

        # Validation mean should be different (scaler was NOT fit on it)
        X_val = val_df[cols].to_numpy(dtype=np.float64)
        val_mean = X_val.mean(axis=0)
        # They can be close but should not be exactly equal
        assert not np.array_equal(scaler.scaler.mean_, val_mean)

    def test_regime_scaler_train_only(self, train_df, val_df):
        """Per-regime scaler uses only training engines in that regime."""
        sensors = get_sensor_config("A")
        cols = OPERATIONAL_COLS + sensors
        scaler = RegimeScaler(fallback="global")
        scaler.fit(train_df, cols, regime_col="operating_regime")

        for rid, sc in scaler.scalers.items():
            # Get train rows of this regime
            train_regime = train_df[train_df["operating_regime"] == rid]
            X = train_regime[cols].to_numpy(dtype=np.float64)
            expected_mean = X.mean(axis=0)
            np.testing.assert_array_almost_equal(sc.mean_, expected_mean)

    def test_val_transform_does_not_alter_scaler(self, train_df, val_df):
        """Transforming val data cannot change the scaler's fitted parameters."""
        sensors = get_sensor_config("A")
        cols = OPERATIONAL_COLS + sensors
        scaler = GlobalScaler()
        scaler.fit(train_df, cols)
        mean_before = scaler.scaler.mean_.copy()

        _ = scaler.transform(val_df)
        np.testing.assert_array_equal(scaler.scaler.mean_, mean_before)


# ===========================================================================
# SCHEMA TESTS
# ===========================================================================


class TestSchema:
    def test_expected_feature_count(self):
        """Config A: 24 features (3 settings + 21 sensors)."""
        sensors = get_sensor_config("A")
        cols = OPERATIONAL_COLS + sensors
        assert len(cols) == 24

    def test_expected_feature_ordering(self):
        """Feature order is deterministic across calls."""
        s1 = get_sensor_config("A")
        s2 = get_sensor_config("A")
        assert s1 == s2

    def test_forbidden_columns_absent(self):
        """No target column appears in the feature list."""
        sensors = get_sensor_config("A")
        cols = OPERATIONAL_COLS + sensors
        for t in FORBIDDEN_FEATURE_COLS:
            assert t not in cols

    def test_validate_feature_columns_raises(self):
        with pytest.raises(ValueError, match="Data contract violation"):
            validate_feature_columns(["sensor_01", "raw_RUL"])

    def test_sensor_config_B(self):
        """Config B removes duplicate sensors."""
        sensors_b = get_sensor_config("B")
        assert "sensor_12" not in sensors_b
        assert "sensor_18" not in sensors_b
        assert "sensor_19" not in sensors_b
        assert len(sensors_b) == 18


# ===========================================================================
# DATA CONTRACT TESTS
# ===========================================================================


class TestDataContract:
    def test_contract_covers_all_sensors(self):
        assert len(SENSOR_COLS) == 21
        assert SENSOR_COLS[0] == "sensor_01"
        assert SENSOR_COLS[-1] == "sensor_21"

    def test_forbidden_includes_targets(self):
        for t in TARGET_COLS:
            assert t in FORBIDDEN_FEATURE_COLS

    def test_feature_columns_helper(self):
        sensors = get_sensor_config("A")
        cols = feature_columns(sensors, include_operational=True, include_regime=False)
        assert len(cols) == 24
        assert "raw_RUL" not in cols
        assert "unit_id" not in cols
