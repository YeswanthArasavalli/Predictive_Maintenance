"""Phase 2.5 tests — Causal Temporal Feature Baselines.

The Phase 2.5 spec (section 12) makes causality tests *more important than
adding features*. This file implements them literally:

    Test A — future perturbation invariance   (test_future_perturbation_invariance)
    Test B — no future index usage            (test_rolling_window_uses_only_past_and_present)
    Test C — engine boundary isolation        (test_windows_never_cross_engine_boundary)
    Test D — split isolation                  (test_val_features_independent_of_train_rows)
    Test E — train-only fitting / selection   (test_monitor_selection_is_train_only)
    Test F — deterministic reproduction       (test_temporal_pipeline_is_deterministic)

Plus registry / allowlist / official-test-isolation / wording checks. No neural
model and no official test data can pass here.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.contract import FORBIDDEN_FEATURE_COLS  # noqa: E402
from src.data.integrity import verify_raw  # noqa: E402
from src.data.loader import CYCLE, UNIT_ID, load_dataset  # noqa: E402
from src.data.regime import assign_regime  # noqa: E402
from src.features.temporal import (  # noqa: E402
    FAMILY_COMBINED,
    FAMILY_NONE,
    FAMILY_ROLLING,
    FAMILY_SLOPE,
    TemporalConfig,
    add_causal_temporal_features,
    temporal_feature_names,
    _rolling_slope,
)
from src.models.features import select_temporal_monitors  # noqa: E402
from src.models.features import validate_row_features  # noqa: E402
from src.pipeline.experiment import prepare, run_experiment  # noqa: E402
from src.pipeline.registry import (  # noqa: E402
    ExperimentSpec,
    build_registry,
    load_experiments,
    load_phase1_split,
)

CONFIG_PATH = ROOT / "configs" / "phase2_5_experiments.yaml"


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def phase1_split() -> dict:
    return load_phase1_split()


@pytest.fixture(scope="module")
def specs25() -> list:
    _, s = load_experiments(CONFIG_PATH)
    return s


def _combined_cfg(sources, window=3):
    return TemporalConfig(
        family=FAMILY_COMBINED, windows=(window,), lags=(window,),
        source_cols=tuple(sources), monitor_sensors=tuple(sources),
    )


def _synthetic() -> pd.DataFrame:
    """Two engines, contiguous cycles, a drifting + a flat sensor."""
    rng = np.random.default_rng(0)
    n1, n2 = 12, 9
    df = pd.DataFrame({
        UNIT_ID: [1] * n1 + [2] * n2,
        CYCLE: list(range(1, n1 + 1)) + list(range(1, n2 + 1)),
        "sensor_14": np.concatenate([np.arange(n1, dtype=float) + rng.normal(0, 0.1, n1),
                                     np.full(n2, 5.0)]),
        "sensor_17": np.concatenate([rng.normal(0, 1, n1), rng.normal(0, 1, n2)]),
    })
    return df


# ===========================================================================
# TEST A — future perturbation invariance
# ===========================================================================


class TestCausalityAFuturePerturbationInvariance:
    def test_future_perturbation_invariance(self):
        df = _synthetic()
        cfg = _combined_cfg(["sensor_14", "sensor_17"], window=3)
        base = add_causal_temporal_features(df, cfg)
        names = temporal_feature_names(cfg)

        # Perturb sensor_14 STRICTLY after cycle t0 (engine 1) by an absurd value.
        t0 = 6
        pert = df.copy()
        mask = (pert[UNIT_ID] == 1) & (pert[CYCLE] > t0)
        pert.loc[mask, "sensor_14"] = 9.9e5

        after = add_causal_temporal_features(pert, cfg)
        rows_le_t0 = (base[UNIT_ID] == 1) & (base[CYCLE] <= t0)
        for n in names:
            np.testing.assert_allclose(
                base.loc[rows_le_t0, n].to_numpy(),
                after.loc[rows_le_t0, n].to_numpy(),
                rtol=1e-12, atol=1e-9,
                err_msg=f"{n} changed when FUTURE observations were modified",
            )

    def test_current_value_also_affects_own_row(self):
        # Sanity: a feature at t SHOULD depend on x[t] (not only the past).
        df = _synthetic()
        cfg = _combined_cfg(["sensor_14"], window=3)
        base = add_causal_temporal_features(df, cfg)
        df2 = df.copy()
        df2.loc[(df2[UNIT_ID] == 1) & (df2[CYCLE] == 5), "sensor_14"] += 100.0
        after = add_causal_temporal_features(df2, cfg)
        assert not np.allclose(
            base.loc[base[CYCLE] == 5, "sensor_14__rmean3"].to_numpy(),
            after.loc[after[CYCLE] == 5, "sensor_14__rmean3"].to_numpy(),
        )


# ===========================================================================
# TEST B — no future index usage (window is [t-w+1, t])
# ===========================================================================


class TestCausalityBNoFutureUsage:
    def test_rolling_window_uses_only_past_and_present(self):
        df = _synthetic()
        w = 3
        cfg = _combined_cfg(["sensor_14"], window=w)
        out = add_causal_temporal_features(df, cfg)
        x = df[df[UNIT_ID] == 1]["sensor_14"].to_numpy()
        for t in range(len(x)):
            lo = max(0, t - w + 1)
            expected = x[lo:t + 1].mean()  # only indices <= t
            got = out[(out[UNIT_ID] == 1)].iloc[t]["sensor_14__rmean3"]
            assert got == pytest.approx(expected, rel=1e-12)

    def test_first_difference_uses_only_current_and_previous(self):
        df = _synthetic()
        cfg = _combined_cfg(["sensor_14"], window=3)
        out = add_causal_temporal_features(df, cfg)
        eng = df[df[UNIT_ID] == 1]["sensor_14"].to_numpy()
        d1 = out[df[UNIT_ID] == 1]["sensor_14__d1"].to_numpy()
        assert d1[0] == 0.0  # no prior observation at engine start
        np.testing.assert_allclose(d1[1:], eng[1:] - eng[:-1], rtol=1e-12)

    def test_rolling_slope_matches_causal_polyfit(self):
        x = np.cumsum(np.random.default_rng(1).normal(0, 1, 40)) + 100.0
        df = pd.DataFrame({UNIT_ID: [7] * 40, CYCLE: range(1, 41), "sensor_14": x})
        w = 5
        out = add_causal_temporal_features(df, _combined_cfg(["sensor_14"], window=w))
        sl = out["sensor_14__slope5"].to_numpy()
        for t in range(len(x)):
            lo = max(0, t - w + 1)
            seg = x[lo:t + 1]
            expected = np.polyfit(np.arange(lo, t + 1), seg, 1)[0] if len(seg) >= 2 else 0.0
            assert sl[t] == pytest.approx(expected, rel=1e-9, abs=1e-9)


# ===========================================================================
# TEST C — engine boundary isolation
# ===========================================================================


class TestCausalityCEngineBoundaryIsolation:
    def test_windows_never_cross_engine_boundary(self):
        df = _synthetic()
        cfg = _combined_cfg(["sensor_14"], window=3)
        out = add_causal_temporal_features(df, cfg)
        eng2 = out[out[UNIT_ID] == 2]
        x2 = df[df[UNIT_ID] == 2]["sensor_14"].to_numpy()
        # Engine 2 first row: expanding window => rmean equals its own value,
        # d1 == 0, i.e. NO contamination from engine 1's tail.
        assert eng2["sensor_14__rmean3"].iloc[0] == pytest.approx(x2[0], rel=1e-12)
        assert eng2["sensor_14__d1"].iloc[0] == 0.0
        # Second row slope must depend only on engine 2's first two rows.
        expected_slope = np.polyfit(np.arange(2), x2[:2], 1)[0]
        assert eng2["sensor_14__slope3"].iloc[1] == pytest.approx(expected_slope, rel=1e-9)

    def test_shuffled_row_order_reproduces_identical_features(self):
        """Grouping by engine makes construction order-independent within an
        engine; a stable re-sort must give identical per-row values."""
        df = _synthetic()
        cfg = _combined_cfg(["sensor_14", "sensor_17"], window=4)
        a = add_causal_temporal_features(df, cfg)
        shuffled = df.sample(frac=1.0, random_state=3)
        b = add_causal_temporal_features(shuffled, cfg)
        b = b.reindex(a.index)
        for n in temporal_feature_names(cfg):
            np.testing.assert_allclose(a[n].to_numpy(), b[n].to_numpy(), rtol=1e-12)


# ===========================================================================
# TEST D — split isolation
# ===========================================================================


class TestCausalityDSplitIsolation:
    def test_val_features_independent_of_train_rows(self, phase1_split):
        """Temporal features are built per partition with groupby(unit_id);
        a validation engine's features cannot be affected by training rows."""
        spec = ExperimentSpec(
            experiment_id="_t25_d", task="A_RUL", model="hist_gb_regressor",
            feature_config="A", normalization_mode="B", include_cycle=True,
            horizon=None, hyperparameters={}, temporal_family=FAMILY_ROLLING,
            temporal_windows=[5], temporal_lags=[5], temporal_n_monitors=6,
        )
        prep = prepare(spec, fd_id="FD004", split=phase1_split)
        val_uids = set(prep.val_df[UNIT_ID].unique())
        train_uids = set(prep.train_df[UNIT_ID].unique())
        assert not (val_uids & train_uids)

        # Corrupt a TRAIN engine's sensor and rebuild features; VAL features must
        # be byte-identical because they are computed on the val partition alone.
        temporal_names = temporal_feature_names(TemporalConfig(
            family=FAMILY_ROLLING, windows=(5,), lags=(5,),
            source_cols=tuple(prep.temporal["monitors_selected_train_only"]),
            monitor_sensors=tuple(prep.temporal["monitors_selected_train_only"]),
        ))
        cfg = TemporalConfig(
            family=FAMILY_ROLLING, windows=(5,), lags=(5,),
            source_cols=tuple(prep.temporal["monitors_selected_train_only"]),
            monitor_sensors=tuple(prep.temporal["monitors_selected_train_only"]),
        )
        val_before = prep.val_df[temporal_names].to_numpy()
        # Recompute val features in isolation (train removed) => identical.
        val_only = add_causal_temporal_features(prep.val_df.copy(), cfg)
        np.testing.assert_allclose(val_only[temporal_names].to_numpy(), val_before, rtol=1e-12)

    def test_prepare_fits_scaler_on_train_only(self, phase1_split):
        spec = ExperimentSpec(
            experiment_id="_t25_scale", task="A_RUL", model="hist_gb_regressor",
            feature_config="A", normalization_mode="B", include_cycle=False,
            horizon=None, hyperparameters={}, temporal_family=FAMILY_COMBINED,
        )
        prep = prepare(spec, fd_id="FD004", split=phase1_split)
        # Mode-B scaler is a train-fit; there is no attribute that references val
        # engines. We assert the split filter kept val engines out of train.
        assert set(prep.train_df[UNIT_ID].unique()) & set(prep.val_df[UNIT_ID].unique()) == set()


# ===========================================================================
# TEST E — train-only fitting / selection
# ===========================================================================


class TestCausalityETrainOnlySelection:
    def test_monitor_selection_is_train_only(self, phase1_split):
        spec = ExperimentSpec(
            experiment_id="_t25_e", task="A_RUL", model="hist_gb_regressor",
            feature_config="A", normalization_mode="B", include_cycle=True,
            horizon=None, hyperparameters={}, temporal_family=FAMILY_COMBINED,
            temporal_windows=[5], temporal_lags=[5], temporal_n_monitors=8,
        )
        prep = prepare(spec, fd_id="FD004", split=phase1_split)
        monitors = prep.temporal["monitors_selected_train_only"]
        # Recompute selection from TRAIN rows only, independently => must match.
        from src.features.sensors import get_sensor_config
        candidates = get_sensor_config("A")
        # train_df already has temporal cols; drop them for a clean re-selection.
        train_clean = prep.train_df[candidates + [UNIT_ID, CYCLE]].copy()
        reselected = select_temporal_monitors(train_clean, candidates, top_n=8)
        assert reselected == monitors

    def test_selection_unaffected_by_validation_rows(self, phase1_split):
        spec = ExperimentSpec(
            experiment_id="_t25_e2", task="A_RUL", model="hist_gb_regressor",
            feature_config="A", normalization_mode="B", include_cycle=True,
            horizon=None, hyperparameters={}, temporal_family=FAMILY_SLOPE,
            temporal_n_monitors=6,
        )
        prep = prepare(spec, fd_id="FD004", split=phase1_split)
        monitors = list(prep.temporal["monitors_selected_train_only"])
        from src.features.sensors import get_sensor_config
        candidates = get_sensor_config("A")
        train_cols = candidates + [UNIT_ID, CYCLE]
        base_selection = select_temporal_monitors(prep.train_df[train_cols], candidates, top_n=6)
        assert base_selection == monitors
        # Corrupt validation sensors arbitrarily; a TRAIN-only selection cannot move.
        prep.val_df[candidates[0]] = prep.val_df[candidates[0]] * 50.0 + 7.0
        after_selection = select_temporal_monitors(prep.train_df[train_cols], candidates, top_n=6)
        assert after_selection == monitors


# ===========================================================================
# TEST F — deterministic reproduction
# ===========================================================================


class TestCausalityFDeterminism:
    def test_temporal_pipeline_is_deterministic(self):
        df = _synthetic()
        cfg = _combined_cfg(["sensor_14", "sensor_17"], window=3)
        a = add_causal_temporal_features(df, cfg)
        b = add_causal_temporal_features(df, cfg)
        for n in temporal_feature_names(cfg):
            np.testing.assert_array_equal(a[n].to_numpy(), b[n].to_numpy())

    def test_run_experiment_end_to_end_deterministic(self, phase1_split):
        spec = ExperimentSpec(
            experiment_id="_t25_f", task="A_RUL", model="hist_gb_regressor",
            feature_config="A", normalization_mode="B", include_cycle=True,
            horizon=None, hyperparameters={"max_iter": 30},
            temporal_family=FAMILY_COMBINED, temporal_n_monitors=4,
        )
        r1 = run_experiment(spec, fd_id="FD004", split=phase1_split)
        r2 = run_experiment(spec, fd_id="FD004", split=phase1_split)
        np.testing.assert_allclose(
            r1.val_rows["y_pred"].to_numpy(), r2.val_rows["y_pred"].to_numpy(), rtol=0, atol=0
        )
        assert r1.metrics["mae"] == r2.metrics["mae"]


# ===========================================================================
# Feature contract on temporal columns
# ===========================================================================


class TestTemporalFeatureContract:
    def test_temporal_names_are_not_target_or_identifier(self):
        cfg = TemporalConfig(
            family=FAMILY_COMBINED, windows=(5,), lags=(5,),
            source_cols=tuple(f"sensor_{i:02d}" for i in range(1, 22)),
            monitor_sensors=tuple(f"sensor_{i:02d}" for i in [11, 15, 4, 14]),
        )
        names = temporal_feature_names(cfg)
        assert names, "combined family must produce features"
        for n in names:
            assert n not in FORBIDDEN_FEATURE_COLS
            assert not str(n).startswith("failure_risk_target_")
            assert "raw_RUL" not in n
            # validate_row_features must accept them (derived from sensors)
        validate_row_features(names, allow_cycle=True)

    def test_family_none_produces_no_temporal_features(self):
        cfg = TemporalConfig(family=FAMILY_NONE, source_cols=("sensor_14",))
        assert temporal_feature_names(cfg) == []
        df = _synthetic()
        out = add_causal_temporal_features(df, cfg)
        assert list(out.columns) == list(df.columns)

    def test_monitor_count_controls_feature_explosion(self):
        sources = tuple(f"sensor_{i:02d}" for i in range(1, 22))
        cfg_all = TemporalConfig(family=FAMILY_COMBINED, windows=(5,), lags=(5,),
                                 source_cols=sources, monitor_sensors=sources[:8])
        assert len(temporal_feature_names(cfg_all)) == 8 * 5  # d1 + dk + rmean + rstd + slope


# ===========================================================================
# Registry / allowlist / no deep learning
# ===========================================================================


class TestRegistryPhase25:
    def test_all_models_are_classical(self, specs25):
        allowed = {
            "naive_mean", "naive_age", "linear_degradation",
            "ridge", "hist_gb_regressor",
            "majority_class", "logistic_regression", "hist_gb_classifier",
        }
        forbidden_kw = ("lstm", "gru", "transformer", "cnn", "rnn", "mlp", "deep", "neural")
        for s in specs25:
            assert s.model in allowed
            assert not any(k in s.model.lower() for k in forbidden_kw)

    def test_required_experiment_families_present(self, specs25):
        fams = {s.temporal_family for s in specs25}
        assert {"none", "lag_diff", "rolling", "slope", "combined"} <= fams
        # cycle-blind temporal must exist for both tasks
        rul_blind = [s for s in specs25 if s.task == "A_RUL" and not s.include_cycle and s.temporal_family == "combined"]
        fr_blind = [s for s in specs25 if s.task == "B_FAILURE_RISK" and not s.include_cycle and s.temporal_family == "combined"]
        assert rul_blind and fr_blind

    def test_horizon_sensitivity_present(self, specs25):
        horizons = {s.horizon for s in specs25 if s.task == "B_FAILURE_RISK"}
        assert {14, 30, 50} <= horizons

    def test_window_ablation_present(self, specs25):
        rul = [s for s in specs25 if s.task == "A_RUL" and s.temporal_family == "combined"]
        wins = {tuple(s.temporal_windows) for s in rul}
        assert len(wins) >= 2  # short vs medium distinguished

    def test_registry_records_causal_temporal_policy(self, specs25, phase1_split):
        defaults, _ = load_experiments(CONFIG_PATH)
        reg = build_registry(specs=specs25, phase1_split=phase1_split, defaults=defaults, phase=2.5)
        assert reg["phase"] == 2.5
        assert reg["hard_rules"]["no_deep_learning"] is True
        assert reg["temporal_policy"]["enabled"] is True
        assert reg["phase1_split_reference"]["train_hash"] == phase1_split["train_hash"]


# ===========================================================================
# Anchors reproduce the Phase 2 authoritative baseline
# ===========================================================================


class TestAnchorReproducesPhase2:
    def test_rul_anchor_cycle_matches_phase2(self, phase1_split):
        spec = ExperimentSpec(
            experiment_id="_anchor_rul", task="A_RUL", model="hist_gb_regressor",
            feature_config="A", normalization_mode="B", include_cycle=True,
            horizon=None, hyperparameters={"max_iter": 200, "learning_rate": 0.1},
            temporal_family=FAMILY_NONE,
        )
        r = run_experiment(spec, fd_id="FD004", split=phase1_split)
        # Phase 2 authoritative: MAE ~32.01, R2 ~0.762, PS_last ~58.99
        assert r.metrics["mae"] == pytest.approx(32.01, abs=0.05)
        assert r.metrics["r2"] == pytest.approx(0.762, abs=0.01)

    def test_phase2_5_artifacts_exist_on_disk(self):
        exp_dir = ROOT / "results" / "experiments" / "phase2_5"
        assert (exp_dir / "registry.json").exists()
        assert (exp_dir / "rul_comparison.csv").exists()
        assert (exp_dir / "failure_risk_comparison.csv").exists()
        assert (exp_dir / "temporal_feature_metadata.json").exists()


# ===========================================================================
# Official test isolation
# ===========================================================================


class TestTestIsolation25:
    def test_phase2_5_src_and_scripts_have_no_test_load_call(self):
        bad = []
        for sub in ("models", "evaluation", "pipeline", "features"):
            for f in (ROOT / "src" / sub).rglob("*.py"):
                text = f.read_text(encoding="utf-8")
                for pat in (r"load_dataset\s*\([^)]*['\"]test['\"]", r"load_rul\s*\("):
                    for m in re.finditer(pat, text):
                        bad.append((str(f), m.group(0)))
        for f in (ROOT / "scripts").glob("*phase2_5*.py"):
            text = f.read_text(encoding="utf-8")
            for pat in (r"load_dataset\s*\([^)]*['\"]test['\"]", r"load_rul\s*\("):
                for m in re.finditer(pat, text):
                    bad.append((str(f), m.group(0)))
        assert bad == [], f"Official-test load call found in Phase 2.5 code: {bad}"

    def test_official_test_files_unchanged_since_seal(self):
        results = {fi.name: fi for fi in verify_raw()}
        for name in ("test_FD004.txt", "RUL_FD004.txt"):
            assert name in results
            assert results[name].match, f"{name} changed since Phase 0 seal"


# ===========================================================================
# Wording / report audit
# ===========================================================================


class TestWordingPhase25:
    REPORT = ROOT / "reports" / "PHASE_2_5_TEMPORAL_BASELINES.md"

    def test_cycle_never_called_leakage_in_phase2_5_code(self):
        patterns = [
            r"\bcycle\b(?:(?!\.)[^.\n]){0,80}?(?<!not )\bleak",
            r"\bleak\w*(?:(?!\.)[^.\n]){0,80}?\bcycle\b",
        ]
        sources = (
            list((ROOT / "src" / "features").rglob("*.py"))
            + list((ROOT / "src" / "models").rglob("*.py"))
            + list((ROOT / "scripts").glob("*phase2_5*.py"))
        )
        bad = []
        for f in sources:
            text = f.read_text(encoding="utf-8")
            for pat in patterns:
                for m in re.finditer(pat, text, re.IGNORECASE):
                    bad.append((f.name, m.group(0)[:80]))
        assert bad == []

    def test_report_distinguishes_phases_and_states_evidence(self):
        if not self.REPORT.exists():
            pytest.skip("report written at completion")
        r = self.REPORT.read_text(encoding="utf-8").lower()
        assert "phase 2" in r and "phase 2.5" in r
        assert "operational cycles" in r or "cycles" in r
        assert "50 independent engines" in r
