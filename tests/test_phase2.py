"""Phase 2 tests.

Every requirement in Phase 2 section 28 (QUALITY CHECKS) has at least one
test here:

    - no train/validation overlap               TestSplitIntegrity
    - no forbidden target features              TestFeatureContract
    - deterministic model training              TestDeterminism
    - correct target alignment                  TestTargetAlignment
    - class-weight / sample-weight behavior     TestImbalanceHandling
    - threshold calculation                     TestThresholdAnalysis
    - no test loading                           TestTestIsolation
    - no future rolling features                TestNoFutureRollingFeatures
    - experiment registry correctness           TestRegistry

Plus focused tests for the causal linear-degradation baseline and for the
prognostic-score function correctness.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.contract import FORBIDDEN_FEATURE_COLS, OPERATIONAL_COLS  # noqa: E402
from src.data.integrity import verify_raw  # noqa: E402
from src.data.loader import CYCLE, UNIT_ID, load_dataset  # noqa: E402
from src.data.regime import assign_regime  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    _prognostic_penalty,
    class_balance_report,
    classification_metrics,
    prognostic_score_last_per_engine,
    prognostic_score_rows,
    rul_metrics,
    suggest_candidate_threshold,
    threshold_sweep,
)
from src.features.sensors import get_sensor_config  # noqa: E402
from src.models.baselines import MajorityClass, NaiveAgeRUL, NaiveMeanRUL  # noqa: E402
from src.models.features import (  # noqa: E402
    HARD_FORBIDDEN,
    build_row_matrix,
    validate_row_features,
)
from src.models.gradient_boosting import (  # noqa: E402
    HistGBClassifierWrapper,
    HistGBRegressorWrapper,
    _inverse_freq_sample_weights,
)
from src.models.linear_degradation import LinearDegradationRUL  # noqa: E402
from src.models.logistic import LogisticRegressionWrapper  # noqa: E402
from src.models.ridge import RidgeRegressor  # noqa: E402
from src.pipeline.experiment import prepare, run_experiment  # noqa: E402
from src.pipeline.registry import (  # noqa: E402
    build_registry,
    load_experiments,
    load_phase1_split,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def fd004_full() -> pd.DataFrame:
    df = load_dataset("FD004", "train")
    df["operating_regime"] = assign_regime(df)
    return df


@pytest.fixture(scope="module")
def phase1_split() -> dict:
    return load_phase1_split()


@pytest.fixture(scope="module")
def specs() -> list:
    _, s = load_experiments()
    return s


@pytest.fixture(scope="module")
def ridge_prep(fd004_full, phase1_split):
    from src.pipeline.registry import ExperimentSpec
    spec = ExperimentSpec(
        experiment_id="_test_ridge", task="A_RUL", model="ridge",
        feature_config="A", normalization_mode="B", include_cycle=False,
        horizon=None, hyperparameters={"alpha": 1.0},
    )
    return prepare(spec, fd_id="FD004", split=phase1_split)


@pytest.fixture(scope="module")
def fr_prep(fd004_full, phase1_split):
    """Prep for Task B tests: same features but with failure_risk_target_H30."""
    from src.pipeline.registry import ExperimentSpec
    spec = ExperimentSpec(
        experiment_id="_test_fr", task="B_FAILURE_RISK", model="logistic_regression",
        feature_config="A", normalization_mode="B", include_cycle=False,
        horizon=30, hyperparameters={},
    )
    return prepare(spec, fd_id="FD004", split=phase1_split)


# ===========================================================================
# SPLIT INTEGRITY
# ===========================================================================


class TestSplitIntegrity:
    def test_train_val_engines_disjoint(self, phase1_split):
        assert set(phase1_split["train_engine_ids"]) & set(phase1_split["val_engine_ids"]) == set()

    def test_phase1_manifest_hash_stable(self, phase1_split):
        """Hash changes must reflect manifest changes; verify hash is present and non-empty."""
        assert len(phase1_split["train_hash"]) == 64
        assert len(phase1_split["val_hash"]) == 64

    def test_preparation_never_loads_test(self, phase1_split, fd004_full):
        from src.pipeline.registry import ExperimentSpec
        spec = ExperimentSpec(
            experiment_id="_t", task="A_RUL", model="ridge",
            feature_config="A", normalization_mode="A", include_cycle=False,
            horizon=None, hyperparameters={},
        )
        prep = prepare(spec, fd_id="FD004", split=phase1_split)
        used = set(prep.train_df[UNIT_ID].unique()) | set(prep.val_df[UNIT_ID].unique())
        # Every engine used is inside the Phase 1 manifest (which is FD004 TRAIN only).
        assert used <= set(fd004_full[UNIT_ID].unique())


# ===========================================================================
# FEATURE CONTRACT
# ===========================================================================


class TestFeatureContract:
    def test_target_leakage_always_rejected(self):
        """Even with allow_cycle=True, targets/unit_id/regime are NEVER permitted."""
        for col in ["raw_RUL", "model_RUL_target", "unit_id", "operating_regime",
                    "failure_risk_target_H30"]:
            with pytest.raises(ValueError, match="Row-level feature contract violation"):
                validate_row_features([col], allow_cycle=True)

    def test_cycle_requires_explicit_flag(self):
        with pytest.raises(ValueError, match="cycle"):
            validate_row_features(["cycle"], allow_cycle=False)
        # And it succeeds with allow_cycle=True
        validate_row_features(["cycle"], allow_cycle=True)

    def test_build_row_matrix_config_A_no_cycle(self, ridge_prep):
        X, names = build_row_matrix(
            ridge_prep.train_df, sensor_config="A", include_cycle=False
        )
        assert len(names) == 24  # 3 ops + 21 sensors
        assert X.shape[1] == 24
        # No forbidden column names in the feature list
        for f in names:
            assert f not in FORBIDDEN_FEATURE_COLS

    def test_build_row_matrix_config_B_drops_duplicates(self, ridge_prep):
        _, names = build_row_matrix(
            ridge_prep.train_df, sensor_config="B", include_cycle=False
        )
        assert "sensor_12" not in names
        assert "sensor_18" not in names
        assert "sensor_19" not in names
        assert len(names) == 21  # 3 + 18

    def test_build_row_matrix_config_Bplus_drops_sensor_16(self, ridge_prep):
        _, names = build_row_matrix(
            ridge_prep.train_df, sensor_config="B+", include_cycle=False
        )
        assert "sensor_16" not in names
        assert len(names) == 20  # 3 + 17

    def test_build_row_matrix_with_cycle_appends(self, ridge_prep):
        X, names = build_row_matrix(
            ridge_prep.train_df, sensor_config="A", include_cycle=True
        )
        assert names[-1] == "cycle"
        assert X.shape[1] == 25


# ===========================================================================
# DETERMINISM
# ===========================================================================


class TestDeterminism:
    def test_ridge_is_deterministic(self, ridge_prep):
        X_tr = ridge_prep.train_df[ridge_prep.feature_names].to_numpy()
        y_tr = ridge_prep.train_df["raw_RUL"].to_numpy()
        m1 = RidgeRegressor(alpha=1.0).fit(X_tr, y_tr)
        m2 = RidgeRegressor(alpha=1.0).fit(X_tr, y_tr)
        X_va = ridge_prep.val_df[ridge_prep.feature_names].to_numpy()
        np.testing.assert_allclose(m1.predict(X_va), m2.predict(X_va))

    def test_histgb_regressor_is_deterministic(self, ridge_prep):
        X_tr = ridge_prep.train_df[ridge_prep.feature_names].to_numpy()
        y_tr = ridge_prep.train_df["raw_RUL"].to_numpy()
        a = HistGBRegressorWrapper(max_iter=30).fit(X_tr, y_tr)
        b = HistGBRegressorWrapper(max_iter=30).fit(X_tr, y_tr)
        X_va = ridge_prep.val_df[ridge_prep.feature_names].to_numpy()
        np.testing.assert_allclose(a.predict(X_va), b.predict(X_va))

    def test_naive_baselines_are_deterministic(self, ridge_prep):
        m1 = NaiveMeanRUL().fit(ridge_prep.train_df)
        m2 = NaiveMeanRUL().fit(ridge_prep.train_df)
        assert m1.mean_rul == m2.mean_rul

        a1 = NaiveAgeRUL().fit(ridge_prep.train_df)
        a2 = NaiveAgeRUL().fit(ridge_prep.train_df)
        assert a1.mean_lifespan == a2.mean_lifespan

    def test_linear_degradation_is_deterministic(self, ridge_prep):
        """LinearDegradationRUL has no RNG at all; must produce identical output twice."""
        m1 = LinearDegradationRUL().fit(ridge_prep.train_df)
        m2 = LinearDegradationRUL().fit(ridge_prep.train_df)
        p1 = m1.predict(ridge_prep.val_df)
        p2 = m2.predict(ridge_prep.val_df)
        np.testing.assert_array_equal(p1, p2)


# ===========================================================================
# TARGET ALIGNMENT
# ===========================================================================


class TestTargetAlignment:
    def test_raw_rul_definition(self, ridge_prep):
        """For every val row, raw_RUL == max_cycle(engine) - cycle."""
        for uid, grp in ridge_prep.val_df.groupby(UNIT_ID):
            expected = grp[CYCLE].max() - grp[CYCLE]
            np.testing.assert_array_equal(grp["raw_RUL"].to_numpy(), expected.to_numpy())

    def test_failure_target_horizon(self, phase1_split):
        """failure_risk_target_H{H} == (raw_RUL <= H) for H in {14, 30, 50}."""
        from src.pipeline.registry import ExperimentSpec
        for h in (14, 30, 50):
            spec = ExperimentSpec(
                experiment_id=f"_h{h}", task="B_FAILURE_RISK",
                model="majority_class", feature_config=None,
                normalization_mode=None, include_cycle=False,
                horizon=h, hyperparameters={},
            )
            prep = prepare(spec, fd_id="FD004", split=phase1_split)
            col = f"failure_risk_target_H{h}"
            for df in (prep.train_df, prep.val_df):
                expected = (df["raw_RUL"] <= h).astype(np.int8)
                np.testing.assert_array_equal(df[col].to_numpy(), expected.to_numpy())

    def test_row_predictions_align_with_inputs(self, ridge_prep):
        """Predictions must be per-row aligned to val_df (no ordering swap)."""
        X_va = ridge_prep.val_df[ridge_prep.feature_names].to_numpy()
        y_va = ridge_prep.val_df["raw_RUL"].to_numpy()
        model = RidgeRegressor(alpha=1.0).fit(
            ridge_prep.train_df[ridge_prep.feature_names].to_numpy(),
            ridge_prep.train_df["raw_RUL"].to_numpy(),
        )
        preds = model.predict(X_va)
        # Predictions should correlate positively with the target
        assert np.corrcoef(preds, y_va)[0, 1] > 0.5


# ===========================================================================
# IMBALANCE HANDLING
# ===========================================================================


class TestImbalanceHandling:
    def test_inverse_freq_sample_weights_balanced(self):
        """Weighted average is equal for both classes with 'balanced' scheme."""
        y = np.array([0] * 90 + [1] * 10, dtype=np.int8)
        w = _inverse_freq_sample_weights(y)
        # Class 0 avg weight and class 1 avg weight should be inversely proportional to counts.
        w0_avg = w[y == 0].mean()
        w1_avg = w[y == 1].mean()
        # Class 1 is 9x rarer, so its weight is 9x higher.
        assert pytest.approx(w1_avg / w0_avg, rel=1e-9) == 9.0

    def test_histgb_classifier_respects_class_weight_toggle(self, fr_prep):
        """Fitting with balanced vs no class_weight must produce different predictions."""
        X_tr = fr_prep.train_df[fr_prep.feature_names].to_numpy()
        y_tr = fr_prep.train_df["failure_risk_target_H30"].to_numpy()
        X_va = fr_prep.val_df[fr_prep.feature_names].to_numpy()
        a = HistGBClassifierWrapper(max_iter=20, class_weight="balanced").fit(X_tr, y_tr)
        b = HistGBClassifierWrapper(max_iter=20, class_weight=None).fit(X_tr, y_tr)
        pa = a.positive_class_probability(X_va)
        pb = b.positive_class_probability(X_va)
        # Same data, same model, different weighting => predictions must differ.
        assert not np.allclose(pa, pb)

    def test_logreg_class_weight_balanced_by_default(self, fr_prep):
        X_tr = fr_prep.train_df[fr_prep.feature_names].to_numpy()
        y_tr = fr_prep.train_df["failure_risk_target_H30"].to_numpy()
        m = LogisticRegressionWrapper(C=1.0, class_weight="balanced", max_iter=200)
        m.fit(X_tr, y_tr)
        assert m._model.class_weight == "balanced"

    def test_class_balance_report_matches_labels(self, fr_prep):
        y = fr_prep.val_df["failure_risk_target_H30"].to_numpy()
        u = fr_prep.val_df[UNIT_ID].to_numpy()
        rep = class_balance_report(y, u)
        pos_rows = int((y == 1).sum())
        assert rep["positive_rows"] == pos_rows
        assert rep["total_rows"] == len(y)
        # Engines with any positive label
        engines_with_pos = {int(x) for x in u[y == 1]}
        assert rep["engines_with_any_positive"] == len(engines_with_pos)

    def test_majority_class_predicts_correct_label(self):
        m = MajorityClass().fit(np.array([0] * 100 + [1] * 20, dtype=np.int8))
        assert m.majority_label == 0
        preds = m.predict(5)
        assert np.all(preds == 0)
        m2 = MajorityClass().fit(np.array([0] * 10 + [1] * 40, dtype=np.int8))
        assert m2.majority_label == 1


# ===========================================================================
# THRESHOLD ANALYSIS
# ===========================================================================


class TestThresholdAnalysis:
    def test_threshold_sweep_shape(self):
        y_true = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        y_prob = np.array([0.1, 0.9, 0.4, 0.6, 0.2])
        sweep = threshold_sweep(y_true, y_prob, n_thresholds=10)
        for k in ("thresholds", "precision", "recall", "f1", "accuracy"):
            assert len(sweep[k]) == 10

    def test_sweep_extremes(self):
        y_true = np.array([0, 1, 0, 1], dtype=np.int8)
        y_prob = np.array([0.2, 0.8, 0.3, 0.9])
        sweep = threshold_sweep(y_true, y_prob, n_thresholds=20)
        # Lowest threshold => everything positive => recall = 1
        assert sweep["recall"][0] == pytest.approx(1.0)
        # Highest threshold => fewer positives; recall non-increasing
        assert sweep["recall"][-1] <= sweep["recall"][0]

    def test_candidate_threshold_maximizes_f1(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1], dtype=np.int8)
        y_prob = np.array([0.1, 0.9, 0.4, 0.5, 0.8, 0.2, 0.7])
        sweep = threshold_sweep(y_true, y_prob, n_thresholds=101)
        thr, m = suggest_candidate_threshold(sweep, criterion="f1")
        # The chosen threshold's F1 equals the max in the sweep
        assert m["f1"] == pytest.approx(max(sweep["f1"]))

    def test_metrics_for_majority_classifier_are_degenerate(self):
        y_true = np.array([0] * 100 + [1] * 10, dtype=np.int8)
        y_pred = np.zeros_like(y_true)  # always predict negative
        m = classification_metrics(y_true, y_prob=None, y_pred=y_pred).to_dict()
        # Recall is exactly 0 for the positive class.
        assert m["recall"] == 0.0
        # Precision is 0 (undefined; sklearn's zero_division=0 gives 0.0).
        assert m["precision"] == 0.0
        # F1 is 0
        assert m["f1"] == 0.0
        # PR-AUC / ROC-AUC are None (not 0 or 1).
        assert m["pr_auc"] is None
        assert m["roc_auc"] is None
        assert any("Deterministic classifier" in n for n in m["notes"])


# ===========================================================================
# OFFICIAL TEST ISOLATION
# ===========================================================================


class TestTestIsolation:
    """The Phase 2 pipeline must NEVER load test_FD004.txt or RUL_FD004.txt."""

    def test_phase2_src_has_no_test_load_call(self):
        """Scan src/models, src/evaluation, src/pipeline for load_dataset(..., 'test')."""
        bad = []
        for sub in ("models", "evaluation", "pipeline"):
            for f in (ROOT / "src" / sub).rglob("*.py"):
                text = f.read_text(encoding="utf-8")
                # Match actual call forms; ignore comments/docstrings that
                # explicitly say "never loaded" or "test set".
                for m in re.finditer(r"load_dataset\s*\([^)]*['\"]test['\"]", text):
                    bad.append((str(f), m.group(0)))
                for m in re.finditer(r"load_rul\s*\(", text):
                    bad.append((str(f), m.group(0)))
        assert bad == [], f"Test data load call found: {bad}"

    def test_phase2_driver_has_no_test_load_call(self):
        bad = []
        for f in (ROOT / "scripts").glob("*phase2*.py"):
            text = f.read_text(encoding="utf-8")
            for m in re.finditer(r"load_dataset\s*\([^)]*['\"]test['\"]", text):
                bad.append((str(f), m.group(0)))
            for m in re.finditer(r"load_rul\s*\(", text):
                bad.append((str(f), m.group(0)))
        assert bad == []


# ===========================================================================
# NO FUTURE ROLLING FEATURES (Phase 2 section 21)
# ===========================================================================


class TestNoFutureRollingFeatures:
    def test_phase2_has_no_rolling_feature_helper(self):
        """Phase 2 uses only raw Phase-1 features. Any rolling helper would
        live in src/features/; assert none was introduced ON THE PHASE 2 PATH.

        Phase 2.5 separately authorizes strictly-causal rolling features in
        ``src/features/temporal.py`` (guarded by tests/test_phase2_5.py
        causality tests A-F). That module is excluded here so the Phase 2
        invariant (raw row-level features, no windowing) is preserved without
        contradicting the later, explicitly-authorized phase."""
        files = list((ROOT / "src" / "features").rglob("*.py"))
        for f in files:
            if f.name == "temporal.py":  # Phase 2.5 authorized causal module
                continue
            text = f.read_text(encoding="utf-8").lower()
            # We deliberately do NOT introduce rolling on the Phase 2 path.
            assert "rolling(" not in text, f"Unexpected rolling in {f}"

    def test_features_are_row_level_only(self, ridge_prep):
        """build_row_matrix uses df[feature_names].to_numpy() directly, no window."""
        X, _ = build_row_matrix(ridge_prep.train_df, sensor_config="A", include_cycle=False)
        # If we ever sneak in a windowing layer, X would be 3D.
        assert X.ndim == 2


# ===========================================================================
# REGISTRY
# ===========================================================================


class TestRegistry:
    def test_all_experiments_have_required_fields(self, specs):
        required = {
            "experiment_id", "task", "model", "feature_config",
            "normalization_mode", "include_cycle", "horizon",
            "hyperparameters",
        }
        for s in specs:
            d = s.to_dict()
            for k in required:
                assert k in d, f"{s.experiment_id} missing {k}"

    def test_no_deep_learning_models_allowed(self, specs):
        """Every registered model name must be from the fixed Phase 2 allowlist."""
        allowed = {
            "naive_mean", "naive_age", "linear_degradation",
            "ridge", "hist_gb_regressor",
            "majority_class", "logistic_regression", "hist_gb_classifier",
        }
        for s in specs:
            assert s.model in allowed, f"{s.experiment_id} uses non-allowed model {s.model}"

    def test_task_a_horizon_must_be_none(self, specs):
        for s in specs:
            if s.task == "A_RUL":
                assert s.horizon is None

    def test_task_b_horizon_must_be_present(self, specs):
        for s in specs:
            if s.task == "B_FAILURE_RISK":
                assert s.horizon in {14, 30, 50}

    def test_experiment_ids_are_unique(self, specs):
        ids = [s.experiment_id for s in specs]
        assert len(ids) == len(set(ids))

    def test_registry_file_matches_specs(self, specs, phase1_split):
        defaults, _ = load_experiments()
        reg = build_registry(specs=specs, phase1_split=phase1_split, defaults=defaults)
        reg_ids = [e["experiment_id"] for e in reg["experiments"]]
        assert set(reg_ids) == {s.experiment_id for s in specs}
        # Reference to Phase 1 hashes must be present
        assert reg["phase1_split_reference"]["train_hash"] == phase1_split["train_hash"]
        assert reg["phase1_split_reference"]["val_hash"] == phase1_split["val_hash"]


# ===========================================================================
# LINEAR DEGRADATION BASELINE — CAUSALITY
# ===========================================================================


class TestLinearDegradationCausality:
    """The Phase 2 spec (section 6) requires the linear-degradation baseline
    to be CAUSAL: predicting RUL at cycle t must use only cycles 1..t of the
    same engine, never the future trajectory."""

    def test_corrupt_last_cycle_does_not_change_early_predictions(self, ridge_prep):
        """Modifying an engine's last observed HI must not alter predictions
        at earlier cycles."""
        val_df = ridge_prep.val_df.copy()
        model = LinearDegradationRUL().fit(ridge_prep.train_df)
        preds_before = model.predict(val_df)

        # Find the largest engine and corrupt its last cycle
        sizes = val_df.groupby(UNIT_ID).size()
        target_uid = int(sizes.idxmax())
        idx = val_df[val_df[UNIT_ID] == target_uid].index[-1]
        corrupted = val_df.copy()
        for c in ["sensor_14", "sensor_17", "sensor_11"]:
            corrupted.loc[idx, c] = 999.0

        preds_after = model.predict(corrupted)

        # All rows for target_uid BEFORE its last cycle must be identical.
        mask_before = (
            (val_df[UNIT_ID] == target_uid).to_numpy()
            & (np.arange(len(val_df)) != list(val_df.index).index(idx))
        )
        np.testing.assert_array_equal(preds_before[mask_before], preds_after[mask_before])


# ===========================================================================
# PROGNOSTIC SCORE (Section 11)
# ===========================================================================


class TestPrognosticScore:
    def test_perfect_prediction_scores_zero(self):
        y = np.array([100, 50, 20, 5], dtype=np.float64)
        p = y.copy()
        assert prognostic_score_rows(y, p) == pytest.approx(0.0)

    def test_asymmetric_penalty(self):
        """Over-prediction (predicted > actual) must be penalized harder than
        under-prediction of the same magnitude (Saxena 2008)."""
        y = np.array([100.0])
        under = np.array([90.0])   # E = -10
        over = np.array([110.0])   # E = +10
        s_under = prognostic_score_rows(y, under)
        s_over = prognostic_score_rows(y, over)
        # exp(10/10)-1 = 1.718; exp(10/13)-1 = 1.153.  Over > Under.
        assert s_over > s_under

    def test_formula_matches_reference(self):
        """Direct element check against the published formula."""
        errs = np.array([-5.0, 0.0, 5.0])
        pen = _prognostic_penalty(errs)
        expected = np.array([
            np.exp(5.0 / 13.0) - 1.0,   # E<0
            0.0,                        # E>=0 with E=0 => exp(0)-1 = 0
            np.exp(5.0 / 10.0) - 1.0,   # E>0
        ])
        np.testing.assert_allclose(pen, expected)

    def test_last_per_engine_variant_uses_one_prediction_per_engine(self):
        # Two engines, three rows total; last row per engine is (index 1, 2).
        y = np.array([100, 50, 20], dtype=np.float64)
        p = np.array([100, 50, 30], dtype=np.float64)
        u = np.array([1, 1, 2])
        # Errors at engine 1 last (index 1): 0. Error at engine 2 last (index 2): +10.
        score = prognostic_score_last_per_engine(y, p, u)
        assert score == pytest.approx(np.exp(10.0 / 10.0) - 1.0)

    def test_rul_metrics_returns_all_keys(self):
        y = np.array([100, 50, 20], dtype=np.float64)
        p = np.array([95, 55, 25], dtype=np.float64)
        m = rul_metrics(y, p, unit_ids=np.array([1, 1, 2]))
        for k in ("mae", "rmse", "r2", "prognostic_score_rows",
                  "prognostic_score_last_per_engine"):
            assert k in m


# ===========================================================================
# End-to-end smoke test
# ===========================================================================


class TestEndToEndSmoke:
    def test_single_experiment_runs_end_to_end(self, phase1_split, specs):
        """One representative experiment of each kind must complete without error."""
        # Pick: naive_mean, ridge with cycle, hist_gb, logreg, majority, linear_degradation
        targets = {
            "RUL_naive_mean",
            "RUL_ridge_globA_withCyc_cfgA",
            "RUL_gb_regimeB_withCyc_cfgA",
            "RUL_linear_degradation",
            "FR_majority_H30",
            "FR_logreg_regimeB_noCyc_cfgA",
            "FR_gb_regimeB_noCyc_cfgA",
        }
        chosen = [s for s in specs if s.experiment_id in targets]
        assert len(chosen) == len(targets)
        for spec in chosen:
            r = run_experiment(spec, fd_id="FD004", split=phase1_split)
            assert r.metrics  # non-empty
            assert len(r.val_rows) > 0


# ===========================================================================
# WORDING / REPORT AUDIT (project-lead corrective review items 1, 4, 6)
# ===========================================================================


class TestWordingAndReport:
    """Enforce the methodological language corrections required by the
    Phase 2 corrective review:

    1. `cycle` must never be characterised as target leakage in Phase 2
       code, tests, or the Phase 2 report (it is a legitimate,
       target-correlated observable).
    2. The majority baseline's undefined PR-AUC/ROC-AUC must not be
       described as a metric it was 'beaten' on.
    3. The report must carry the row-level metric dependence limitation.
    """

    PHASE2_TEXT_SOURCES = (
        list((ROOT / "src" / "models").rglob("*.py"))
        + list((ROOT / "src" / "evaluation").rglob("*.py"))
        + list((ROOT / "src" / "pipeline").rglob("*.py"))
        + list((ROOT / "scripts").glob("*phase2*.py"))
        + [ROOT / "reports" / "PHASE_2_BASELINE_RESULTS.md"]
    )

    def test_cycle_never_described_as_target_leakage(self):
        """Phase 2 sources must never assert that `cycle` is leakage. We scan
        for leak-phrases collocated with 'cycle' inside the same sentence
        ("cycle ... leak", "leakage of (the) cycle", "leaks the ... cycle").
        A sentence that explicitly DENIES leakage ("not leakage") contains
        no such collocation and passes."""
        patterns = [
            r"\bcycle\b(?:(?!\.)[^.\n]){0,80}?(?<!not )\bleak",
            r"\bleak\w*(?:(?!\.)[^.\n]){0,80}?\bcycle\b",
        ]
        bad = []
        for f in self.PHASE2_TEXT_SOURCES:
            if not f.exists():
                continue
            text = f.read_text(encoding="utf-8")
            for pat in patterns:
                for m in re.finditer(pat, text, re.IGNORECASE):
                    bad.append((f.name, m.group(0)[:100]))
        assert bad == [], f"cycle described near 'leak' in: {bad}"

    def test_beats_majority_except_accuracy_phrase_absent(self):
        report = (ROOT / "reports" / "PHASE_2_BASELINE_RESULTS.md").read_text(encoding="utf-8")
        assert "except accuracy" not in report.lower()
        # And the corrected statement IS present.
        assert "all defined predictive metrics" in report.lower()

    def test_report_contains_row_level_dependence_limitation(self):
        report = (ROOT / "reports" / "PHASE_2_BASELINE_RESULTS.md").read_text(encoding="utf-8")
        assert "not independent" in report
        assert "50 independent engines" in report
        assert "row-weighted" in report

    def test_official_test_files_remain_unmodified(self):
        """Sealed-test audit: every raw C-MAPSS file, including
        test_FD004.txt and RUL_FD004.txt, must still match the
        Phase 0 checksum registry (unchanged since the seal)."""
        results = {fi.name: fi for fi in verify_raw()}
        for name in ("test_FD004.txt", "RUL_FD004.txt"):
            assert name in results, f"{name} missing from checksum registry"
            assert results[name].match, f"{name} content changed since Phase 0 seal!"

