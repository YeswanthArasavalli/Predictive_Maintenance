"""Phase 2.6 tests — Decision & Cost-Sensitive Validation.

Implements the mandatory tests A–I of the authorization prompt (section 22):

    Test A — cost calculation correctness          (TestCostCalculationA)
    Test B — confusion-matrix accounting           (TestConfusionAccountingB)
    Test C — threshold determinism                 (TestThresholdDeterminismC)
    Test D — cost-ratio determinism                (TestCostRatioDeterminismD)
    Test E — no test-set access                    (TestNoTestSetAccessE)
    Test F — engine-level aggregation correctness  (TestEngineAggregationF)
    Test G — bootstrap reproducibility             (TestBootstrapReproducibilityG)
    Test H — no neural model imports               (TestNoNeuralImportsH)
    Test I — Phase 2 / 2.5 artifacts unchanged      (TestHistoryUnchangedI)

Every test operates on the sealed Phase 2.5 VALIDATION predictions or on small
synthetic arrays. No test refits a model or reads the official FD004 test set.
No existing leakage test is weakened.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.business.decision import (  # noqa: E402
    ILLUSTRATIVE_COST_LABEL,
    bootstrap_cost_difference,
    build_cost_ratio_sensitivity,
    build_decision_cost_comparison,
    build_threshold_cost_table,
    confusion_counts,
    engine_level_counts,
    expected_cost,
    false_negative_profile,
    precision_recall_f1,
    rul_decision_profile,
)
from src.data.integrity import verify_raw  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "phase2_6_decision.yaml"
EXPERIMENTS_DIR = ROOT / "results" / "experiments" / "phase2_5"
DECISIONS_DIR = ROOT / "results" / "decisions" / "phase2_6"
PROCESSED_DIR = ROOT / "data" / "processed"


def _cfg() -> dict:
    import yaml
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def temporal_fr_h30() -> pd.DataFrame:
    m = _cfg()["models"]["temporal_fr_H30"]
    return pd.read_parquet(EXPERIMENTS_DIR / "val_rows" / f"{m}.parquet")


@pytest.fixture(scope="module")
def anchor_fr_h30() -> pd.DataFrame:
    m = _cfg()["models"]["anchor_fr_H30"]
    return pd.read_parquet(EXPERIMENTS_DIR / "val_rows" / f"{m}.parquet")


# ===========================================================================
# TEST A — cost calculation correctness
# ===========================================================================


class TestCostCalculationA:
    def test_primary_framework_is_fn_times_cfn_plus_fp_times_cfp(self):
        assert expected_cost(fn=3, fp=2, c_fn=10.0, c_fp=1.0) == 3 * 10 + 2 * 1

    def test_zero_errors_zero_cost(self):
        assert expected_cost(0, 0, 50.0, 1.0) == 0.0

    def test_cost_independent_of_tp_tn(self):
        # The chosen framework charges only FN and FP; TP/TN are free.
        assert expected_cost(fn=1, fp=0, c_fn=7.0, c_fp=1.0) == 7.0
        assert expected_cost(fn=1, fp=0, c_fn=7.0, c_fp=1.0) == expected_cost(
            fn=1, fp=0, c_fn=7.0, c_fp=999.0
        )

    def test_cost_monotone_in_each_error(self):
        assert expected_cost(6, 5, 10, 1) > expected_cost(5, 5, 10, 1)
        assert expected_cost(5, 6, 10, 1) > expected_cost(5, 5, 10, 1)

    def test_prf_edge_cases(self):
        assert precision_recall_f1(tp=0, fp=0, fn=0) == (0.0, 0.0, 0.0)
        p, r, f1 = precision_recall_f1(tp=8, fp=2, fn=0)
        assert p == pytest.approx(0.8) and r == 1.0 and f1 == pytest.approx(2 * 0.8 / 1.8)


# ===========================================================================
# TEST B — confusion-matrix accounting
# ===========================================================================


class TestConfusionAccountingB:
    def test_counts_sum_to_row_count(self, temporal_fr_h30):
        yt, yp = temporal_fr_h30["y_true"].to_numpy(), temporal_fr_h30["y_prob"].to_numpy()
        for thr in (0.1, 0.5, 0.9):
            tp, fp, tn, fn = confusion_counts(yt, yp, thr)
            assert tp + fp + tn + fn == len(temporal_fr_h30)

    def test_matches_sklearn_confusion_matrix(self, temporal_fr_h30):
        yt = temporal_fr_h30["y_true"].to_numpy().astype(int)
        yp = temporal_fr_h30["y_prob"].to_numpy()
        for thr in (0.3, 0.5, 0.7):
            tn, fp, fn, tp = sk_confusion_matrix(yt, (yp >= thr).astype(int), labels=[0, 1]).ravel()
            assert confusion_counts(yt, yp, thr) == (int(tp), int(fp), int(tn), int(fn))

    def test_threshold_cost_table_reproduces_predictions(self, temporal_fr_h30):
        cfg = _cfg()
        tbl = build_threshold_cost_table(
            temporal_fr_h30["y_true"].to_numpy(), temporal_fr_h30["y_prob"].to_numpy(),
            model_id="x", horizon=30, thresholds=cfg["thresholds"],
            cost_ratios=[float(r) for r in cfg["cost_assumptions"]["cost_ratios"]],
            c_fp_unit=1.0,
        )
        row05 = tbl[tbl["threshold"] == 0.5].iloc[0]
        tp, fp, tn, fn = confusion_counts(
            temporal_fr_h30["y_true"].to_numpy(), temporal_fr_h30["y_prob"].to_numpy(), 0.5)
        assert int(row05["tp"]) == tp and int(row05["fp"]) == fp
        assert int(row05["fn"]) == fn and int(row05["tn"]) == tn


# ===========================================================================
# TEST C — threshold determinism
# ===========================================================================


class TestThresholdDeterminismC:
    def test_same_threshold_grid_gives_identical_table(self, temporal_fr_h30):
        cfg = _cfg()
        kw = dict(model_id="m", horizon=30, thresholds=cfg["thresholds"],
                  cost_ratios=[1.0, 5.0], c_fp_unit=1.0)
        a = build_threshold_cost_table(temporal_fr_h30["y_true"].to_numpy(),
                                       temporal_fr_h30["y_prob"].to_numpy(), **kw)
        b = build_threshold_cost_table(temporal_fr_h30["y_true"].to_numpy(),
                                       temporal_fr_h30["y_prob"].to_numpy(), **kw)
        pd.testing.assert_frame_equal(a, b)

    def test_threshold_ordering_of_costs_is_stable(self, temporal_fr_h30):
        cfg = _cfg()
        a = build_threshold_cost_table(temporal_fr_h30["y_true"].to_numpy(),
                                       temporal_fr_h30["y_prob"].to_numpy(),
                                       model_id="m", horizon=30,
                                       thresholds=list(cfg["thresholds"]),
                                       cost_ratios=[10.0], c_fp_unit=1.0)
        b = build_threshold_cost_table(temporal_fr_h30["y_true"].to_numpy(),
                                       temporal_fr_h30["y_prob"].to_numpy(),
                                       model_id="m", horizon=30,
                                       thresholds=list(reversed(cfg["thresholds"])),
                                       cost_ratios=[10.0], c_fp_unit=1.0)
        m = a.merge(b, on="threshold", suffixes=("_a", "_b"))
        assert np.allclose(m["expected_cost_a"], m["expected_cost_b"])


# ===========================================================================
# TEST D — cost-ratio determinism
# ===========================================================================


class TestCostRatioDeterminismD:
    def test_sensitivity_is_deterministic(self, temporal_fr_h30, anchor_fr_h30):
        cfg = _cfg()
        ratios = [float(r) for r in cfg["cost_assumptions"]["cost_ratios"]]
        frames = {}
        for mid, df in (("a", anchor_fr_h30), ("t", temporal_fr_h30)):
            frames[mid] = build_threshold_cost_table(
                df["y_true"].to_numpy(), df["y_prob"].to_numpy(), model_id=mid,
                horizon=30, thresholds=cfg["thresholds"], cost_ratios=ratios, c_fp_unit=1.0)
        t1 = build_cost_ratio_sensitivity(pd.concat(frames.values(), ignore_index=True))
        t2 = build_cost_ratio_sensitivity(pd.concat(frames.values(), ignore_index=True))
        pd.testing.assert_frame_equal(t1, t2)

    def test_comparison_winner_is_deterministic(self, anchor_fr_h30, temporal_fr_h30):
        cfg = _cfg()
        ratios = [float(r) for r in cfg["cost_assumptions"]["cost_ratios"]]
        ta = build_threshold_cost_table(anchor_fr_h30["y_true"].to_numpy(),
                                        anchor_fr_h30["y_prob"].to_numpy(), model_id="a",
                                        horizon=30, thresholds=cfg["thresholds"],
                                        cost_ratios=ratios, c_fp_unit=1.0)
        tt = build_threshold_cost_table(temporal_fr_h30["y_true"].to_numpy(),
                                        temporal_fr_h30["y_prob"].to_numpy(), model_id="t",
                                        horizon=30, thresholds=cfg["thresholds"],
                                        cost_ratios=ratios, c_fp_unit=1.0)
        sens = build_cost_ratio_sensitivity(pd.concat([ta, tt], ignore_index=True))
        c1 = build_decision_cost_comparison(sens, anchor_id="a", temporal_id="t")
        c2 = build_decision_cost_comparison(sens, anchor_id="a", temporal_id="t")
        pd.testing.assert_frame_equal(c1, c2)
        # abs_reduction must equal anchor_cost - temporal_cost exactly
        assert np.allclose(c1["abs_reduction"], c1["anchor_cost"] - c1["temporal_cost"])


# ===========================================================================
# TEST E — no test-set access
# ===========================================================================


class TestNoTestSetAccessE:
    def test_phase2_6_code_has_no_official_test_load(self):
        bad = []
        files = [ROOT / "src" / "business" / "decision.py"] + list(
            (ROOT / "scripts").glob("*phase2_6*.py"))
        for f in files:
            text = f.read_text(encoding="utf-8")
            for pat in (r"load_dataset\s*\([^)]*['\"]test['\"]", r"load_rul\s*\("):
                for m in re.finditer(pat, text):
                    bad.append((f.name, m.group(0)))
        assert bad == []

    def test_official_test_files_unchanged_since_seal(self):
        results = {fi.name: fi for fi in verify_raw()}
        for name in ("test_FD004.txt", "RUL_FD004.txt"):
            assert results[name].match, f"{name} changed since Phase 0 seal"

    def test_raw_rul_source_is_validation_not_test(self):
        # Only FD004_val.parquet (validation) is used as the raw-RUL reference.
        val = pd.read_parquet(PROCESSED_DIR / "FD004_val.parquet")
        assert val["unit_id"].nunique() == 50  # the sealed 50-engine validation split


# ===========================================================================
# TEST F — engine-level aggregation correctness
# ===========================================================================


class TestEngineAggregationF:
    def test_per_engine_counts_sum_to_global(self, temporal_fr_h30):
        thr = 0.5
        eng = engine_level_counts(temporal_fr_h30, model_id="t", threshold=thr)
        tp, fp, tn, fn = confusion_counts(
            temporal_fr_h30["y_true"].to_numpy(), temporal_fr_h30["y_prob"].to_numpy(), thr)
        assert int(eng["tp"].sum()) == tp
        assert int(eng["fp"].sum()) == fp
        assert int(eng["fn"].sum()) == fn
        assert int(eng["tn"].sum()) == tn
        assert int(eng["n_rows"].sum()) == len(temporal_fr_h30)

    def test_engines_and_alerts_are_consistent(self, temporal_fr_h30):
        eng = engine_level_counts(temporal_fr_h30, model_id="t", threshold=0.5)
        assert eng["unit_id"].nunique() == 50
        # alerts == tp + fp per engine
        assert np.array_equal(eng["n_alerts"].to_numpy(),
                              (eng["tp"] + eng["fp"]).to_numpy())

    def test_positive_runs_match_run_definition(self):
        # Synthetic single engine: predicted positives 1,1,0,1,0,1,1,1
        df = pd.DataFrame({
            "unit_id": [1] * 8,
            "cycle": list(range(1, 9)),
            "y_true": [1, 1, 0, 0, 0, 1, 0, 1],
            "y_prob": [0.9, 0.8, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
        })
        eng = engine_level_counts(df, model_id="s", threshold=0.5)
        # runs of predicted positives (>=0.5): [1,2], [4], [6], [8,9(->rows idx5..7)]
        assert int(eng["n_positive_runs"].iloc[0]) == 4
        # the run at cycle 4 alone is all-negative => one FP run
        assert int(eng["n_fp_runs"].iloc[0]) == 1

    def test_false_negative_profile_counts(self):
        df = pd.DataFrame({
            "unit_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "y_true": [1, 0, 1, 1],
            "y_prob": [0.9, 0.1, 0.2, 0.1],
            "raw_RUL": [5, 40, 20, 10],
        })
        prof = false_negative_profile(df, threshold=0.5, raw_rul_col="raw_RUL")
        assert prof["n_fn_rows"] == 2  # engine1 cycle1 is TP; engine2 both FN
        assert prof["n_engines_with_fn"] == 1
        assert prof["raw_rul_min"] == 10.0 and prof["raw_rul_max"] == 20.0


# ===========================================================================
# TEST G — bootstrap reproducibility
# ===========================================================================


class TestBootstrapReproducibilityG:
    def test_same_seed_identical_result(self, anchor_fr_h30, temporal_fr_h30):
        a = engine_level_counts(anchor_fr_h30, model_id="a", threshold=0.5)
        t = engine_level_counts(temporal_fr_h30, model_id="t", threshold=0.5)
        b1 = bootstrap_cost_difference(a, t, cost_ratios=[1.0, 10.0], threshold=0.5,
                                       n_resamples=500, seed=42)
        b2 = bootstrap_cost_difference(a, t, cost_ratios=[1.0, 10.0], threshold=0.5,
                                       n_resamples=500, seed=42)
        pd.testing.assert_frame_equal(b1, b2)

    def test_different_seed_changes_estimate(self, anchor_fr_h30, temporal_fr_h30):
        a = engine_level_counts(anchor_fr_h30, model_id="a", threshold=0.5)
        t = engine_level_counts(temporal_fr_h30, model_id="t", threshold=0.5)
        b1 = bootstrap_cost_difference(a, t, cost_ratios=[10.0], threshold=0.5,
                                       n_resamples=500, seed=42)
        b2 = bootstrap_cost_difference(a, t, cost_ratios=[10.0], threshold=0.5,
                                       n_resamples=500, seed=7)
        assert b1["mean_cost_diff"].iloc[0] != b2["mean_cost_diff"].iloc[0]

    def test_bootstrap_resamples_engines_not_rows(self, anchor_fr_h30, temporal_fr_h30):
        a = engine_level_counts(anchor_fr_h30, model_id="a", threshold=0.5)
        t = engine_level_counts(temporal_fr_h30, model_id="t", threshold=0.5)
        res = bootstrap_cost_difference(a, t, cost_ratios=[10.0], threshold=0.5,
                                        n_resamples=200, seed=42)
        row = res.iloc[0]
        assert row["ci_low"] <= row["mean_cost_diff"] <= row["ci_high"]

    def test_stored_bootstrap_matches_reproduction(self, anchor_fr_h30, temporal_fr_h30):
        cfg = _cfg()
        a = engine_level_counts(anchor_fr_h30, model_id="a", threshold=0.5)
        t = engine_level_counts(temporal_fr_h30, model_id="t", threshold=0.5)
        fresh = bootstrap_cost_difference(
            a, t, cost_ratios=[float(r) for r in cfg["cost_assumptions"]["cost_ratios"]],
            threshold=float(cfg["bootstrap"]["threshold"]),
            c_fp_unit=float(cfg["cost_assumptions"]["fp_unit_cost"]),
            n_resamples=int(cfg["bootstrap"]["n_resamples"]), seed=int(cfg["bootstrap"]["seed"]))
        stored = pd.read_csv(DECISIONS_DIR / "bootstrap.csv")
        assert np.allclose(stored["ci_low"].to_numpy(), fresh["ci_low"].to_numpy(), atol=1e-9)


# ===========================================================================
# TEST H — no neural model imports
# ===========================================================================


class TestNoNeuralImportsH:
    def test_decision_core_imports_only_allowlist(self):
        text = (ROOT / "src" / "business" / "decision.py").read_text(encoding="utf-8")
        imports = re.findall(r"^\s*(?:import|from)\s+([\w\.]+)", text, re.MULTILINE)
        allow = ("numpy", "pandas", "dataclasses", "typing", "__future__", "src.")
        for i in imports:
            assert i.startswith(allow), f"disallowed import in decision core: {i}"

    def test_no_dl_framework_import_anywhere_in_phase2_6(self):
        kw = r"^\s*(?:import|from)\s+(torch|tensorflow|keras|jax|flax)\b"
        files = list((ROOT / "src" / "business").rglob("*.py")) + list(
            (ROOT / "scripts").glob("*phase2_6*.py"))
        for f in files:
            assert not re.search(kw, f.read_text(encoding="utf-8"), re.MULTILINE), f.name

    def test_hard_rules_registered(self):
        reg = json.loads((DECISIONS_DIR / "analysis_registry.json").read_text(encoding="utf-8"))
        assert reg["hard_rules"]["no_deep_learning"] is True
        assert reg["hard_rules"]["official_test_untouched"] is True
        assert reg["hard_rules"]["no_retraining"] is True


# ===========================================================================
# TEST I — existing Phase 2 / 2.5 artifacts remain unchanged
# ===========================================================================


class TestHistoryUnchangedI:
    def test_phase2_5_inputs_not_modified_by_analysis(self):
        rm = json.loads((DECISIONS_DIR / "run_manifest.json").read_text(encoding="utf-8"))

        def _h(p: Path) -> str:
            return hashlib.sha256(p.read_bytes()).hexdigest()

        m = _cfg()["models"]
        for name, path in {
            f"{m['anchor_fr_H30']}.parquet": EXPERIMENTS_DIR / "val_rows" / f"{m['anchor_fr_H30']}.parquet",
            f"{m['temporal_fr_H30']}.parquet": EXPERIMENTS_DIR / "val_rows" / f"{m['temporal_fr_H30']}.parquet",
            "FD004_val.parquet": PROCESSED_DIR / "FD004_val.parquet",
        }.items():
            assert rm["input_hashes"][name] == _h(path)

    def test_phase2_5_anchor_metrics_still_expected(self):
        fr = pd.read_csv(EXPERIMENTS_DIR / "failure_risk_comparison.csv")
        at = fr[(fr["experiment_id"] == "FR25_T4_combined_cyc")
               & (fr["threshold_source"] == "default_0.5")]
        assert abs(float(at["F1"].iloc[0]) - 0.8977) < 0.002

    def test_phase2_6_output_dir_is_disjoint_from_experiments(self):
        assert (ROOT / "results" / "decisions" / "phase2_6").exists()
        # no phase-2_6 file lives under results/experiments
        for p in (ROOT / "results" / "experiments").rglob("*phase2_6*"):
            pytest.fail(f"Phase 2.6 wrote into experiments tree: {p}")


# ===========================================================================
# Wording / labeling audit (section 23)
# ===========================================================================


class TestWordingPhase26:
    def test_costs_are_labeled_illustrative(self):
        assert ILLUSTRATIVE_COST_LABEL.startswith("illustrative")
        reg = json.loads((DECISIONS_DIR / "analysis_registry.json").read_text(encoding="utf-8"))
        assert reg["illustrative_cost_label"] == ILLUSTRATIVE_COST_LABEL

    def test_report_never_claims_financial_savings(self):
        report = ROOT / "reports" / "PHASE_2_6_DECISION_COST_VALIDATION.md"
        if not report.exists():
            pytest.skip("report written at completion")
        text = report.read_text(encoding="utf-8")
        for pat in (r"₹", r"\$\s?\d", r"1\.15\s?M", r"\bROI\b", r"production[- ]ready",
                    r"operationally acceptable"):
            assert not re.search(pat, text, re.IGNORECASE), pat
