"""Phase 2 Final Gate Verification.

Run:  python scripts/verify_phase2_gate.py

Mirrors the layout of scripts/verify_phase1_gate.py but checks the Phase 2
exit criteria enumerated in the Phase 2 spec (sections 2, 3, 26, 27, 28, 31).

The script only READS artifacts written by scripts/run_phase2_experiments.py;
no model is refit here and the official test data is never loaded.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.contract import FORBIDDEN_FEATURE_COLS  # noqa: E402
from src.pipeline.registry import (  # noqa: E402
    load_experiments,
    load_phase1_split,
)

EXPERIMENTS_DIR = ROOT / "results" / "experiments" / "phase2"
FIGDIR = ROOT / "results" / "figures" / "phase_2"
MANIFEST_PATH = ROOT / "results" / "audits" / "phase1_dataset_manifest.json"

ALLOWED_MODELS = {
    "naive_mean", "naive_age", "linear_degradation",
    "ridge", "hist_gb_regressor",
    "majority_class", "logistic_regression", "hist_gb_classifier",
}
FORBIDDEN_MODEL_KEYWORDS = ("lstm", "gru", "transformer", "cnn", "rnn", "mlp", "deep", "neural")

PASS = "PASS"
FAIL = "FAIL"
issues: list[str] = []


def _hdr(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        issues.append(name)


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def section_registry() -> None:
    _hdr("SECTION 1: EXPERIMENT REGISTRY")
    defaults, specs = load_experiments()
    print(f"  Registered experiments: {len(specs)}")
    print(f"  Primary dataset: {defaults.get('primary_dataset')}")
    print(f"  Random seed: {defaults.get('random_seed')}")
    print(f"  Failure horizon (primary): {defaults.get('failure_horizon_primary')}")
    print(f"  Failure horizon (sensitivity): {defaults.get('failure_horizon_sensitivity')}")

    required = {
        "experiment_id", "task", "model", "feature_config",
        "normalization_mode", "include_cycle", "horizon", "hyperparameters",
    }
    ids = [s.experiment_id for s in specs]
    _check("experiment_ids unique", len(ids) == len(set(ids)))
    for s in specs:
        d = s.to_dict()
        missing = required - set(d)
        if missing:
            _check(f"{s.experiment_id} has required fields", False, str(missing))
            break
    else:
        _check("all experiments have required fields", True)

    bad = [s for s in specs if s.model not in ALLOWED_MODELS]
    _check("all models within Phase 2 allowlist", not bad, str([s.experiment_id for s in bad]))

    dl_hits = [
        s.experiment_id for s in specs
        if any(k in s.model.lower() for k in FORBIDDEN_MODEL_KEYWORDS)
    ]
    _check("no deep-learning model registered", not dl_hits, str(dl_hits))

    task_a = [s for s in specs if s.task == "A_RUL"]
    task_b = [s for s in specs if s.task == "B_FAILURE_RISK"]
    _check("Task A (RUL) experiments present", len(task_a) >= 4, f"n={len(task_a)}")
    _check("Task B (Failure Risk) experiments present", len(task_b) >= 3, f"n={len(task_b)}")

    # Registry JSON on disk agrees with YAML
    reg_path = EXPERIMENTS_DIR / "registry.json"
    _check("registry.json exists", reg_path.exists())
    if reg_path.exists():
        reg = _load_json(reg_path)
        disk_ids = [e["experiment_id"] for e in reg["experiments"]]
        _check("registry.json matches YAML", disk_ids == ids)
        _check("registry has hard_rules block", "hard_rules" in reg)
        _check("registry records phase1_split_reference", "phase1_split_reference" in reg)


def section_split_isolation() -> None:
    _hdr("SECTION 2: SPLIT ISOLATION & OFFICIAL-TEST FIREWALL")
    split = load_phase1_split()
    tr = set(split["train_engine_ids"])
    va = set(split["val_engine_ids"])
    _check("train/val engines disjoint", not (tr & va),
           f"overlap={sorted(tr & va) if (tr & va) else '[]'}")
    _check("train engines count matches Phase 1 manifest",
           len(tr) == _load_json(MANIFEST_PATH)["engine_split"]["n_train_engines"],
           f"n={len(tr)}")
    _check("val engines count matches Phase 1 manifest",
           len(va) == _load_json(MANIFEST_PATH)["engine_split"]["n_val_engines"],
           f"n={len(va)}")
    print(f"  Train IDs hash: {split['train_hash'][:16]}...")
    print(f"  Val IDs hash:   {split['val_hash'][:16]}...")
    print(f"  Manifest hash:  {split['manifest_hash'][:16]}...")

    # Every experiment JSON must reference the same manifest hash.
    manifest_hash = split["manifest_hash"]
    seen_hash_ok = True
    seen_hash_bad = []
    for sub in ("rul", "failure_risk"):
        for p in (EXPERIMENTS_DIR / sub).glob("*.json"):
            d = _load_json(p)
            ref = d.get("phase1_reference", {})
            if ref.get("manifest_hash") != manifest_hash:
                seen_hash_ok = False
                seen_hash_bad.append(p.name)
    _check("all experiments share Phase 1 manifest_hash", seen_hash_ok,
           str(seen_hash_bad[:5]))

    # Source-code scan: Phase 2 code paths must not load official test.
    # `src/data/loader.py` legitimately defines `load_rul` (Phase 0/1
    # infrastructure) — we only scan the Phase 2 owned directories.
    hits: list[tuple[str, str]] = []
    for sub in ("models", "evaluation", "pipeline", "features"):
        for f in (ROOT / "src" / sub).rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            for pat in (r"load_dataset\s*\([^)]*['\"]test['\"]", r"load_rul\s*\("):
                for m in re.finditer(pat, text):
                    hits.append((str(f), m.group(0)))
    for f in (ROOT / "scripts").glob("*phase2*.py"):
        text = f.read_text(encoding="utf-8")
        for pat in (r"load_dataset\s*\([^)]*['\"]test['\"]", r"load_rul\s*\("):
            for m in re.finditer(pat, text):
                hits.append((str(f), m.group(0)))
    _check("no Phase 2 code loads official test set", not hits, str(hits[:5]))


def section_feature_contract() -> None:
    _hdr("SECTION 3: FEATURE CONTRACT")
    all_ok = True
    offenders: list[tuple[str, str]] = []
    cycle_variants = {"with": [], "without": []}
    for sub in ("rul", "failure_risk"):
        for p in (EXPERIMENTS_DIR / sub).glob("*.json"):
            d = _load_json(p)
            feats = d.get("feature_names") or []
            include_cyc = bool(d.get("include_cycle"))
            # Forbidden columns must NEVER appear (cycle is handled separately
            # because Phase 2 has an explicit, gated ablation for it).
            hard_forbidden = set(FORBIDDEN_FEATURE_COLS) - {"cycle"}
            for f in feats:
                if f in hard_forbidden or str(f).startswith("failure_risk_target_"):
                    all_ok = False
                    offenders.append((p.name, f))
            # Cycle consistency: only enforced for experiments that build a
            # feature matrix. Naive / linear-degradation baselines consume
            # the dataframe directly and may legitimately set
            # include_cycle=True without listing 'cycle' in feature_names.
            has_cycle_in_feats = "cycle" in feats
            if feats and has_cycle_in_feats != include_cyc:
                all_ok = False
                offenders.append((p.name, f"cycle_flag_mismatch(include_cycle={include_cyc}, in_features={has_cycle_in_feats})"))
            # Track ablation coverage
            if include_cyc:
                cycle_variants["with"].append(d["experiment_id"])
            else:
                cycle_variants["without"].append(d["experiment_id"])
    _check("no forbidden columns leak into features", all_ok, str(offenders[:5]))
    _check("cycle ablation executed (with-cycle experiments exist)",
           len(cycle_variants["with"]) > 0,
           f"n={len(cycle_variants['with'])}")
    _check("cycle ablation executed (no-cycle experiments exist)",
           len(cycle_variants["without"]) > 0,
           f"n={len(cycle_variants['without'])}")
    print(f"  with-cycle:  {cycle_variants['with']}")
    print(f"  no-cycle:    {cycle_variants['without']}")


def section_metrics() -> None:
    _hdr("SECTION 4: METRICS COVERAGE")
    for sub, required_keys in (
        ("rul", {"mae", "rmse", "r2", "prognostic_score_rows", "prognostic_score_last_per_engine"}),
        ("failure_risk", set()),  # Task B has a different structure; checked below
    ):
        paths = sorted((EXPERIMENTS_DIR / sub).glob("*.json"))
        _check(f"{sub}/ experiment JSONs exist", len(paths) > 0, f"n={len(paths)}")
        for p in paths:
            d = _load_json(p)
            m = d.get("metrics") or {}
            if sub == "rul":
                missing = required_keys - set(m)
                if missing:
                    _check(f"{p.name} RUL metrics complete", False, str(missing))
            else:
                at_d = m.get("at_default_threshold") or {}
                need = {"accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"}
                missing = need - set(at_d)
                if missing:
                    _check(f"{p.name} Task B metrics complete", False, str(missing))


def section_prognostic_score_formula() -> None:
    _hdr("SECTION 5: SAXENA PROGNOSTIC SCORE — FORMULA DOCUMENTED")
    # Re-import to check docstring presence and evaluate against known values.
    from src.evaluation import metrics as metrics_mod
    from src.evaluation.metrics import prognostic_score_rows, _prognostic_penalty

    # Reference values from Saxena 2008 (verified: for E=0 penalty=0;
    # E=-13 (underprediction magnitude 13) -> exp(1)-1; E=+10 -> exp(1)-1).
    pen = _prognostic_penalty(np.array([-13.0, 10.0, 0.0]))
    _check("E=0 penalty = 0", abs(pen[2]) < 1e-12)
    _check("E=-13 penalty = exp(1)-1", abs(pen[0] - (np.exp(1) - 1)) < 1e-9)
    _check("E=+10 penalty = exp(1)-1", abs(pen[1] - (np.exp(1) - 1)) < 1e-9)
    doc_all = (metrics_mod.__doc__ or "") + (prognostic_score_rows.__doc__ or "")
    _check("documented as LOWER IS BETTER", "lower is better" in doc_all.lower())
    _check("s1=13 and s2=10 documented",
           "13" in doc_all and "10" in doc_all)


def section_class_imbalance() -> None:
    _hdr("SECTION 6: CLASS IMBALANCE COVERAGE")
    required_horizons = {14, 30, 50}
    seen = set()
    for p in (EXPERIMENTS_DIR / "failure_risk").glob("*.json"):
        d = _load_json(p)
        h = d.get("horizon")
        cb = d.get("class_balance")
        if h and cb:
            seen.add(h)
            for k in ("positive_rows", "negative_rows", "positive_rate",
                      "engines_with_any_positive"):
                if k not in cb:
                    _check(f"{p.name} class_balance has {k}", False)
    _check("class balance reported for H=14/30/50",
           required_horizons <= seen, str(sorted(seen)))


def section_threshold_analysis() -> None:
    _hdr("SECTION 7: FAILURE-RISK THRESHOLD ANALYSIS")
    non_majority = 0
    with_sweep = 0
    with_candidate = 0
    for p in (EXPERIMENTS_DIR / "failure_risk").glob("*.json"):
        d = _load_json(p)
        if d["model"] == "majority_class":
            # Degenerate case: threshold sweep is not meaningful (no probabilities).
            at_default = d["metrics"]["at_default_threshold"]
            _check(f"{p.name} PR-AUC/ROC-AUC flagged as undefined",
                   at_default["pr_auc"] is None and at_default["roc_auc"] is None)
            continue
        non_majority += 1
        if d.get("threshold_sweep") is not None:
            with_sweep += 1
        if d.get("candidate_threshold") is not None:
            with_candidate += 1
    _check("threshold sweep present for all probabilistic FR models",
           with_sweep == non_majority, f"{with_sweep}/{non_majority}")
    _check("candidate threshold recorded for all probabilistic FR models",
           with_candidate == non_majority, f"{with_candidate}/{non_majority}")


def section_comparison_csvs() -> None:
    _hdr("SECTION 8: COMPARISON TABLES")
    rul_path = EXPERIMENTS_DIR / "rul_comparison.csv"
    fr_path = EXPERIMENTS_DIR / "failure_risk_comparison.csv"
    _check("rul_comparison.csv exists", rul_path.exists())
    _check("failure_risk_comparison.csv exists", fr_path.exists())

    if rul_path.exists():
        df = pd.read_csv(rul_path)
        need = {"experiment_id", "model", "feature_config", "normalization",
                "cycle_feature", "MAE", "RMSE", "R2",
                "prognostic_score_rows", "prognostic_score_last_per_engine"}
        missing = need - set(df.columns)
        _check("RUL CSV columns complete", not missing, str(missing))
        # Check best-by-MAE ordering
        _check("RUL CSV sorted by MAE ascending",
               df["MAE"].is_monotonic_increasing,
               f"best={df.iloc[0]['experiment_id']} MAE={df.iloc[0]['MAE']:.3f}")

    if fr_path.exists():
        df = pd.read_csv(fr_path)
        need = {"experiment_id", "model", "horizon", "feature_config",
                "normalization", "threshold", "recall", "precision", "F1",
                "PR_AUC", "ROC_AUC"}
        missing = need - set(df.columns)
        _check("FR CSV columns complete", not missing, str(missing))


def section_figures() -> None:
    _hdr("SECTION 9: FIGURES")
    _check(f"{FIGDIR.relative_to(ROOT)} exists", FIGDIR.exists())
    if FIGDIR.exists():
        pns = sorted(FIGDIR.glob("fig*.png"))
        _check("10 Phase 2 figures written", len(pns) == 10,
               f"n={len(pns)}: {[p.name for p in pns]}")


def section_sealed_test_audit() -> None:
    _hdr("SECTION 10: SEALED OFFICIAL-TEST FILE AUDIT")
    # Positive evidence that Phase 2 did not touch the official test files:
    # their content hashes must still match the Phase 0 checksum registry.
    # (Reading a file to hash it does not load it into any experiment; the
    # Phase 2 pipeline itself never opens these files.)
    from src.data.integrity import verify_raw

    results = {fi.name: fi for fi in verify_raw()}
    for name in ("test_FD004.txt", "RUL_FD004.txt"):
        _check(f"{name} unchanged since Phase 0 seal",
               name in results and results[name].match)


def _print_summary() -> None:
    _hdr("SUMMARY")
    if issues:
        print(f"  {len(issues)} GATE CHECK(S) FAILED:")
        for i in issues:
            print(f"    - {i}")
        raise SystemExit(1)
    print("  ALL PHASE 2 GATE CHECKS PASSED")


def main() -> int:
    print("== Phase 2 Final Gate Verification ==")
    section_registry()
    section_split_isolation()
    section_feature_contract()
    section_metrics()
    section_prognostic_score_formula()
    section_class_imbalance()
    section_threshold_analysis()
    section_comparison_csvs()
    section_figures()
    section_sealed_test_audit()
    _print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
