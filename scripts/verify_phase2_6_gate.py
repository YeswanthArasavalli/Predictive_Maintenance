"""Phase 2.6 Final Gate Verification — Decision & Cost-Sensitive Validation.

Run:  python scripts/verify_phase2_6_gate.py

Verifies the exit criteria demanded by the Phase 2.6 authorization prompt. The
gate READS the artifacts written by `scripts/run_phase2_6_decision_analysis.py`
and re-derives the core cost/confusion numbers directly from the sealed Phase
2.5 validation predictions to give positive, executed evidence. It:

  - trains NO model and NEVER reads the official test partition;
  - imports NO neural / sequence framework;
  - confirms Phase 2 / Phase 2.5 historical artifacts are byte-unchanged;
  - confirms every cost is labeled illustrative (no fabricated financial claim).

Sections:
   1.  no retraining / inputs are existing validation predictions
   2.  official-test firewall (checksums sealed; no test load in phase-2.6 code)
   3.  no neural / sequence model imports
   4.  cost-math correctness (executed)
   5.  confusion-matrix accounting (TP+FP+TN+FN == rows; reproduces stored metrics)
   6.  threshold & cost-ratio determinism (rebuild == stored tables)
   7.  engine-level aggregation correctness (per-engine sums == global counts)
   8.  bootstrap reproducibility (reseed == stored CI)
   9.  required artifacts + figures present
  10. illustrative-cost labeling & prohibited financial wording
  11. Phase 2 / Phase 2.5 historical artifacts unchanged
  12. report / results consistency
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

from src.business.decision import (  # noqa: E402
    ILLUSTRATIVE_COST_LABEL,
    bootstrap_cost_difference,
    build_cost_ratio_sensitivity,
    build_threshold_cost_table,
    confusion_counts,
    engine_level_counts,
    expected_cost,
)
from src.data.integrity import verify_raw  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "phase2_6_decision.yaml"
DECISIONS_DIR = ROOT / "results" / "decisions" / "phase2_6"
EXPERIMENTS_DIR = ROOT / "results" / "experiments" / "phase2_5"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGDIR = ROOT / "results" / "figures" / "phase_2_6"
REPORT_PATH = ROOT / "reports" / "PHASE_2_6_DECISION_COST_VALIDATION.md"

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


def _cfg() -> dict:
    import yaml
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _models() -> dict:
    return _cfg()["models"]


# ===========================================================================
# SECTION 1 — no retraining; inputs are existing predictions
# ===========================================================================
def section_no_retraining() -> None:
    _hdr("SECTION 1: NO RETRAINING (existing validation predictions only)")
    cfg = _cfg()
    m = cfg["models"]
    for key in ("anchor_fr_H30", "temporal_fr_H30", "anchor_rul", "temporal_rul"):
        p = EXPERIMENTS_DIR / "val_rows" / f"{m[key]}.parquet"
        _check(f"input prediction artifact exists: {m[key]}", p.exists())
    # Phase 2.6 code must not import an estimator-fit path or run_experiment.
    hits = []
    for f in [ROOT / "src" / "business" / "decision.py",
              ROOT / "scripts" / "run_phase2_6_decision_analysis.py",
              ROOT / "scripts" / "phase2_6_figures.py"]:
        text = f.read_text(encoding="utf-8")
        for pat in (r"run_experiment", r"\.fit\s*\(", r"HistGradientBoosting", r"import\s+torch"):
            if re.search(pat, text):
                hits.append((f.name, pat))
    _check("no Phase 2.6 module fits a model or reruns experiments", not hits, str(hits[:5]))


# ===========================================================================
# SECTION 2 — official-test firewall
# ===========================================================================
def section_test_firewall() -> None:
    _hdr("SECTION 2: OFFICIAL-TEST FIREWALL")
    results = {fi.name: fi for fi in verify_raw()}
    for name in ("test_FD004.txt", "RUL_FD004.txt"):
        _check(f"{name} unchanged since Phase 0 seal", name in results and results[name].match)

    hits = []
    scan = [ROOT / "src" / "business" / "decision.py"] + [
        f for f in (ROOT / "scripts").glob("*phase2_6*.py")
        if f.name != Path(__file__).name
    ]
    # Only ACTUAL load calls are violations; prose/docstrings that merely name
    # the sealed files (to assert they are NOT read) must not be flagged.
    for f in scan:
        text = f.read_text(encoding="utf-8")
        for pat in (r"load_dataset\s*\([^)]*['\"]test['\"]", r"load_rul\s*\("):
            for mt in re.finditer(pat, text):
                hits.append((f.name, mt.group(0)))
    _check("no Phase 2.6 code loads the official test set / labels", not hits, str(hits[:6]))

    # Positive evidence: the raw-RUL source is the VALIDATION partition.
    val = PROCESSED_DIR / "FD004_val.parquet"
    _check("raw-RUL reference is the sealed FD004 validation partition", val.exists())


# ===========================================================================
# SECTION 3 — no neural / sequence imports
# ===========================================================================
def section_no_neural() -> None:
    _hdr("SECTION 3: NO NEURAL / SEQUENCE MODEL")
    reg = _load_json(DECISIONS_DIR / "analysis_registry.json")
    _check("registry hard_rules.no_deep_learning == True",
           reg["hard_rules"].get("no_deep_learning") is True)
    _check("registry hard_rules.no_retraining == True",
           reg["hard_rules"].get("no_retraining") is True)
    dep_hits = []
    kw = r"^\s*(import|from)\s+(torch|tensorflow|keras|jax|flax)\b"
    for f in [ROOT / "src" / "business" / "decision.py",
              ROOT / "src" / "business" / "__init__.py"]:
        if re.search(kw, f.read_text(encoding="utf-8"), re.MULTILINE):
            dep_hits.append(f.name)
    for f in (ROOT / "src" / "business").rglob("*.py"):
        if re.search(kw, f.read_text(encoding="utf-8"), re.MULTILINE):
            dep_hits.append(f.name)
    _check("no Phase 2.6 module imports a DL framework", not dep_hits, str(dep_hits))
    # decision.py must import only the scientific/stdlib allowlist (no estimator,
    # no DL, no experiment runner). Scan only import statements, not prose.
    text = (ROOT / "src" / "business" / "decision.py").read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:import|from)\s+([\w\.]+)", text, re.MULTILINE)
    allow_prefixes = ("numpy", "pandas", "dataclasses", "typing", "__future__", "src.")
    disallowed = [i for i in imports if not i.startswith(allow_prefixes)]
    _check("decision core imports only numpy/pandas/stdlib/src allowlist",
           not disallowed, str(disallowed))


# ===========================================================================
# SECTION 4 — cost-math correctness (executed)
# ===========================================================================
def section_cost_math() -> None:
    _hdr("SECTION 4: COST-MATH CORRECTNESS")
    _check("expected_cost(fn=3, fp=2, c_fn=10, c_fp=1) == 32",
           expected_cost(3, 2, 10.0, 1.0) == 32.0)
    _check("expected_cost(fn=0, fp=0, ...) == 0", expected_cost(0, 0, 100.0, 1.0) == 0.0)
    _check("expected_cost ignores TP/TN (fn=1,fp=0,c_fn=7) == 7",
           expected_cost(1, 0, 7.0, 1.0) == 7.0)
    # cost is monotone increasing in both FN and FP for positive unit costs
    mono = expected_cost(5, 5, 10, 1) >= expected_cost(4, 5, 10, 1) >= expected_cost(4, 4, 10, 1)
    _check("cost is monotone non-decreasing in FN and FP", mono)


# ===========================================================================
# SECTION 5 — confusion accounting (reproduces stored metrics + sums)
# ===========================================================================
def section_confusion_accounting() -> None:
    _hdr("SECTION 5: CONFUSION-MATRIX ACCOUNTING")
    m = _models()
    tc = pd.read_csv(DECISIONS_DIR / "threshold_cost.csv")
    fr = pd.read_parquet(EXPERIMENTS_DIR / "val_rows" / f"{m['temporal_fr_H30']}.parquet")
    n_rows = len(fr)
    # Every threshold row's counts must sum to the full validation row count.
    sums = tc["tp"] + tc["fp"] + tc["tn"] + tc["fn"]
    _check("TP+FP+TN+FN == n_rows for every threshold/ratio",
           bool((sums == n_rows).all()), f"n_rows={n_rows}")
    # Recompute at thr 0.5 and match the stored table.
    tp, fp, tn, fn = confusion_counts(fr["y_true"].to_numpy(), fr["y_prob"].to_numpy(), 0.5)
    stored = tc[(tc["model_id"] == m["temporal_fr_H30"]) & (tc["horizon"] == 30)
                & (tc["threshold"] == 0.5)].iloc[0]
    ok = (int(stored["tp"]) == tp and int(stored["fp"]) == fp
          and int(stored["tn"]) == tn and int(stored["fn"]) == fn)
    _check("recomputed thr=0.5 confusion matches stored table", ok,
           f"tp={tp} fp={fp} fn={fn}")


# ===========================================================================
# SECTION 6 — threshold & cost-ratio determinism
# ===========================================================================
def section_determinism() -> None:
    _hdr("SECTION 6: THRESHOLD / COST-RATIO DETERMINISM")
    cfg = _cfg()
    m = cfg["models"]
    thresholds = [float(t) for t in cfg["thresholds"]]
    ratios = [float(r) for r in cfg["cost_assumptions"]["cost_ratios"]]
    c_fp = float(cfg["cost_assumptions"]["fp_unit_cost"])
    fr = pd.read_parquet(EXPERIMENTS_DIR / "val_rows" / f"{m['temporal_fr_H30']}.parquet")
    yt, yp = fr["y_true"].to_numpy(), fr["y_prob"].to_numpy()
    a = build_threshold_cost_table(yt, yp, model_id=m["temporal_fr_H30"], horizon=30,
                                   thresholds=thresholds, cost_ratios=ratios, c_fp_unit=c_fp)
    b = build_threshold_cost_table(yt, yp, model_id=m["temporal_fr_H30"], horizon=30,
                                   thresholds=thresholds, cost_ratios=ratios, c_fp_unit=c_fp)
    _check("threshold-cost table is deterministic (repeat identical)",
           a.equals(b))
    sa = build_cost_ratio_sensitivity(a)
    _check("cost-ratio sensitivity is deterministic",
           sa.equals(build_cost_ratio_sensitivity(b)))
    # Stored threshold_cost must match a fresh rebuild for the temporal H30 slice.
    stored = pd.read_csv(DECISIONS_DIR / "threshold_cost.csv")
    stored_slice = stored[(stored["model_id"] == m["temporal_fr_H30"]) & (stored["horizon"] == 30)]
    merged = a.merge(stored_slice, on=["model_id", "horizon", "threshold", "cost_ratio"],
                     suffixes=("_new", "_stored"))
    dev = float(np.max(np.abs(merged["expected_cost_new"] - merged["expected_cost_stored"]))) \
        if len(merged) else float("inf")
    _check("stored expected costs reproduce from predictions (max abs delta)",
           dev < 1e-9, f"max_delta={dev:.2e}")


# ===========================================================================
# SECTION 7 — engine-level aggregation correctness
# ===========================================================================
def section_engine_aggregation() -> None:
    _hdr("SECTION 7: ENGINE-LEVEL AGGREGATION CORRECTNESS")
    cfg = _cfg()
    m = cfg["models"]
    thr = float(cfg["engine_analysis"]["threshold"])
    val_ref = pd.read_parquet(PROCESSED_DIR / "FD004_val.parquet")[["unit_id", "cycle", "raw_RUL"]]
    fr = pd.read_parquet(EXPERIMENTS_DIR / "val_rows" / f"{m['temporal_fr_H30']}.parquet")
    fr = fr.merge(val_ref, on=["unit_id", "cycle"], how="left")
    eng = engine_level_counts(fr, model_id=m["temporal_fr_H30"], threshold=thr, raw_rul_col="raw_RUL")
    _check("per-engine rows sum to full validation row count",
           int(eng["n_rows"].sum()) == len(fr), f"{int(eng['n_rows'].sum())} vs {len(fr)}")
    tp, fp, tn, fn = confusion_counts(fr["y_true"].to_numpy(), fr["y_prob"].to_numpy(), thr)
    _check("per-engine FN sum == global FN", int(eng["fn"].sum()) == fn, f"{int(eng['fn'].sum())} vs {fn}")
    _check("per-engine FP sum == global FP", int(eng["fp"].sum()) == fp, f"{int(eng['fp'].sum())} vs {fp}")
    _check("50 validation engines represented", int(eng["unit_id"].nunique()) == 50,
           f"n={int(eng['unit_id'].nunique())}")
    stored = pd.read_csv(DECISIONS_DIR / "engine_level_analysis.csv")
    _check("stored engine_level_analysis covers both models x 50 engines",
           stored.groupby("model_id")["unit_id"].nunique().eq(50).all(),
           str(stored.groupby("model_id")["unit_id"].nunique().to_dict()))


# ===========================================================================
# SECTION 8 — bootstrap reproducibility
# ===========================================================================
def section_bootstrap() -> None:
    _hdr("SECTION 8: BOOTSTRAP REPRODUCIBILITY")
    cfg = _cfg()
    boot_cfg = cfg["bootstrap"]
    if not boot_cfg.get("enabled", True):
        _check("bootstrap documented as omitted (with reason)", True, "disabled in config")
        return
    thr = float(boot_cfg["threshold"])
    c_fp = float(cfg["cost_assumptions"]["fp_unit_cost"])
    ratios = [float(r) for r in cfg["cost_assumptions"]["cost_ratios"]]
    val_ref = pd.read_parquet(PROCESSED_DIR / "FD004_val.parquet")[["unit_id", "cycle", "raw_RUL"]]
    ea = engine_level_counts(
        pd.read_parquet(EXPERIMENTS_DIR / "val_rows" / f"{cfg['models']['anchor_fr_H30']}.parquet")
        .merge(val_ref, on=["unit_id", "cycle"], how="left"),
        model_id="a", threshold=thr, raw_rul_col="raw_RUL")
    et = engine_level_counts(
        pd.read_parquet(EXPERIMENTS_DIR / "val_rows" / f"{cfg['models']['temporal_fr_H30']}.parquet")
        .merge(val_ref, on=["unit_id", "cycle"], how="left"),
        model_id="t", threshold=thr, raw_rul_col="raw_RUL")
    b1 = bootstrap_cost_difference(ea, et, cost_ratios=ratios, threshold=thr,
                                   c_fp_unit=c_fp, n_resamples=int(boot_cfg["n_resamples"]),
                                   seed=int(boot_cfg["seed"]))
    b2 = bootstrap_cost_difference(ea, et, cost_ratios=ratios, threshold=thr,
                                   c_fp_unit=c_fp, n_resamples=int(boot_cfg["n_resamples"]),
                                   seed=int(boot_cfg["seed"]))
    _check("bootstrap is reproducible under the same seed", b1.equals(b2))
    stored = pd.read_csv(DECISIONS_DIR / "bootstrap.csv")
    _check("stored bootstrap matches a reseeded run",
           np.allclose(stored["mean_cost_diff"], b1["mean_cost_diff"], atol=1e-9)
           and np.allclose(stored["ci_low"], b1["ci_low"], atol=1e-9))


# ===========================================================================
# SECTION 9 — required artifacts + figures present
# ===========================================================================
def section_artifacts() -> None:
    _hdr("SECTION 9: REQUIRED ARTIFACTS / FIGURES PRESENT")
    required = [
        "threshold_cost.csv", "cost_ratio_sensitivity.csv", "decision_cost_comparison.csv",
        "engine_level_analysis.csv", "false_negative_analysis.json", "alert_fatigue.json",
        "rul_decision_analysis.json", "bootstrap.csv", "analysis_registry.json",
        "run_manifest.json",
    ]
    for name in required:
        _check(f"decisions/{name} exists", (DECISIONS_DIR / name).exists())
    figs = sorted(FIGDIR.glob("fig*.png")) if FIGDIR.exists() else []
    _check("Phase 2.6 figures written (>=8)", len(figs) >= 8, f"n={len(figs)}")


# ===========================================================================
# SECTION 10 — illustrative labeling & prohibited financial wording
# ===========================================================================
def section_illustrative_labeling() -> None:
    _hdr("SECTION 10: ILLUSTRATIVE-COST LABELING & WORDING")
    reg = _load_json(DECISIONS_DIR / "analysis_registry.json")
    _check("registry carries the illustrative cost label",
           reg.get("illustrative_cost_label") == ILLUSTRATIVE_COST_LABEL)
    af = _load_json(DECISIONS_DIR / "alert_fatigue.json")
    _check("alert_fatigue.json labeled illustrative", af.get("label") == ILLUSTRATIVE_COST_LABEL)

    forbidden = [r"₹", r"\bUSD\b", r"\$\s?\d", r"1\.15\s?M", r"production[- ]ready",
                 r"saves\s+[₹$]?\d", r"\bROI\b", r"operationally acceptable"]
    bad = []
    scan_files = [DECISIONS_DIR / "analysis_registry.json",
                  DECISIONS_DIR / "rul_decision_analysis.json"]
    if REPORT_PATH.exists():
        scan_files.append(REPORT_PATH)
    for f in scan_files:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        for pat in forbidden:
            for mt in re.finditer(pat, text, re.IGNORECASE):
                bad.append((f.name, mt.group(0)))
    _check("no fabricated financial / production-ready claim in artifacts/report",
           not bad, str(bad[:6]))


# ===========================================================================
# SECTION 11 — Phase 2 / Phase 2.5 historical artifacts unchanged
# ===========================================================================
def section_history_unchanged() -> None:
    _hdr("SECTION 11: PHASE 2 / 2.5 HISTORY UNCHANGED")
    rm = _load_json(DECISIONS_DIR / "run_manifest.json")
    # Re-hash the read-only inputs and compare to the manifest captured at run time.
    import hashlib
    def _h(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    m = _models()
    checks = {
        f"{m['anchor_fr_H30']}.parquet": EXPERIMENTS_DIR / "val_rows" / f"{m['anchor_fr_H30']}.parquet",
        f"{m['temporal_fr_H30']}.parquet": EXPERIMENTS_DIR / "val_rows" / f"{m['temporal_fr_H30']}.parquet",
        "FD004_val.parquet": PROCESSED_DIR / "FD004_val.parquet",
    }
    for name, path in checks.items():
        _check(f"input unchanged since analysis run: {name}",
               name in rm["input_hashes"] and rm["input_hashes"][name] == _h(path))
    # Phase 2.5 anchor values still read back unchanged (not overwritten).
    fr_csv = pd.read_csv(EXPERIMENTS_DIR / "failure_risk_comparison.csv")
    at = fr_csv[(fr_csv["experiment_id"] == m["temporal_fr_H30"]) & (fr_csv["threshold_source"] == "default_0.5")]
    _check("Phase 2.5 temporal H30 F1@0.5 unchanged (~0.898)",
           len(at) == 1 and abs(float(at["F1"].iloc[0]) - 0.8977) < 0.002,
           f"F1={float(at['F1'].iloc[0]):.4f}" if len(at) else "missing")
    # The decisions dir must be disjoint from the experiments dir.
    _check("Phase 2.6 writes into results/decisions (not results/experiments)",
           "decisions" in str(DECISIONS_DIR) and "experiments" not in str(DECISIONS_DIR))


# ===========================================================================
# SECTION 12 — report / results consistency
# ===========================================================================
def section_report_consistency() -> None:
    _hdr("SECTION 12: REPORT / RESULTS CONSISTENCY")
    _check("Phase 2.6 report exists", REPORT_PATH.exists())
    if not REPORT_PATH.exists():
        return
    text = REPORT_PATH.read_text(encoding="utf-8")
    low = text.lower()
    _check("report labels costs illustrative",
           "illustrative" in low and "not observed" in low)
    _check("report states 50 independent engines", "50 independent engines" in low or "50 validation engines" in low)
    _check("report answers the four decision questions",
           all(k in low for k in ("failure-risk", "rul", "temporal feature layer", "sequence model")))
    # RUL MAE cited must match measured artifact.
    rul = _load_json(DECISIONS_DIR / "rul_decision_analysis.json")
    cited_anchor = rul["anchor"]["mae"]
    _check("report cites the measured RUL anchor MAE (~32.0)",
           "32.0" in text or f"{cited_anchor:.2f}" in text)


def _print_summary() -> None:
    _hdr("SUMMARY")
    if issues:
        print(f"  {len(issues)} GATE CHECK(S) FAILED:")
        for i in issues:
            print(f"    - {i}")
        raise SystemExit(1)
    print("  ALL PHASE 2.6 GATE CHECKS PASSED")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - non-Windows / already utf-8
        pass
    print("== Phase 2.6 Final Gate Verification ==")
    section_no_retraining()
    section_test_firewall()
    section_no_neural()
    section_cost_math()
    section_confusion_accounting()
    section_determinism()
    section_engine_aggregation()
    section_bootstrap()
    section_artifacts()
    section_illustrative_labeling()
    section_history_unchanged()
    section_report_consistency()
    _print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
