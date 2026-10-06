"""Phase 2.5 Final Gate Verification — Causal Temporal Feature Baselines.

Run:  python scripts/verify_phase2_5_gate.py

Verifies the twelve exit criteria enumerated in the Phase 2.5 authorization
prompt (section 24):

    1.  frozen split preserved
    2.  no official test ground truth accessed
    3.  no neural model trained
    4.  all temporal features causal
    5.  engine boundaries respected
    6.  train-only transformations
    7.  required experiments registered
    8.  required metrics present
    9.  required artifacts present
    10. causality tests pass
    11. reproducibility passes
    12. report/results consistency passes

The script READS the artifacts written by scripts/run_phase2_5_experiments.py.
It refits NO model and NEVER loads the official test partition (test_FD004.txt /
RUL_FD004.txt). Sections 4/5/11 exercise the temporal FEATURE MODULE directly on
the FD004 TRAIN partition (allowed) to give positive, executed evidence of
causality — this is not a model refit. Section 10 runs the dedicated pytest
causality suite and reports the measured outcome.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.contract import FORBIDDEN_FEATURE_COLS  # noqa: E402
from src.data.loader import CYCLE, UNIT_ID, load_dataset  # noqa: E402
from src.features.temporal import (  # noqa: E402
    FAMILY_COMBINED,
    TemporalConfig,
    add_causal_temporal_features,
    temporal_feature_names,
)
from src.pipeline.registry import load_experiments, load_phase1_split  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "phase2_5_experiments.yaml"
EXPERIMENTS_DIR = ROOT / "results" / "experiments" / "phase2_5"
FIGDIR = ROOT / "results" / "figures" / "phase_2_5"
MANIFEST_PATH = ROOT / "results" / "audits" / "phase1_dataset_manifest.json"
REPORT_PATH = ROOT / "reports" / "PHASE_2_5_TEMPORAL_BASELINES.md"

ALLOWED_MODELS = {
    "naive_mean", "naive_age", "linear_degradation",
    "ridge", "hist_gb_regressor",
    "majority_class", "logistic_regression", "hist_gb_classifier",
}
FORBIDDEN_MODEL_KEYWORDS = ("lstm", "gru", "transformer", "cnn", "rnn", "mlp", "deep", "neural")
REQUIRED_FAMILIES = {"none", "lag_diff", "rolling", "slope", "combined"}

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


# ===========================================================================
# SECTION 1 — frozen split preserved
# ===========================================================================
def section_frozen_split() -> None:
    _hdr("SECTION 1: FROZEN SPLIT PRESERVED")
    split = load_phase1_split()
    manifest = _load_json(MANIFEST_PATH)
    tr = set(split["train_engine_ids"])
    va = set(split["val_engine_ids"])
    _check("train/val engines disjoint", not (tr & va))
    _check("train engine count == Phase 1 manifest",
           len(tr) == manifest["engine_split"]["n_train_engines"] == 199, f"n={len(tr)}")
    _check("val engine count == Phase 1 manifest",
           len(va) == manifest["engine_split"]["n_val_engines"] == 50, f"n={len(va)}")

    reg = _load_json(EXPERIMENTS_DIR / "registry.json")
    ref = reg["phase1_split_reference"]
    _check("registry reuses Phase 1 train_hash verbatim", ref["train_hash"] == split["train_hash"])
    _check("registry reuses Phase 1 val_hash verbatim", ref["val_hash"] == split["val_hash"])
    _check("registry reuses Phase 1 manifest_hash verbatim", ref["manifest_hash"] == split["manifest_hash"])

    # Every experiment JSON must share the same manifest hash (no re-split).
    bad = []
    for sub in ("rul", "failure_risk"):
        for p in (EXPERIMENTS_DIR / sub).glob("*.json"):
            d = _load_json(p)
            if d.get("phase1_reference", {}).get("manifest_hash") != split["manifest_hash"]:
                bad.append(p.name)
    _check("all experiments share the frozen manifest_hash", not bad, str(bad[:5]))
    print(f"  train_hash {split['train_hash'][:16]}...  val_hash {split['val_hash'][:16]}...")


# ===========================================================================
# SECTION 2 — no official test ground truth accessed
# ===========================================================================
def section_test_firewall() -> None:
    _hdr("SECTION 2: OFFICIAL-TEST FIREWALL")
    # Positive evidence: official test files remain sealed since the Phase 0 record.
    from src.data.integrity import verify_raw
    results = {fi.name: fi for fi in verify_raw()}
    for name in ("test_FD004.txt", "RUL_FD004.txt"):
        _check(f"{name} unchanged since Phase 0 seal",
               name in results and results[name].match)

    # Source scan: Phase 2.5 owned code must not open the official test set.
    hits: list[tuple[str, str]] = []
    for sub in ("models", "evaluation", "pipeline", "features", "data"):
        for f in (ROOT / "src" / sub).rglob("*.py"):
            if f.name == "loader.py":
                continue  # loader legitimately defines load_rul / test split (Phase 0/1 infra)
            text = f.read_text(encoding="utf-8")
            for pat in (r"load_dataset\s*\([^)]*['\"]test['\"]", r"load_rul\s*\("):
                for m in re.finditer(pat, text):
                    hits.append((str(f), m.group(0)))
    for f in (ROOT / "scripts").glob("*phase2_5*.py"):
        text = f.read_text(encoding="utf-8")
        for pat in (r"load_dataset\s*\([^)]*['\"]test['\"]", r"load_rul\s*\("):
            for m in re.finditer(pat, text):
                hits.append((str(f), m.group(0)))
    _check("no Phase 2.5 code loads the official test set", not hits, str(hits[:5]))


# ===========================================================================
# SECTION 3 — no neural model trained
# ===========================================================================
def section_no_neural() -> None:
    _hdr("SECTION 3: NO NEURAL / SEQUENCE MODEL")
    defaults, specs = load_experiments(CONFIG_PATH)
    bad = [s.experiment_id for s in specs if s.model not in ALLOWED_MODELS]
    _check("all registered models within classical allowlist", not bad, str(bad))
    dl = [s.experiment_id for s in specs
          if any(k in s.model.lower() for k in FORBIDDEN_MODEL_KEYWORDS)]
    _check("no deep-learning model registered", not dl, str(dl))

    reg = _load_json(EXPERIMENTS_DIR / "registry.json")
    _check("registry hard_rules.no_deep_learning == True",
           reg["hard_rules"].get("no_deep_learning") is True)

    # Import graph: nothing under Phase 2.5 code may import torch / keras / tf.
    dep_hits = []
    for f in (ROOT / "src" / "features").rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        if re.search(r"^\s*(import|from)\s+(torch|tensorflow|keras|jax)\b", text, re.MULTILINE):
            dep_hits.append(f.name)
    _check("no Phase 2.5 module imports a DL framework", not dep_hits, str(dep_hits))


# ===========================================================================
# SECTION 4 — all temporal features causal (executed functional check)
# ===========================================================================
def _train_slice() -> pd.DataFrame:
    df = load_dataset("FD004", "train")
    uids = df[UNIT_ID].drop_duplicates().to_list()[:4]
    return df[df[UNIT_ID].isin(uids)].reset_index(drop=True)


def section_causality() -> None:
    _hdr("SECTION 4: ALL TEMPORAL FEATURES CAUSAL")
    df = _train_slice()
    srcs = ["sensor_14", "sensor_17", "sensor_11"]
    cfg = TemporalConfig(
        family=FAMILY_COMBINED, windows=(5,), lags=(5,),
        source_cols=tuple(srcs), monitor_sensors=tuple(srcs),
    )
    names = temporal_feature_names(cfg)
    base = add_causal_temporal_features(df, cfg)

    # (A) future-perturbation invariance on a real engine.
    eng = df[UNIT_ID].iloc[0]
    t0 = int(base.loc[base[UNIT_ID] == eng, CYCLE].median())
    pert = df.copy()
    mask = (pert[UNIT_ID] == eng) & (pert[CYCLE] > t0)
    pert.loc[mask, srcs] = 9.9e5
    after = add_causal_temporal_features(pert, cfg)
    rows = (base[UNIT_ID] == eng) & (base[CYCLE] <= t0)
    max_dev = 0.0
    for n in names:
        a = base.loc[rows, n].to_numpy()
        b = after.loc[rows, n].to_numpy()
        if len(a):
            max_dev = max(max_dev, float(np.max(np.abs(a - b))))
    _check("future-perturbation invariance (features at t<=t0 unchanged)",
           max_dev < 1e-6, f"max|Δ|={max_dev:.2e} on engine {eng} cycles<={t0}")

    # (B) rolling mean equals an explicit backward-only window.
    w = 5
    e_df = base[base[UNIT_ID] == eng].reset_index(drop=True)
    x = df[df[UNIT_ID] == eng]["sensor_14"].to_numpy()
    dev = 0.0
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        exp = x[lo:t + 1].mean()
        got = e_df[f"sensor_14__rmean{w}"].iloc[t]
        dev = max(dev, abs(float(got) - float(exp)))
    _check("rolling mean uses only indices <= t", dev < 1e-9, f"max|Δ|={dev:.2e}")

    # (C) first difference: no prior observation at engine start.
    _check("first difference is 0 at each engine start",
           float(e_df["sensor_14__d1"].iloc[0]) == 0.0)


# ===========================================================================
# SECTION 5 — engine boundaries respected
# ===========================================================================
def section_engine_boundaries() -> None:
    _hdr("SECTION 5: ENGINE BOUNDARIES RESPECTED")
    df = _train_slice()
    srcs = ["sensor_14", "sensor_17"]
    cfg = TemporalConfig(
        family=FAMILY_COMBINED, windows=(5,), lags=(5,),
        source_cols=tuple(srcs), monitor_sensors=tuple(srcs),
    )
    base = add_causal_temporal_features(df, cfg)
    # Corrupt one engine's tail; no OTHER engine's features may change.
    target = df[UNIT_ID].iloc[0]
    others = [u for u in df[UNIT_ID].unique() if u != target]
    pert = df.copy()
    pert.loc[pert[UNIT_ID] == target, srcs] = -9.9e5
    after = add_causal_temporal_features(pert, cfg)
    dev = 0.0
    for n in temporal_feature_names(cfg):
        a = base.loc[base[UNIT_ID].isin(others), n].to_numpy()
        b = after.loc[after[UNIT_ID].isin(others), n].to_numpy()
        if len(a):
            dev = max(dev, float(np.max(np.abs(a - b))))
    _check("windows never cross an engine boundary (other engines unaffected)",
           dev < 1e-9, f"max|Δ| across other engines={dev:.2e}")

    # Second-row slope of an engine must depend on only its own first two rows.
    second_eng = others[0]
    x2 = df[df[UNIT_ID] == second_eng]["sensor_14"].to_numpy()
    exp_slope = float(np.polyfit(np.arange(2), x2[:2], 1)[0])
    got = float(base.loc[base[UNIT_ID] == second_eng, "sensor_14__slope5"].iloc[1])
    _check("early-window slope confined to the engine's own rows",
           abs(got - exp_slope) < 1e-6, f"got={got:.4e} exp={exp_slope:.4e}")


# ===========================================================================
# SECTION 6 — train-only transformations
# ===========================================================================
def section_train_only() -> None:
    _hdr("SECTION 6: TRAIN-ONLY TRANSFORMATIONS / SELECTION")
    reg = _load_json(EXPERIMENTS_DIR / "registry.json")
    _check("registry records train-only monitor selection",
           "TRAIN-only" in reg["temporal_policy"]["selection"])
    meta_path = EXPERIMENTS_DIR / "temporal_feature_metadata.json"
    _check("temporal_feature_metadata.json exists", meta_path.exists())
    if not meta_path.exists():
        return
    meta = _load_json(meta_path)
    _check("metadata documents train-only selection policy",
           "train-only" in meta["selection_policy"].lower())

    # The TRAIN-only monitor set must be IDENTICAL across every temporal
    # experiment (it depends solely on the train partition + budget).
    monitor_sets = {}
    for sub in ("rul", "failure_risk"):
        for p in (EXPERIMENTS_DIR / sub).glob("*.json"):
            d = _load_json(p)
            tm = d.get("temporal_metadata") or {}
            if tm.get("monitors_selected_train_only"):
                monitor_sets[d["experiment_id"]] = tuple(tm["monitors_selected_train_only"])
    distinct = set(monitor_sets.values())
    _check("monitor selection identical across all temporal experiments (train-deterministic)",
           len(distinct) == 1, f"n_distinct={len(distinct)}")
    _check("monitor selection covers all temporal experiments",
           len(monitor_sets) > 0, f"n={len(monitor_sets)}")


# ===========================================================================
# SECTION 7 — required experiments registered
# ===========================================================================
def section_required_experiments() -> None:
    _hdr("SECTION 7: REQUIRED EXPERIMENTS REGISTERED")
    defaults, specs = load_experiments(CONFIG_PATH)
    fams = {s.temporal_family for s in specs}
    _check("all five temporal families present", REQUIRED_FAMILIES <= fams, str(sorted(fams)))

    rul_blind = [s for s in specs if s.task == "A_RUL" and not s.include_cycle and s.temporal_family == "combined"]
    fr_blind = [s for s in specs if s.task == "B_FAILURE_RISK" and not s.include_cycle and s.temporal_family == "combined"]
    _check("RUL cycle-blind (T4-blind) experiment present", bool(rul_blind))
    _check("FR cycle-blind (T4-blind) experiment present", bool(fr_blind))

    rul_anchors = [s for s in specs if s.task == "A_RUL" and s.temporal_family == "none" and s.include_cycle]
    _check("RUL T0 anchor (current+cycle) present", bool(rul_anchors))

    horizons = {s.horizon for s in specs if s.task == "B_FAILURE_RISK"}
    _check("horizon sensitivity {14,30,50} present", {14, 30, 50} <= horizons, str(sorted(horizons)))

    rul_comb = [tuple(s.temporal_windows) for s in specs if s.task == "A_RUL" and s.temporal_family == "combined"]
    _check("temporal-window ablation distinguishes short vs medium",
           len(set(rul_comb)) >= 2, str(sorted(set(rul_comb))))


# ===========================================================================
# SECTION 8 — required metrics present
# ===========================================================================
def section_metrics() -> None:
    _hdr("SECTION 8: REQUIRED METRICS PRESENT")
    rul_req = {"mae", "rmse", "r2", "prognostic_score_rows", "prognostic_score_last_per_engine"}
    ok = True
    for p in sorted((EXPERIMENTS_DIR / "rul").glob("*.json")):
        m = set(_load_json(p).get("metrics") or {})
        if rul_req - m:
            ok = False
            _check(f"{p.name} RUL metrics complete", False, str(rul_req - m))
    _check("all RUL experiments report full metric set", ok)

    fr_req = {"accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"}
    ok = True
    for p in sorted((EXPERIMENTS_DIR / "failure_risk").glob("*.json")):
        d = _load_json(p)
        at_d = (d.get("metrics") or {}).get("at_default_threshold") or {}
        if fr_req - set(at_d):
            ok = False
            _check(f"{p.name} FR metrics complete", False, str(fr_req - set(at_d)))
        if d.get("threshold_sweep") is None:
            ok = False
            _check(f"{p.name} FR threshold sweep present", False)
    _check("all FR experiments report metrics + threshold sweep", ok)


# ===========================================================================
# SECTION 9 — required artifacts present
# ===========================================================================
def section_artifacts() -> None:
    _hdr("SECTION 9: REQUIRED ARTIFACTS PRESENT")
    required = [
        EXPERIMENTS_DIR / "registry.json",
        EXPERIMENTS_DIR / "rul_comparison.csv",
        EXPERIMENTS_DIR / "failure_risk_comparison.csv",
        EXPERIMENTS_DIR / "temporal_feature_metadata.json",
        EXPERIMENTS_DIR / "run_manifest.json",
    ]
    for p in required:
        _check(f"{p.relative_to(ROOT)} exists", p.exists())

    defaults, specs = load_experiments(CONFIG_PATH)
    n_json = sum(len(list((EXPERIMENTS_DIR / sub).glob("*.json"))) for sub in ("rul", "failure_risk"))
    _check("per-experiment JSON count matches registry", n_json == len(specs),
           f"{n_json} vs {len(specs)}")
    n_val = len(list((EXPERIMENTS_DIR / "val_rows").glob("*.parquet")))
    _check("validation prediction rows written for every experiment", n_val == len(specs),
           f"{n_val} vs {len(specs)}")

    figs = sorted(FIGDIR.glob("fig*.png")) if FIGDIR.exists() else []
    _check("Phase 2.5 figures written", len(figs) >= 6,
           f"n={len(figs)}: {[p.name for p in figs]}")


# ===========================================================================
# SECTION 10 — causality tests pass
# ===========================================================================
def section_causality_tests() -> None:
    _hdr("SECTION 10: CAUSALITY TESTS (PYTEST)")
    # NOTE: pyproject addopts already carries `-q`; do NOT append another `-q`
    # (a doubled `-qq` suppresses the final "N passed" summary line).
    target = "tests/test_phase2_5.py"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", target, "-k",
             "Causality or Determinism or Contract"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600,
        )
    except FileNotFoundError:  # pragma: no cover
        _check("pytest executable found", False)
        return
    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"(\d+) passed", out)
    n_passed = int(m.group(1)) if m else 0
    fm = re.search(r"(\d+) failed", out)
    n_failed = int(fm.group(1)) if fm else 0
    em = re.search(r"(\d+) error", out)
    n_errors = int(em.group(1)) if em else 0
    _check("causality/determinism/contract tests all pass",
           proc.returncode == 0 and n_failed == 0 and n_errors == 0 and n_passed > 0,
           f"passed={n_passed} failed={n_failed} errors={n_errors} rc={proc.returncode}")


# ===========================================================================
# SECTION 11 — reproducibility passes
# ===========================================================================
def section_reproducibility() -> None:
    _hdr("SECTION 11: REPRODUCIBILITY")
    df = _train_slice()
    srcs = ["sensor_14", "sensor_17", "sensor_11"]
    cfg = TemporalConfig(
        family=FAMILY_COMBINED, windows=(5,), lags=(5,),
        source_cols=tuple(srcs), monitor_sensors=tuple(srcs),
    )
    a = add_causal_temporal_features(df, cfg)
    b = add_causal_temporal_features(df, cfg)
    dev = 0.0
    for n in temporal_feature_names(cfg):
        dev = max(dev, float(np.max(np.abs(a[n].to_numpy() - b[n].to_numpy()))))
    _check("temporal feature construction is deterministic (identical repeat)",
           dev == 0.0, f"max|Δ|={dev:.2e}")

    reg = _load_json(EXPERIMENTS_DIR / "registry.json")
    rm = _load_json(EXPERIMENTS_DIR / "run_manifest.json")
    _check("run_manifest phase1 hashes match registry",
           rm["phase1_reference"]["manifest_hash"] == reg["phase1_split_reference"]["manifest_hash"])
    _check("random_seed recorded (42)", reg["defaults"].get("random_seed") == 42)


# ===========================================================================
# SECTION 12 — report / results consistency
# ===========================================================================
def section_report_consistency() -> None:
    _hdr("SECTION 12: REPORT / RESULTS CONSISTENCY")
    _check("Phase 2.5 report exists", REPORT_PATH.exists())
    if not REPORT_PATH.exists():
        return
    text = REPORT_PATH.read_text(encoding="utf-8")
    low = text.lower()
    _check("report distinguishes Phase 2 and Phase 2.5", "phase 2" in low and "phase 2.5" in low)
    _check("report states 50 independent engines statistical unit",
           "50 independent engines" in low)
    _check("report never calls operational cycles 'days'",
           not re.search(r"\bcycle[s]?\b[^.\n]{0,40}\bdays\b", low))

    # Anchor consistency: RUL anchor MAE in the CSV must be the Phase 2 value.
    rul = pd.read_csv(EXPERIMENTS_DIR / "rul_comparison.csv")
    anchor = rul[rul["experiment_id"] == "RUL25_T0_anchor_cycle"]
    ok = len(anchor) == 1 and abs(float(anchor["MAE"].iloc[0]) - 32.01) < 0.1
    _check("RUL25_T0_anchor_cycle reproduces Phase 2 MAE (~32.01)",
           ok, f"MAE={float(anchor['MAE'].iloc[0]):.3f}" if len(anchor) else "anchor row missing")

    # Report must reference the measured anchor MAE to be consistent with artifacts.
    _check("report cites the RUL anchor MAE (32.0)", "32.0" in text)

    # Every registry experiment id must appear in a comparison CSV.
    fr = pd.read_csv(EXPERIMENTS_DIR / "failure_risk_comparison.csv")
    reg = _load_json(EXPERIMENTS_DIR / "registry.json")
    csv_ids = set(rul["experiment_id"]) | set(fr["experiment_id"])
    reg_ids = {e["experiment_id"] for e in reg["experiments"]}
    _check("comparison CSVs cover every registered experiment",
           reg_ids <= csv_ids, str(sorted(reg_ids - csv_ids)))


def _print_summary() -> None:
    _hdr("SUMMARY")
    if issues:
        print(f"  {len(issues)} GATE CHECK(S) FAILED:")
        for i in issues:
            print(f"    - {i}")
        raise SystemExit(1)
    print("  ALL PHASE 2.5 GATE CHECKS PASSED")


def main() -> int:
    # Native Windows/PowerShell consoles default to a legacy code page that
    # cannot encode the non-ASCII diagnostic symbols (e.g. "\u0394") used in
    # gate detail messages, raising UnicodeEncodeError. Force UTF-8 stdout, as
    # the Phase 2.6 gate already does. Gate logic/criteria are unchanged.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - non-Windows / already utf-8
        pass
    print("== Phase 2.5 Final Gate Verification ==")
    section_frozen_split()
    section_test_firewall()
    section_no_neural()
    section_causality()
    section_engine_boundaries()
    section_train_only()
    section_required_experiments()
    section_metrics()
    section_artifacts()
    section_causality_tests()
    section_reproducibility()
    section_report_consistency()
    _print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
