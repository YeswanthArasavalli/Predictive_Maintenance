"""Phase 2.6 driver — Decision & Cost-Sensitive Validation.

Run:  python scripts/run_phase2_6_decision_analysis.py

Reads ONLY the existing Phase 2.5 validation prediction rows and the sealed
FD004 validation partition, then writes decision/cost artifacts under
`results/decisions/phase2_6/`:

  threshold_cost.csv             per (model x threshold x cost_ratio) confusion + cost
  cost_ratio_sensitivity.csv     per (model x horizon x cost_ratio) min-cost threshold
  decision_cost_comparison.csv   per (horizon x cost_ratio) anchor-vs-temporal value
  engine_level_analysis.csv      per (model x engine) decision + alert accounting
  alert_fatigue.json             descriptive positive-run / false-alarm aggregates
  false_negative_analysis.json   FN row counts + raw-RUL distribution + engines
  rul_decision_analysis.json     RUL MAE / worst-case / bias / end-of-life profile
  bootstrap.csv                  engine-level bootstrap CI for the cost difference
  analysis_registry.json         audit registry (section 19), separate from experiments
  run_manifest.json              hashes of the read-only inputs + output inventory

HARD RULES (verified by scripts/verify_phase2_6_gate.py):
  - NO model is retrained; only existing validation predictions are analyzed.
  - The official test partition (test_FD004.txt / RUL_FD004.txt) is NEVER read.
  - NO neural / sequence model is defined or imported.
  - Every cost is an ILLUSTRATIVE scenario assumption, never a financial figure.
  - No Phase 2 / Phase 2.5 historical artifact is modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.business.decision import (  # noqa: E402
    ILLUSTRATIVE_COST_LABEL,
    bootstrap_cost_difference,
    build_cost_ratio_sensitivity,
    build_decision_cost_comparison,
    build_threshold_cost_table,
    engine_level_counts,
    false_negative_profile,
    rul_decision_profile,
    summarize_engine_alert_fatigue,
)

CONFIG_PATH = ROOT / "configs" / "phase2_6_decision.yaml"

RAW_RUL_COL = "raw_RUL"


def _load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_fr_rows(experiments_dir: Path, model_id: str) -> pd.DataFrame:
    """Load a failure-risk experiment's validation prediction rows.

    The Phase 2.5 val_rows parquet carries unit_id, cycle, operating_regime,
    y_true, y_pred, y_prob. raw RUL is joined from the sealed FD004 VALIDATION
    partition (never the official test partition) for the FN profile.
    """
    p = experiments_dir / "val_rows" / f"{model_id}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing Phase 2.5 prediction artifact: {p}")
    return pd.read_parquet(p)


def _attach_raw_rul(fr_rows: pd.DataFrame, val_reference: pd.DataFrame) -> pd.DataFrame:
    cols = ["unit_id", "cycle", RAW_RUL_COL]
    ref = val_reference[cols].drop_duplicates(subset=["unit_id", "cycle"])
    out = fr_rows.merge(ref, on=["unit_id", "cycle"], how="left", validate="one_to_one")
    if out[RAW_RUL_COL].isna().any():
        raise ValueError("raw_RUL join produced NaN (validation rows misaligned)")
    return out


def main() -> int:
    print("== Phase 2.6 — Decision & Cost-Sensitive Validation ==")
    cfg = _load_config()
    paths = cfg["paths"]
    experiments_dir = ROOT / paths["experiments_dir"]
    decisions_dir = ROOT / paths["decisions_dir"]
    decisions_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(t) for t in cfg["thresholds"]]
    cost = cfg["cost_assumptions"]
    ratios = [float(r) for r in cost["cost_ratios"]]
    c_fp_unit = float(cost["fp_unit_cost"])
    models = cfg["models"]

    anchor_id = models["anchor_fr_H30"]
    temporal_id = models["temporal_fr_H30"]
    primary_h = int(cfg["horizons"]["primary"])

    # --- Load sealed FD004 validation reference (raw RUL), NEVER the test set.
    val_ref = pd.read_parquet(ROOT / paths["processed_dir"] / "FD004_val.parquet")
    print(f"  Validation partition: {len(val_ref)} rows, "
          f"{val_ref['unit_id'].nunique()} engines (raw RUL source)")

    fr_anchor = _attach_raw_rul(_load_fr_rows(experiments_dir, anchor_id), val_ref)
    fr_temporal = _attach_raw_rul(_load_fr_rows(experiments_dir, temporal_id), val_ref)

    # H14 / H50 temporal sensitivity (no matched cycle-aware anchor exists).
    horizon_frames: dict[int, list[tuple[str, pd.DataFrame]]] = {
        primary_h: [(anchor_id, fr_anchor), (temporal_id, fr_temporal)],
    }
    for key, h in (("temporal_fr_H14", 14), ("temporal_fr_H50", 50)):
        mid = models.get(key)
        if mid:
            horizon_frames.setdefault(h, []).append(
                (mid, _attach_raw_rul(_load_fr_rows(experiments_dir, mid), val_ref))
            )

    # ===== Section 6/7 — threshold x cost-ratio tables =====
    tc_frames = []
    for h, entries in horizon_frames.items():
        for mid, frame in entries:
            tc_frames.append(build_threshold_cost_table(
                frame["y_true"].to_numpy(), frame["y_prob"].to_numpy(),
                model_id=mid, horizon=h, thresholds=thresholds,
                cost_ratios=ratios, c_fp_unit=c_fp_unit,
            ))
    threshold_cost = pd.concat(tc_frames, ignore_index=True)
    threshold_cost.to_csv(decisions_dir / "threshold_cost.csv", index=False)

    sensitivity = build_cost_ratio_sensitivity(threshold_cost)
    sensitivity.to_csv(decisions_dir / "cost_ratio_sensitivity.csv", index=False)

    # ===== Section 8 — decision value comparison (matched H30 anchor/temporal) =====
    comparison = build_decision_cost_comparison(
        sensitivity, anchor_id=anchor_id, temporal_id=temporal_id
    )
    comparison.to_csv(decisions_dir / "decision_cost_comparison.csv", index=False)

    # ===== Section 10/11/12/13 — engine-level + alert fatigue =====
    eng_thr = float(cfg["engine_analysis"]["threshold"])
    eng_anchor = engine_level_counts(fr_anchor, model_id=anchor_id, threshold=eng_thr,
                                     raw_rul_col=RAW_RUL_COL)
    eng_temporal = engine_level_counts(fr_temporal, model_id=temporal_id, threshold=eng_thr,
                                       raw_rul_col=RAW_RUL_COL)
    engine_level = pd.concat([eng_anchor, eng_temporal], ignore_index=True)
    engine_level.to_csv(decisions_dir / "engine_level_analysis.csv", index=False)

    alert_fatigue = {
        "threshold": eng_thr,
        "label": ILLUSTRATIVE_COST_LABEL,
        "note": "Descriptive raw-prediction aggregates; NOT a production alert policy.",
        "anchor": summarize_engine_alert_fatigue(eng_anchor),
        "temporal": summarize_engine_alert_fatigue(eng_temporal),
    }
    (decisions_dir / "alert_fatigue.json").write_text(
        json.dumps(alert_fatigue, indent=2), encoding="utf-8")

    fn_analysis = {"threshold": eng_thr, "models": {}}
    for mid, frame in ((anchor_id, fr_anchor), (temporal_id, fr_temporal)):
        fn_analysis["models"][mid] = {
            "false_negatives": false_negative_profile(
                frame, threshold=eng_thr, raw_rul_col=RAW_RUL_COL),
            "false_positives": {
                "n_fp_rows": int((eng_anchor if mid == anchor_id else eng_temporal)["fp"].sum()),
                "fp_rate_of_negatives": float(
                    (eng_anchor if mid == anchor_id else eng_temporal)["fp"].sum()
                    / max(1, int((frame["y_true"].to_numpy() == 0).sum()))),
                "n_engines_with_fp": int((eng_anchor if mid == anchor_id else eng_temporal)["fp"].gt(0).sum()),
            },
        }
    (decisions_dir / "false_negative_analysis.json").write_text(
        json.dumps(fn_analysis, indent=2), encoding="utf-8")

    # ===== Section 16 — engine-level bootstrap of the cost difference =====
    boot_cfg = cfg["bootstrap"]
    if boot_cfg.get("enabled", True):
        bootstrap = bootstrap_cost_difference(
            eng_anchor, eng_temporal,
            cost_ratios=ratios, threshold=float(boot_cfg["threshold"]),
            c_fp_unit=c_fp_unit, n_resamples=int(boot_cfg["n_resamples"]),
            seed=int(boot_cfg["seed"]),
        )
        bootstrap.to_csv(decisions_dir / "bootstrap.csv", index=False)
    else:
        bootstrap = pd.DataFrame()

    # ===== Section 9 — RUL decision analysis (no binary cost forcing) =====
    eol = float(cfg["engine_analysis"]["end_of_life_rul"])
    rul_anchor = _load_fr_rows(experiments_dir, models["anchor_rul"])
    rul_temporal = _load_fr_rows(experiments_dir, models["temporal_rul"])
    rul_anchor = rul_anchor.merge(val_ref[["unit_id", "cycle", RAW_RUL_COL]],
                                  on=["unit_id", "cycle"], how="left", validate="one_to_one")
    rul_temporal = rul_temporal.merge(val_ref[["unit_id", "cycle", RAW_RUL_COL]],
                                       on=["unit_id", "cycle"], how="left", validate="one_to_one")
    rul_analysis = {
        "note": "RUL is NOT forced into the binary cost framework (section 9). "
                "These are predictive-quality diagnostics only; no monetary value.",
        "end_of_life_rul": eol,
        "anchor": rul_decision_profile(rul_anchor, end_of_life_rul=eol),
        "temporal": rul_decision_profile(rul_temporal, end_of_life_rul=eol),
    }
    a_mae = rul_analysis["anchor"]["mae"]
    t_mae = rul_analysis["temporal"]["mae"]
    rul_analysis["mae_change"] = {
        "abs_cycles": t_mae - a_mae,
        "pct": (t_mae - a_mae) / a_mae * 100.0 if a_mae else float("nan"),
    }
    (decisions_dir / "rul_decision_analysis.json").write_text(
        json.dumps(rul_analysis, indent=2), encoding="utf-8")

    # ===== Section 19 — separate analysis registry =====
    artifact_paths = sorted(p.name for p in decisions_dir.glob("*") if p.is_file())
    analysis_registry = {
        "schema_version": "1.0",
        "phase": 2.6,
        "primary_dataset": cfg["primary_dataset"],
        "purpose": "Decision & cost-sensitive validation (NOT a new model phase).",
        "hard_rules": cfg["hard_rules"],
        "cost_assumptions": cost,
        "illustrative_cost_label": ILLUSTRATIVE_COST_LABEL,
        "inputs_read_only": {
            "prediction_artifacts_dir": paths["experiments_dir"],
            "raw_rul_source": f"{paths['processed_dir']}/FD004_val.parquet",
            "val_hash_note": "validation partition only; official test never read",
        },
        "models_evaluated": models,
        "thresholds": thresholds,
        "cost_ratios": ratios,
        "engine_aggregation_method": (
            f"per-engine confusion + positive-run counts at threshold {eng_thr}; "
            "bootstrap resamples the 50 validation engines (rows within an engine "
            "are correlated and are NOT treated as independent events)"
        ),
        "bootstrap": boot_cfg,
        "analyses": [
            {"analysis_id": "threshold_cost", "artifact": "threshold_cost.csv",
             "horizons": sorted(horizon_frames), "models": sorted(
                 m for e in horizon_frames.values() for m, _ in e)},
            {"analysis_id": "cost_ratio_sensitivity", "artifact": "cost_ratio_sensitivity.csv"},
            {"analysis_id": "decision_cost_comparison", "artifact": "decision_cost_comparison.csv",
             "primary_horizon": primary_h, "anchor": anchor_id, "temporal": temporal_id},
            {"analysis_id": "engine_level", "artifact": "engine_level_analysis.csv",
             "threshold": eng_thr},
            {"analysis_id": "alert_fatigue", "artifact": "alert_fatigue.json"},
            {"analysis_id": "false_negative", "artifact": "false_negative_analysis.json"},
            {"analysis_id": "rul_decision", "artifact": "rul_decision_analysis.json"},
            {"analysis_id": "bootstrap", "artifact": "bootstrap.csv",
             "seed": int(boot_cfg["seed"]), "n_resamples": int(boot_cfg["n_resamples"])},
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (decisions_dir / "analysis_registry.json").write_text(
        json.dumps(analysis_registry, indent=2), encoding="utf-8")

    # ===== run manifest (input hashes -> proves no history mutation) =====
    run_manifest = {
        "phase": 2.6,
        "cost_framework": cost["framework"],
        "illustrative_cost_label": ILLUSTRATIVE_COST_LABEL,
        "input_hashes": {
            f"{anchor_id}.parquet": _sha256_file(experiments_dir / "val_rows" / f"{anchor_id}.parquet"),
            f"{temporal_id}.parquet": _sha256_file(experiments_dir / "val_rows" / f"{temporal_id}.parquet"),
            "FD004_val.parquet": _sha256_file(ROOT / paths["processed_dir"] / "FD004_val.parquet"),
        },
        "outputs": artifact_paths,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (decisions_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8")

    # ===== console summary =====
    print("\n== Decision value (H30, best illustrative cost per ratio) ==")
    h30 = comparison[comparison["horizon"] == primary_h]
    with pd.option_context("display.width", 160):
        print(h30[["cost_ratio", "anchor_cost", "temporal_cost", "abs_reduction",
                   "pct_reduction", "lower_cost_model"]].to_string(index=False))
    print("\n== RUL decision profile (MAE cycles) ==")
    print(f"  anchor MAE={a_mae:.3f}  temporal MAE={t_mae:.3f}  "
          f"change={rul_analysis['mae_change']['abs_cycles']:+.3f} "
          f"({rul_analysis['mae_change']['pct']:+.2f}%)")
    if not bootstrap.empty:
        print("\n== Engine bootstrap: P(temporal cost < anchor) ==")
        with pd.option_context("display.width", 160):
            print(bootstrap[["cost_ratio", "mean_cost_diff", "ci_low", "ci_high",
                             "prob_temporal_lower"]].to_string(index=False))
    print(f"\nWrote {len(artifact_paths)} artifacts to {decisions_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
