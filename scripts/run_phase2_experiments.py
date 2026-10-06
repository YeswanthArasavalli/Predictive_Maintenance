"""Phase 2 driver: run every baseline in configs/phase2_experiments.yaml.

Run:  python scripts/run_phase2_experiments.py

Outputs:
  results/experiments/phase2/registry.json
  results/experiments/phase2/rul/{experiment_id}.json
  results/experiments/phase2/failure_risk/{experiment_id}.json
  results/experiments/phase2/val_rows/{experiment_id}.parquet
  results/experiments/phase2/rul_comparison.csv
  results/experiments/phase2/failure_risk_comparison.csv

HARD RULES enforced here (and verified by verify_phase2_gate.py):
  - Uses the Phase 1 manifest engine split verbatim (no re-split).
  - Never loads test_FD004.txt or RUL_FD004.txt.
  - Never fits a scaler on validation rows (all scalers fit on train).
  - Never trains any LSTM/GRU/Transformer (spec-driven allowlist in
    src/pipeline/registry.py rejects unknown models at load time).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.experiment import run_experiment  # noqa: E402
from src.pipeline.registry import (  # noqa: E402
    build_registry,
    load_experiments,
    load_phase1_split,
    write_registry,
)

EXPERIMENTS_DIR = ROOT / "results" / "experiments" / "phase2"
VAL_ROWS_DIR = EXPERIMENTS_DIR / "val_rows"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def persist_result(result, out_dir_rul: Path, out_dir_fr: Path) -> Path:
    """Write one per-experiment JSON. Returns path."""
    spec = result.spec
    if spec.task == "A_RUL":
        out_dir = out_dir_rul
        # For Task A the comparison table uses the flat metrics directly
        summary_metrics = result.metrics
    else:
        out_dir = out_dir_fr
        summary_metrics = result.metrics.get("at_default_threshold", {})

    payload = {
        "experiment_id": spec.experiment_id,
        "task": spec.task,
        "model": spec.model,
        "feature_config": spec.feature_config,
        "normalization_mode": spec.normalization_mode,
        "include_cycle": spec.include_cycle,
        "horizon": spec.horizon,
        "hyperparameters": spec.hyperparameters,
        "random_seed": 42,
        "description": spec.description,
        "feature_names": result.feature_names,
        "model_state": result.model_state,
        "metrics": result.metrics,
        "summary_metrics_flat": summary_metrics,
        "class_balance": result.class_balance,
        "candidate_threshold": result.candidate_threshold,
        "threshold_sweep": result.threshold_sweep,
        "worst10": result.worst10.to_dict(orient="records") if result.worst10 is not None else None,
        "phase1_reference": result.data_hash,
        "timing_seconds": result.timing,
        "executed_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.experiment_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def persist_val_rows(result) -> Path:
    VAL_ROWS_DIR.mkdir(parents=True, exist_ok=True)
    path = VAL_ROWS_DIR / f"{result.spec.experiment_id}.parquet"
    result.val_rows.to_parquet(path, index=False)
    return path


def build_rul_comparison(results) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.spec.task != "A_RUL":
            continue
        rows.append({
            "experiment_id": r.spec.experiment_id,
            "model": r.spec.model,
            "feature_config": r.spec.feature_config or "",
            "normalization": r.spec.normalization_mode or "",
            "cycle_feature": bool(r.spec.include_cycle),
            "MAE": r.metrics["mae"],
            "RMSE": r.metrics["rmse"],
            "R2": r.metrics["r2"],
            "prognostic_score_rows": r.metrics["prognostic_score_rows"],
            "prognostic_score_last_per_engine": r.metrics.get("prognostic_score_last_per_engine"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("MAE").reset_index(drop=True)


def build_failure_risk_comparison(results) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.spec.task != "B_FAILURE_RISK":
            continue
        at_default = r.metrics.get("at_default_threshold", {}) or {}
        at_candidate = r.metrics.get("at_candidate_threshold", {}) or {}
        candidate_thr = r.metrics.get("candidate_threshold")
        # Report row: prefer candidate-threshold metrics for the CSV (Phase 2
        # section 17 says "identify a reasonable candidate threshold"); still
        # record the default-threshold row separately as a distinct experiment
        # row via the `threshold_source` column so nothing is hidden.
        for source, m in (("default_0.5", at_default), ("candidate_f1_argmax", at_candidate)):
            if not m:
                continue
            rows.append({
                "experiment_id": r.spec.experiment_id,
                "model": r.spec.model,
                "horizon": r.spec.horizon,
                "feature_config": r.spec.feature_config or "",
                "normalization": r.spec.normalization_mode or "",
                "cycle_feature": bool(r.spec.include_cycle),
                "threshold_source": source,
                "threshold": m.get("threshold"),
                "accuracy": m.get("accuracy"),
                "recall": m.get("recall"),
                "precision": m.get("precision"),
                "F1": m.get("f1"),
                "PR_AUC": m.get("pr_auc"),
                "ROC_AUC": m.get("roc_auc"),
                "TP": m.get("tp"),
                "FP": m.get("fp"),
                "TN": m.get("tn"),
                "FN": m.get("fn"),
                "positive_rate_val": (r.class_balance or {}).get("positive_rate"),
                "candidate_threshold_overall": candidate_thr,
            })
    df = pd.DataFrame(rows)
    # Order by (horizon primary first, then model, then source)
    df = df.sort_values(
        by=["horizon", "model", "threshold_source"],
        key=lambda s: s.map({30: 0, 14: 1, 50: 2}) if s.name == "horizon" else s,
    ).reset_index(drop=True)
    return df


def main() -> int:
    print("== Phase 2 — Baseline Modeling ==")
    defaults, specs = load_experiments()
    split = load_phase1_split()
    fd_id = defaults.get("primary_dataset", "FD004")

    print(f"  Dataset: {fd_id}")
    print(f"  Phase 1 manifest: {split['manifest_path']}")
    print(f"  Train engines: {len(split['train_engine_ids'])} (hash {split['train_hash'][:12]}...)")
    print(f"  Val engines:   {len(split['val_engine_ids'])} (hash {split['val_hash'][:12]}...)")
    print(f"  Registered experiments: {len(specs)}")

    registry = build_registry(specs=specs, phase1_split=split, defaults=defaults)
    reg_path = write_registry(registry, EXPERIMENTS_DIR)
    print(f"  Registry: {reg_path.relative_to(ROOT)}")

    out_rul = EXPERIMENTS_DIR / "rul"
    out_fr = EXPERIMENTS_DIR / "failure_risk"

    results = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i:>2}/{len(specs)}] {spec.experiment_id:<40} task={spec.task} model={spec.model}")
        r = run_experiment(spec, fd_id=fd_id, split=split)
        p = persist_result(r, out_rul, out_fr)
        persist_val_rows(r)
        results.append(r)
        # Compact summary line
        if spec.task == "A_RUL":
            print(
                f"        MAE={r.metrics['mae']:.3f}  RMSE={r.metrics['rmse']:.3f}  "
                f"R2={r.metrics['r2']:.3f}  PS_rows={r.metrics['prognostic_score_rows']:.3e}"
            )
        else:
            at_d = r.metrics.get("at_default_threshold", {})
            at_c = r.metrics.get("at_candidate_threshold") or {}
            thr = r.metrics.get("candidate_threshold")
            print(
                f"        thr=0.5: recall={at_d.get('recall', 0):.3f}  precision={at_d.get('precision', 0):.3f}  "
                f"F1={at_d.get('f1', 0):.3f}  PR-AUC={at_d.get('pr_auc') if at_d.get('pr_auc') is not None else 'n/a'}"
            )
            if thr is not None:
                print(
                    f"        cand thr={thr:.3f}: recall={at_c.get('recall', 0):.3f}  "
                    f"precision={at_c.get('precision', 0):.3f}  F1={at_c.get('f1', 0):.3f}"
                )

    # Comparison tables
    rul_df = build_rul_comparison(results)
    fr_df = build_failure_risk_comparison(results)
    rul_path = EXPERIMENTS_DIR / "rul_comparison.csv"
    fr_path = EXPERIMENTS_DIR / "failure_risk_comparison.csv"
    rul_df.to_csv(rul_path, index=False)
    fr_df.to_csv(fr_path, index=False)

    # Also persist a compact run manifest for the gate script
    run_manifest = {
        "n_experiments": len(specs),
        "phase1_reference": {
            "train_hash": split["train_hash"],
            "val_hash": split["val_hash"],
            "manifest_hash": split["manifest_hash"],
        },
        "rul_csv": str(rul_path.relative_to(ROOT)),
        "failure_risk_csv": str(fr_path.relative_to(ROOT)),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (EXPERIMENTS_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, default=_json_default), encoding="utf-8"
    )

    print("\n== Summary ==")
    print(rul_df.to_string(index=False))
    print()
    print(fr_df.to_string(index=False))
    print(f"\nWrote: {rul_path.relative_to(ROOT)}")
    print(f"Wrote: {fr_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
