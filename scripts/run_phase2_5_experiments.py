"""Phase 2.5 driver: run every causal-temporal baseline in configs/phase2_5_experiments.yaml.

Run:  python scripts/run_phase2_5_experiments.py

Outputs (under results/experiments/phase2_5/):
  registry.json
  rul/{experiment_id}.json
  failure_risk/{experiment_id}.json
  val_rows/{experiment_id}.parquet
  rul_comparison.csv
  failure_risk_comparison.csv
  temporal_feature_metadata.json
  run_manifest.json

HARD RULES (also verified by scripts/verify_phase2_5_gate.py):
  - Reuses the Phase 1 manifest engine split verbatim (no re-split).
  - Never loads test_FD004.txt or RUL_FD004.txt.
  - Never fits a scaler or selects a feature on validation rows.
  - Trains no LSTM/GRU/Transformer/CNN/RNN (registry allowlist rejects any
    non-classical model at load time).
  - Every temporal feature is causal (see src/features/temporal.py).

This phase does NOT modify or overwrite any Phase 2 artifact.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

CONFIG_PATH = ROOT / "configs" / "phase2_5_experiments.yaml"
EXPERIMENTS_DIR = ROOT / "results" / "experiments" / "phase2_5"
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
    spec = result.spec
    if spec.task == "A_RUL":
        out_dir = out_dir_rul
        summary_metrics = result.metrics
    else:
        out_dir = out_dir_fr
        summary_metrics = result.metrics.get("at_default_threshold", {})

    payload = {
        "experiment_id": spec.experiment_id,
        "phase": 2.5,
        "task": spec.task,
        "model": spec.model,
        "feature_config": spec.feature_config,
        "normalization_mode": spec.normalization_mode,
        "include_cycle": spec.include_cycle,
        "horizon": spec.horizon,
        "temporal_family": spec.temporal_family,
        "temporal_windows": spec.temporal_windows,
        "temporal_lags": spec.temporal_lags,
        "temporal_n_monitors": spec.temporal_n_monitors,
        "temporal_metadata": result.temporal,
        "hyperparameters": spec.hyperparameters,
        "random_seed": 42,
        "description": spec.description,
        "feature_names": result.feature_names,
        "n_features": len(result.feature_names),
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
        tm = r.temporal or {}
        rows.append({
            "experiment_id": r.spec.experiment_id,
            "model": r.spec.model,
            "feature_config": r.spec.feature_config or "",
            "normalization": r.spec.normalization_mode or "",
            "cycle_feature": bool(r.spec.include_cycle),
            "temporal_family": r.spec.temporal_family,
            "temporal_window": ",".join(str(w) for w in r.spec.temporal_windows),
            "temporal_lag": ",".join(str(k) for k in r.spec.temporal_lags),
            "n_temporal_features": tm.get("n_temporal_features", 0),
            "n_features": len(r.feature_names),
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
        at_candidate = r.metrics.get("at_candidate_threshold") or {}
        candidate_thr = r.metrics.get("candidate_threshold")
        tm = r.temporal or {}
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
                "temporal_family": r.spec.temporal_family,
                "temporal_window": ",".join(str(w) for w in r.spec.temporal_windows),
                "n_temporal_features": tm.get("n_temporal_features", 0),
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
    df = df.sort_values(
        by=["horizon", "model", "threshold_source"],
        key=lambda s: s.map({30: 0, 14: 1, 50: 2}) if s.name == "horizon" else s,
    ).reset_index(drop=True)
    return df


def main() -> int:
    print("== Phase 2.5 — Causal Temporal Feature Baselines ==")
    defaults, specs = load_experiments(CONFIG_PATH)
    split = load_phase1_split()
    fd_id = defaults.get("primary_dataset", "FD004")

    print(f"  Dataset: {fd_id}")
    print(f"  Phase 1 manifest: {split['manifest_path']}")
    print(f"  Train engines: {len(split['train_engine_ids'])} (hash {split['train_hash'][:12]}...)")
    print(f"  Val engines:   {len(split['val_engine_ids'])} (hash {split['val_hash'][:12]}...)")
    print(f"  Registered experiments: {len(specs)}")

    registry = build_registry(specs=specs, phase1_split=split, defaults=defaults, phase=2.5)
    reg_path = write_registry(registry, EXPERIMENTS_DIR)
    print(f"  Registry: {reg_path.relative_to(ROOT)}")

    out_rul = EXPERIMENTS_DIR / "rul"
    out_fr = EXPERIMENTS_DIR / "failure_risk"

    results = []
    temporal_meta: dict[str, Any] = {}
    for i, spec in enumerate(specs, start=1):
        print(f"[{i:>2}/{len(specs)}] {spec.experiment_id:<32} task={spec.task} fam={spec.temporal_family}")
        r = run_experiment(spec, fd_id=fd_id, split=split)
        persist_result(r, out_rul, out_fr)
        persist_val_rows(r)
        results.append(r)
        if r.temporal:
            temporal_meta[spec.experiment_id] = r.temporal
        if spec.task == "A_RUL":
            print(
                f"        MAE={r.metrics['mae']:.3f}  RMSE={r.metrics['rmse']:.3f}  "
                f"R2={r.metrics['r2']:.3f}  PS_last={r.metrics.get('prognostic_score_last_per_engine', float('nan')):.2f}"
            )
        else:
            at_d = r.metrics.get("at_default_threshold", {})
            print(
                f"        thr=0.5: recall={at_d.get('recall', 0):.3f}  precision={at_d.get('precision', 0):.3f}  "
                f"F1={at_d.get('f1', 0):.3f}  PR-AUC={at_d.get('pr_auc')}"
            )

    rul_df = build_rul_comparison(results)
    fr_df = build_failure_risk_comparison(results)
    rul_path = EXPERIMENTS_DIR / "rul_comparison.csv"
    fr_path = EXPERIMENTS_DIR / "failure_risk_comparison.csv"
    rul_df.to_csv(rul_path, index=False)
    fr_df.to_csv(fr_path, index=False)

    # Temporal feature metadata: the TRAIN-only monitor selection is identical
    # across every temporal experiment (it depends only on the train partition
    # and the monitor budget), so it is captured once for audit.
    (EXPERIMENTS_DIR / "temporal_feature_metadata.json").write_text(
        json.dumps(
            {
                "selection_policy": "train-only within-engine drift (see src/models/features.select_temporal_monitors)",
                "early_cycle_policy": "expanding window; std/slope require >=2 obs else 0; diff/delta fill 0",
                "causality": "same-engine observations at cycle <= t only; no future; no engine-boundary crossing",
                "per_experiment": temporal_meta,
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    run_manifest = {
        "phase": 2.5,
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

    print("\n== RUL Summary (sorted by MAE) ==")
    print(rul_df[["experiment_id", "temporal_family", "cycle_feature", "MAE", "R2", "prognostic_score_last_per_engine"]].to_string(index=False))
    print("\n== Failure-Risk Summary (H30, thr=0.5) ==")
    h30 = fr_df[(fr_df["horizon"] == 30) & (fr_df["threshold_source"] == "default_0.5")]
    print(h30[["experiment_id", "temporal_family", "cycle_feature", "recall", "precision", "F1", "PR_AUC"]].to_string(index=False))
    print(f"\nWrote: {rul_path.relative_to(ROOT)}")
    print(f"Wrote: {fr_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
