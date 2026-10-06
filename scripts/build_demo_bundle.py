"""Build the public Streamlit demo bundle from frozen research artifacts.

This script is a READ-ONLY transform over already-frozen scientific outputs.
It never retrains, re-splits, re-metrics, or modifies anything under `results/`.
It writes a SMALL, public-safe bundle under `demo/` that the Streamlit app reads
as its only data source.

Provenance guarantee written into every bundle manifest:

    Derived from frozen validation artifacts. No official test partition used.

The bundle contains ONLY validation-engine aggregate metrics and per-cycle
validation predictions. It contains NO raw NASA data, NO sealed official test
labels (test_FD004.txt / RUL_FD004.txt), NO sensor rows, NO machine paths.

Run (in the locked scientific env, which already reads the frozen artifacts):

    .venv/Scripts/python.exe scripts/build_demo_bundle.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEMO = ROOT / "demo"

# Selected showcase models (frozen Phase 2.5 selections).
ANCHOR_RUL = "RUL25_T0_anchor_cycle"
TEMPORAL_RUL = "RUL25_T4_combined_cyc"       # combined, W=5 (paired with FR temporal)
RUL_TEMPORAL_BEST = "RUL25_T4_combined_cyc_W3"  # combined, W=3 (lowest frozen RUL MAE)
ANCHOR_FR = "FR25_T0_anchor_cycle"
TEMPORAL_FR = "FR25_T4_combined_cyc"

PROVENANCE = "Derived from frozen validation artifacts. No official test partition used."


def _read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _copy_decisions() -> None:
    src = RESULTS / "decisions" / "phase2_6"
    dst = DEMO / "decision"
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "cost_ratio_sensitivity.csv",
        "decision_cost_comparison.csv",
        "engine_level_analysis.csv",
        "threshold_cost.csv",
        "alert_fatigue.json",
        "false_negative_analysis.json",
        "rul_decision_analysis.json",
    ]:
        data = (src / name).read_bytes()
        (dst / name).write_bytes(data)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    DEMO.mkdir(parents=True, exist_ok=True)

    # ---- Overview facts -------------------------------------------------
    inventory = _read_json("results/audits/inventory.json")["FD004"]
    metadata = _read_json("data/processed/FD004_metadata.json")
    env = _read_json("results/audits/environment.json")

    overview = {
        "provenance": PROVENANCE,
        "dataset": "NASA C-MAPSS FD004",
        "n_train_engines_total": inventory["train"]["n_engines"],
        "n_test_engines_official": inventory["test"]["n_engines"],
        "n_operating_regimes": _read_json("results/audits/operating_conditions.json")["FD004_train"]["n_unique_regimes"],
        "n_fault_modes": 2,  # FD004 combines two fault modes (documented in docs/dataset.md)
        "n_train_engines_used": len(metadata["train_engines"]),
        "n_val_engines": len(metadata["val_engines"]),
        "primary_horizon_H": metadata["failure_horizon_H"],
        "scientific_python": env["python"],
        "environment_packages": env["packages"],
        "seed": env["random_seed_policy"]["global_seed"],
        "gates": {
            "phase1": "PASS",
            "phase2": "PASS",
            "phase2_5": "PASS",
            "phase2_6": "PASS",
        },
        "scientific_test_count": 147,
    }
    (DEMO / "overview.json").write_text(
        json.dumps(overview, indent=2), encoding="utf-8"
    )

    # ---- Frozen metrics -------------------------------------------------
    rul = pd.read_csv(RESULTS / "experiments" / "phase2_5" / "rul_comparison.csv")
    fr = pd.read_csv(RESULTS / "experiments" / "phase2_5" / "failure_risk_comparison.csv")

    def _rul_row(exp_id: str) -> dict:
        r = rul[rul["experiment_id"] == exp_id].iloc[0]
        return {
            "experiment_id": exp_id,
            "MAE": float(r["MAE"]),
            "RMSE": float(r["RMSE"]),
            "R2": float(r["R2"]),
            "prognostic_score_last_per_engine": float(r["prognostic_score_last_per_engine"]),
        }

    def _fr_row(exp_id: str) -> dict:
        r = fr[
            (fr["experiment_id"] == exp_id)
            & (fr["horizon"] == 30)
            & (fr["threshold_source"] == "candidate_f1_argmax")
        ].iloc[0]
        return {
            "experiment_id": exp_id,
            "threshold": float(r["threshold"]),
            "recall": float(r["recall"]),
            "precision": float(r["precision"]),
            "F1": float(r["F1"]),
            "PR_AUC": float(r["PR_AUC"]),
            "ROC_AUC": float(r["ROC_AUC"]),
            "TP": int(r["TP"]),
            "FP": int(r["FP"]),
            "TN": int(r["TN"]),
            "FN": int(r["FN"]),
            "positive_rate_val": float(r["positive_rate_val"]),
        }

    majority = _read_json("results/experiments/phase2/failure_risk/FR_majority_H30.json")
    metrics = {
        "provenance": PROVENANCE,
        "note": "All metrics are engine-grouped VALIDATION metrics, not official test-set metrics.",
        "rul": {
            "anchor": _rul_row(ANCHOR_RUL),
            "temporal": _rul_row(TEMPORAL_RUL),
            "temporal_best_mae": _rul_row(RUL_TEMPORAL_BEST),
        },
        "failure_risk": {"anchor": _fr_row(ANCHOR_FR), "temporal": _fr_row(TEMPORAL_FR)},
        "labels": {
            "anchor_rul": "Classical anchor (cycle, no temporal)",
            "temporal_rul": "Causal temporal - paired (combined, W=5)",
            "temporal_best_rul": "Causal temporal - best-MAE (combined, W=3)",
            "anchor_fr": "Classical anchor (cycle, H30)",
            "temporal_fr": "Causal temporal (combined, H30)",
        },
        "selection_note": (
            "The RUL 'temporal' card is the combined W=5 configuration paired with the "
            "selected failure-risk model; the best-MAE temporal RUL is the combined W=3 "
            "configuration (reported separately). No single configuration is presented as "
            "the sole 'best RUL'."
        ),
        "majority_baseline": {
            "accuracy": float(majority["summary_metrics_flat"]["accuracy"]),
            "recall": float(majority["summary_metrics_flat"]["recall"]),
            "positive_rate": float(majority["class_balance"]["positive_rate"]),
        },
        "decision": _read_json("results/decisions/phase2_6/rul_decision_analysis.json"),
    }
    (DEMO / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # ---- Per-cycle validation predictions (anchor + temporal) ----------
    def _rul(exp_id: str) -> pd.DataFrame:
        return pd.read_parquet(
            RESULTS / "experiments" / "phase2_5" / "val_rows" / f"{exp_id}.parquet"
        )[["unit_id", "cycle", "operating_regime", "y_true", "y_pred"]].rename(
            columns={"y_true": "actual_rul", "y_pred": "predicted_rul"}
        )

    def _fr(exp_id: str) -> pd.DataFrame:
        return pd.read_parquet(
            RESULTS / "experiments" / "phase2_5" / "val_rows" / f"{exp_id}.parquet"
        )[["unit_id", "cycle", "operating_regime", "y_true", "y_prob"]].rename(
            columns={"y_true": "actual_state", "y_prob": "failure_prob"}
        )

    rul_pred = _rul(TEMPORAL_RUL)
    fr_pred = _fr(TEMPORAL_FR)
    # Back-compatible temporal files (existing pages/tests) + explicit model-keyed set.
    rul_pred.to_parquet(DEMO / "rul_val_predictions.parquet", index=False)
    rul_pred.to_parquet(DEMO / "rul_temporal_predictions.parquet", index=False)
    _rul(ANCHOR_RUL).to_parquet(DEMO / "rul_anchor_predictions.parquet", index=False)
    fr_pred.to_parquet(DEMO / "fr_val_predictions.parquet", index=False)
    fr_pred.to_parquet(DEMO / "fr_temporal_predictions.parquet", index=False)
    _fr(ANCHOR_FR).to_parquet(DEMO / "fr_anchor_predictions.parquet", index=False)

    _copy_decisions()

    # ---- Manifest -------------------------------------------------------
    files = sorted(p.relative_to(DEMO).as_posix() for p in DEMO.rglob("*") if p.is_file())
    manifest = {
        "provenance": PROVENANCE,
        "generated_from": "results/ (frozen Phase 2.5 / 2.6 validation artifacts)",
        "n_files": len(files),
        "files": files,
        "sha256": {
            "rul_val_predictions.parquet": _sha256(DEMO / "rul_val_predictions.parquet"),
            "fr_val_predictions.parquet": _sha256(DEMO / "fr_val_predictions.parquet"),
        },
        "n_val_engines": int(rul_pred["unit_id"].nunique()),
        "n_rows": int(len(rul_pred)),
    }
    (DEMO / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (DEMO / "README.md").write_text(
        "# Public Streamlit demo bundle\n\n"
        f"> {PROVENANCE}\n\n"
        "Generated by `scripts/build_demo_bundle.py` from frozen `results/` validation\n"
        "artifacts. This is the Streamlit app's only data source. It contains aggregate\n"
        "validation metrics and per-cycle validation predictions for the 50 held-out\n"
        "validation engines. It contains no raw data and no sealed official test labels.\n",
        encoding="utf-8",
    )

    return manifest


if __name__ == "__main__":
    m = build()
    print(json.dumps(m, indent=2))
