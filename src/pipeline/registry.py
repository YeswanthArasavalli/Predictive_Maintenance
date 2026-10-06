"""Phase 2 experiment registry.

Loads `configs/phase2_experiments.yaml`, resolves the Phase 1 train/val
engine IDs from `results/audits/phase1_dataset_manifest.json`, and emits
`results/experiments/phase2/registry.json` with per-experiment metadata
including engine-ID SHA-256 hashes, dataset manifest hash, git commit,
execution timestamp, and every declared hyperparameter.

The registry is the audit trail. Every subsequent experiment result file
references an `experiment_id` that MUST appear in this registry.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ExperimentSpec:
    """One row of a Phase 2 / Phase 2.5 experiment YAML.

    Fields match the Phase 2 spec (section 3). Optional modeling fields are
    `None` for baselines that use no features at all (naive, majority-class).

    The trailing ``temporal_*`` fields are Phase 2.5 additions. Their defaults
    reproduce the Phase 2 (non-temporal) behavior exactly, so every existing
    Phase 2 experiment spec is unchanged when those keys are absent.
    """

    experiment_id: str
    task: str                      # "A_RUL" or "B_FAILURE_RISK"
    model: str
    feature_config: str | None     # "A" | "B" | "B+" | None
    normalization_mode: str | None # "A" | "B" | None
    include_cycle: bool
    horizon: int | None            # only meaningful for Task B
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    # --- Phase 2.5 causal temporal fields (defaults == Phase 2 behavior) ---
    temporal_family: str = "none"              # none|lag_diff|rolling|slope|combined
    temporal_windows: list[int] = field(default_factory=lambda: [5])
    temporal_lags: list[int] = field(default_factory=lambda: [5])
    temporal_n_monitors: int = 8
    temporal_include_range: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def load_experiments(config_path: Path | None = None) -> tuple[dict[str, Any], list[ExperimentSpec]]:
    """Parse `configs/phase2_experiments.yaml`.

    Returns (defaults, list_of_specs). Raises on duplicate experiment_id
    or on an unknown `model` name. The set of allowed model names is
    frozen here so no deep-learning model can sneak in via YAML typo.
    """
    path = config_path or (ROOT / "configs" / "phase2_experiments.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {}) or {}
    specs: list[ExperimentSpec] = []
    seen: set[str] = set()

    allowed_models = {
        # Task A
        "naive_mean", "naive_age", "linear_degradation",
        "ridge", "hist_gb_regressor",
        # Task B
        "majority_class", "logistic_regression", "hist_gb_classifier",
    }
    allowed_tasks = {"A_RUL", "B_FAILURE_RISK"}
    allowed_temporal_families = {
        "none", "lag_diff", "rolling", "slope", "combined",
    }

    for entry in raw.get("experiments", []):
        spec = ExperimentSpec(
            experiment_id=str(entry["experiment_id"]),
            task=str(entry["task"]),
            model=str(entry["model"]),
            feature_config=entry.get("feature_config"),
            normalization_mode=entry.get("normalization_mode"),
            include_cycle=bool(entry.get("include_cycle", False)),
            horizon=entry.get("horizon"),
            hyperparameters=dict(entry.get("hyperparameters") or {}),
            description=str(entry.get("description", "")).strip(),
            temporal_family=str(entry.get("temporal_family", "none")),
            temporal_windows=[int(w) for w in (entry.get("temporal_windows") or [5])],
            temporal_lags=[int(k) for k in (entry.get("temporal_lags") or [5])],
            temporal_n_monitors=int(entry.get("temporal_n_monitors", 8)),
            temporal_include_range=bool(entry.get("temporal_include_range", False)),
        )
        if spec.experiment_id in seen:
            raise ValueError(f"Duplicate experiment_id: {spec.experiment_id}")
        seen.add(spec.experiment_id)
        if spec.model not in allowed_models:
            raise ValueError(
                f"Phase 2 hard rule violation: model {spec.model!r} "
                f"(experiment {spec.experiment_id}) is not in the allowed baseline list."
            )
        if spec.temporal_family not in allowed_temporal_families:
            raise ValueError(
                f"Unknown temporal_family {spec.temporal_family!r} in {spec.experiment_id}."
            )
        if spec.task not in allowed_tasks:
            raise ValueError(f"Unknown task {spec.task!r} in {spec.experiment_id}")
        if spec.task == "A_RUL" and spec.horizon not in (None,):
            raise ValueError(f"Task A must have horizon=None: {spec.experiment_id}")
        if spec.task == "B_FAILURE_RISK" and spec.horizon is None:
            raise ValueError(f"Task B requires a horizon: {spec.experiment_id}")
        specs.append(spec)
    return defaults, specs


def load_phase1_split(manifest_path: Path | None = None) -> dict[str, Any]:
    """Return the authoritative Phase 1 engine split (never re-derived here)."""
    path = manifest_path or (ROOT / "results" / "audits" / "phase1_dataset_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split = manifest["engine_split"]
    train_ids = list(split["train_engine_ids"])
    val_ids = list(split["val_engine_ids"])
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"Phase 1 manifest reports overlapping engines: {sorted(overlap)}")
    return {
        "train_engine_ids": train_ids,
        "val_engine_ids": val_ids,
        "train_hash": _sha256_json(sorted(train_ids)),
        "val_hash": _sha256_json(sorted(val_ids)),
        "manifest_hash": _sha256_json(manifest),
        "manifest_path": str(path.relative_to(ROOT)),
    }


def _git_commit() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(ROOT),
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def build_registry(
    *,
    specs: list[ExperimentSpec],
    phase1_split: dict[str, Any],
    defaults: dict[str, Any],
    phase: int | float = 2,
) -> dict[str, Any]:
    """Assemble the full registry dict for the phase's `registry.json`."""
    return {
        "schema_version": "1.1",
        "phase": phase,
        "primary_dataset": defaults.get("primary_dataset", "FD004"),
        "hard_rules": {
            "no_deep_learning": True,
            "official_test_untouched": True,
            "uses_phase1_engine_split_verbatim": True,
            "temporal_features_causal": bool(phase == 2.5) or None,
        },
        "temporal_policy": {
            "enabled": bool(phase == 2.5),
            "families": sorted({s.temporal_family for s in specs}),
            "causality": "every temporal feature uses only same-engine observations at cycle <= t",
            "selection": "monitor sensors ranked on TRAIN-only within-engine drift",
            "early_cycle_policy": "expanding window; std/slope need >=2 obs else 0; diff/delta fill 0",
        } if phase == 2.5 else None,
        "phase1_split_reference": {
            "manifest_path": phase1_split["manifest_path"],
            "manifest_hash": phase1_split["manifest_hash"],
            "train_hash": phase1_split["train_hash"],
            "val_hash": phase1_split["val_hash"],
            "n_train_engines": len(phase1_split["train_engine_ids"]),
            "n_val_engines": len(phase1_split["val_engine_ids"]),
        },
        "defaults": {
            "random_seed": defaults.get("random_seed", 42),
            "failure_horizon_primary": defaults.get("failure_horizon_primary", 30),
            "failure_horizon_sensitivity": defaults.get("failure_horizon_sensitivity", [14, 50]),
        },
        "experiments": [s.to_dict() for s in specs],
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }


def write_registry(registry: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "registry.json"
    path.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    return path
