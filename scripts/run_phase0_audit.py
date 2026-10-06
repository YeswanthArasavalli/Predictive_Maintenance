"""Phase-0 audit driver: computes and writes machine-readable artifacts.

Run:  python scripts/run_phase0_audit.py

Outputs (all regenerated deterministically from raw data):
  results/audits/integrity.json
  results/audits/inventory.json
  results/audits/data_quality.json
  results/audits/operating_conditions.json
  results/audits/degradation_rul.json
  results/audits/target_design.json
  results/tables/dataset_inventory.csv
  results/tables/sensor_summary_<FD>.csv
  results/tables/sensor_correlation_<FD>.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import integrity, load_dataset, load_rul  # noqa: E402
from src.data import audit  # noqa: E402
from src.data.loader import README_ENGINE_CLAIMS, UNIT_ID  # noqa: E402

SEED = 42
np.random.seed(SEED)

AUDITS = ROOT / "results" / "audits"
TABLES = ROOT / "results" / "tables"
AUDITS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

PRIMARY = "FD004"
ALL_FD = ["FD001", "FD002", "FD003", "FD004"]
HORIZONS = [14, 30, 50]


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


def main() -> int:
    print("== Phase 0 audit ==")

    # 1. Integrity (§5)
    results = integrity.verify_raw()
    integrity_obj = {
        "all_match": all(r.match for r in results),
        "n_files": len(results),
        "checksum_source": "data/.checksums.txt",
        "files": [r.to_dict() for r in results],
    }
    _write_json(AUDITS / "integrity.json", integrity_obj)
    print(f"  integrity all_match={integrity_obj['all_match']}")

    # 2. Inventory (§7)
    loaded = {}
    inventory = {}
    rows = []
    for fd in ALL_FD:
        tr = load_dataset(fd, "train")
        te = load_dataset(fd, "test")
        ru = load_rul(fd)
        loaded[fd] = (tr, te, ru)
        claims = README_ENGINE_CLAIMS[fd]
        inv = audit.inventory_entry(
            train=tr, test=te, rul=ru,
            readme_train=claims["train"], readme_test=claims["test"],
        )
        inventory[fd] = inv
        rows.append({
            "fd_id": fd,
            "train_engines": inv["train"]["n_engines"],
            "train_rows": inv["train"]["n_rows"],
            "test_engines": inv["test"]["n_engines"],
            "test_rows": inv["test"]["n_rows"],
            "rul_lines": inv["rul"]["n_test_engines"],
            "test_eq_rul": inv["test_matches_rul_count"],
            "readme_train": inv["readme_train_engines"],
            "readme_test": inv["readme_test_engines"],
            "train_eq_readme": inv["train_matches_readme"],
            "test_eq_readme": inv["test_matches_readme"],
        })
    _write_json(AUDITS / "inventory.json", inventory)
    pd.DataFrame(rows).to_csv(TABLES / "dataset_inventory.csv", index=False)
    print("  inventory written for", ", ".join(ALL_FD))

    # 3. Data quality (§8)
    quality = {}
    for fd in ALL_FD:
        tr, te, _ = loaded[fd]
        quality[f"{fd}_train"] = audit.quality_audit(tr, f"{fd}_train")
        quality[f"{fd}_test"] = audit.quality_audit(te, f"{fd}_test")
    _write_json(AUDITS / "data_quality.json", quality)

    # 4. Sensor analysis (§9) — full detail for primary, summary for all
    sensor_tables = {}
    sensor_audit = {}
    for fd in ALL_FD:
        tr, _, _ = loaded[fd]
        summary = audit.sensor_summary(tr)
        corr = audit.sensor_correlation(tr)
        summary.to_csv(TABLES / f"sensor_summary_{fd}.csv")
        corr.to_csv(TABLES / f"sensor_correlation_{fd}.csv")
        sensor_audit[fd] = {
            "high_correlation_pairs_ge_0.95": audit.correlation_pairs(corr, 0.95),
            "constant_sensors": [s for s in summary.index if summary.loc[s, "std"] == 0],
            "near_constant_sensors": summary[
                (summary["std"] > 0) & (summary["std"] < 0.1)
            ].index.tolist(),
        }
        sensor_tables[fd] = summary
    _write_json(AUDITS / "sensor_analysis.json", sensor_audit)

    # 5. Operating conditions (§10) — FD002 & FD004 (six regimes)
    opcond = {}
    for fd in ["FD002", "FD004"]:
        tr, te, _ = loaded[fd]
        opcond[f"{fd}_train"] = audit.operating_condition_analysis(tr)
        opcond[f"{fd}_test"] = audit.operating_condition_analysis(te)
    _write_json(AUDITS / "operating_conditions.json", opcond)

    # 6. Degradation + RUL (§11/§12)
    degradation = {}
    for fd in ALL_FD:
        tr, _, _ = loaded[fd]
        degradation[fd] = audit.degradation_and_rul(tr)
    _write_json(AUDITS / "degradation_rul.json", degradation)

    # 7. Target design (§15)
    targets = {}
    for fd in ALL_FD:
        tr, _, _ = loaded[fd]
        targets[fd] = audit.target_horizon_simulation(tr, HORIZONS)
    _write_json(AUDITS / "target_design.json", targets)

    print("== done ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
