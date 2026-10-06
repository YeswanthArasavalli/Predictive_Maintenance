"""Phase-0 FINAL GATE analysis: computes evidence for Conditions 1, 2, 3, 5.

Run:  python scripts/phase0_gate_analysis.py
Writes results/audits/gate_fd004_discrepancy.json
      results/audits/gate_rul_cap.json
      results/audits/gate_variance_decomposition.json
      results/audits/gate_horizon_detail.json

No models are trained; nothing is written to data/raw.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_dataset, load_rul  # noqa: E402
from src.data.loader import CYCLE, UNIT_ID, OPS_COLS, derive_train_rul, raw_dir  # noqa: E402
from src.data.audit import SENSORS, _regime_key  # noqa: E402

OUT = ROOT / "results" / "audits"
OUT.mkdir(parents=True, exist_ok=True)
PRIMARY = "FD004"
HORIZONS = [14, 30, 50]
CAPS = [125, 100]


def _write(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print("  wrote", name)


# --------------------------------------------------------------------------- #
# Condition 1 — FD004 train/test discrepancy
# --------------------------------------------------------------------------- #
def fd004_discrepancy() -> dict:
    tr = load_dataset("FD004", "train")
    te = load_dataset("FD004", "test")
    ru = load_rul("FD004")

    train_units = sorted(int(u) for u in tr[UNIT_ID].unique())
    test_units = sorted(int(u) for u in te[UNIT_ID].unique())

    # Does the 249th training engine look like a complete run-to-failure trajectory?
    g249 = tr[tr[UNIT_ID] == 249].sort_values(CYCLE)
    cyc249 = g249[CYCLE].to_numpy()
    engine249 = {
        "present_in_train": bool(249 in train_units),
        "n_cycles": int(len(g249)),
        "cycles_consecutive_1n": bool(np.array_equal(cyc249, np.arange(1, len(cyc249) + 1))),
        "first_cycle": int(cyc249.min()),
        "last_cycle": int(cyc249.max()),
        "ends_at_zero_rul": bool(int(derive_train_rul(g249).iloc[-1]) == 0),
        "has_all_26_columns": bool(g249.shape[1] == 26),
        "no_nulls": bool(g249[OPS_COLS + SENSORS].isna().sum().sum() == 0),
    }

    # Train vs test behavioural signature (are the FILES swapped, or just the readme numbers?)
    train_rul_at_end = [int(derive_train_rul(g).iloc[-1]) for _, g in tr.groupby(UNIT_ID, sort=False)]
    test_lens = te.groupby(UNIT_ID).size()
    train_lens = tr.groupby(UNIT_ID).size()

    # RUL file facts
    rul_values = ru["rul"].to_numpy()
    test_last_cycle = te.groupby(UNIT_ID)[CYCLE].max()
    # A test engine's implied failure cycle = last observed cycle + supplied RUL.
    aligned = pd.DataFrame({
        "last_cycle": test_last_cycle,
        "rul": ru["rul"],
    }).dropna()
    implied_fail = (aligned["last_cycle"] + aligned["rul"])

    return {
        "readme_stated": {"train": 248, "test": 249},
        "observed": {
            "train_units": len(train_units),
            "test_units": len(test_units),
            "train_unit_range": [train_units[0], train_units[-1]],
            "test_unit_range": [test_units[0], test_units[-1]],
            "train_units_contiguous_1n": train_units == list(range(1, len(train_units) + 1)),
            "test_units_contiguous_1n": test_units == list(range(1, len(test_units) + 1)),
        },
        "rul_file": {
            "n_values": int(len(rul_values)),
            "min": int(rul_values.min()),
            "max": int(rul_values.max()),
            "all_positive": bool((rul_values > 0).all()),
            "one_rul_per_test_engine": bool(len(rul_values) == len(test_units)),
            "test_ids_match_rul_ids": bool(test_units == sorted(int(u) for u in ru.index)),
        },
        "engine_249": engine249,
        "behavioural_signature": {
            "train_engines_all_end_at_rul0": bool(all(v == 0 for v in train_rul_at_end)),
            "train_mean_length": round(float(train_lens.mean()), 1),
            "test_mean_length": round(float(test_lens.mean()), 1),
            "test_shorter_than_train_on_avg": bool(test_lens.mean() < train_lens.mean()),
            "implied_failure_cycle_min": int(implied_fail.min()),
            "implied_failure_cycle_max": int(implied_fail.max()),
        },
        "interpretation": (
            "train file holds full run-to-failure trajectories (every engine ends at "
            "RUL=0); test file holds truncated trajectories whose count matches the 248 "
            "RUL values one-to-one. The FILES are internally consistent as train=249/"
            "test=248; only the readme.txt two numbers are transposed."
        ),
    }


# --------------------------------------------------------------------------- #
# Condition 2 — RUL raw vs clipped target
# --------------------------------------------------------------------------- #
def rul_cap_analysis() -> dict:
    out = {}
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        tr = load_dataset(fd, "train")
        rul = derive_train_rul(tr).to_numpy()
        n = len(rul)
        lives = tr.groupby(UNIT_ID).size().to_numpy()
        entry = {
            "n_train_rows": int(n),
            "n_engines": int(len(lives)),
            "raw_rul_min": int(rul.min()),
            "raw_rul_max": int(rul.max()),
            "raw_rul_mean": round(float(rul.mean()), 2),
            "raw_rul_median": float(np.median(rul)),
            "raw_rul_q90": float(np.quantile(rul, 0.90)),
            "raw_rul_q99": float(np.quantile(rul, 0.99)),
            "pct_rows_rul_gt_125": round(float((rul > 125).mean() * 100), 2),
            "pct_rows_rul_gt_100": round(float((rul > 100).mean() * 100), 2),
            "pct_rows_rul_gt_50": round(float((rul > 50).mean() * 100), 2),
            "caps": {},
        }
        for cap in CAPS:
            clipped = np.minimum(rul, cap)
            entry["caps"][cap] = {
                "pct_observations_clipped": round(float((rul > cap).mean() * 100), 2),
                "n_observations_clipped": int((rul > cap).sum()),
                "engines_with_any_clipped_row": int(((lives - 1) > cap).sum()),
                "clipped_target_mean": round(float(clipped.mean()), 2),
                "clipped_target_std": round(float(clipped.std(ddof=1)), 2),
                "raw_target_std": round(float(rul.std(ddof=1)), 2),
                "rows_at_cap_mass_pct": round(float((clipped == cap).mean() * 100), 2),
            }
        out[fd] = entry
    return out


# --------------------------------------------------------------------------- #
# Condition 3 — variance decomposition audit
# --------------------------------------------------------------------------- #
def _between_total_fraction(values: np.ndarray, groups: np.ndarray) -> float:
    """ANOVA-style: fraction of a single sensor's total variance that is BETWEEN groups."""
    total = float(np.var(values, ddof=1))
    if total == 0:
        return 0.0
    grand = float(np.mean(values))
    df = pd.DataFrame({"v": values, "g": groups})
    between = 0.0
    for _, grp in df.groupby("g"):
        between += len(grp) * (float(grp["v"].mean()) - grand) ** 2
    between_var = between / (len(values) - 1)
    return between_var / total


def variance_decomposition() -> dict:
    out = {}
    for fd in ["FD002", "FD004"]:
        tr = load_dataset(fd, "train")
        groups = _regime_key(tr).to_numpy()
        per_sensor = {}
        for s in SENSORS:
            v = tr[s].to_numpy(dtype=float)
            if np.std(v) == 0:
                per_sensor[s] = None  # constant sensor: undefined
                continue
            per_sensor[s] = round(_between_total_fraction(v, groups), 4)
        valid = {k: v for k, v in per_sensor.items() if v is not None}
        fr = np.array(list(valid.values()))

        # Aggregate across sensors: RAW sum-of-variances (scale-distorted) vs
        # STANDARDIZED (each sensor unit-variance before pooling).
        raw_between_sum = 0.0
        raw_total_sum = 0.0
        std_between_sum = 0.0
        std_total_sum = 0.0
        for s in valid:
            v = tr[s].to_numpy(dtype=float)
            tot = float(np.var(v, ddof=1))
            bet = tot * valid[s]
            raw_between_sum += bet
            raw_total_sum += tot
            # standardized: divide sensor variance by itself -> between frac unchanged,
            # but each sensor now contributes equal weight (1.0 total variance)
            std_between_sum += valid[s]
            std_total_sum += 1.0
        out[fd] = {
            "n_regimes": int(len(np.unique(groups))),
            "per_sensor_between_fraction": per_sensor,
            "sensors_excluded_constant": [k for k, v in per_sensor.items() if v is None],
            "distribution_of_between_fraction": {
                "n_sensors": int(len(fr)),
                "min": float(fr.min()),
                "median": float(np.median(fr)),
                "max": float(fr.max()),
                "mean": round(float(fr.mean()), 4),
                "pct_sensors_above_0_95": round(float((fr > 0.95).mean() * 100), 1),
            },
            "aggregate_between_fraction_raw_scale": round(raw_between_sum / raw_total_sum, 4),
            "aggregate_between_fraction_standardized": round(std_between_sum / std_total_sum, 4),
            "representative_sensors_from_phase0": {
                s: valid.get(s) for s in ["sensor_02", "sensor_17", "sensor_21"] if s in valid
            },
        }
    return out


# --------------------------------------------------------------------------- #
# Condition 5 — failure-risk horizon detail
# --------------------------------------------------------------------------- #
def horizon_detail() -> dict:
    out = {}
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        tr = load_dataset(fd, "train")
        rul = derive_train_rul(tr)
        n_rows = len(rul)
        work = tr[[UNIT_ID]].assign(_rul=rul.to_numpy())
        entry = {"n_rows": int(n_rows), "n_engines": int(tr[UNIT_ID].nunique()), "horizons": {}}
        for h in HORIZONS:
            pos = int((work["_rul"] <= h).sum())
            neg = n_rows - pos
            # per-engine consecutive positive run statistics
            max_runs, sum_runs, engines_pos, engines_neg = [], [], 0, 0
            earliest_offsets = []
            for _, g in work.groupby(UNIT_ID, sort=False):
                mask = (g["_rul"].to_numpy() <= h)
                if mask.any():
                    engines_pos += 1
                    # longest consecutive run of positives
                    best = cur = 0
                    first_idx = None
                    for i, m in enumerate(mask):
                        cur = cur + 1 if m else 0
                        best = max(best, cur)
                        if m and first_idx is None:
                            first_idx = i
                    max_runs.append(best)
                    sum_runs.append(int(mask.sum()))
                    # earliest positive label = its RUL value (cycles before failure)
                    earliest_offsets.append(int(g["_rul"].to_numpy()[first_idx]))
                else:
                    engines_neg += 1
            entry["horizons"][h] = {
                "candidate_windows_rowlevel": n_rows,
                "positive_windows": pos,
                "negative_windows": neg,
                "positive_pct": round(pos / n_rows * 100, 2),
                "positive_engines": engines_pos,
                "negative_engines": engines_neg,
                "pos_windows_per_engine_min": int(min(sum_runs)) if sum_runs else 0,
                "pos_windows_per_engine_max": int(max(sum_runs)) if sum_runs else 0,
                "pos_windows_per_engine_mean": round(float(np.mean(sum_runs)), 2) if sum_runs else 0,
                "max_consecutive_positive_windows": int(max(max_runs)) if max_runs else 0,
                "avg_consecutive_run_length": round(float(np.mean(max_runs)), 2) if max_runs else 0,
                "earliest_positive_label_rul": int(max(earliest_offsets)) if earliest_offsets else None,
                "label_overlap_ratio": round(float(np.mean(max_runs) / (h + 1)), 3) if max_runs else None,
                "every_engine_has_positives": bool(engines_neg == 0),
            }
        out[fd] = entry
    return out


def main() -> int:
    print("== Phase 0 gate analysis ==")
    _write("gate_fd004_discrepancy.json", fd004_discrepancy())
    _write("gate_rul_cap.json", rul_cap_analysis())
    _write("gate_variance_decomposition.json", variance_decomposition())
    _write("gate_horizon_detail.json", horizon_detail())
    print("== done ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
