"""Read-only data-access layer for the Streamlit portfolio demo.

Every value shown in the app comes from the small, committed, public-safe demo
bundle in `demo/`, which `scripts/build_demo_bundle.py` derived from the frozen
`results/` validation artifacts.

Hard guarantees enforced here:

* The app never touches `data/raw/`, the sealed official test partition
  (`test_FD004.txt` / `RUL_FD004.txt`), or anything outside `demo/`.
* The app performs NO training, tuning or re-metric computation — only lightweight
  display transforms (filtering by engine, recomputing a predicted state from a
  stored probability against a chosen threshold).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo"
DECISION = DEMO / "decision"

PROVENANCE = "Derived from frozen validation artifacts. No official test partition used."

# Substrings that must NEVER appear in any path the demo app reads.
FORBIDDEN_PATH_TOKENS = (
    "CMAPSSData",
    "data/raw",
    "test_FD004",
    "RUL_FD004",
    "Damage Propagation",
)


class MissingArtifact(Exception):
    """Raised when a required demo-bundle file is unavailable."""


# Frozen prediction artifacts available for the interactive lab. Each maps to a
# committed validation-prediction file (anchor + causal-temporal, no new models).
MODEL_RUL_FILES = {
    "anchor": "demo/rul_anchor_predictions.parquet",
    "temporal": "demo/rul_temporal_predictions.parquet",
}
MODEL_FR_FILES = {
    "anchor": "demo/fr_anchor_predictions.parquet",
    "temporal": "demo/fr_temporal_predictions.parquet",
}
MODEL_LABELS = {
    "anchor": "Classical anchor (cycle, no temporal)",
    "temporal": "Causal temporal (combined, W=5)",
}
MODEL_KEYS = ("temporal", "anchor")


def _check_model(model: str) -> None:
    if model not in MODEL_KEYS:
        raise ValueError(f"unknown frozen prediction view {model!r}; expected {MODEL_KEYS}")


def _resolve(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    s = str(p).replace("\\", "/")
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok.lower() in s.lower():
            raise ValueError(f"Refusing to access forbidden path containing {tok!r}")
    return p


def _require(p: Path) -> Path:
    if not p.exists():
        raise MissingArtifact(str(p.relative_to(ROOT)))
    return p


def bundle_exists() -> bool:
    return DEMO.is_dir() and (DEMO / "manifest.json").exists()


def load_json(rel: str) -> dict:
    p = _require(_resolve(rel))
    return json.loads(p.read_text(encoding="utf-8"))


def load_overview() -> dict:
    return load_json("demo/overview.json")


def load_metrics() -> dict:
    return load_json("demo/metrics.json")


def load_rul_predictions(model: str = "temporal") -> pd.DataFrame:
    _check_model(model)
    return pd.read_parquet(_require(_resolve(MODEL_RUL_FILES[model])))


def load_fr_predictions(model: str = "temporal") -> pd.DataFrame:
    _check_model(model)
    return pd.read_parquet(_require(_resolve(MODEL_FR_FILES[model])))


def load_decision_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(_require(_resolve(f"demo/decision/{name}")))


def load_decision_json(name: str) -> dict:
    return load_json(f"demo/decision/{name}")


def validation_engine_ids(df: pd.DataFrame) -> list[int]:
    return sorted(int(u) for u in df["unit_id"].unique())


def engine_slice_predictions(df: pd.DataFrame, unit_id: int) -> pd.DataFrame:
    """Filter the cached validation predictions to one engine (display-only)."""
    return df[df["unit_id"] == int(unit_id)].sort_values("cycle").reset_index(drop=True)


def predicted_state_from_threshold(fr_engine: pd.DataFrame, threshold: float) -> pd.Series:
    """Recompute the binary predicted failure state at a chosen threshold.

    This is a pure display transform over a stored probability; it is NOT a
    re-prediction or retraining.
    """
    return (fr_engine["failure_prob"] >= threshold).astype(int)


def rul_engine_summary(engine_df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(engine_df)),
        "cycle_min": int(engine_df["cycle"].min()),
        "cycle_max": int(engine_df["cycle"].max()),
        "mae": float((engine_df["actual_rul"] - engine_df["predicted_rul"]).abs().mean()),
    }


# --------------------------------------------------------------------------
# Interactive Prediction Lab helpers (deterministic transforms over frozen
# predictions only — no training, no re-metric of the scientific results).
# --------------------------------------------------------------------------
def cycle_options(engine_df: pd.DataFrame) -> list[int]:
    return sorted(int(c) for c in engine_df["cycle"].unique())


def row_at_cycle(engine_df: pd.DataFrame, cycle: int) -> dict:
    """Return the single frozen prediction row for an engine at a given cycle."""
    row = engine_df[engine_df["cycle"] == int(cycle)]
    if row.empty:
        raise MissingArtifact(f"no row for cycle {cycle}")
    return row.iloc[0].to_dict()


def rul_point(rul_engine: pd.DataFrame, cycle: int) -> dict:
    r = row_at_cycle(rul_engine, cycle)
    actual = float(r["actual_rul"])
    pred = float(r["predicted_rul"])
    return {
        "cycle": int(cycle),
        "actual_rul": actual,
        "predicted_rul": pred,
        "signed_error": pred - actual,
        "abs_error": abs(pred - actual),
    }


def fr_point(fr_engine: pd.DataFrame, cycle: int, threshold: float) -> dict:
    r = row_at_cycle(fr_engine, cycle)
    prob = float(r["failure_prob"])
    predicted_state = int(prob >= threshold)
    actual_state = int(r["actual_state"])
    return {
        "cycle": int(cycle),
        "failure_prob": prob,
        "threshold": float(threshold),
        "predicted_state": predicted_state,
        "actual_state": actual_state,
        "correct": predicted_state == actual_state,
    }


def threshold_metrics(fr_df: pd.DataFrame, threshold: float) -> dict:
    """Deterministic precision/recall/F1 + confusion counts for the displayed
    prediction set at a chosen threshold. Pure transform of stored probabilities
    and frozen actual labels."""
    pred = (fr_df["failure_prob"] >= threshold).astype(int)
    act = fr_df["actual_state"].astype(int)
    tp = int(((pred == 1) & (act == 1)).sum())
    fp = int(((pred == 1) & (act == 0)).sum())
    tn = int(((pred == 0) & (act == 0)).sum())
    fn = int(((pred == 0) & (act == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "predicted_positive": int(pred.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def engine_alert_count(fr_engine: pd.DataFrame, threshold: float) -> int:
    return int((fr_engine["failure_prob"] >= threshold).sum())


def engine_false_alerts(fr_engine: pd.DataFrame, threshold: float) -> int:
    pred = (fr_engine["failure_prob"] >= threshold).astype(int)
    return int(((pred == 1) & (fr_engine["actual_state"].astype(int) == 0)).sum())


def _final_cycle_rul(rul_df: pd.DataFrame) -> pd.DataFrame:
    idx = rul_df.groupby("unit_id")["cycle"].idxmax()
    return rul_df.loc[idx, ["unit_id", "cycle", "actual_rul", "predicted_rul"]]


def _cycle_near_threshold(fr_engine: pd.DataFrame, target: float = 0.5) -> int:
    """Pick the engine's cycle whose failure probability is closest to `target`
    (an instructive point for the threshold slider). Falls back to max cycle."""
    if fr_engine.empty:
        return 0
    d = (fr_engine["failure_prob"] - target).abs()
    return int(fr_engine.loc[d.idxmin(), "cycle"])


def derive_examples(rul_df: pd.DataFrame, fr_df: pd.DataFrame) -> list[dict]:
    """Recommended validation engines, derived (and therefore verified) from the
    frozen predictions themselves — nothing is invented."""
    finals = _final_cycle_rul(rul_df).set_index("unit_id")
    fr_by_engine = {u: g for u, g in fr_df.groupby("unit_id")}

    def example(label: str, unit: int, note: str) -> dict:
        cyc = _cycle_near_threshold(fr_by_engine.get(unit, pd.DataFrame()), 0.5)
        return {"label": label, "unit_id": int(unit), "cycle": cyc, "note": note}

    out: list[dict] = []
    if finals.empty:
        return out
    med_target = float(finals["cycle"].median())

    fp_rows = (
        fr_df[(fr_df["failure_prob"] >= 0.5) & (fr_df["actual_state"].astype(int) == 0)]
        .groupby("unit_id")
        .size()
        .sort_values(ascending=False)
    )
    fn_rows = (
        fr_df[(fr_df["failure_prob"] < 0.5) & (fr_df["actual_state"].astype(int) == 1)]
        .groupby("unit_id")
        .size()
        .sort_values(ascending=False)
    )

    all_units = list(finals.index)
    short_order = list(finals["cycle"].sort_values().index)
    long_order = list(finals["cycle"].sort_values(ascending=False).index)
    rep_order = list((finals["cycle"] - med_target).abs().sort_values().index)
    fp_order = [int(u) for u in fp_rows.index] + [u for u in all_units]
    fn_order = [int(u) for u in fn_rows.index] + [u for u in all_units]

    used: set[int] = set()

    def pick(order):
        for u in order:
            if int(u) not in used:
                used.add(int(u))
                return int(u)
        return int(order[0])

    short = pick(short_order)
    long = pick(long_order)
    rep = pick(rep_order)
    high_fp = pick(fp_order)
    missed = pick(fn_order)

    out.append(example("Shorter-life engine", short, f"finals near cycle {int(finals.loc[short, 'cycle'])}"))
    out.append(example("Longer-life engine", long, f"runs to cycle {int(finals.loc[long, 'cycle'])}"))
    out.append(example("Representative engine", rep, f"mid-life, near {med_target:.0f}-cycle median"))
    out.append(example("High false-alarm engine", high_fp, "many low-risk cycles still cross 0.5"))
    out.append(example("Late / missed-risk engine", missed, "true at-risk cycles stay under 0.5"))
    return out
