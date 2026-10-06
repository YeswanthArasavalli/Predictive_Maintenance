"""Phase 2.6 — Decision & Cost-Sensitive Validation: pure analysis core.

This module turns the *existing* Phase 2 / Phase 2.5 validation predictions
(`results/experiments/phase2_5/val_rows/*.parquet`) into decision-level,
cost-sensitive evidence. It performs NO model training, imports NO
neural/DL framework, and NEVER reads the official FD004 test partition
(`test_FD004.txt` / `RUL_FD004.txt`). Every function here is a deterministic
transformation of arrays that the caller has already loaded from the sealed
validation artifacts.

COST MODEL — READ THIS BEFORE INTERPRETING ANY NUMBER
------------------------------------------------------
There is no validated real-world cost model for this dataset. Every cost in
this phase is an *illustrative scenario assumption*, NOT observed financial
data. The primary framework (authorization prompt section 4) is deliberately
minimal and transparent::

    Expected Cost = FN * C_FN + FP * C_FP

`C_FP` is fixed to a unit value (`c_fp_unit`, default ``1.0``) so all costs are
expressed in "units of one false positive"; `C_FN = cost_ratio * c_fp_unit`.
Because only the RATIO ``C_FN / C_FP`` matters, this normalization changes no
ranking and fabricates no dollar amount. A positive ``expected_cost`` is a
relative, unitless scenario score — never a currency figure and never a claim
of operational savings.

References
----------
Confusion-matrix and prognostic conventions follow `src.evaluation.metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

# Every cost artifact must carry this exact label (verified by the gate).
ILLUSTRATIVE_COST_LABEL = (
    "illustrative / scenario assumption, not observed real-world financial data."
)


# ===========================================================================
# Section 4/6 — confusion accounting & per-threshold cost
# ===========================================================================


def confusion_counts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn) for hard labels ``y_prob >= threshold``.

    Positive class is 1 (imminent failure-risk event). A row predicted 1 while
    truth 1 is a true positive; predicted 1 while truth 0 is a false positive;
    predicted 0 while truth 1 is a false negative; predicted 0 while truth 0 is
    a true negative. Deterministic and order-independent.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    pred = (np.asarray(y_prob, dtype=np.float64) >= float(threshold)).astype(np.int8)
    tp = int(np.count_nonzero((pred == 1) & (y_true == 1)))
    fp = int(np.count_nonzero((pred == 1) & (y_true == 0)))
    fn = int(np.count_nonzero((pred == 0) & (y_true == 1)))
    tn = int(np.count_nonzero((pred == 0) & (y_true == 0)))
    return tp, fp, tn, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision/recall/F1 from counts, with 0.0 for empty denominators."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def expected_cost(fn: int, fp: int, c_fn: float, c_fp: float) -> float:
    """Primary illustrative decision cost: ``FN*C_FN + FP*C_FP``."""
    return float(fn) * float(c_fn) + float(fp) * float(c_fp)



# ===========================================================================
# Section 6/7/8 — threshold x cost-ratio tables (illustrative)
# ===========================================================================


def build_threshold_cost_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    model_id: str,
    horizon: int,
    thresholds: Sequence[float],
    cost_ratios: Sequence[float],
    c_fp_unit: float = 1.0,
) -> pd.DataFrame:
    """Long-format table: one row per (threshold x cost_ratio) for one model.

    Columns: model_id, horizon, threshold, tp, fp, tn, fn, precision, recall,
    f1, cost_ratio, c_fn, c_fp, expected_cost. Every ``expected_cost`` is an
    illustrative scenario score in units of ``c_fp_unit`` (one false positive).
    """
    rows: list[dict[str, object]] = []
    for thr in thresholds:
        tp, fp, tn, fn = confusion_counts(y_true, y_prob, thr)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        for ratio in cost_ratios:
            c_fn = float(ratio) * float(c_fp_unit)
            rows.append({
                "model_id": model_id,
                "horizon": int(horizon),
                "threshold": float(thr),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cost_ratio": float(ratio),
                "c_fn": c_fn,
                "c_fp": float(c_fp_unit),
                "expected_cost": expected_cost(fn, fp, c_fn, c_fp_unit),
            })
    return pd.DataFrame(rows)


def build_cost_ratio_sensitivity(
    threshold_cost: pd.DataFrame,
) -> pd.DataFrame:
    """Per (model_id, horizon, cost_ratio): the min-cost threshold on the grid.

    Adds ``best_threshold`` (grid threshold minimizing expected cost; ties
    broken toward the lower threshold for determinism), the cost there, and the
    confusion metrics at that operating point.
    """
    out: list[dict[str, object]] = []
    for (model_id, horizon, ratio), grp in threshold_cost.groupby(
        ["model_id", "horizon", "cost_ratio"], sort=True
    ):
        # Deterministic argmin: lowest cost, then lowest threshold.
        g = grp.sort_values(["expected_cost", "threshold"])
        best = g.iloc[0]
        out.append({
            "model_id": model_id,
            "horizon": int(horizon),
            "cost_ratio": float(ratio),
            "best_threshold": float(best["threshold"]),
            "best_expected_cost": float(best["expected_cost"]),
            "tp": int(best["tp"]), "fp": int(best["fp"]),
            "tn": int(best["tn"]), "fn": int(best["fn"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
        })
    return pd.DataFrame(out).sort_values(
        ["horizon", "cost_ratio", "model_id"]
    ).reset_index(drop=True)


def build_decision_cost_comparison(
    sensitivity: pd.DataFrame,
    *,
    anchor_id: str,
    temporal_id: str,
) -> pd.DataFrame:
    """Per (horizon, cost_ratio): anchor vs temporal best-case illustrative cost.

    ``abs_reduction = anchor_cost - temporal_cost`` (positive => temporal is
    cheaper under that scenario). ``pct_reduction`` is reported only when the
    anchor cost is strictly positive; otherwise it is ``nan`` (a 0/0 or 0-x
    case is not mathematically meaningful). ``lower_cost_model`` names the
    winner; a tie is labeled ``tie``.
    """
    rows: list[dict[str, object]] = []
    for (horizon, ratio), grp in sensitivity.groupby(["horizon", "cost_ratio"], sort=True):
        a = grp[grp["model_id"] == anchor_id]
        t = grp[grp["model_id"] == temporal_id]
        if a.empty or t.empty:
            continue
        a_cost = float(a["best_expected_cost"].iloc[0])
        t_cost = float(t["best_expected_cost"].iloc[0])
        diff = a_cost - t_cost
        pct = (diff / a_cost * 100.0) if a_cost > 0 else float("nan")
        if diff > 0:
            winner = "temporal"
        elif diff < 0:
            winner = "anchor"
        else:
            winner = "tie"
        rows.append({
            "horizon": int(horizon),
            "cost_ratio": float(ratio),
            "anchor_model_id": anchor_id,
            "temporal_model_id": temporal_id,
            "anchor_threshold": float(a["best_threshold"].iloc[0]),
            "anchor_cost": a_cost,
            "temporal_threshold": float(t["best_threshold"].iloc[0]),
            "temporal_cost": t_cost,
            "abs_reduction": diff,
            "pct_reduction": pct,
            "lower_cost_model": winner,
        })
    return pd.DataFrame(rows).sort_values(["horizon", "cost_ratio"]).reset_index(drop=True)


# ===========================================================================
# Section 10/11/12/13 — engine-level decision & alert analysis
# ===========================================================================


def _positive_runs(pred: np.ndarray) -> list[tuple[int, int]]:
    """Return maximal runs of consecutive ``pred==1`` as (start, end) index
    pairs (inclusive, 0-based) in the row order supplied (must be one engine's
    rows already sorted by cycle)."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, p in enumerate(pred):
        if p == 1 and start is None:
            start = i
        elif p == 0 and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(pred) - 1))
    return runs


def engine_level_counts(
    df: pd.DataFrame,
    *,
    model_id: str,
    threshold: float,
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    raw_rul_col: str | None = None,
) -> pd.DataFrame:
    """Per-engine decision accounting at a fixed threshold.

    Each input row is one validation observation belonging to ``unit_col``.
    Rows within an engine are temporally correlated, so all quantities here are
    *per-engine descriptive aggregates*, not independent events.

    Definition of every column (documented per section 12 — no rule is invented
    silently):
      * ``n_rows``            rows in the engine's validation trajectory.
      * ``n_positive_rows``   rows whose ``y_true == 1`` (in the H-horizon risk
                              window).
      * ``n_alerts``          rows predicted positive (``y_prob >= threshold``).
      * ``tp/fp/tn/fn``       confusion counts within the engine.
      * ``ever_detected``     1 if the engine has >=1 true positive (a risk row
                              was flagged); else 0.
      * ``ever_missed``       1 if the engine has >=1 false negative (a risk row
                              was NOT flagged); else 0.
      * ``earliest_detection_cycle``  the smallest cycle at which a TP occurs
                              (first cycle the engine was correctly alerted while
                              genuinely in risk); ``nan`` if never detected.
      * ``n_positive_runs``   count of maximal consecutive predicted-positive
                              runs in the engine (section 13, descriptive).
      * ``n_fp_runs``         positive runs containing zero true positives
                              (a run that is entirely false alarms).
      * ``fn_raw_rul_min/max/mean``  raw RUL distribution among that engine's
                              false negatives (only when ``raw_rul_col`` is
                              supplied); else nan.
    """
    pred_all = (df[y_prob_col].to_numpy(dtype=np.float64) >= float(threshold)).astype(np.int8)
    yt_all = df[y_true_col].to_numpy().astype(np.int8)
    cyc_all = df[cycle_col].to_numpy()
    rul_all = df[raw_rul_col].to_numpy(dtype=np.float64) if raw_rul_col else None

    rows: list[dict[str, object]] = []
    for uid, idx in df.groupby(unit_col, sort=True).indices.items():
        idx = np.asarray(idx)
        yt = yt_all[idx]
        pr = pred_all[idx]
        cyc = cyc_all[idx]
        tp = int(np.count_nonzero((pr == 1) & (yt == 1)))
        fp = int(np.count_nonzero((pr == 1) & (yt == 0)))
        tn = int(np.count_nonzero((pr == 0) & (yt == 0)))
        fn = int(np.count_nonzero((pr == 0) & (yt == 1)))
        # Sort this engine's rows by cycle before computing runs / earliest TP.
        order = np.argsort(cyc, kind="stable")
        cyc_sorted = cyc[order]
        yt_sorted = yt[order]
        pr_sorted = pr[order]
        runs = _positive_runs(pr_sorted)
        n_fp_runs = 0
        for s, e in runs:
            if np.count_nonzero(yt_sorted[s:e + 1]) == 0:
                n_fp_runs += 1
        tp_cycles = cyc_sorted[(pr_sorted == 1) & (yt_sorted == 1)]
        earliest = float(tp_cycles.min()) if tp_cycles.size else float("nan")
        d: dict[str, object] = {
            "model_id": model_id,
            "unit_id": int(uid),
            "n_rows": int(len(idx)),
            "n_positive_rows": int(np.count_nonzero(yt == 1)),
            "n_alerts": int(np.count_nonzero(pr == 1)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "ever_detected": int(tp > 0),
            "ever_missed": int(fn > 0),
            "earliest_detection_cycle": earliest,
            "n_positive_runs": len(runs),
            "n_fp_runs": n_fp_runs,
        }
        if rul_all is not None:
            fn_mask = (pr == 0) & (yt == 1)
            vals = rul_all[idx][fn_mask]
            d["fn_raw_rul_count"] = int(vals.size)
            d["fn_raw_rul_min"] = float(vals.min()) if vals.size else float("nan")
            d["fn_raw_rul_max"] = float(vals.max()) if vals.size else float("nan")
            d["fn_raw_rul_mean"] = float(vals.mean()) if vals.size else float("nan")
        rows.append(d)
    return pd.DataFrame(rows)


def summarize_engine_alert_fatigue(engine_counts: pd.DataFrame) -> dict[str, object]:
    """Descriptive alert-fatigue aggregates over per-engine counts.

    These quantities describe raw prediction behavior at the analysis threshold;
    they are NOT a production alerting/suppression policy (section 13)."""
    return {
        "n_engines": int(engine_counts["unit_id"].nunique()),
        "total_alert_rows": int(engine_counts["n_alerts"].sum()),
        "total_false_positive_rows": int(engine_counts["fp"].sum()),
        "total_positive_runs": int(engine_counts["n_positive_runs"].sum()),
        "total_fp_runs": int(engine_counts["n_fp_runs"].sum()),
        "mean_positive_runs_per_engine": float(engine_counts["n_positive_runs"].mean()),
        "mean_fp_runs_per_engine": float(engine_counts["n_fp_runs"].mean()),
        "engines_ever_detected": int(engine_counts["ever_detected"].sum()),
        "engines_ever_missed": int(engine_counts["ever_missed"].sum()),
    }


def false_negative_profile(
    df: pd.DataFrame,
    *,
    threshold: float,
    raw_rul_col: str,
    unit_col: str = "unit_id",
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
) -> dict[str, object]:
    """Row- and engine-level profile of false negatives at a threshold.

    Reports the count of missed positive rows, the raw-RUL distribution among
    them, and how many distinct engines contribute them. Row counts are NOT
    independent failure events (engine correlation is preserved).
    """
    yt = df[y_true_col].to_numpy().astype(np.int8)
    pr = (df[y_prob_col].to_numpy(dtype=np.float64) >= float(threshold)).astype(np.int8)
    fn_mask = (pr == 0) & (yt == 1)
    rul = df[raw_rul_col].to_numpy(dtype=np.float64)[fn_mask]
    units = df[unit_col].to_numpy()[fn_mask]
    if rul.size == 0:
        return {
            "threshold": float(threshold),
            "n_fn_rows": 0,
            "n_engines_with_fn": 0,
            "raw_rul_min": float("nan"),
            "raw_rul_max": float("nan"),
            "raw_rul_mean": float("nan"),
            "raw_rul_median": float("nan"),
            "raw_rul_p90": float("nan"),
        }
    return {
        "threshold": float(threshold),
        "n_fn_rows": int(rul.size),
        "n_engines_with_fn": int(len(set(int(u) for u in units))),
        "raw_rul_min": float(rul.min()),
        "raw_rul_max": float(rul.max()),
        "raw_rul_mean": float(rul.mean()),
        "raw_rul_median": float(np.median(rul)),
        "raw_rul_p90": float(np.percentile(rul, 90)),
    }


# ===========================================================================
# Section 16 — engine-level bootstrap of the cost difference
# ===========================================================================


@dataclass
class BootstrapResult:
    cost_ratio: float
    threshold: float
    n_resamples: int
    seed: int
    mean_cost_diff: float
    ci_low: float
    ci_high: float
    prob_temporal_lower: float


def bootstrap_cost_difference(
    engine_anchor: pd.DataFrame,
    engine_temporal: pd.DataFrame,
    *,
    cost_ratios: Sequence[float],
    threshold: float,
    c_fp_unit: float = 1.0,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Engine-level bootstrap CI for ``temporal_cost - anchor_cost``.

    Resampling is at the ENGINE level (rows within an engine are correlated, so
    row resampling would understate variance). For each ratio, per-engine cost
    is ``c_fn*fn_i + c_fp*fp_i`` with ``c_fn = ratio*c_fp_unit``. The engines
    are resampled with replacement ``n_resamples`` times using a deterministic
    ``numpy.random.default_rng(seed)``. The reported difference is
    ``sum(temporal) - sum(anchor)`` on each resample; ``ci_low``/``ci_high`` are
    the 2.5/97.5 percentiles. ``prob_temporal_lower`` is the fraction of
    resamples where temporal cost < anchor cost (diff < 0).

    ``engine_anchor`` and ``engine_temporal`` must be ``engine_level_counts``
    outputs sharing the same ``unit_id`` set; they are aligned on unit_id.
    """
    a = engine_anchor.sort_values("unit_id").reset_index(drop=True)
    t = engine_temporal.sort_values("unit_id").reset_index(drop=True)
    if not np.array_equal(a["unit_id"].to_numpy(), t["unit_id"].to_numpy()):
        raise ValueError("engine_anchor and engine_temporal must cover the same units")
    n_engines = len(a)
    fn_a = a["fn"].to_numpy(dtype=np.float64)
    fp_a = a["fp"].to_numpy(dtype=np.float64)
    fn_t = t["fn"].to_numpy(dtype=np.float64)
    fp_t = t["fp"].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    # One shared index stream across ratios keeps the comparison paired.
    idx_streams = rng.integers(0, n_engines, size=(int(n_resamples), n_engines))

    rows: list[dict[str, object]] = []
    for ratio in cost_ratios:
        c_fn = float(ratio) * float(c_fp_unit)
        cost_a = c_fn * fn_a + float(c_fp_unit) * fp_a
        cost_t = c_fn * fn_t + float(c_fp_unit) * fp_t
        diffs = cost_t[idx_streams].sum(axis=1) - cost_a[idx_streams].sum(axis=1)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({
            "cost_ratio": float(ratio),
            "threshold": float(threshold),
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "resample_level": "engine",
            "mean_cost_diff": float(diffs.mean()),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "prob_temporal_lower": float((diffs < 0).mean()),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Section 9 — RUL decision analysis (no binary cost forcing)
# ===========================================================================


def rul_decision_profile(
    df: pd.DataFrame,
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    raw_rul_col: str = "raw_RUL",
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
    end_of_life_rul: float = 10.0,
) -> dict[str, object]:
    """Row-level RUL diagnostics used for the decision (non-cost) analysis.

    Returns MAE, worst-case absolute errors (p90/p95/p99/max), the mean signed
    residual overall and split by engine lifespan (short vs long, defined by a
    median split of each engine's maximum observed cycle — documented, not
    arbitrary), and the end-of-life MAE restricted to rows with
    ``raw_RUL <= end_of_life_rul``. No monetary value is implied.
    """
    y_true = df[y_true_col].to_numpy(dtype=np.float64)
    y_pred = df[y_pred_col].to_numpy(dtype=np.float64)
    resid = y_pred - y_true
    abs_err = np.abs(resid)
    raw_rul = df[raw_rul_col].to_numpy(dtype=np.float64)

    # Engine lifespan = the engine's last observed cycle within this partition.
    lifespan = df.groupby(unit_col)[cycle_col].max()
    median_life = float(np.median(lifespan.to_numpy()))
    long_engines = set(int(u) for u in lifespan[lifespan > median_life].index)
    units = df[unit_col].to_numpy()
    is_long = np.array([int(u) in long_engines for u in units])

    eol_mask = raw_rul <= end_of_life_rul
    out: dict[str, object] = {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "worst_abs_p90": float(np.percentile(abs_err, 90)),
        "worst_abs_p95": float(np.percentile(abs_err, 95)),
        "worst_abs_p99": float(np.percentile(abs_err, 99)),
        "worst_abs_max": float(abs_err.max()),
        "mean_signed_residual": float(resid.mean()),
        "mean_signed_residual_short_life": (
            float(resid[~is_long].mean()) if np.any(~is_long) else float("nan")
        ),
        "mean_signed_residual_long_life": (
            float(resid[is_long].mean()) if np.any(is_long) else float("nan")
        ),
        "end_of_life_mae": float(abs_err[eol_mask].mean()) if np.any(eol_mask) else float("nan"),
        "end_of_life_rows": int(eol_mask.sum()),
        "lifespan_median_cycle": median_life,
        "n_long_life_engines": int(len(long_engines)),
    }
    return out
