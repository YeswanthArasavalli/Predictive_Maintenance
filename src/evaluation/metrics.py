"""Metric functions for Phase 2 baselines.

Task A (RUL):
    - MAE, RMSE, R² (sklearn defaults)
    - NASA C-MAPSS prognostic score (Saxena et al., 2008), "lower is better"

Task B (Failure Risk):
    - accuracy, precision, recall, F1 (zero_division handling explicit)
    - PR-AUC (average_precision) and ROC-AUC
    - confusion matrix
    - threshold sweep (precision/recall/F1 as a function of cut)

All metrics operate at the ROW level on the validation partition. There is
NO test-set metric function in this module — Phase 2 must never compute one.

Prognostic Score Reference:
    A. Saxena, K. Goebel, D. Simon, N. Eklund,
    "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Scenario",
    International Journal of Prognostics and Health Management, 2008.

Formula (per prediction):
    E = predicted_RUL - actual_RUL
    s(E) = exp(-E / 13) - 1     if E < 0    (under-prediction, less severe)
    s(E) = exp(E  / 10) - 1     if E >= 0   (over-prediction, operationally dangerous)

Total score = sum_i s(E_i). LOWER IS BETTER. A perfect model scores 0.

The published Saxena formulation scores one RUL prediction per engine at
its final observed cycle. Phase 2 evaluates on the FD004 TRAIN partition
(train/validation engines from the Phase 1 split) at row level, so we report
TWO variants:
    prognostic_score_rows      : sum of s(E) over every validation row (a
                                  direct row-level extension of the formula;
                                  row-weighted, rows are not independent)
    prognostic_score_last_per_engine : Saxena prognostic score computed on
                                  one prediction per validation engine,
                                  using that engine's final observed
                                  validation row. This is the nearest
                                  analogue of the single-prediction-per-
                                  engine scoring on the FD004 training
                                  partition; it is NOT an official-test
                                  score and does not imply the 125-cap
                                  convention.

Neither variant touches the official test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Task A — RUL
# ---------------------------------------------------------------------------


def _prognostic_penalty(errors: np.ndarray) -> np.ndarray:
    """Elementwise Saxena 2008 penalty for RUL errors E = pred - actual.

    The published formula is applied unchanged. To keep the aggregate finite
    under float64 when a baseline catastrophically over-predicts (|E| > 700
    causes np.exp(E/10) to overflow to inf), we cap the EXPONENT ARGUMENT at
    +/- 500 before exponentiating. This preserves monotonicity in E and the
    published s1=13 / s2=10 asymmetry; it only prevents `inf` at the extreme
    tail where the model is already unrecoverably wrong.
    """
    errors = np.asarray(errors, dtype=np.float64)
    out = np.zeros_like(errors)
    under = errors < 0
    over = ~under
    out[under] = np.exp(np.clip(-errors[under] / 13.0, -500.0, 500.0)) - 1.0
    out[over] = np.exp(np.clip(errors[over] / 10.0, -500.0, 500.0)) - 1.0
    return out


def prognostic_score_rows(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sum of Saxena 2008 penalties over every evaluated row. LOWER IS BETTER."""
    errors = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(_prognostic_penalty(errors).sum())


def prognostic_score_last_per_engine(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit_ids: np.ndarray,
) -> float:
    """Saxena prognostic score computed on one prediction per validation
    engine, using each engine's final observed validation row.

    LOWER IS BETTER. This is the nearest analogue of Saxena-style
    single-prediction-per-engine scoring, applied to the FD004 training
    partition; it is not an official-test score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    unit_ids = np.asarray(unit_ids)
    last_idx: dict[int, int] = {}
    for i, uid in enumerate(unit_ids):
        last_idx[int(uid)] = i  # overwrite => keep the last occurrence
    idx = np.array(sorted(last_idx.values()), dtype=np.int64)
    errors = y_pred[idx] - y_true[idx]
    return float(_prognostic_penalty(errors).sum())


def rul_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """Full regression metric pack, including both prognostic-score variants
    when unit_ids are supplied."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out: dict[str, float] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "prognostic_score_rows": prognostic_score_rows(y_true, y_pred),
    }
    if unit_ids is not None:
        out["prognostic_score_last_per_engine"] = prognostic_score_last_per_engine(
            y_true, y_pred, unit_ids
        )
    return out


# ---------------------------------------------------------------------------
# Task B — Failure Risk
# ---------------------------------------------------------------------------


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float | None
    roc_auc: float | None
    tp: int
    fp: int
    tn: int
    fn: int
    threshold: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "threshold": self.threshold,
            "notes": self.notes,
        }


def _safe_binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """Compute precision/recall/F1/accuracy with explicit degenerate handling.

    Precision and recall are defined as 0.0 when their denominators are 0.
    sklearn's `zero_division=0.0` already handles that; we also guard accuracy.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_pred = np.asarray(y_pred).astype(np.int8)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    accuracy = float((y_true == y_pred).mean())
    # Positive class = 1 (imminent failure)
    precision = float(precision_score(y_true, y_pred, zero_division=0, average="binary"))
    recall = float(recall_score(y_true, y_pred, zero_division=0, average="binary"))
    f1 = float(f1_score(y_true, y_pred, zero_division=0, average="binary"))
    return accuracy, precision, recall, f1


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray | None,
    *,
    threshold: float = 0.5,
    y_pred: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute the full classification metric set at a given threshold.

    Parameters
    ----------
    y_true : np.ndarray of {0,1}
    y_prob : np.ndarray of float in [0,1] or None
        If provided, PR-AUC and ROC-AUC are computed from probabilities.
        If the labels are single-class, both are set to None and a note
        is added.
    threshold : float
        Cut applied to y_prob when y_pred is not provided directly.
    y_pred : np.ndarray of {0,1}, optional
        For deterministic classifiers (majority class) that only produce hard
        labels, callers can pass y_pred directly.

    Notes
    -----
    For the majority-class baseline, `y_prob` is undefined; PR-AUC and
    ROC-AUC are reported as `None` (not as 0.0 or 1.0) and an explanatory
    note is attached. This satisfies Phase 2 section 14 ("explain which
    metrics are undefined or degenerate").
    """
    y_true = np.asarray(y_true).astype(np.int8)
    if y_pred is None:
        if y_prob is None:
            raise ValueError("Either y_prob or y_pred must be provided.")
        y_pred = (np.asarray(y_prob) >= threshold).astype(np.int8)
    else:
        y_pred = np.asarray(y_pred).astype(np.int8)

    accuracy, precision, recall, f1 = _safe_binary_prf(y_true, y_pred)

    notes: list[str] = []

    pr_auc: float | None = None
    roc_auc: float | None = None
    if y_prob is not None:
        if len(np.unique(y_true)) < 2:
            notes.append("y_true is single-class; PR-AUC/ROC-AUC are undefined.")
        else:
            try:
                pr_auc = float(average_precision_score(y_true, y_prob))
            except Exception as exc:  # pragma: no cover - defensive
                notes.append(f"PR-AUC failed: {exc}")
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception as exc:  # pragma: no cover - defensive
                notes.append(f"ROC-AUC failed: {exc}")
    else:
        notes.append("Deterministic classifier: PR-AUC/ROC-AUC are undefined without probabilities.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        threshold=float(threshold),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_thresholds: int = 200,
) -> dict[str, np.ndarray]:
    """Sweep thresholds across the probability range and return per-threshold
    precision/recall/F1/accuracy.

    Returns dict with keys: thresholds, precision, recall, f1, accuracy, tp, fp, fn, tn.

    Thresholds are evenly spaced in [0, 1] but bounded to the observed
    probability range so the curve is meaningful (does not waste samples
    on regions where all predictions are one class).
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    lo = float(min(y_prob.min(), 0.0))
    hi = float(max(y_prob.max(), 1.0))
    thresholds = np.linspace(lo, hi, n_thresholds)

    precision = np.zeros(n_thresholds)
    recall = np.zeros(n_thresholds)
    f1 = np.zeros(n_thresholds)
    accuracy = np.zeros(n_thresholds)
    tp = np.zeros(n_thresholds, dtype=np.int64)
    fp = np.zeros(n_thresholds, dtype=np.int64)
    fn = np.zeros(n_thresholds, dtype=np.int64)
    tn = np.zeros(n_thresholds, dtype=np.int64)

    n = len(y_true)
    for i, thr in enumerate(thresholds):
        pred = (y_prob >= thr).astype(np.int8)
        tp_i = int(((pred == 1) & (y_true == 1)).sum())
        fp_i = int(((pred == 1) & (y_true == 0)).sum())
        fn_i = int(((pred == 0) & (y_true == 1)).sum())
        tn_i = int(((pred == 0) & (y_true == 0)).sum())
        tp[i], fp[i], fn[i], tn[i] = tp_i, fp_i, fn_i, tn_i
        precision[i] = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
        recall[i] = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        accuracy[i] = (tp_i + tn_i) / n if n > 0 else 0.0

    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def suggest_candidate_threshold(
    sweep: dict[str, np.ndarray],
    *,
    criterion: str = "f1",
) -> tuple[float, dict[str, float]]:
    """Return the threshold on the sweep that maximizes the chosen criterion.

    This is a CANDIDATE (Phase 2 section 17: "Do NOT select a final business
    threshold yet"). The returned dict contains the metric values at that
    threshold.
    """
    if criterion not in {"f1", "precision", "recall", "accuracy"}:
        raise ValueError(f"Unknown criterion {criterion!r}")
    scores = sweep[criterion]
    idx = int(np.argmax(scores))
    thr = float(sweep["thresholds"][idx])
    metrics = {
        "threshold": thr,
        "precision": float(sweep["precision"][idx]),
        "recall": float(sweep["recall"][idx]),
        "f1": float(sweep["f1"][idx]),
        "accuracy": float(sweep["accuracy"][idx]),
        "tp": int(sweep["tp"][idx]),
        "fp": int(sweep["fp"][idx]),
        "fn": int(sweep["fn"][idx]),
        "tn": int(sweep["tn"][idx]),
    }
    return thr, metrics


# ---------------------------------------------------------------------------
# Class imbalance report (Phase 2 section 19)
# ---------------------------------------------------------------------------


def class_balance_report(
    y_true: np.ndarray,
    unit_ids: np.ndarray,
) -> dict[str, Any]:
    """Report positive/negative rates and affected engines.

    Returns:
        total_rows, positive_rows, negative_rows,
        positive_rate, negative_rate,
        total_engines, engines_with_any_positive, engines_all_negative,
        horizon_inferred_note
    """
    y_true = np.asarray(y_true).astype(np.int8)
    unit_ids = np.asarray(unit_ids)
    pos_rows = int((y_true == 1).sum())
    neg_rows = int((y_true == 0).sum())
    total = pos_rows + neg_rows
    pos_engines = sorted({int(u) for u, y in zip(unit_ids, y_true) if y == 1})
    all_engines = sorted({int(u) for u in unit_ids})
    return {
        "total_rows": total,
        "positive_rows": pos_rows,
        "negative_rows": neg_rows,
        "positive_rate": (pos_rows / total) if total else 0.0,
        "negative_rate": (neg_rows / total) if total else 0.0,
        "total_engines": len(all_engines),
        "engines_with_any_positive": len(pos_engines),
        "engines_all_negative": len(all_engines) - len(pos_engines),
        "positive_engine_ids_sample": pos_engines[:10],
    }
