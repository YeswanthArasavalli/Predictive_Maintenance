"""Phase 2 experiment runner.

Given an ExperimentSpec, this module:

1. Filters FD004 to the Phase 1 train / validation engine sets (loaded
   verbatim from the Phase 1 manifest — no re-split).
2. Assigns regimes and computes raw_RUL and per-horizon failure targets.
3. Fits the requested normalization (Mode A global or Mode B regime
   conditioned) on TRAIN rows only, then transforms both partitions.
4. Builds the row-level feature matrix (Config A/B/B+ plus optional cycle)
   via the controlled gate in `src.models.features`.
5. Trains the specified baseline (naive / linear_degradation / ridge /
   hist_gb / logreg / majority), predicts on the validation partition,
   and computes Phase 2 metrics.
6. Returns a RunResult carrying the metrics, per-row validation records,
   worst-10 errors, class-balance report, threshold sweep, candidate
   threshold, model-state, timing, and Phase 1 split hashes.

The official test set is never referenced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data.contract import OPERATIONAL_COLS
from ..data.loader import CYCLE, UNIT_ID, load_dataset
from ..data.regime import assign_regime
from ..evaluation.metrics import (
    classification_metrics,
    class_balance_report,
    rul_metrics,
    suggest_candidate_threshold,
    threshold_sweep,
)
from ..features.normalization import GlobalScaler, RegimeScaler
from ..features.sensors import get_sensor_config
from ..features.targets import (
    compute_failure_risk_target,
    compute_model_rul_target,
    compute_raw_rul,
)
from ..features.temporal import (
    TemporalConfig,
    add_causal_temporal_features,
    temporal_feature_names,
)
from ..models.baselines import MajorityClass, NaiveAgeRUL, NaiveMeanRUL
from ..models.features import (
    build_row_matrix,
    build_temporal_matrix,
    select_temporal_monitors,
)
from ..models.gradient_boosting import HistGBClassifierWrapper, HistGBRegressorWrapper
from ..models.linear_degradation import LinearDegradationRUL
from ..models.logistic import LogisticRegressionWrapper
from ..models.ridge import RidgeRegressor
from .registry import ExperimentSpec, load_phase1_split


@dataclass
class PreparedData:
    """Row-level train/val frames with targets and normalization applied."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    feature_names: list[str]
    normalization_mode: str | None
    sensor_config: str | None
    include_cycle: bool
    horizon: int | None
    temporal: dict[str, Any] | None = None  # Phase 2.5 temporal metadata


@dataclass
class RunResult:
    spec: ExperimentSpec
    metrics: dict[str, Any]
    val_rows: pd.DataFrame          # per-row unit_id, cycle, regime, y_true, y_pred, error, y_prob?
    worst10: pd.DataFrame | None    # only for Task A
    class_balance: dict[str, Any] | None
    threshold_sweep: dict[str, Any] | None
    candidate_threshold: float | None
    model_state: dict[str, Any]
    timing: dict[str, float]
    data_hash: dict[str, str]
    feature_names: list[str]
    temporal: dict[str, Any] | None = None
    prepared: PreparedData = field(repr=False, default=None)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _load_full_train(fd_id: str) -> pd.DataFrame:
    df = load_dataset(fd_id, "train")
    df["operating_regime"] = assign_regime(df)
    return df


def prepare(spec: ExperimentSpec, *, fd_id: str, split: dict[str, Any]) -> PreparedData:
    """Filter by Phase 1 manifest IDs, fit scaler on TRAIN, transform both.

    Also computes raw_RUL (never overwritten) and the horizon-specific
    failure target on BOTH partitions. Val engines are run-to-failure, so
    their raw_RUL is well defined exactly as in training.
    """
    full = _load_full_train(fd_id)
    train_engines = set(split["train_engine_ids"])
    val_engines = set(split["val_engine_ids"])
    if train_engines & val_engines:
        raise ValueError("Phase 2 loader: manifest engine sets overlap.")

    train_df = full[full[UNIT_ID].isin(train_engines)].copy().reset_index(drop=True)
    val_df = full[full[UNIT_ID].isin(val_engines)].copy().reset_index(drop=True)

    # Targets are computed from UNSCALED data (regime and cycle are unaffected
    # by scaling; RUL is defined by cycle only).
    train_df["raw_RUL"] = compute_raw_rul(train_df)
    train_df["model_RUL_target"] = compute_model_rul_target(train_df["raw_RUL"], cap=None)
    val_df["raw_RUL"] = compute_raw_rul(val_df)
    val_df["model_RUL_target"] = compute_model_rul_target(val_df["raw_RUL"], cap=None)

    if spec.horizon is not None:
        col = f"failure_risk_target_H{spec.horizon}"
        train_df[col] = compute_failure_risk_target(train_df["raw_RUL"], horizon=spec.horizon)
        val_df[col] = compute_failure_risk_target(val_df["raw_RUL"], horizon=spec.horizon)

    # Fit normalization on TRAIN only.
    norm_mode = spec.normalization_mode
    if norm_mode is None:
        # Naive / majority baselines do not need features.
        return PreparedData(
            train_df=train_df,
            val_df=val_df,
            feature_names=[],
            normalization_mode=None,
            sensor_config=None,
            include_cycle=bool(spec.include_cycle),
            horizon=spec.horizon,
        )

    sensors = get_sensor_config(spec.feature_config or "A")
    scale_cols = OPERATIONAL_COLS + sensors

    if norm_mode == "A":
        scaler = GlobalScaler()
        scaler.fit(train_df, scale_cols)
        train_scaled = scaler.transform(train_df)
        val_scaled = scaler.transform(val_df)
    elif norm_mode == "B":
        scaler = RegimeScaler(fallback="global")
        scaler.fit(train_df, scale_cols, regime_col="operating_regime")
        train_scaled = scaler.transform(train_df, regime_col="operating_regime")
        val_scaled = scaler.transform(val_df, regime_col="operating_regime")
    else:
        raise ValueError(f"Unknown normalization_mode {norm_mode!r}")

    # Build row-level feature names + matrix (with cycle exception gate).
    # Phase 2.5: if a causal temporal family is requested, select monitor
    # sensors on TRAIN only, materialise causal temporal columns on each
    # partition independently (engine boundaries and the frozen split are
    # respected because train and val are separate frames with disjoint
    # engines), then build a row + temporal matrix.
    sensor_config = spec.feature_config or "A"
    temporal_meta: dict[str, Any] | None = None
    if getattr(spec, "temporal_family", "none") not in ("none", None):
        candidates = get_sensor_config(sensor_config)
        monitors = select_temporal_monitors(
            train_scaled, candidates, top_n=int(spec.temporal_n_monitors)
        )
        tcfg = TemporalConfig(
            family=spec.temporal_family,
            windows=tuple(spec.temporal_windows),
            lags=tuple(spec.temporal_lags),
            include_range=bool(spec.temporal_include_range),
            source_cols=tuple(candidates),
            monitor_sensors=tuple(monitors),
        )
        train_scaled = add_causal_temporal_features(train_scaled, tcfg)
        val_scaled = add_causal_temporal_features(val_scaled, tcfg)
        tnames = temporal_feature_names(tcfg)
        _, feat_names = build_temporal_matrix(
            train_scaled,
            sensor_config=sensor_config,
            include_cycle=bool(spec.include_cycle),
            temporal_names=tnames,
        )
        temporal_meta = {
            "family": tcfg.family,
            "windows": list(tcfg.windows),
            "lags": list(tcfg.lags),
            "include_range": tcfg.include_range,
            "n_monitors_requested": int(spec.temporal_n_monitors),
            "monitors_selected_train_only": list(monitors),
            "temporal_feature_names": list(tnames),
            "n_temporal_features": len(tnames),
        }
    else:
        _, feat_names = build_row_matrix(
            train_scaled,
            sensor_config=sensor_config,
            include_cycle=bool(spec.include_cycle),
        )

    return PreparedData(
        train_df=train_scaled,
        val_df=val_scaled,
        feature_names=feat_names,
        normalization_mode=norm_mode,
        sensor_config=sensor_config,
        include_cycle=bool(spec.include_cycle),
        horizon=spec.horizon,
        temporal=temporal_meta,
    )


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------


def _x_matrix(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return df[feature_names].to_numpy(dtype=np.float64)


def _make_val_rows(
    val_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    target_col: str,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "unit_id": val_df[UNIT_ID].to_numpy(),
        "cycle": val_df[CYCLE].to_numpy(),
        "operating_regime": val_df["operating_regime"].to_numpy(),
        "y_true": val_df[target_col].to_numpy(),
        "y_pred": y_pred,
    })
    out["error"] = out["y_pred"] - out["y_true"] if np.issubdtype(out["y_pred"].dtype, np.floating) else 0
    if y_prob is not None:
        out["y_prob"] = y_prob
    return out


def _run_task_a(spec: ExperimentSpec, prep: PreparedData) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit + evaluate on the RUL target. Returns (metrics, val_rows, worst10, model_state)."""
    target_col = "raw_RUL"
    if spec.model == "naive_mean":
        model = NaiveMeanRUL().fit(prep.train_df, target_col=target_col)
        y_pred = model.predict(prep.val_df)
        state = model.describe()
    elif spec.model == "naive_age":
        model = NaiveAgeRUL().fit(prep.train_df)
        y_pred = model.predict(prep.val_df)
        state = model.describe()
    elif spec.model == "linear_degradation":
        hi_sensors = tuple(spec.hyperparameters.get("hi_sensors", ["sensor_14", "sensor_17", "sensor_11"]))
        min_history = int(spec.hyperparameters.get("min_history", 15))
        slope_floor = float(spec.hyperparameters.get("slope_floor", 1e-6))
        model = LinearDegradationRUL(
            hi_sensors=hi_sensors, min_history=min_history, slope_floor=slope_floor
        ).fit(prep.train_df, target_col=target_col)
        y_pred = model.predict(prep.val_df)
        state = model.describe()
    elif spec.model == "ridge":
        X_tr = _x_matrix(prep.train_df, prep.feature_names)
        y_tr = prep.train_df[target_col].to_numpy(dtype=np.float64)
        X_va = _x_matrix(prep.val_df, prep.feature_names)
        model = RidgeRegressor(alpha=float(spec.hyperparameters.get("alpha", 1.0)), feature_names=list(prep.feature_names))
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        state = model.describe()
    elif spec.model == "hist_gb_regressor":
        X_tr = _x_matrix(prep.train_df, prep.feature_names)
        y_tr = prep.train_df[target_col].to_numpy(dtype=np.float64)
        X_va = _x_matrix(prep.val_df, prep.feature_names)
        model = HistGBRegressorWrapper(
            max_iter=int(spec.hyperparameters.get("max_iter", 200)),
            learning_rate=float(spec.hyperparameters.get("learning_rate", 0.1)),
            random_state=int(spec.hyperparameters.get("random_state", 42)),
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        state = model.describe()
    else:
        raise ValueError(f"Unknown Task A model: {spec.model!r}")

    y_true = prep.val_df[target_col].to_numpy()
    metrics = rul_metrics(y_true, y_pred, unit_ids=prep.val_df[UNIT_ID].to_numpy())

    val_rows = _make_val_rows(prep.val_df, y_pred, y_prob=None, target_col=target_col)
    val_rows["abs_error"] = (val_rows["y_pred"] - val_rows["y_true"]).abs()
    worst10 = val_rows.sort_values("abs_error", ascending=False).head(10).reset_index(drop=True)

    return metrics, val_rows, worst10, state


def _run_task_b(spec: ExperimentSpec, prep: PreparedData) -> tuple[dict[str, Any], pd.DataFrame, dict, dict, dict[str, Any]]:
    """Fit + evaluate on failure_risk_target_H{h}. Returns (metrics, val_rows, balance, sweep, state)."""
    horizon = int(spec.horizon)
    target_col = f"failure_risk_target_H{horizon}"
    y_tr = prep.train_df[target_col].to_numpy(dtype=np.int8)
    y_va = prep.val_df[target_col].to_numpy(dtype=np.int8)

    y_prob: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    state: dict[str, Any]

    if spec.model == "majority_class":
        model = MajorityClass().fit(y_tr)
        y_pred = model.predict(len(prep.val_df))
        state = model.describe()
    elif spec.model == "logistic_regression":
        X_tr = _x_matrix(prep.train_df, prep.feature_names)
        X_va = _x_matrix(prep.val_df, prep.feature_names)
        model = LogisticRegressionWrapper(
            C=float(spec.hyperparameters.get("C", 1.0)),
            class_weight=spec.hyperparameters.get("class_weight", "balanced"),
            max_iter=int(spec.hyperparameters.get("max_iter", 1000)),
        )
        model.fit(X_tr, y_tr)
        y_prob = model.positive_class_probability(X_va)
        state = model.describe()
    elif spec.model == "hist_gb_classifier":
        X_tr = _x_matrix(prep.train_df, prep.feature_names)
        X_va = _x_matrix(prep.val_df, prep.feature_names)
        model = HistGBClassifierWrapper(
            max_iter=int(spec.hyperparameters.get("max_iter", 200)),
            learning_rate=float(spec.hyperparameters.get("learning_rate", 0.1)),
            random_state=int(spec.hyperparameters.get("random_state", 42)),
            class_weight=spec.hyperparameters.get("class_weight", "balanced"),
        )
        model.fit(X_tr, y_tr)
        y_prob = model.positive_class_probability(X_va)
        state = model.describe()
    else:
        raise ValueError(f"Unknown Task B model: {spec.model!r}")

    # Threshold sweep + candidate threshold selection (F1 criterion).
    # For majority-class (no probabilities) we skip the sweep and evaluate at
    # the deterministic y_pred.
    balance = class_balance_report(y_va, prep.val_df[UNIT_ID].to_numpy())

    if y_prob is not None:
        sweep = threshold_sweep(y_va, y_prob, n_thresholds=200)
        candidate_thr, cand_metrics = suggest_candidate_threshold(sweep, criterion="f1")
        default_metrics = classification_metrics(y_va, y_prob, threshold=0.5).to_dict()
        cand_row = classification_metrics(y_va, y_prob, threshold=candidate_thr).to_dict()
        sweep_serializable = {
            "thresholds": sweep["thresholds"].tolist(),
            "precision": sweep["precision"].tolist(),
            "recall": sweep["recall"].tolist(),
            "f1": sweep["f1"].tolist(),
            "accuracy": sweep["accuracy"].tolist(),
        }
        metrics = {
            "at_default_threshold": default_metrics,
            "at_candidate_threshold": cand_row,
            "candidate_threshold": float(candidate_thr),
        }
        y_pred_final = (y_prob >= 0.5).astype(np.int8)
    else:
        # Deterministic (majority) classifier
        default_metrics = classification_metrics(y_va, y_prob=None, y_pred=y_pred).to_dict()
        sweep_serializable = None
        candidate_thr = None
        metrics = {
            "at_default_threshold": default_metrics,
            "at_candidate_threshold": None,
            "candidate_threshold": None,
        }
        y_pred_final = y_pred

    val_rows = _make_val_rows(
        prep.val_df,
        y_pred_final,
        y_prob=y_prob,
        target_col=target_col,
    )
    val_rows["abs_error"] = 0  # classification rows keep the column for uniformity

    return metrics, val_rows, balance, sweep_serializable, state


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_experiment(spec: ExperimentSpec, *, fd_id: str, split: dict[str, Any]) -> RunResult:
    t0 = time.perf_counter()
    prep = prepare(spec, fd_id=fd_id, split=split)
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    if spec.task == "A_RUL":
        metrics, val_rows, worst10, state = _run_task_a(spec, prep)
        balance: dict[str, Any] | None = None
        sweep: dict[str, Any] | None = None
    elif spec.task == "B_FAILURE_RISK":
        metrics, val_rows, balance, sweep, state = _run_task_b(spec, prep)
        worst10 = None
    else:
        raise ValueError(f"Unknown task {spec.task!r}")
    t_fit_predict = time.perf_counter() - t1

    return RunResult(
        spec=spec,
        metrics=metrics,
        val_rows=val_rows,
        worst10=worst10,
        class_balance=balance,
        threshold_sweep=sweep,
        candidate_threshold=(metrics.get("candidate_threshold") if isinstance(metrics, dict) else None),
        model_state=state,
        timing={"prepare_seconds": t_prep, "fit_predict_seconds": t_fit_predict},
        data_hash={
            "train_hash": split["train_hash"],
            "val_hash": split["val_hash"],
            "manifest_hash": split["manifest_hash"],
        },
        feature_names=prep.feature_names,
        temporal=prep.temporal,
        prepared=prep,
    )
