"""Phase 2 figures (spec section 25).

Run:  python scripts/phase2_figures.py

Writes 10 PNGs to `results/figures/phase_2/`:
    RUL:      fig1_pred_vs_actual, fig2_residual_distribution,
              fig3_error_vs_rul, fig4_error_by_regime
    Failure:  fig5_precision_recall, fig6_confusion_matrix,
              fig7_recall_vs_threshold, fig8_precision_vs_threshold,
              fig9_f1_vs_threshold, fig10_performance_by_horizon

Figures consume ONLY the artifacts written by `run_phase2_experiments.py`
(per-experiment JSON + val_rows/*.parquet). No model is refit here, and
the official test set is never read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import classification_metrics  # noqa: E402

EXPERIMENTS_DIR = ROOT / "results" / "experiments" / "phase2"
VAL_ROWS = EXPERIMENTS_DIR / "val_rows"
FIGDIR = ROOT / "results" / "figures" / "phase_2"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})

# Featured models (chosen by observed validation performance in Phase 2).
BEST_RUL_MODEL = "RUL_gb_regimeB_withCyc_cfgA"
BEST_FR_MODEL = "FR_gb_regimeB_withCyc_cfgA"
HORIZON_MODELS_GB = {
    14: "FR_gb_regimeB_noCyc_cfgA_H14",
    30: "FR_gb_regimeB_noCyc_cfgA",
    50: "FR_gb_regimeB_noCyc_cfgA_H50",
}
HORIZON_MODELS_LR = {
    14: "FR_logreg_regimeB_noCyc_cfgA_H14",
    30: "FR_logreg_regimeB_noCyc_cfgA",
    50: "FR_logreg_regimeB_noCyc_cfgA_H50",
}


def _load_experiment(exp_id: str) -> dict:
    for sub in ("rul", "failure_risk"):
        p = EXPERIMENTS_DIR / sub / f"{exp_id}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(exp_id)


def _load_val_rows(exp_id: str) -> pd.DataFrame:
    return pd.read_parquet(VAL_ROWS / f"{exp_id}.parquet")


def _save(fig, name: str) -> None:
    fig.savefig(FIGDIR / name)
    plt.close(fig)
    print("  fig:", name)


# ---------------------------------------------------------------------------
# RUL figures
# ---------------------------------------------------------------------------


def fig1_pred_vs_actual():
    df = _load_val_rows(BEST_RUL_MODEL)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["y_true"], df["y_pred"], s=4, alpha=0.35, edgecolors="none")
    lim = [min(df["y_true"].min(), df["y_pred"].min()),
           max(df["y_true"].max(), df["y_pred"].max())]
    ax.plot(lim, lim, "r--", linewidth=1, label="y = x (perfect)")
    ax.set_xlabel("Actual raw_RUL (operational cycles)")
    ax.set_ylabel("Predicted RUL (operational cycles)")
    ax.set_title(f"Fig 1 — Predicted vs actual RUL on validation\n({BEST_RUL_MODEL})")
    ax.legend()
    _save(fig, "fig1_pred_vs_actual.png")


def fig2_residual_distribution():
    df = _load_val_rows(BEST_RUL_MODEL)
    resid = df["y_pred"] - df["y_true"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(resid, bins=80, color="steelblue", edgecolor="white")
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.axvline(float(resid.mean()), color="orange", linestyle="--", linewidth=1,
               label=f"mean={float(resid.mean()):.2f}")
    ax.set_xlabel("Residual (predicted − actual, cycles)")
    ax.set_ylabel("Validation rows")
    ax.set_title(f"Fig 2 — RUL residual distribution\n({BEST_RUL_MODEL})")
    ax.legend()
    _save(fig, "fig2_residual_distribution.png")


def fig3_error_vs_actual_rul():
    df = _load_val_rows(BEST_RUL_MODEL)
    df = df.copy()
    df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    # Bin actual RUL into 15 bins to average within
    df["rul_bin"] = pd.cut(df["y_true"], bins=15)
    grouped = df.groupby("rul_bin", observed=True).agg(
        mean_rul=("y_true", "mean"),
        mean_abs_err=("abs_err", "mean"),
        p90_abs_err=("abs_err", lambda s: float(np.percentile(s, 90))),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grouped["mean_rul"], grouped["mean_abs_err"], "o-", label="Mean |error|")
    ax.plot(grouped["mean_rul"], grouped["p90_abs_err"], "s--", label="P90 |error|")
    ax.set_xlabel("Actual raw_RUL (operational cycles)")
    ax.set_ylabel("|Error| (cycles)")
    ax.set_title(f"Fig 3 — RUL error magnitude vs actual RUL\n({BEST_RUL_MODEL})")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "fig3_error_vs_rul.png")


def fig4_error_by_regime():
    df = _load_val_rows(BEST_RUL_MODEL)
    df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    order = sorted(df["operating_regime"].unique())
    means = [df[df["operating_regime"] == r]["abs_err"].mean() for r in order]
    p90s = [df[df["operating_regime"] == r]["abs_err"].quantile(0.90) for r in order]
    counts = [int((df["operating_regime"] == r).sum()) for r in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(order))
    ax.bar(x - 0.2, means, width=0.4, label="Mean |error|", color="steelblue")
    ax.bar(x + 0.2, p90s, width=0.4, label="P90 |error|", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{r}\n(n={c})" for r, c in zip(order, counts)])
    ax.set_xlabel("Operating regime")
    ax.set_ylabel("|Error| (cycles)")
    ax.set_title(f"Fig 4 — RUL error by operating regime\n({BEST_RUL_MODEL})")
    ax.legend()
    _save(fig, "fig4_error_by_regime.png")


# ---------------------------------------------------------------------------
# Failure-risk figures
# ---------------------------------------------------------------------------


def _load_prob_and_truth(exp_id: str) -> tuple[np.ndarray, np.ndarray]:
    df = _load_val_rows(exp_id)
    return df["y_prob"].to_numpy(), df["y_true"].to_numpy(dtype=np.int8)


def fig5_precision_recall():
    exp = _load_experiment(BEST_FR_MODEL)
    sweep = exp.get("threshold_sweep") or {}
    # Reconstruct PR from sweep
    precision = np.asarray(sweep["precision"])
    recall = np.asarray(sweep["recall"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=1.5, label=BEST_FR_MODEL)
    # Also draw the naive baseline: PR-AUC ≈ positive rate (majority-class prior)
    m = _load_experiment(BEST_FR_MODEL)
    positive_rate = (m.get("class_balance") or {}).get("positive_rate", 0.0)
    ax.axhline(positive_rate, color="gray", linestyle="--", linewidth=1,
               label=f"No-skill prior = {positive_rate:.3f}")
    pr_auc = exp["metrics"]["at_default_threshold"]["pr_auc"]
    ax.set_xlabel("Recall (fraction of failures caught)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Fig 5 — Precision-Recall curve\n({BEST_FR_MODEL}, PR-AUC={pr_auc:.3f})")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "fig5_precision_recall.png")


def fig6_confusion_matrix():
    exp = _load_experiment(BEST_FR_MODEL)
    at_cand = exp["metrics"].get("at_candidate_threshold") or exp["metrics"]["at_default_threshold"]
    cm = np.array([[at_cand["tn"], at_cand["fp"]],
                   [at_cand["fn"], at_cand["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center",
                    color=text_color, fontsize=12)
    ax.set_xticks([0, 1], ["No failure", "Failure"])
    ax.set_yticks([0, 1], ["No failure", "Failure"])
    ax.set_xlabel("Predicted (at candidate threshold)")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Fig 6 — Confusion matrix\n"
        f"({BEST_FR_MODEL} @ threshold={at_cand['threshold']:.3f})"
    )
    _save(fig, "fig6_confusion_matrix.png")


def _threshold_curve(exp_id: str, metric: str, ylabel: str, title: str, name: str,
                     marker_annotation: str | None = None):
    exp = _load_experiment(exp_id)
    sweep = exp["threshold_sweep"]
    thr = np.asarray(sweep["thresholds"])
    ys = np.asarray(sweep[metric])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thr, ys, linewidth=1.5)
    cand = exp.get("candidate_threshold")
    if cand is not None:
        ax.axvline(cand, color="red", linestyle="--", linewidth=1,
                   label=f"Candidate thr={cand:.3f}")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Default thr=0.5")
    ax.set_xlabel("Decision threshold on P(failure)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n({exp_id})")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, name)


def fig7_recall_vs_threshold():
    _threshold_curve(BEST_FR_MODEL, "recall", "Recall",
                     "Fig 7 — Recall vs decision threshold",
                     "fig7_recall_vs_threshold.png")


def fig8_precision_vs_threshold():
    _threshold_curve(BEST_FR_MODEL, "precision", "Precision",
                     "Fig 8 — Precision vs decision threshold",
                     "fig8_precision_vs_threshold.png")


def fig9_f1_vs_threshold():
    _threshold_curve(BEST_FR_MODEL, "f1", "F1",
                     "Fig 9 — F1 vs decision threshold",
                     "fig9_f1_vs_threshold.png")


def fig10_performance_by_horizon():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    metrics = [("recall", "Recall"), ("precision", "Precision"), ("f1", "F1")]
    horizons = [14, 30, 50]

    for ax, key, label in zip(axes, *zip(*metrics)):
        gb_vals = []
        lr_vals = []
        for h in horizons:
            gb_exp = _load_experiment(HORIZON_MODELS_GB[h])
            lr_exp = _load_experiment(HORIZON_MODELS_LR[h])
            # Compare at the CANDIDATE threshold (the F1 argmax on val).
            gb_vals.append(gb_exp["metrics"]["at_candidate_threshold"][key])
            lr_vals.append(lr_exp["metrics"]["at_candidate_threshold"][key])
        ax.plot(horizons, gb_vals, "o-", label="HistGB", color="steelblue")
        ax.plot(horizons, lr_vals, "s--", label="LogReg", color="indianred")
        ax.set_xticks(horizons)
        ax.set_xticklabels([f"H={h}" for h in horizons])
        ax.set_ylabel(label)
        ax.set_title(f"{label} at candidate threshold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Fig 10 — Failure-risk performance by horizon (H14 / H30 / H50)", y=1.02)
    _save(fig, "fig10_performance_by_horizon.png")


def main() -> int:
    print("== Phase 2 Figures ==")
    if not VAL_ROWS.exists():
        print("ERROR: run scripts/run_phase2_experiments.py first.")
        return 1
    for fn in (
        fig1_pred_vs_actual,
        fig2_residual_distribution,
        fig3_error_vs_actual_rul,
        fig4_error_by_regime,
        fig5_precision_recall,
        fig6_confusion_matrix,
        fig7_recall_vs_threshold,
        fig8_precision_vs_threshold,
        fig9_f1_vs_threshold,
        fig10_performance_by_horizon,
    ):
        fn()
    print(f"\nWrote 10 figures to {FIGDIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
