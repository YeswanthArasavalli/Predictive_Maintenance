"""Phase 2.5 figures (authorization prompt section 21).

Run:  python scripts/phase2_5_figures.py

Writes PNGs to `results/figures/phase_2_5/`:
    fig1_rul_family_comparison      — RUL MAE / R2 / PS_last by temporal family
    fig2_fr_family_comparison       — H30 failure-risk F1 / PR-AUC by temporal family
    fig3_temporal_window_ablation   — RUL & FR robustness across W=3/5/10
    fig4_cycle_blind_vs_aware       — cycle-aware vs cycle-blind temporal contribution
    fig5_rul_error_distribution     — residual distribution for the best temporal RUL model
    fig6_fr_pr_tradeoff             — precision/recall at default vs candidate threshold
    fig7_horizon_sensitivity        — combined temporal F1 across H14/H30/H50
    fig8_temporal_feature_trace     — a representative causal feature over one engine

Figures consume ONLY artifacts written by `run_phase2_5_experiments.py` plus,
for fig8, the causal temporal FEATURE MODULE applied to a single FD004 TRAIN
engine (no model is refit and the official test set is never read).

They are descriptive renderings of measured validation numbers; no visual
conclusion is manufactured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import CYCLE, UNIT_ID, load_dataset  # noqa: E402
from src.features.temporal import (  # noqa: E402
    FAMILY_COMBINED,
    TemporalConfig,
    add_causal_temporal_features,
)

EXPERIMENTS_DIR = ROOT / "results" / "experiments" / "phase2_5"
FIGDIR = ROOT / "results" / "figures" / "phase_2_5"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})

RUL_CSV = EXPERIMENTS_DIR / "rul_comparison.csv"
FR_CSV = EXPERIMENTS_DIR / "failure_risk_comparison.csv"

# Best temporal RUL model by MAE and best FR model, chosen by measured val numbers.
ANCHOR_RUL = "RUL25_T0_anchor_cycle"
ANCHOR_FR = "FR25_T0_anchor_cycle"

FAMILY_ORDER = ["none", "lag_diff", "rolling", "slope", "combined"]


def _read_rul_frame() -> pd.DataFrame:
    return pd.read_csv(RUL_CSV)


def _load_fr() -> pd.DataFrame:
    fr = pd.read_csv(FR_CSV)
    return fr[fr["horizon"] == 30]


def _save(fig, name: str) -> None:
    fig.savefig(FIGDIR / name)
    plt.close(fig)
    print("  fig:", name)


# ---------------------------------------------------------------------------
# fig1 — RUL comparison by temporal family
# ---------------------------------------------------------------------------
def fig1_rul_family_comparison():
    df = _read_rul_frame()
    cyc = df[df["cycle_feature"]].copy()
    # Anchor (none) is the Phase 2 controlled baseline.
    order = [e for e in ["RUL25_T0_anchor_cycle", "RUL25_T1_lagdiff_cyc",
                        "RUL25_T2_rolling_cyc", "RUL25_T3_slope_cyc",
                        "RUL25_T4_combined_cyc"] if e in set(cyc["experiment_id"])]
    sub = cyc.set_index("experiment_id").loc[order].reset_index()
    short = {
        "RUL25_T0_anchor_cycle": "T0 anchor\n(cur+cyc)",
        "RUL25_T1_lagdiff_cyc": "T1\nlag/diff",
        "RUL25_T2_rolling_cyc": "T2\nrolling",
        "RUL25_T3_slope_cyc": "T3\nslope",
        "RUL25_T4_combined_cyc": "T4\ncombined",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(sub))
    ax1.bar(x, sub["MAE"], color="steelblue")
    ax1.axhline(sub.loc[sub["experiment_id"] == ANCHOR_RUL, "MAE"].iloc[0],
                color="red", linestyle="--", linewidth=1, label="Phase 2 anchor MAE")
    ax1.set_xticks(x, [short[e] for e in sub["experiment_id"]])
    ax1.set_ylabel("RUL MAE (cycles, lower is better)")
    ax1.set_title("RUL MAE by causal temporal family")
    ax1.set_ylim(bottom=min(sub["MAE"]) - 2)
    ax1.legend(fontsize=8)
    for xi, v in zip(x, sub["MAE"]):
        ax1.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(x - 0.2, sub["R2"], width=0.4, color="seagreen", label="R²")
    ax2b = ax2.twinx()
    ax2b.bar(x + 0.2, sub["prognostic_score_last_per_engine"], width=0.4,
             color="indianred", label="PS_last")
    ax2.set_xticks(x, [short[e] for e in sub["experiment_id"]])
    ax2.set_ylabel("R² (higher is better)", color="seagreen")
    ax2b.set_ylabel("PS_last (lower is better)", color="indianred")
    ax2.set_title("RUL R² and prognostic score by family")
    fig.suptitle("Fig 1 — RUL: current+cycle baseline vs causal temporal features (cycle-aware)")
    fig.tight_layout()
    _save(fig, "fig1_rul_family_comparison.png")


# ---------------------------------------------------------------------------
# fig2 — Failure-risk comparison by temporal family
# ---------------------------------------------------------------------------
def fig2_fr_family_comparison():
    df = _load_fr()
    d05 = df[df["threshold_source"] == "default_0.5"].copy()
    order = ["FR25_T0_anchor_cycle", "FR25_T1_lagdiff_cyc", "FR25_T2_rolling_cyc",
             "FR25_T3_slope_cyc", "FR25_T4_combined_cyc"]
    order = [e for e in order if e in set(d05["experiment_id"])]
    sub = d05.set_index("experiment_id").loc[order].reset_index()
    short = {
        "FR25_T0_anchor_cycle": "T0 anchor",
        "FR25_T1_lagdiff_cyc": "T1 lag/diff",
        "FR25_T2_rolling_cyc": "T2 rolling",
        "FR25_T3_slope_cyc": "T3 slope",
        "FR25_T4_combined_cyc": "T4 combined",
    }
    x = np.arange(len(sub))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(x - 0.2, sub["F1"], width=0.4, color="steelblue", label="F1")
    ax1.bar(x + 0.2, sub["PR_AUC"], width=0.4, color="darkorange", label="PR-AUC")
    ax1.axhline(sub.loc[sub["experiment_id"] == ANCHOR_FR, "F1"].iloc[0],
                color="red", linestyle="--", linewidth=1, label="anchor F1")
    ax1.set_xticks(x, [short[e] for e in sub["experiment_id"]])
    ax1.set_ylabel("score (higher is better)")
    ax1.set_ylim(0.8, 1.0)
    ax1.set_title("H30 failure risk @ thr=0.5")
    ax1.legend(fontsize=8)

    ax2.bar(x - 0.2, sub["precision"], width=0.4, color="seagreen", label="precision")
    ax2.bar(x + 0.2, sub["recall"], width=0.4, color="indianred", label="recall")
    ax2.set_xticks(x, [short[e] for e in sub["experiment_id"]])
    ax2.set_ylim(0.7, 1.0)
    ax2.set_ylabel("rate")
    ax2.set_title("Recall held ~0.95; gains come from precision")
    ax2.legend(fontsize=8)
    fig.suptitle("Fig 2 — Failure risk: current+cycle baseline vs causal temporal features (H30)")
    fig.tight_layout()
    _save(fig, "fig2_fr_family_comparison.png")


# ---------------------------------------------------------------------------
# fig3 — Temporal window ablation (short vs medium)
# ---------------------------------------------------------------------------
def fig3_temporal_window_ablation():
    rul = _read_rul_frame()
    # Window ablation is a cycle-aware comparison; drop the cycle-blind row.
    rul_c = rul[(rul["temporal_family"] == "combined") & (rul["cycle_feature"])].copy()
    rul_c["win"] = rul_c["temporal_window"].astype(str).map(
        {"3": "W=3 (short)", "5": "W=5 (default)", "10": "W=10 (medium)"})
    fr = _load_fr()
    fr_c = fr[(fr["temporal_family"] == "combined") & (fr["cycle_feature"])
              & (fr["threshold_source"] == "default_0.5")].copy()
    fr_c["win"] = fr_c["temporal_window"].astype(str).map(
        {"3": "W=3 (short)", "5": "W=5 (default)", "10": "W=10 (medium)"})

    order = ["W=3 (short)", "W=5 (default)", "W=10 (medium)"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    r = rul_c.set_index("win").reindex(order)
    ax1.plot(order, r["MAE"], "o-", color="steelblue")
    ax1.axhline(rul[rul["experiment_id"] == ANCHOR_RUL]["MAE"].iloc[0],
                color="red", linestyle="--", linewidth=1, label="anchor MAE")
    ax1.set_ylabel("RUL MAE (cycles)")
    ax1.set_title("RUL combined-features across window size")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    for xi, v in zip(order, r["MAE"]):
        if not np.isnan(v):
            ax1.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    f = fr_c.set_index("win").reindex(order)
    ax2.plot(order, f["F1"], "s-", color="darkorange")
    ax2.axhline(fr[(fr["experiment_id"] == ANCHOR_FR) &
                   (fr["threshold_source"] == "default_0.5")]["F1"].iloc[0],
                color="red", linestyle="--", linewidth=1, label="anchor F1")
    ax2.set_ylabel("H30 F1 @ thr=0.5")
    ax2.set_title("Failure risk combined-features across window size")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    for xi, v in zip(order, f["F1"]):
        if not np.isnan(v):
            ax2.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Fig 3 — Temporal-window robustness (short vs medium context)")
    fig.tight_layout()
    _save(fig, "fig3_temporal_window_ablation.png")


# ---------------------------------------------------------------------------
# fig4 — Cycle-blind vs cycle-aware
# ---------------------------------------------------------------------------
def fig4_cycle_blind_vs_aware():
    rul = _read_rul_frame()
    fr = _load_fr()
    d05 = fr[fr["threshold_source"] == "default_0.5"]

    rul_groups = {
        "current\n(no cyc,\nno temp)": rul[rul["experiment_id"] == "RUL25_T0_current"]["MAE"].iloc[0],
        "cycle only\n(anchor)": rul[rul["experiment_id"] == ANCHOR_RUL]["MAE"].iloc[0],
        "temporal only\n(cycle-blind)": rul[rul["experiment_id"] == "RUL25_T4_combined_nocyc"]["MAE"].iloc[0],
        "cycle +\ntemporal": rul[rul["experiment_id"] == "RUL25_T4_combined_cyc"]["MAE"].iloc[0],
    }
    fr_groups = {
        "current\n(no cyc,\nno temp)": d05[d05["experiment_id"] == "FR25_T0_current"]["F1"].iloc[0],
        "cycle only\n(anchor)": d05[d05["experiment_id"] == ANCHOR_FR]["F1"].iloc[0],
        "temporal only\n(cycle-blind)": d05[d05["experiment_id"] == "FR25_T4_combined_nocyc"]["F1"].iloc[0],
        "cycle +\ntemporal": d05[d05["experiment_id"] == "FR25_T4_combined_cyc"]["F1"].iloc[0],
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    lbl = list(rul_groups)
    ax1.bar(lbl, list(rul_groups.values()), color=["gray", "steelblue", "mediumpurple", "seagreen"])
    ax1.set_ylabel("RUL MAE (lower is better)")
    ax1.set_title("RUL: does temporal help WITHOUT cycle?")
    for i, v in enumerate(rul_groups.values()):
        ax1.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(lbl, list(fr_groups.values()), color=["gray", "steelblue", "mediumpurple", "seagreen"])
    ax2.set_ylim(0.8, 0.95)
    ax2.set_ylabel("H30 F1 @ thr=0.5 (higher is better)")
    ax2.set_title("Failure risk: does temporal help WITHOUT cycle?")
    for i, v in enumerate(fr_groups.values()):
        ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Fig 4 — Cycle-blind vs cycle-aware contribution of causal temporal features")
    fig.tight_layout()
    _save(fig, "fig4_cycle_blind_vs_aware.png")


# ---------------------------------------------------------------------------
# fig5 — Error distribution for best temporal RUL model
# ---------------------------------------------------------------------------
def fig5_rul_error_distribution():
    best = _read_rul_frame().nsmallest(1, "MAE")["experiment_id"].iloc[0]
    df = pd.read_parquet(EXPERIMENTS_DIR / "val_rows" / f"{best}.parquet")
    resid = df["y_pred"] - df["y_true"]
    anchor = pd.read_parquet(EXPERIMENTS_DIR / "val_rows" / f"{ANCHOR_RUL}.parquet")
    a_resid = anchor["y_pred"] - anchor["y_true"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(a_resid, bins=80, alpha=0.5, color="steelblue", label=f"{ANCHOR_RUL} (mean={a_resid.mean():.2f})")
    ax.hist(resid, bins=80, alpha=0.5, color="seagreen", label=f"{best} (mean={resid.mean():.2f})")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (predicted − actual, operational cycles)")
    ax.set_ylabel("Validation rows")
    ax.set_title("Fig 5 — RUL residual distribution: anchor vs best temporal model")
    ax.legend(fontsize=8)
    _save(fig, "fig5_rul_error_distribution.png")


# ---------------------------------------------------------------------------
# fig6 — Precision/recall tradeoff at default vs candidate threshold
# ---------------------------------------------------------------------------
def fig6_fr_pr_tradeoff():
    fr = _load_fr()
    ids = [ANCHOR_FR, "FR25_T2_rolling_cyc", "FR25_T4_combined_cyc", "FR25_T4_combined_cyc_W10"]
    fig, ax = plt.subplots(figsize=(6, 5))
    markers = ["o", "s", "^", "d"]
    for eid, mk in zip(ids, markers):
        d05 = fr[(fr["experiment_id"] == eid) & (fr["threshold_source"] == "default_0.5")]
        dc = fr[(fr["experiment_id"] == eid) & (fr["threshold_source"] == "candidate_f1_argmax")]
        if len(d05):
            ax.scatter(d05["recall"].iloc[0], d05["precision"].iloc[0], marker=mk, s=70,
                       color="steelblue", label=f"{eid} @0.5")
        if len(dc):
            ax.scatter(dc["recall"].iloc[0], dc["precision"].iloc[0], marker=mk, s=70,
                       color="darkorange", label=f"{eid} @cand")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.85, 0.98)
    ax.set_ylim(0.75, 0.95)
    ax.set_title("Fig 6 — H30 precision/recall operating points\n(blue=thr 0.5, orange=candidate F1-argmax)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, "fig6_fr_pr_tradeoff.png")


# ---------------------------------------------------------------------------
# fig7 — Horizon sensitivity for combined temporal features
# ---------------------------------------------------------------------------
def fig7_horizon_sensitivity():
    fr = pd.read_csv(FR_CSV)
    comb = fr[(fr["temporal_family"] == "combined") & (fr["cycle_feature"])
              & (fr["threshold_source"] == "default_0.5")]
    # Anchor cycle exists only at H30; compare combined across H14/H30/H50.
    hs = [14, 30, 50]
    rec = [comb[comb["horizon"] == h]["recall"].mean() for h in hs]
    prec = [comb[comb["horizon"] == h]["precision"].mean() for h in hs]
    f1 = [comb[comb["horizon"] == h]["F1"].mean() for h in hs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hs, f1, "o-", label="F1", color="steelblue")
    ax.plot(hs, prec, "s-", label="precision", color="seagreen")
    ax.plot(hs, rec, "^-", label="recall", color="indianred")
    ax.set_xticks(hs, [f"H={h}" for h in hs])
    ax.set_ylabel("rate @ thr=0.5")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Fig 7 — Combined temporal features across failure-risk horizons (cycle-aware)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "fig7_horizon_sensitivity.png")


# ---------------------------------------------------------------------------
# fig8 — Representative causal temporal feature trace on one TRAIN engine
# ---------------------------------------------------------------------------
def fig8_temporal_feature_trace():
    df = load_dataset("FD004", "train")
    eng = df[UNIT_ID].drop_duplicates().iloc[0]
    e = df[df[UNIT_ID] == eng].sort_values(CYCLE).reset_index(drop=True)
    cfg = TemporalConfig(
        family=FAMILY_COMBINED, windows=(5,), lags=(5,),
        source_cols=("sensor_14",), monitor_sensors=("sensor_14",),
    )
    out = add_causal_temporal_features(e, cfg)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(out[CYCLE], out["sensor_14"], label="sensor_14 (raw)", color="black", linewidth=1)
    ax1.plot(out[CYCLE], out["sensor_14__rmean5"], label="causal rolling mean w=5",
             color="steelblue", linewidth=1.2)
    ax1.set_ylabel("sensor_14")
    ax1.set_title("Fig 8 — Causal temporal features on one FD004 TRAIN engine (sensor_14)")
    ax1.legend(fontsize=8)
    ax2.plot(out[CYCLE], out["sensor_14__slope5"], label="causal slope w=5",
             color="seagreen", linewidth=1.2)
    ax2.plot(out[CYCLE], out["sensor_14__d1"], label="first difference",
             color="darkorange", linewidth=1, alpha=0.8)
    ax2.set_xlabel("operational cycle")
    ax2.set_ylabel("trend feature")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig8_temporal_feature_trace.png")


def main() -> int:
    print("== Phase 2.5 Figures ==")
    if not RUL_CSV.exists() or not FR_CSV.exists():
        print("ERROR: run scripts/run_phase2_5_experiments.py first.")
        return 1
    for fn in (
        fig1_rul_family_comparison,
        fig2_fr_family_comparison,
        fig3_temporal_window_ablation,
        fig4_cycle_blind_vs_aware,
        fig5_rul_error_distribution,
        fig6_fr_pr_tradeoff,
        fig7_horizon_sensitivity,
        fig8_temporal_feature_trace,
    ):
        fn()
    print(f"\nWrote 8 figures to {FIGDIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
