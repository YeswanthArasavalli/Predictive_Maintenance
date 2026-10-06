"""Phase 2.6 figures (authorization prompt section 21).

Run:  python scripts/phase2_6_figures.py

Writes PNGs to `results/figures/phase_2_6/`:
    fig1_threshold_cost_anchor       — threshold vs illustrative expected cost (anchor)
    fig2_threshold_cost_temporal     — threshold vs illustrative expected cost (temporal)
    fig3_cost_ratio_comparison       — anchor vs temporal best cost across cost ratios
    fig4_pr_vs_threshold             — precision/recall vs threshold (both models)
    fig5_fp_vs_threshold             — false positives vs threshold (both models)
    fig6_fn_vs_threshold             — false negatives vs threshold (both models)
    fig7_engine_level                — engine-level detection / false-alarm aggregates
    fig8_bootstrap_uncertainty       — engine-bootstrap CI of the temporal-vs-anchor cost diff

Every cost axis is an ILLUSTRATIVE scenario score (units of one false positive),
never a financial figure. The figures read ONLY artifacts produced by
`run_phase2_6_decision_analysis.py`. No model is refit and the official test
partition is never read.
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

DECISIONS_DIR = ROOT / "results" / "decisions" / "phase2_6"
FIGDIR = ROOT / "results" / "figures" / "phase_2_6"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})

ILLUSTRATIVE = "Illustrative scenario assumptions — not observed financial costs."
ANCHOR = "FR25_T0_anchor_cycle"
TEMPORAL = "FR25_T4_combined_cyc"
H_PRIMARY = 30

RATIO_COLORS = plt.cm.viridis(np.linspace(0.0, 0.9, 7))


def _threshold_cost() -> pd.DataFrame:
    return pd.read_csv(DECISIONS_DIR / "threshold_cost.csv")


def _save(fig, name: str) -> None:
    fig.savefig(FIGDIR / name)
    plt.close(fig)
    print("  fig:", name)


# ---------------------------------------------------------------------------
# fig1 / fig2 — threshold vs expected cost for a single model
# ---------------------------------------------------------------------------
def _threshold_cost_panel(model_id: str, fname: str, title: str) -> None:
    tc = _threshold_cost()
    sub = tc[(tc["model_id"] == model_id) & (tc["horizon"] == H_PRIMARY)]
    ratios = sorted(sub["cost_ratio"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    thresholds = sorted(sub["threshold"].unique())
    for i, r in enumerate(ratios):
        rs = sub[sub["cost_ratio"] == r].set_index("threshold").loc[thresholds]
        ax.plot(thresholds, rs["expected_cost"], "o-", color=RATIO_COLORS[i],
                label=f"C_FN:C_FP = {int(r)}:1")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost (illustrative, units of one FP)")
    ax.set_title(title)
    ax.legend(fontsize=7, title="cost ratio", title_fontsize=7)
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.02, ILLUSTRATIVE, ha="center", fontsize=7, style="italic", color="firebrick")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, fname)


def fig1_threshold_cost_anchor() -> None:
    _threshold_cost_panel(ANCHOR, "fig1_threshold_cost_anchor.png",
                          "Fig 1 — Threshold vs expected cost (Phase 2 anchor, H30)")


def fig2_threshold_cost_temporal() -> None:
    _threshold_cost_panel(TEMPORAL, "fig2_threshold_cost_temporal.png",
                          "Fig 2 — Threshold vs expected cost (temporal W5, H30)")


# ---------------------------------------------------------------------------
# fig3 — cost ratio vs best expected cost, anchor vs temporal
# ---------------------------------------------------------------------------
def fig3_cost_ratio_comparison() -> None:
    comp = pd.read_csv(DECISIONS_DIR / "decision_cost_comparison.csv")
    comp = comp[comp["horizon"] == H_PRIMARY].sort_values("cost_ratio")
    ratios = comp["cost_ratio"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(ratios, comp["anchor_cost"], "o-", color="steelblue", label="anchor (best thr)")
    ax1.plot(ratios, comp["temporal_cost"], "s-", color="seagreen", label="temporal (best thr)")
    ax1.set_xscale("log")
    ax1.set_xticks(ratios, [int(r) for r in ratios])
    ax1.set_xlabel("Cost ratio C_FN : C_FP")
    ax1.set_ylabel("Expected cost at best threshold\n(illustrative, units of one FP)")
    ax1.set_title("Best-case illustrative cost by ratio")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8)

    colors = np.where(comp["abs_reduction"] > 0, "seagreen", "indianred")
    ax2.bar([int(r) for r in ratios], comp["abs_reduction"], color=colors)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel("Cost ratio C_FN : C_FP")
    ax2.set_ylabel("anchor_cost − temporal_cost")
    ax2.set_title("Decision value (positive ⇒ temporal cheaper)")
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("Fig 3 — Anchor vs temporal across illustrative cost ratios (H30)")
    fig.text(0.5, 0.02, ILLUSTRATIVE, ha="center", fontsize=7, style="italic", color="firebrick")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save(fig, "fig3_cost_ratio_comparison.png")


# ---------------------------------------------------------------------------
# fig4 — precision / recall vs threshold
# ---------------------------------------------------------------------------
def fig4_pr_vs_threshold() -> None:
    tc = _threshold_cost()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mid, mk, col in ((ANCHOR, "o", "steelblue"), (TEMPORAL, "s", "seagreen")):
        sub = tc[(tc["model_id"] == mid) & (tc["horizon"] == H_PRIMARY)].sort_values("threshold")
        ax.plot(sub["threshold"], sub["precision"], mk + "-", color=col, alpha=0.9,
                label=f"{mid.split('_')[1]} precision")
        ax.plot(sub["threshold"], sub["recall"], mk + "--", color=col, alpha=0.6,
                label=f"{mid.split('_')[1]} recall")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("rate")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Fig 4 — Precision / recall vs threshold (H30)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, "fig4_pr_vs_threshold.png")


# ---------------------------------------------------------------------------
# fig5 / fig6 — FP / FN vs threshold
# ---------------------------------------------------------------------------
def _error_vs_threshold(col: str, fname: str, title: str, ylab: str) -> None:
    tc = _threshold_cost()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mid, mk, cc in ((ANCHOR, "o", "steelblue"), (TEMPORAL, "s", "seagreen")):
        sub = tc[(tc["model_id"] == mid) & (tc["horizon"] == H_PRIMARY)].sort_values("threshold")
        ax.plot(sub["threshold"], sub[col], mk + "-", color=cc, label=mid)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, fname)


def fig5_fp_vs_threshold() -> None:
    _error_vs_threshold("fp", "fig5_fp_vs_threshold.png",
                        "Fig 5 — False positives vs threshold (H30)",
                        "FP count (rows)")


def fig6_fn_vs_threshold() -> None:
    _error_vs_threshold("fn", "fig6_fn_vs_threshold.png",
                        "Fig 6 — False negatives vs threshold (H30)",
                        "FN count (rows)")


# ---------------------------------------------------------------------------
# fig7 — engine-level aggregates
# ---------------------------------------------------------------------------
def fig7_engine_level() -> None:
    eng = pd.read_csv(DECISIONS_DIR / "engine_level_analysis.csv")
    labels = ["engines_ever\ndetected", "engines_ever\nmissed",
              "engines_with\nany FP", "total false-\nalarm runs"]
    vals_anchor = [
        int(eng[eng["model_id"] == ANCHOR]["ever_detected"].sum()),
        int(eng[eng["model_id"] == ANCHOR]["ever_missed"].sum()),
        int(eng[eng["model_id"] == ANCHOR]["fp"].gt(0).sum()),
        int(eng[eng["model_id"] == ANCHOR]["n_fp_runs"].sum()),
    ]
    vals_temporal = [
        int(eng[eng["model_id"] == TEMPORAL]["ever_detected"].sum()),
        int(eng[eng["model_id"] == TEMPORAL]["ever_missed"].sum()),
        int(eng[eng["model_id"] == TEMPORAL]["fp"].gt(0).sum()),
        int(eng[eng["model_id"] == TEMPORAL]["n_fp_runs"].sum()),
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, vals_anchor, width=0.4, color="steelblue", label="anchor")
    ax.bar(x + 0.2, vals_temporal, width=0.4, color="seagreen", label="temporal")
    ax.set_xticks(x, labels)
    ax.set_ylabel("count (of 50 engines / total runs)")
    ax.set_title("Fig 7 — Engine-level decision & false-alarm aggregates (thr=0.5, H30)")
    for xi, v in zip(x - 0.2, vals_anchor):
        ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + 0.2, vals_temporal):
        ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8)
    _save(fig, "fig7_engine_level.png")


# ---------------------------------------------------------------------------
# fig8 — bootstrap uncertainty
# ---------------------------------------------------------------------------
def fig8_bootstrap() -> None:
    boot = pd.read_csv(DECISIONS_DIR / "bootstrap.csv")
    if boot.empty:
        print("  fig8 skipped (bootstrap disabled)")
        return
    boot = boot.sort_values("cost_ratio")
    ratios = boot["cost_ratio"].to_numpy()
    mean = boot["mean_cost_diff"].to_numpy()
    lo = boot["ci_low"].to_numpy()
    hi = boot["ci_high"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.errorbar(ratios, mean, yerr=[mean - lo, hi - mean], fmt="o-",
                 color="darkorange", capsize=4)
    ax1.axhline(0, color="black", linewidth=1, linestyle="--")
    ax1.set_xscale("log")
    ax1.set_xticks(ratios, [int(r) for r in ratios])
    ax1.set_xlabel("Cost ratio C_FN : C_FP")
    ax1.set_ylabel("temporal − anchor cost (95% engine-bootstrap CI)")
    ax1.set_title("Cost difference (negative ⇒ temporal cheaper)")
    ax1.grid(alpha=0.3, which="both")

    ax2.bar([int(r) for r in ratios], boot["prob_temporal_lower"] * 100, color="seagreen")
    ax2.axhline(50, color="firebrick", linewidth=1, linestyle="--", label="coin-flip (50%)")
    ax2.set_xlabel("Cost ratio C_FN : C_FP")
    ax2.set_ylabel("% bootstrap resamples where temporal cheaper")
    ax2.set_ylim(0, 105)
    ax2.set_title("Confidence temporal is cheaper")
    ax2.legend(fontsize=8)
    fig.suptitle("Fig 8 — Engine-level bootstrap uncertainty (thr=0.5, H30)")
    fig.text(0.5, 0.02, ILLUSTRATIVE, ha="center", fontsize=7, style="italic", color="firebrick")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save(fig, "fig8_bootstrap_uncertainty.png")


def main() -> int:
    print("== Phase 2.6 Figures ==")
    if not (DECISIONS_DIR / "threshold_cost.csv").exists():
        print("ERROR: run scripts/run_phase2_6_decision_analysis.py first.")
        return 1
    for fn in (
        fig1_threshold_cost_anchor,
        fig2_threshold_cost_temporal,
        fig3_cost_ratio_comparison,
        fig4_pr_vs_threshold,
        fig5_fp_vs_threshold,
        fig6_fn_vs_threshold,
        fig7_engine_level,
        fig8_bootstrap,
    ):
        fn()
    print(f"\nWrote figures to {FIGDIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
