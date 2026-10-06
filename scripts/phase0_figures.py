"""Phase-0 figures (§17): publication-quality, non-misleading plots.

Run:  python scripts/phase0_figures.py
Writes PNGs to results/figures/phase_0/.

Every figure has a title, axis labels (+ units where applicable) and a legend
where necessary. Figures are generated deterministically from raw data.
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

from src.data import load_dataset, load_rul  # noqa: E402
from src.data.audit import SENSORS, sensor_correlation  # noqa: E402
from src.data.loader import CYCLE, UNIT_ID, derive_train_rul  # noqa: E402
from src.data.audit import _regime_key  # noqa: E402

FIGDIR = ROOT / "results" / "figures" / "phase_0"
FIGDIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

PRIMARY = "FD004"
DEGRAD_SENSORS = ["sensor_02", "sensor_08", "sensor_14", "sensor_17"]  # candidate indicators

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})


def _save(fig, name: str) -> None:
    fig.savefig(FIGDIR / name)
    plt.close(fig)
    print("  fig:", name)


def fig1_trajectory_lengths():
    fig, ax = plt.subplots(figsize=(7, 4))
    data = []
    labels = []
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        tr = load_dataset(fd, "train")
        data.append(tr.groupby(UNIT_ID)[CYCLE].max().to_numpy())
        labels.append(f"{fd} train")
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("Total operational cycles per engine (cycles)")
    ax.set_xlabel("Dataset")
    ax.set_title("Fig 1 — Engine trajectory-length distribution (train)")
    _save(fig, "fig1_trajectory_length_distribution.png")


def fig2_rul_distribution():
    fig, ax = plt.subplots(figsize=(7, 4))
    tr = load_dataset(PRIMARY, "train")
    rul = derive_train_rul(tr)
    ax.hist(rul.to_numpy(), bins=50, color="#2b6cb0")
    ax.set_xlabel("Training RUL = engine_max_cycle - current_cycle (operational cycles)")
    ax.set_ylabel("Row count")
    ax.set_title(f"Fig 2 — RUL distribution ({PRIMARY} train)")
    _save(fig, "fig2_rul_distribution.png")


def fig3_sensor_variance():
    fig, ax = plt.subplots(figsize=(8, 4))
    tr = load_dataset(PRIMARY, "train")
    var = tr[SENSORS].var(ddof=1).sort_values()
    ax.barh(var.index, np.log10(var.to_numpy() + 1e-12), color="#2c7a7b")
    ax.set_xlabel("log10(sensor variance)  [raw units]")
    ax.set_ylabel("Sensor")
    ax.set_title(f"Fig 3 — Sensor variance distribution ({PRIMARY} train)")
    _save(fig, "fig3_sensor_variance_distribution.png")


def fig4_correlation_matrix():
    tr = load_dataset(PRIMARY, "train")
    corr = sensor_correlation(tr)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels([c.replace("sensor_", "s") for c in corr.columns], rotation=90, fontsize=7)
    ax.set_yticklabels([c.replace("sensor_", "s") for c in corr.columns], fontsize=7)
    ax.set_title(f"Fig 4 — Sensor correlation matrix ({PRIMARY} train, pooled)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
    _save(fig, "fig4_sensor_correlation_matrix.png")


def fig5_degradation_trajectories():
    # FD001 = single operating condition → clean degradation illustration.
    tr = load_dataset("FD001", "train")
    engines = tr[UNIT_ID].unique()
    rng = np.random.default_rng(42)
    chosen = rng.choice(engines, size=4, replace=False)
    fig, axes = plt.subplots(1, len(DEGRAD_SENSORS), figsize=(13, 3.2), sharex=True)
    for ax, s in zip(axes, DEGRAD_SENSORS):
        for e in chosen:
            g = tr[tr[UNIT_ID] == e].sort_values(CYCLE)
            ax.plot(g[CYCLE], g[s], lw=1, label=f"engine {e}")
        ax.set_xlabel("Operational cycle")
        ax.set_ylabel(s.replace("sensor_", "Sensor "))
        ax.legend(fontsize=6)
    axes[0].set_title("Fig 5 — Representative degradation trajectories (FD001, one condition)")
    _save(fig, "fig5_degradation_trajectories.png")


def fig6_operating_condition_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, fd in zip(axes, ["FD002", "FD004"]):
        tr = load_dataset(fd, "train")
        key = _regime_key(tr).value_counts()
        ax.bar(range(len(key)), key.to_numpy(), color="#b7791f")
        ax.set_xticks(range(len(key)))
        ax.set_xticklabels([k.replace("|", ",") for k in key.index], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Rows")
        ax.set_xlabel(f"{fd} regime (setting1, setting2, setting3)")
        ax.set_title(f"Fig 6 — Operating-condition distribution ({fd})")
    _save(fig, "fig6_operating_condition_distribution.png")


def fig7_sensor_by_regime():
    tr = load_dataset(PRIMARY, "train")
    key = _regime_key(tr)
    sensor = "sensor_17"
    fig, ax = plt.subplots(figsize=(8, 4))
    regimes = sorted(key.unique())
    for i, r in enumerate(regimes):
        v = tr.loc[key == r, sensor].to_numpy()
        ax.boxplot([v], positions=[i], widths=0.6)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([r.replace("|", ",") for r in regimes], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"{sensor} value")
    ax.set_xlabel("Operating regime (setting1, setting2, setting3)")
    ax.set_title(f"Fig 7 — {sensor} distribution by regime ({PRIMARY} train)")
    _save(fig, "fig7_sensor_by_regime.png")


def fig8_class_balance_simulation():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    tr = load_dataset(PRIMARY, "train")
    rul = derive_train_rul(tr).to_numpy()
    horizons = [7, 14, 21, 30, 40, 50]
    pos_rate = [float((rul <= h).mean() * 100) for h in horizons]
    ax.plot(horizons, pos_rate, marker="o", color="#2b6cb0")
    for h in [14, 30, 50]:
        pr = float((rul <= h).mean() * 100)
        ax.annotate(f"H={h}\n{pr:.1f}%", (h, pr), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel("Failure-risk horizon H (operational cycles)")
    ax.set_ylabel("Positive rate (% of rows with RUL <= H)")
    ax.set_title(f"Fig 8 — Failure-risk class-balance vs horizon ({PRIMARY} train)")
    _save(fig, "fig8_failure_risk_class_balance.png")


def main() -> int:
    print("== Phase 0 figures ==")
    fig1_trajectory_lengths()
    fig2_rul_distribution()
    fig3_sensor_variance()
    fig4_correlation_matrix()
    fig5_degradation_trajectories()
    fig6_operating_condition_distribution()
    fig7_sensor_by_regime()
    fig8_class_balance_simulation()
    print("== done ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
