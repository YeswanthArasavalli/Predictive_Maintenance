# Predictive Maintenance Intelligence — NASA C-MAPSS FD004

**Leakage-safe Remaining-Useful-Life and failure-risk forecasting on simulated
turbofan engines — with the discipline to stop at the simplest model the evidence
supports (no LSTM, because the data did not earn one).**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](docs/reproducibility.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Read this first:** this is a **completed research / portfolio ML
> implementation**, version-frozen after an audited gate process. It is **not** a
> production system — there is no API, no online inference, no deployment, and no
> real industrial savings claim. Every number below comes from a
> **leakage-safe, engine-grouped validation partition**.

---

## Demo

A **read-only Streamlit portfolio demo** presents the frozen validation artifacts
(nothing is retrained on load). It is a presentation layer over validated outputs —
**separate from** the research repository itself.

| | |
|---|---|
| **Live interactive demo** | `LIVE_DEMO_URL: <TO_BE_FILLED_AFTER_DEPLOYMENT>` |
| **GitHub repository** | This repository (authoritative research implementation) |

Run it locally:

```powershell
.venv-app\Scripts\python.exe -m streamlit run streamlit_app.py
```

The live URL is intentionally a placeholder until a real Streamlit Community Cloud
deployment exists — it is **not** invented here. See
[docs/portfolio_evidence.md](docs/portfolio_evidence.md).

---

## The problem (60 seconds)

Engines (and industrial equipment generally) degrade over time. Predictive
maintenance asks two things from sensor telemetry:

1. **RUL (Remaining Useful Life):** how many *operational cycles* until failure?
2. **Failure-risk forecasting:** will this engine fail within the next *H* cycles?

The trap: time-series ML **leaks the future** easily (scaling on all data,
randomly splitting rows from the same engine, using later cycles to predict
earlier ones). Leakage produces beautiful, meaningless metrics. This project's
entire structure exists to prevent that — and to report honestly when a fancy
model *isn't* warranted.

An **operational cycle** is a sampling step in a degradation trajectory, **not a
calendar day**. Nothing here predicts "2 weeks ahead."

## Dataset

NASA **C-MAPSS** (Turbofan Engine Degradation Simulation). Sub-set **FD004**:

| | |
|---|---|
| Training engines | **249** |
| Test engines | **248** (sealed — never used) |
| Operating regimes | **6** |
| Fault modes | **2** (HPC + Fan degradation) |
| Raw columns | **26** (unit, cycle, 3 settings, 21 sensors) |

The dataset is **not redistributed here**; see [docs/dataset.md](docs/dataset.md)
for how to obtain it, the schema, and the documented FD004 readme *transposition*
discrepancy (the raw files are treated as authoritative).

## Methodology

A gated, audit-first pipeline (full reasoning in
[docs/methodology.md](docs/methodology.md)):

1. Data-integrity audit + leakage audit (Phase 0)
2. **Engine-grouped** train/validation split — frozen (Phase 1)
3. **Train-only, per-regime** normalization (Phase 1)
4. **Strictly causal** window & temporal features (Phase 1 / 2.5)
5. Classical baselines (Phase 2)
6. Causal temporal-feature baselines (Phase 2.5)
7. Failure-risk evaluation (Phase 2 / 2.5)
8. Decision & **cost-sensitive** analysis (Phase 2.6)
9. **Engine-level bootstrap** (Phase 2.6)
10. Final **freeze** decision (Phase 2.7A)

## Models

The validated solution is **classical + causal-temporal** only:

- **Ridge**, **LogisticRegression**
- **HistGradientBoostingRegressor / Classifier**
- naive + linear-degradation reference baselines
- **causal temporal features**: lag/difference, rolling, slope, combined

No neural model is implemented, because none was justified by the evidence (see
below and [docs/model_selection.md](docs/model_selection.md)).

## Results (frozen, engine-grouped **validation**)

Full detail: [docs/evaluation.md](docs/evaluation.md).

### RUL (operational cycles)

| Model | MAE | R² |
|---|---:|---:|
| naive_mean | 72.9 | ~0.0 |
| HistGB + per-regime + cycle (baseline) | **32.0** | 0.762 |
| HistGB + causal temporal (combined, W=3) | **31.5** | 0.762 |

### Failure risk — H30 (30 operational cycles)

| Model | Precision | Recall | F1 | PR-AUC | False alarms |
|---|---:|---:|---:|---:|---:|
| majority class | 0.00 | 0.00 | 0.00 | — | — |
| HistGB + cycle (anchor) | 0.898 | 0.975 | 0.905 | 0.968 | 146 |
| HistGB + combined temporal | **0.906** | 0.977 | **0.912** | **0.974** | **135** |

The majority-class baseline reaches **accuracy 0.874 with F1 = 0** — exactly why
this project reports PR-AUC/F1 and **never** leads with "accuracy."

### Decision / cost

Temporal features lower decision cost by ~5–15% across moderate
(missed-failure : false-alarm) cost ratios, but the **no-temporal anchor is
cheaper at extreme ratios (50:1, 100:1)** — the advantage is **not robust to
every cost assumption**. Costs are illustrative validation units, not dollars.

### Validation & uncertainty

Engine-grouped validation + a **50-replicate engine-level bootstrap**. The
independent unit is the engine, never the row.

## Important finding — why there is no LSTM

> Classical and causal-temporal approaches were sufficiently validated under
> leakage-safe evaluation. Temporal features gave modest, sometimes
> cost-dependent improvements, but the evidence did **not** establish a robust
> unresolved predictive/decision gap requiring a neural network. The project was
> therefore **intentionally frozen without adding an LSTM**.

This is a **feature, not an apology**: disciplined model selection beats
complexity added for resume appearance. Earlier, non-reproducible portfolio claims
("87% accuracy", "2 weeks ahead", "$1.15M/year savings", "LSTM superiority") were
retired and are **not** reintroduced anywhere.

## Reproducibility

- Python 3.10+ (validated on 3.14.6); exact pins in `requirements.lock`
- Determinism: `PYTHONHASHSEED=0`, explicit `random_state`, seeded RNG
- **147 automated tests** + four phase gates (Phase 1 / 2 / 2.5 / 2.6)
- Captured environment: `results/audits/environment.json`

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -r requirements.lock
python -m pytest                              # 147 passed
python scripts/verify_phase1_gate.py         # exit 0  (repeat for 2 / 2_5 / 2_6)
```

> Run the suite/gates **inside the locked `.venv`**. Refit-and-compare tests are
> pinned to scikit-learn 1.9.1 and will drift on other versions. See
> [docs/reproducibility.md](docs/reproducibility.md).

## Repository status

> **Research implementation complete and version-frozen.**

This is distinct from production deployment: there is no serving layer, model
persistence for inference, monitoring, or online scoring.

## Limitations

- NASA **simulated** dataset — validation is internal, **not external real-world**
  validation
- **No production deployment** and **no real industrial savings measurement**
- Predictions are in **operational cycles**, not calendar days
- Performance **may not transfer** to real equipment
- The sealed official test set was never used; results are validation-set results

## Future work (honest, not a checklist of buzzwords)

- External validation on the sealed test set / a second dataset
- Model persistence + a real inference path
- Probability calibration and monitoring
- Real industrial telemetry
- A neural model **only** if it demonstrably beats these frozen baselines

## Documentation

| Doc | Purpose |
|---|---|
| [docs/methodology.md](docs/methodology.md) | The full scientific reasoning chain |
| [docs/model_selection.md](docs/model_selection.md) | Why there is no LSTM |
| [docs/evaluation.md](docs/evaluation.md) | Metrics, unit of independence, bootstrap |
| [docs/dataset.md](docs/dataset.md) | C-MAPSS / FD004, schema, sealing policy |
| [docs/architecture.md](docs/architecture.md) | Pipeline layers and boundaries |
| [docs/reproducibility.md](docs/reproducibility.md) | Tested setup / run commands |
| [docs/portfolio_summary.md](docs/portfolio_summary.md) | Recruiter-oriented summary |
| [docs/portfolio_evidence.md](docs/portfolio_evidence.md) | Streamlit demo + screenshot plan |
| [docs/resume_claims.md](docs/resume_claims.md) | Defensible vs unsupported claims |
| [docs/release_status.md](docs/release_status.md) | Current vs historical status |
| [docs/FINAL_RELEASE_REPORT.md](docs/FINAL_RELEASE_REPORT.md) | Release-hardening report |

## Anti-fabrication policy

The dataset and reproducible experiments are ground truth. Existing portfolio
claims are not; they are re-evaluated against the data and reported as observed.
No metric, sensor behavior, or dataset property is stated without computation.

## Citation

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation. *Proceedings of the 1st
International Conference on Prognostics and Health Management*, Denver, CO.
