# Evaluation

All metrics below are computed on the **engine-grouped validation partition** of
FD004. The official test set is sealed and untouched (see `docs/dataset.md`).
Every number here is the frozen authoritative value recorded in
`results/experiments/**` and `results/decisions/phase2_6/**`.

## The unit of independence

The independent observational unit is the **engine**, not the row. A single
engine contributes many cycle-rows that are strongly correlated. Row-level
samples are therefore **not** independent observations, and no metric here is
presented as if they were. Uncertainty is quantified with an **engine-level
bootstrap** (below).

## RUL metrics (Task A)

Reported per standard C-MAPSS practice:

- **MAE** — mean absolute error in operational cycles (lower is better).
- **RMSE** — root mean squared error (penalizes large misses).
- **R²** — coefficient of determination on validation.
- **Prognostic score** — the asymmetric C-MAPSS exponential penalty
  (over-estimating remaining life is punished harder than under-estimating).
  Reported both row-level and last-cycle-per-engine.

Frozen RUL results (validation):

| Model | Features | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| naive_mean | — | 72.90 | 88.62 | ~0.00 |
| naive_age | age only | 61.15 | 71.53 | 0.349 |
| Ridge (global norm) | snapshot | 44.87 | 55.90 | 0.602 |
| **HistGB (per-regime + cycle)** | snapshot | **32.01** | 43.25 | 0.762 |
| HistGB + combined temporal (W=3) | snapshot+temporal | **31.52** | 43.24 | 0.762 |
| HistGB + lag_diff temporal (W=5) | snapshot+temporal | 31.64 | 42.93 | 0.765 |
| linear_degradation | extrapolation | 323.77 | — | negative (pathological) |

Reading: classical HistGB with per-regime normalization and the cycle index is
the strong baseline (MAE ≈ 32.0). Causal temporal features shave ~0.1–0.5 cycles
off MAE — a real but **modest** improvement, and not uniform across windows.

## Failure-risk metrics (Task B)

Binary "will fail within H operational cycles" (H = 30 primary; 14 and 50 for
sensitivity). Because the class is imbalanced (positive rate ≈ 12.6% at H30),
**accuracy is explicitly de-emphasized**:

- Recall, Precision, F1
- PR-AUC (average precision — the informative curve under imbalance)
- ROC-AUC
- Full confusion matrix (TP/FP/TN/FN)

Frozen H30 results at the F1-argmax threshold:

| Model | Features | Precision | Recall | F1 | PR-AUC | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| majority_class | — | 0.00 | 0.00 | 0.00 | — | 0 | 1550 |
| HistGB + cycle (anchor) | snapshot | 0.898 | 0.975 | 0.905 | 0.968 | 146 | 158 |
| HistGB + combined temporal | snapshot+temporal | **0.906** | 0.977 | **0.912** | **0.974** | **135** | 145 |
| HistGB + combined (H14) | snapshot+temporal | 0.908 | 0.871* | 0.872 | 0.961 | 100 | 69 |
| HistGB + combined (H50) | snapshot+temporal | 0.916 | 0.876* | 0.876 | 0.970 | 330 | 213 |

*The majority-class baseline reaches **accuracy 0.874 with F1 = 0** — the exact
reason this project never leads with accuracy and never claims "87% accuracy."
Temporal features raise precision/F1 and cut false alarms (FP 146 → 135) at
H30.

## Decision metrics (Phase 2.6)

Beyond classification scores, the project evaluates the maintenance *decision*:

- **False positives / false alarms** and **false negatives / missed failures**.
- **Alert fatigue** — repeated back-to-back alarms on the same engine.
- **Engine-level detection** — did the model flag a failing engine in time,
  aggregated per engine rather than per row.
- **Cost-sensitive thresholds** — total decision cost across a sweep of the
  (missed-failure : false-alarm) cost ratio.

Frozen cost-ratio comparison (H30, anchor vs temporal, lower cost is better):

| Miss:false-alarm cost ratio | Anchor cost | Temporal cost | Change | Lower-cost model |
|---:|---:|---:|---:|---|
| 1 | 309 | 293 | −5.2% | temporal |
| 2 | 456 | 386 | −15.4% | temporal |
| 5 | 662 | 591 | −10.7% | temporal |
| 10 | 844 | 715 | −15.3% | temporal |
| 20 | 916 | 831 | −9.3% | temporal |
| 50 | 1096 | 1131 | +3.2% | **anchor** |
| 100 | 1396 | 1631 | +16.8% | **anchor** |

Reading: temporal features reduce decision cost at moderate cost ratios
(~5–15%), but the advantage **reverses at extreme missed-failure cost (50:1,
100:1)**. All costs are **illustrative units on the validation partition**, not
dollars and not a business-ROI claim.

## Statistical analysis — 50-engine bootstrap

A **50-replicate bootstrap resampled at the engine level** was used to estimate
uncertainty on the decision/validation metrics. What it establishes: a
reasonable interval around validation performance *given the engines we have*.
What it does **not** establish: external/production generalization, or real-world
data drift. The resampling unit is the engine precisely so that pseudo-replicated
rows are not mistaken for independent evidence.
