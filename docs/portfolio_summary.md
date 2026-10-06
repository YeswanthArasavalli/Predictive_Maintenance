# Portfolio Summary

**Project:** Predictive Maintenance Intelligence — NASA C-MAPSS FD004

**One-liner:** A leakage-safe, fully reproducible prognostics study that
estimates engine Remaining Useful Life and forecasts failure risk over an
operational-cycle horizon — and that deliberately stops at classical +
causal-temporal models because the evidence did not justify a neural network.

## Problem

- **RUL estimation:** predict remaining operational cycles before failure.
- **Failure-risk forecasting:** predict whether an engine fails within H
  operational cycles (H=30 primary; H=14/H=50 sensitivity).
- Done under **leakage-safe** evaluation, because naive time-series ML leaks the
  future and produces impressive-but-fake numbers.

## Technical highlights

- Python · pandas · NumPy · scikit-learn
- **Engine-grouped validation** (the independent unit is the engine, not the row)
- **Train-only, per-regime normalization** across 6 operating conditions
- **Strictly causal temporal features** (lag/diff, rolling, slope) with a
  dedicated pytest causality suite
- **Decision / cost-sensitive evaluation** (missed-failure vs false-alarm cost
  sweep) and a **50-replicate engine-level bootstrap**
- **Sealed official test set** enforced by a firewall in every gate
- 147 automated tests + four phase gates + dependency-locked environment

## Key results (frozen, engine-grouped **validation**)

| Task | Selected model | Metric |
|---|---|---|
| RUL | HistGB + causal temporal (combined, W=3) | MAE ≈ **31.5** cycles (R² ≈ 0.76) |
| RUL baseline | HistGB + per-regime + cycle | MAE ≈ 32.0 |
| Failure-risk H30 | HistGB + combined temporal | **F1 0.912**, PR-AUC 0.974, false alarms 146→135 |
| Decision | temporal vs anchor | ~5–15% lower cost at moderate ratios; **anchor cheaper at extreme miss-cost** |

The majority-class baseline scores **accuracy 0.874 with F1 = 0** — the explicit
reason this project reports PR-AUC/F1 and never "accuracy."

## Engineering quality

- 147 passing tests; Phase 1/2/2.5/2.6 gates all pass (exit 0)
- Git version freeze with commit provenance in the run manifest
- `requirements.lock` exact pins + captured `environment.json`
- Full documentation set (`docs/`) and phase-by-phase audit reports

## Honest limitations

- NASA **simulated** data; validation is internal, not external real-world
- **No production deployment**, no real industrial savings measurement
- Predictions are in **operational cycles**, not calendar days
- Performance may not transfer to real equipment
- Research/portfolio implementation, **not** a serving system

## The headline judgment

Temporal features helped at the margin, but not enough — and not robustly
enough across cost assumptions — to justify escalating to an LSTM. Freezing the
project there is the demonstration of maturity this portfolio is meant to show.
