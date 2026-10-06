# Architecture

This is a **research/portfolio ML implementation**, not a serving system. The
architecture is a reproducible offline pipeline with hard leakage boundaries and
gate-verified artifacts. There is no API, online inference, or deployed
monitoring.

```
                 ┌─────────────────────────────────────────────┐
                 │  data/raw/CMAPSSData_v1.0  (immutable)       │
                 │  sealed: test_FD004.txt, RUL_FD004.txt       │
                 └───────────────┬─────────────────────────────┘
                                 │  load + integrity (src/data)
                                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  DATA LAYER   src/data                                        │
   │   loader · integrity · checks · splitter(engine-grouped) ·    │
   │   regime(6 conditions) · contract(feature column policy)      │
   └───────────────┬────────────────────────────────────────────┘
                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  FEATURE LAYER  src/features                                  │
   │   sensors · normalization(train-only, per-regime) ·           │
   │   windows · targets(RUL cap 125, H30/14/50 labels) ·         │
   │   temporal(causal lag/diff/rolling/slope)                     │
   └───────────────┬────────────────────────────────────────────┘
                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  MODELING LAYER  src/models                                   │
   │   ridge · logistic · gradient_boosting(HistGB) ·             │
   │   linear_degradation · baselines(naive) · features            │
   └───────────────┬────────────────────────────────────────────┘
                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  EVALUATION LAYER  src/evaluation                             │
   │   metrics: MAE/RMSE/R²/prognostic · recall/precision/F1/     │
   │   PR-AUC/ROC-AUC/confusion                                   │
   └───────────────┬────────────────────────────────────────────┘
                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  DECISION LAYER  src/business                                 │
   │   decision.py: thresholds · cost-ratio sweep · alert fatigue ·│
   │   engine-level detection · false-neg analysis                │
   └───────────────┬────────────────────────────────────────────┘
                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  REPORTING / RESULTS LAYER  src/pipeline + results/          │
   │   experiment runner · registry · manifest(git provenance) ·  │
   │   artifacts → results/{experiments,decisions,audits,tables,  │
   │   figures,preprocessing}                                      │
   └──────────────────────────────────────────────────────────────┘

   Cross-cutting: tests/ (147) + scripts/verify_phase*_gate.py enforce
   leakage safety, causality, determinism, and the sealed-test firewall.
```

## Layers

- **Data layer (`src/data`)** — parsing, schema/integrity checks, regime
  assignment, the frozen engine-grouped splitter, and the feature-column
  contract that forbids leakage-prone columns.
- **Feature layer (`src/features`)** — train-only per-regime normalization,
  window/target construction (RUL cap 125; failure-risk H14/H30/H50), and
  strictly **causal** temporal features.
- **Modeling layer (`src/models`)** — classical learners only (Ridge, Logistic,
  HistGradientBoosting, naive/linear degradation baselines). `describe()`
  metadata is recorded; **fitted models are not persisted as inference
  binaries**.
- **Evaluation layer (`src/evaluation`)** — RUL and classification metrics.
- **Decision layer (`src/business`)** — turns predictions into maintenance
  decisions: thresholds, cost-ratio sensitivity, alert fatigue, engine-level
  detection.
- **Reporting/results layer (`src/pipeline`, `results/`)** — experiment runner,
  registry, manifest (with git commit provenance), and all serialized JSON/CSV
  artifacts and figures.

## Persistence note

`results/preprocessing/regime_scalers.joblib` persists *preprocessing* objects
(scalers/mappings) because they are part of the reproducible dataset contract.
It is a genuine pipeline output, **not** a shipped model binary. No trained-model
serialization or inference path exists; scoring brand-new data requires re-running
the pipeline. Model persistence/inference is listed as future work.

## Not an enterprise architecture

There is deliberately no message queue, container orchestrator, feature store,
experiment tracker, or serving stack. Adding any of those "for appearance" would
be misleading for a frozen research implementation.
