# Model Selection — Why There Is No LSTM

This project's original portfolio concept mentioned an LSTM. It does not contain
one. That is a **deliberate, evidence-based decision**, not an omission.

## Initial concept

An LSTM-based deep-learning prognostics project was originally envisioned —
the common "RUL prediction with an LSTM" portfolio framing.

## What changed

Before choosing any neural architecture, the project first did the work a neural
model would have to beat:

1. A leakage-safe data audit and frozen engine-grouped split (Phase 0/1).
2. Strong **classical baselines** (Ridge, Logistic, HistGradientBoosting) under
   train-only per-regime normalization (Phase 2).
3. **Causal temporal-feature baselines** (lag/diff, rolling, slope, combined)
   layered on the same classical learner (Phase 2.5).
4. A **decision & cost-sensitive** evaluation, not just accuracy (Phase 2.6).

Only after that evidence would a neural escalation be justified — if it were
justified at all.

## Evidence

On the engine-grouped validation partition:

- Temporal features **modestly improved RUL**: MAE ≈ 32.0 → ≈ 31.5 (combined,
  W=3), with no uniform win across window sizes.
- Temporal features **improved some failure-risk metrics** at H30: F1 ≈ 0.905 →
  0.912, PR-AUC 0.968 → 0.974, and **false alarms fell** (FP 146 → 135).
- The benefit was **not robust across all cost assumptions**: under extreme
  missed-failure cost ratios (50:1, 100:1) the no-temporal **anchor was the
  lower-cost model**.
- No experiment revealed a large, persistent predictive/decision gap of the kind
  that a sequence model would be needed to close.

## Decision

**Freeze the project without adding an LSTM.** The evidence did not demonstrate
an unresolved problem that additional model complexity would solve. Adding a
neural network to make the repository look more advanced would have:

- broken the "beatable-baseline first" discipline,
- made the results harder to reproduce deterministically,
- and — critically — echoed the earlier, unjustified over-claims
  (LSTM / "87% accuracy" / "2 weeks ahead" / "$1.15M/year savings") that this
  project explicitly retired as non-reproducible.

## How to read this

Choosing the simplest model that the evidence supports, and *stopping there*, is
the point. The strongest signal in this repository is not a metric — it is the
demonstration of leakage-safe evaluation, honest uncertainty (engine-level
bootstrap), decision/cost awareness, and the restraint to not add complexity the
data does not earn.

Future neural work is legitimate **only** as an explicitly-labelled experiment
that must beat these frozen baselines on a properly sealed test set — see
"Future work" in the README. It is not automatically required.
