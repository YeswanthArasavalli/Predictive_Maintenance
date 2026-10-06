# Methodology

This document explains the full scientific reasoning chain — not just **what**
was built, but **why** each step exists. The guiding principle throughout:
*scientific correctness, leakage prevention, and honest reporting are
prioritized over impressive metrics.*

```
Raw data
  → integrity checks
  → split design (engine-grouped)
  → normalization (train-only, per-regime)
  → feature construction (causal)
  → baseline models (classical)
  → temporal features (causal)
  → failure-risk analysis
  → cost sensitivity
  → bootstrap
  → final freeze decision
```

## 1. Raw data → integrity checks (Phase 0)

Before any modeling, the dataset was audited and frozen as immutable
(`data/raw/`, SHA-256 checksums). This phase established the ground-truth facts
the rest of the project relies on:

- The **FD004 readme transposition** (249 train / 248 test actual, not the
  readme's claim) and the decision that **raw files are authoritative**.
- The **RUL cap at 125 operational cycles**.
- Which sensors are duplicate/uninformative (e.g. flat `sensor_16`).
- The 6 operating regimes and 2 fault modes.
- A written **leakage audit** enumerating every place future information could
  leak into features.

Why: a model is only as trustworthy as the data contract. Over-claims in a
prior version of this project's narrative (87% accuracy, 2-week prediction,
$1.15M savings, LSTM superiority) were retired because they could not be
reproduced from data. The audit-first posture is what keeps the remaining
claims defensible.

## 2. Split design — engine-grouped validation (Phase 1)

Rows are **not** independent: all cycles from one engine share that engine's
degradation history. A random row split would leak a engine's future cycles
into training and inflate scores. The project uses a **frozen engine-grouped
train/validation split** — every cycle of a given engine lands on exactly one
side. The official test set remains sealed.

Why: this is the single most important leakage control. It makes validation
scores a realistic estimate of "predict an engine we have never seen."

## 3. Normalization — train-only, per-regime (Phase 1)

- **Train-only statistics**: scalers are fit on the training engines only and
  applied to validation; validation never influences the scaler.
- **Per-regime normalization (mode B)**: because sensor distributions differ
  sharply across the 6 operating conditions, each regime gets its own scaler
  (`results/preprocessing/regime_scalers.joblib`) rather than one global scaler
  (mode A).

Why: global scaling across regimes washes out the signal; fitting on validation
data would leak. The Phase 1 gate verifies both properties.

## 4. Feature construction — strictly causal (Phase 1 / 2.5)

Window/target construction (`src/features/`) uses only information available up
and including the current cycle:

- **Current-snapshot** features (Phase 2): sensor + setting values at cycle *t*
  (plus the cycle index).
- **Causal temporal** features (Phase 2.5): lagged values, first differences,
  rolling means/std, and degradation slopes — all computed **backward in time**
  within an engine. A dedicated pytest causality suite proves no forward-looking
  value enters any feature.

Why: prognostics is a time-series task; using a future cycle to predict the
present is the classic leak. Causality is enforced and *tested*, not assumed.

## 5. Classical baselines (Phase 2)

The first models are deliberately simple and strong:

- RUL: `naive_mean`, `naive_age`, `linear_degradation`, `Ridge`,
  `HistGradientBoostingRegressor`.
- Failure-risk: `majority_class`, `LogisticRegression`,
  `HistGradientBoostingClassifier`.

Feature ablations (config A/B/B+, normalization A/B, with/without cycle) were
run on the validation set. Result: HistGradientBoosting with per-regime
normalization and the cycle index is the strongest classical RUL model
(MAE ≈ 32.0); the majority-class baseline (accuracy 0.874 but F1 = 0) shows why
raw accuracy is meaningless on imbalanced failure data.

Why baselines first: any fancy model must earn its complexity against a proper
classical baseline. Establishing that baseline is what makes the later "why not
an LSTM?" decision honest.

## 6. Causal temporal features (Phase 2.5)

Lag/diff, rolling, slope, and combined causal feature families were layered on
top of the classical learner. Observed effect (validation):

- RUL MAE improved modestly, ~32.0 → ~31.5–31.9 (best combined W=3: 31.52).
- Failure-risk H30 F1 improved ~0.905 → ~0.912 and **false alarms fell**
  (FP 146 → 135 at the F1-argmax threshold).
- No family produced a robust win across **all** horizons and cost assumptions.

Why: temporal features are the natural next step for a degradation task, and
they helped — but the improvement was incremental, not a step change.

## 7. Failure-risk analysis & 8. cost sensitivity (Phase 2.6)

Decision metrics (not just ML metrics) were computed on the existing validation
predictions: confusion matrices, false-negative/false-alarm behavior, alert
fatigue, and a **cost-ratio sensitivity sweep** (cost of a missed failure vs a
false alarm, from 1:1 up to 100:1).

Finding: the temporal model is lower-cost across moderate ratios (~5–15% cheaper
at 1:1–20:1), but the **anchor (no temporal features) is cheaper at extreme
miss-cost ratios (50:1, 100:1)**. So temporal features help the decision but are
not dominant under every assumption.

Why: in maintenance, the *business* cost of a false alarm vs a missed failure
drives the threshold. Evaluating only F1 would hide that the advantage is
cost-dependent.

## 9. Bootstrap analysis (Phase 2.6)

Because the independent unit is the **engine** (249 total, a subset in
validation), not the row, uncertainty was quantified with a **50-replicate
engine-level bootstrap**. Row-level samples are explicitly *not* treated as
independent observations.

Why: over-claiming statistical confidence from pseudo-replicated rows is a
common and serious error. Engine-level bootstrapping gives honest intervals.

## 10. Final freeze decision (Phase 2.7 / 2.7A)

The evidence showed classical + causal-temporal approaches were sufficiently
validated under leakage-safe evaluation, and did **not** establish an unresolved
predictive/decision gap that would justify escalating to a neural model. The
project was therefore **intentionally frozen without adding an LSTM**. See
`docs/model_selection.md`.

Why: stopping when the evidence says the simple model is adequate is disciplined
engineering, not a shortfall. Adding a neural net "for the resume" would have
reintroduced exactly the unjustified-complexity problem the earlier over-claims
came from.
