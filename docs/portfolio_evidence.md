# Portfolio Evidence — Streamlit Demo Screenshot Plan

This document specifies the screenshots to capture from the **working** Streamlit
demo for portfolio/recruiter evidence.

> Do **not** commit any screenshot until it has actually been captured from the
> running application. No image files are committed by this task.

## How to run the demo locally

```powershell
# 1. (Re)build the public demo bundle from frozen artifacts (optional; bundle is committed)
.venv\Scripts\python.exe scripts\build_demo_bundle.py

# 2. Launch the app in the app environment (Python 3.12 + Streamlit)
.venv-app\Scripts\python.exe -m streamlit run streamlit_app.py
```

Recommended viewport: **1440 × 900**, browser zoom 100%.

## Required screenshots

Capture into `docs/evidence/` once the app is running.

| # | Filename | Page | Viewport | What must be visible | Claim it proves |
|---|---|---|---|---|---|
| 1 | `01_overview.png` | Overview | 1440×900 | Title, RUL/failure-risk one-liner, metric cards (249 / 248 / 6 / 2 / 50 / H30 / 147), "Research implementation complete and version-frozen" | The project is understood in ~30 seconds |
| 2 | `02_rul_performance.png` | Model Performance | 1440×900 | RUL comparison table (MAE/RMSE/R²/prognostic), anchor vs causal temporal, "validation" label | Real, frozen regression metrics |
| 3 | `03_failure_risk.png` | Failure Risk | 1440×900 | Recall/Precision/F1/PR-AUC/ROC-AUC cards + confusion matrix + majority-baseline note | Imbalance-aware evaluation, not accuracy |
| 4 | `04_engine_explorer.png` | Engine Explorer | 1440×900 | Engine selector, cycle range, observation count | An interactive, real validation result |
| 5 | `05_rul_trajectory.png` | Engine Explorer | 1440×900 | Actual vs predicted RUL line chart | Real per-engine RUL predictions |
| 6 | `06_failure_probability.png` | Engine Explorer | 1440×900 | Failure-risk probability + threshold line + predicted/actual states | Real failure-risk trajectory + threshold behavior |
| 7 | `07_decision_analysis.png` | Decision Analysis | 1440×900 | Cost-ratio sensitivity curve + "illustrative scenario" warning + engine-level table | Decision/cost awareness without dollar fabrication |
| 8 | `08_model_selection.png` | Why No LSTM? | 1440×900 | Evidence bullets + "no neural escalation was justified" | Disciplined, evidence-based model selection |
| 9 | `09_reproducibility.png` | Reproducibility | 1440×900 | Environment packages, 147 tests, four gate PASS, leakage controls | Reproducibility + leakage-prevention evidence |
| 10 | `10_limitations.png` | Limitations | 1440×900 | All limitation bullets (simulated data, no deployment, cycles ≠ days) | Honest scope, obvious limitations |

## Suggested directory

```text
docs/evidence/
    01_overview.png
    02_rul_performance.png
    03_failure_risk.png
    04_engine_explorer.png
    05_rul_trajectory.png
    06_failure_probability.png
    07_decision_analysis.png
    08_model_selection.png
    09_reproducibility.png
    10_limitations.png
```

## Integrity reminders for every capture

- Metrics shown must match the frozen `demo/metrics.json` (validated by the test
  `test_metric_consistency_confusion_matrix`).
- Never show official test labels or raw data — the app cannot load them by design.
- Never add a dollar figure to any cost chart.
