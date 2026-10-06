# Resume / Portfolio Claim Safety

Defensible bullets based **strictly** on the frozen project and its validation
results. The resume itself is **not** modified here — these are ready-to-use
phrasings that stay inside the evidence.

## Do use

- Built a **leakage-safe** prognostics study on NASA C-MAPSS FD004: engine-
  grouped validation, train-only per-regime normalization, and strictly causal
  temporal features, backed by **147 automated tests** and four phase gates.
- Estimated turbofan **Remaining Useful Life** and **H30 failure risk**, reaching
  RUL **MAE ≈ 31.5 cycles (R² ≈ 0.76)** and failure-risk **F1 ≈ 0.91 / PR-AUC ≈
  0.97** on an engine-grouped validation partition.
- Added a **cost-sensitive decision analysis** across missed-failure :
  false-alarm cost ratios and a **50-replicate engine-level bootstrap**, showing
  the causal-temporal gain is real at moderate cost ratios but **not robust** at
  extremes.
- Made an evidence-based **model-selection decision to freeze without an LSTM**,
  demonstrating that classical + causal-temporal models met the bar and extra
  neural complexity was not justified.

## Do NOT use (not supported by this project)

- ❌ "87% accuracy" (the majority-class baseline scores 0.874 accuracy with F1 = 0;
  accuracy is deliberately not reported as a headline).
- ❌ "2 weeks / days-ahead forecasting" (predictions are in **operational cycles**,
  not calendar time).
- ❌ "$1.15M/year savings" or any dollar figure (decision costs are **illustrative,
  unitless scenario assumptions**).
- ❌ "Production deployment / real-time monitoring" (this is a research/portfolio
  implementation with a read-only demo, not a serving system).
- ❌ Any real-world industrial validation (data is a **simulated benchmark**).
