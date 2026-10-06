# Data Directory

- `raw/` — **immutable** source dataset. Never modified by any script.
  Integrity is verified via `data/.checksums.txt`.
- `interim/` — intermediate artifacts (e.g., parsed CSVs, per-dataset tables)
  regenerated from `raw/`. Safe to delete; re-run the loader.
- `processed/` — final prepared datasets (features/targets, scalers) for model
  training. Regenerated from `interim/`.

All derived data is created with fixed seeds so runs are reproducible.
