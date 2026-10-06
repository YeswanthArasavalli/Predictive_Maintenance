# Reproducibility

Every command below was actually run and verified in the maintained development
environment (Windows 11, PowerShell). Commands are **not** listed unless they
were tested. Where something was deliberately *not* re-run, that is stated.

## Environment

- **Python:** 3.14.6 (CPython), Windows-11 / AMD64.
- **Pinned stack (`requirements.lock`)** — this is the environment the frozen
  results were produced in and the one you must use to reproduce them:

  ```
  numpy==2.5.3
  pandas==3.0.6
  scipy==1.18.1
  scikit-learn==1.9.1
  matplotlib==3.11.2
  pyyaml==6.0.3
  pyarrow==25.0.1
  joblib==1.6.0
  pytest==9.1.1
  ```

- `requirements.txt` keeps human-facing **lower bounds** only; `requirements.lock`
  is the authoritative reproducibility constraint set.

> **Critical caveat (verified in this pass):** some tests and gates *refit
> models* and compare against the frozen numbers. These pass **only** in an
> environment matching `requirements.lock`. In an older scikit-learn (e.g.
> `1.8.0` with `numpy 2.4.4`), HistGradientBoosting produces slightly different
> trees and `tests/test_phase2_5.py::TestAnchorReproducesPhase2` fails on a ~0.06
> MAE drift. **Always create the venv from `requirements.lock` before running the
> suite/gates.** Do not run them against a stray global interpreter.

## Setup

```powershell
# from the repository root
python -m venv .venv
.venv\Scripts\Activate.ps1           # PowerShell (Windows)
pip install -r requirements.lock
$env:PYTHONHASHSEED = "0"            # deterministic hashing
python scripts\record_env.py         # writes results/audits/environment.json
```

Expected: `record_env.py` prints a package table matching `requirements.lock`.

## Dataset

Place the official NASA C-MAPSS files at `data/raw/CMAPSSData_v1.0/` (see
`docs/dataset.md` and `data/raw/README.md`). The repo does **not** ship the raw
data. Verify integrity:

```powershell
cd data\raw\CMAPSSData_v1.0
Get-FileHash ..\..\..\.checksums.txt   # or use sha256sum under WSL/git-bash
```

(`sha256sum -c ../../.checksums.txt` from that directory is the canonical check
shown in `data/raw/README.md`.)

## Tests — VERIFIED

```powershell
.venv\Scripts\python.exe -m pytest
```

Verified result in this pass: **147 passed** in the locked `.venv`.

## Phase gates — VERIFIED

All four exit code **0** in the locked `.venv`:

```powershell
.venv\Scripts\python.exe scripts\verify_phase1_gate.py
.venv\Scripts\python.exe scripts\verify_phase2_gate.py
.venv\Scripts\python.exe scripts\verify_phase2_5_gate.py
.venv\Scripts\python.exe scripts\verify_phase2_6_gate.py
```

Note: `verify_phase2_5_gate.py` prints a `Δ` symbol; on a native Windows console
the script reconfigures stdout to UTF-8, so it runs cleanly **without output
redirection** (this was the P1-B fix, confirmed here).

## Full pipeline (regenerates scientific artifacts) — DOCUMENTED, NOT RE-RUN HERE

The end-to-end order is:

```powershell
python scripts\run_phase0_audit.py            # Phase 0 audit JSON/tables
python scripts\phase0_figures.py
python scripts\phase0_gate_analysis.py
python scripts\run_phase1_pipeline.py         # data/processed/*, results/preprocessing/*
python scripts\run_phase2_experiments.py      # results/experiments/phase2/*
python scripts\run_phase2_5_experiments.py    # results/experiments/phase2_5/*
python scripts\run_phase2_6_decision_analysis.py  # results/decisions/phase2_6/*
python scripts\phase2_figures.py / phase2_5_figures.py / phase2_6_figures.py
```

These commands exist and are referenced by the README runbook, but they were
**not re-executed in this release pass** because they overwrite the frozen
Phase 2/2.5/2.6 experiment artifacts — a scientific re-run that this hardening
pass is explicitly not authorized to perform. The committed artifacts under
`results/` remain the authoritative frozen outputs.

## Determinism

- `PYTHONHASHSEED=0`, explicit `random_state`/`seed` wherever a learner/splitter
  is stochastic, and `numpy.random.default_rng(42)` for sampling.
- The frozen split is fixed by hash in `results/audits/phase1_dataset_manifest.json`.

## Honest status statement

> Reproducibility workflow documented and **validated in the maintained
> development environment** for the test suite and all four phase gates. A
> from-scratch clean-environment rebuild and full artifact regeneration were not
> independently re-run in this pass (regeneration would rewrite frozen
> scientific outputs). Do not describe this as "fully reproduced from a clean
> clone" unless a clean-machine rebuild has actually been performed.
