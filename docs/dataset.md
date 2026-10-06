# Dataset — NASA C-MAPSS FD004

## What the dataset is

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) is a
simulated turbofan-engine degradation dataset produced by the NASA C-MAPSS
engine model. Each engine is run "run-to-failure": a sequence of operational
cycles, each cycle containing 3 operational settings and 21 sensor
measurements, ending when the engine fails.

**Reference:** Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage
Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.
*Proceedings of the 1st International Conference on Prognostics and Health
Management*, Denver, CO.

## Why FD004

C-MAPSS has four sub-sets (FD001–FD004). This project uses **FD004**, the most
operationally complex set:

| Property | FD004 value |
|---|---|
| Training engines | **249** |
| Test engines | **248** |
| Operating conditions (regimes) | **6** |
| Fault modes | **2** (HPC degradation + Fan degradation) |
| Raw columns per row | **26** (unit, cycle, 3 settings, 21 sensors) |

FD004 mixes multiple operating regimes and two fault modes, so a naive
train-only-normalization or single-model approach is strongly penalized. That
makes it a hard, honest test of leakage-safe methodology.

## Raw schema (26 columns)

1. `unit_id` — engine index
2. `cycle` — operational cycle number (time step)
3–5. `operational_setting_1..3` — throttle/altitude/commanded-mach settings
6–26. `sensor_01..21` — sensor measurements

`readme.txt` words the last line as "26) sensor measurement 26"; this is a
documentation quirk. The real sensor count is **21**. Some sensors are
constant or near-noise (e.g. several are flat); their treatment is documented in
the Phase 0 audit.

## Train / test structure and the RUL definition

- `train_FD004.txt` — full run-to-failure trajectories (RUL is derivable: at
  cycle *t* of an engine with last cycle *L*, `RUL = L − t`).
- `test_FD004.txt` — trajectories truncated before failure.
- `RUL_FD004.txt` — the true RUL at the **last provided cycle** of each test
  engine (the official test labels). **These are sealed and never used during
  Phases 0–2.6** (see sealing policy below).

RUL is **clipped at 125 operational cycles** (the standard C-MAPSS convention):
once an engine's true remaining life exceeds 125 cycles it is treated as 125,
because very-early degradation is not reliably predictable.

## Operational cycles (not calendar days)

A cycle is a sampling interval in the degradation trajectory, **not a day**.
There is no calendar-time axis in C-MAPSS. Statements like "predicts 2 weeks
ahead" are **not appropriate** for this dataset; the correct unit is
**operational cycles**. Failure-risk horizons in this project are expressed as
H14 / H30 / H50 **operational cycles**.

## The FD004 readme transposition discrepancy

The supplied `readme.txt` states FD004 has 248 train / 249 test engines. The
**actual file contents** are 249 train / 248 test — the readme appears to swap
the two. The test count (248) matches `RUL_FD004.txt` exactly (248 lines), so
test/RUL alignment is sound.

Resolution used in this project: **the raw supplied files are computationally
authoritative**, not the readme prose. Full evidence:
`reports/FD004_DISCREPANCY.md` and `reports/PHASE_0_DATA_AUDIT.md` §3/§13.

## Six operating regimes & two fault modes

The three operational settings define 6 discrete operating conditions. Because
sensor distributions shift strongly across regimes, normalization is done
**per-regime** (mode B) using **train-only statistics**, not a single global
scaler. Two fault modes (HPC and Fan degradation) are mixed without per-engine
fault labels, which is part of why FD004 is hard.

## Duplicate / uninformative sensors & `sensor_16`

Several sensors are near-constant or pure noise and are excluded from modeling
(see `results/preprocessing/sensor_config.json`). `sensor_16` in particular is a
flat/uninformative channel and is not used as a predictive feature. The exact
dropped set is defined in the Phase 0 sensor analysis and encoded in the feature
schema.

## Train/validation split

A **frozen, engine-grouped** train/validation split is used (Phase 1). All
cycles of a given engine stay on one side of the split — this is the single most
important leakage control in the project (see `docs/evaluation.md`). The split
is fixed in `results/audits/phase1_dataset_manifest.json` and never changed.

## Test-set sealing policy

The official FD004 test partition (`test_FD004.txt` + `RUL_FD004.txt`) is the
held-out ground truth for a final external evaluation. In this project it is:

- **Never** loaded by any Phase 0–2.6 experiment or gate.
- Verified by an explicit "official-test firewall" in each gate script.
- Excluded from Git publication (`data/raw/*` is gitignored; only
  `data/raw/README.md` and `data/.checksums.txt` are tracked).

All reported metrics in this repository come from the **engine-grouped
validation partition**, not the sealed official test set.

## How to obtain and place the data

The dataset is **not redistributed by this repository**. Obtain the official
C-MAPSS package from NASA / the PHM Society, then place the files at:

```
data/raw/CMAPSSData_v1.0/
    train_FD001.txt ... train_FD004.txt
    test_FD001.txt  ... test_FD004.txt
    RUL_FD001.txt   ... RUL_FD004.txt
    readme.txt
```

Integrity can be re-checked against `data/.checksums.txt` (see
`data/raw/README.md` for the per-file SHA-256 table).
