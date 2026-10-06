# Data Raw — NASA C-MAPSS Turbofan Engine Degradation Simulation

## Dataset name

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Turbofan
Engine Degradation Simulation dataset.

## Source

NASA Ames Research Center and Penn State University — provided via the PHM
Society Data Challenge. The original dataset is distributed as a zip-compressed
package containing training, test, and RUL text files plus this readme.

**Reference:** A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage
Propagation Modeling for Aircraft Engine Degradation Simulation", in the
Proceedings of the 1st International Conference on Prognostics and Health
Management (PHM08), Denver CO, Oct 2008.

## Dataset version / accession information

Recorded as supplied to this project:

- Package contents: `train_FD00{1,2,3,4}.txt`, `test_FD00{1,2,3,4}.txt`,
  `RUL_FD00{1,2,3,4}.txt`, `readme.txt`, `Damage Propagation Modeling.pdf`
- Recorded version label: `CMAPSSData_v1.0`

**No downloaded copy was used.** The files in this directory are an exact,
unmodified copy of the dataset package supplied to the project. No other
version of the C-MAPSS dataset was substituted.

## Acquisition date

Copied into the project on 2026-10-05.

## File inventory and SHA-256 checksums

Checksums recorded at copy time (2026-10-05). These are the single source of
truth for data-integrity verification. Use the file `../.checksums.txt`
(`sha256sum`) for verification.

| Filename | Size (bytes) | SHA-256 |
|---|---:|---|
| RUL_FD001.txt | 429 | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |
| RUL_FD002.txt | 1110 | `c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b` |
| RUL_FD003.txt | 428 | `df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb` |
| RUL_FD004.txt | 1084 | `196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1` |
| readme.txt | 2442 | `4f5270554b775c67e73aff383c5436fd329d6e4cc3d3a116913276fae511269b` |
| test_FD001.txt | 2228855 | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| test_FD002.txt | 5734587 | `de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02` |
| test_FD003.txt | 2826651 | `299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790` |
| test_FD004.txt | 6957759 | `1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5` |
| train_FD001.txt | 3515356 | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| train_FD002.txt | 9082480 | `dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f` |
| train_FD003.txt | 4213862 | `2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2` |
| train_FD004.txt | 10350705 | `27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218` |
| Damage Propagation Modeling.pdf | 434158 | (PDF binary; see `sha256sum` above for a text listing) |

## Attributions and known dataset documentation

- Saxena et al. (2008), PHM Society Conference, Denver, CO — the canonical
  reference.
- `readme.txt` (copied above) documents the four experimental scenarios:

  - FD001 — 100 train / 100 test trajectories, ONE condition (sea level), ONE
    fault mode (HPC degradation).
  - FD002 — 260 train / 259 test trajectories, SIX conditions, ONE fault mode
    (HPC degradation).
  - FD003 — 100 train / 100 test trajectories, ONE condition, TWO fault modes
    (HPC degradation, Fan degradation).
  - FD004 — 248 train / 249 test trajectories (as *stated in readme.txt*), SIX
    conditions, TWO fault modes (HPC degradation, Fan degradation).

  > **Observed discrepancy (Phase 0):** the actual file contents differ from
  > the readme for FD004 — `train_FD004.txt` contains **249** engines and
  > `test_FD004.txt` contains **248** (the readme appears to swap the two).
  > The FD004 test count (248) matches `RUL_FD004.txt` (248 lines) exactly, so
  > test/RUL alignment is sound. See `reports/PHASE_0_DATA_AUDIT.md` §3/§13.

## File schema (per `readme.txt`)

Each row is a snapshot taken during a single operational cycle. Each file has
26 space-separated numeric columns:

1. unit number (unit_id)
2. time, in cycles (cycle)
3. operational setting 1 (operational_setting_1)
4. operational setting 2 (operational_setting_2)
5. operational setting 3 (operational_setting_3)
6. sensor measurement 1 (sensor_01)
7. sensor measurement 2 (sensor_02)
...
26. sensor measurement 21 (sensor_21)

There are 26 columns total: 2 index columns (unit, cycle) + 3 operational
settings + **21 sensors** (columns 6-26). `readme.txt` words the final line as
"26) sensor measurement 26", which is a documentation quirk; the sensor count
is 21, and the Phase-0 schema uses `sensor_01`..`sensor_21`.

## Access / licensing notes

The C-MAPSS dataset is a NASA simulation dataset released for prognostics
research. The specific licensing terms of the original distribution were not
reviewed in detail in this project; the dataset is treated as a research
resource shared by the PHM community. Redistribution of these exact copies from
this repository should carry the Saxena et al. (2008) citation.

## Handling note: source artifact form

The supplied dataset was delivered as 13 individual `.txt` files (not as a
`.zip` archive). The "ZIP" described in the phase plan therefore corresponds to
this file bundle. Integrity is tracked via SHA-256 checksums rather than an
archive hash.

## Provenance policy

- These files are **immutable**. No code in this project writes to
  `data/raw`.
- Any transformation writes to `data/interim` or `data/processed` only.
- Before any analysis, integrity can be re-verified with:

  ```bash
  cd data/raw/CMAPSSData_v1.0
  sha256sum -c ../../.checksums.txt
  ```
