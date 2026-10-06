# FD004 Train/Test Count Discrepancy — Investigation (Condition 1)

**Scope:** Explain the mismatch between the supplied `readme.txt` and the actual
FD004 file contents. **No raw file was modified, renamed, swapped, or corrected.**
All numbers below are computed by `scripts/phase0_gate_analysis.py` and stored in
`results/audits/gate_fd004_discrepancy.json`.

---

## 1. Exact counts in the supplied files (observed)

| Quantity | Observed value |
|---|---:|
| `train_FD004.txt` distinct engines | **249** (unit ids 1–249, contiguous) |
| `train_FD004.txt` rows | 61,249 |
| `test_FD004.txt` distinct engines | **248** (unit ids 1–248, contiguous) |
| `test_FD004.txt` rows | 41,214 |
| `RUL_FD004.txt` values | **248** |

## 2. Exact counts stated in the documentation

`readme.txt` (and the transcribed `data/raw/README.md`) state:

```
Data Set: FD004
Train trjectories: 248
Test trajectories: 249
```

So the readme claims **248 train / 249 test**, while the files contain
**249 train / 248 test**. The two numbers are **transposed** relative to the data.

## 3. Were the files actually swapped, or is this a package inconsistency?

Evidence that the **files are correct and internally consistent** (i.e. the
readme numbers are transposed, NOT the files swapped):

- **Every training engine terminates at RUL = 0** (run-to-failure). This is the
  defining property of a C-MAPSS *training* file. `train_FD004.txt` satisfies it
  for all 249 engines. If the files had been swapped, the "train" file would
  contain truncated trajectories that do **not** reach failure — it does not.
- **Every test engine is truncated** (ends before failure). Mean trajectory length:
  train **246.0** cycles vs test **166.2** cycles (test is shorter, as expected
  for a truncated hold-out).
- **The test engine count (248) equals the RUL vector length (248) exactly**, and
  the test unit ids match the RUL ids one-to-one. A test file can only be scored
  against a RUL vector of the same length; this 248↔248 pairing confirms the
  248-engine file is genuinely the **test** set.
- Implied failure cycle for test engines (`last_cycle + supplied_RUL`) ranges
  126–554, consistent with real run-to-failure horizons — again confirming these
  are truncated test trajectories, not full training runs.

**Conclusion:** this is a **documentation/package inconsistency in `readme.txt`**,
not a file swap. The archive's file contents are coherent.

## 4. Does `RUL_FD004.txt` contain 248 values?

**Yes.** 248 integer values, all positive, min = 6, max = 195.

## 5. Do all 248 test engines have exactly one RUL value?

**Yes.** 248 test engines ↔ 248 RUL values, aligned one-to-one by engine id
(`test_ids_match_rul_ids = true`). No engine is missing a label and none has two.

## 6. Is the 249th training engine a valid, complete run-to-failure trajectory?

**Yes.** Unit 249 in `train_FD004.txt`:

| Property | Value |
|---|---|
| Present in train | yes |
| Number of cycles | 255 |
| Cycles consecutive 1..n | yes (1…255) |
| Ends at RUL = 0 | yes |
| All 26 columns present | yes |
| Nulls | 0 |

It is a complete, well-formed engine identical in structure to the other 248
training engines. There is no evidence it is a stray/partial record.

## 7. Source / version information

- The only in-package documentation is `readme.txt` and the Saxena et al. (2008)
  paper (`Damage Propagation Modeling.pdf`). The paper describes the four
  scenarios but does not enumerate per-file engine counts, so it cannot adjudicate
  the count.
- The publicly mirrored C-MAPSS readme on the NASA repository is commonly quoted
  as "248 train / 249 test" for FD004 — i.e. the **same transposed wording** seen
  here. Per the project rules we did **not** download or substitute any external
  copy; the supplied archive remains the sole source artifact.
- The discrepancy is therefore a long-standing quirk of the distributed
  `readme.txt`, not damage introduced in this project. The supplied **file
  contents** are the authoritative record.

---

## Conclusion

- **Computationally authoritative files:** `train_FD004.txt` (**249** engines),
  `test_FD004.txt` (**248** engines), `RUL_FD004.txt` (**248** values). All
  SHA-256 verified against `data/.checksums.txt` (`integrity.json`, all match).
  These define the real dataset used for every computation in this project.
- **Documentary reference:** `readme.txt` states 248 train / 249 test. This is
  treated as an **erroneous (transposed) documentation value**, retained verbatim
  for provenance but **not** used to drive any code or split.
- **Is any correction justified?** **No correction to the data is justified.**
  The files are internally consistent and must remain unchanged. The only action
  taken is **documentation**: `data/raw/README.md` now records the observed
  249/248 counts and this discrepancy, and the Phase-0 reports cite the file
  contents (not the readme) as authoritative. No file was renamed, swapped,
  padded, or trimmed.

**Downstream rule for Phase 1:** engine counts, splits, and scoring must be
derived from the parsed file contents at runtime (via `src.data.loader`), never
hard-coded from `readme.txt`.
