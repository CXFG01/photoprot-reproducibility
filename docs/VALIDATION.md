# Release validation

Validated on 2026-09-20 against the live Brev production artifacts:

- The independent standard-library verifier recomputed all six metric subsets
  across 479 queries and checked all 9,580 top-20 rows.
- The portable ranking runner searched all 836,399 reference embeddings again:
  all 479 source ranks and all 479 complete top-20 lists matched the frozen
  results. Maximum score difference was exactly zero on the original hardware.
- Fresh inference loaded the released checkpoint with `weights_only=True`,
  encoded the existing public 3I3W example, and returned 3I3W at rank 1 with
  score 0.936663806438446.
- The released checkpoint, index, render manifest and split hashes match the
  hashes recorded in the original benchmark protocol.

These are implementation/reproduction checks, not additional scientific test
cohorts. See the dataset card for benchmark limitations.

All 479 crop pairs were reconstructed from local original figures, with exact
SHA-256 matches to both the frozen 512-pixel and 224-pixel PNG files.
