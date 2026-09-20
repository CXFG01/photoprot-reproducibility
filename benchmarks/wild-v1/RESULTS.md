# DINOv2 wild-image benchmark — 20 September 2026

All **479 approved crops from 433 source figures** were encoded afresh and searched on Brev against the deployed **836,399-image / 38,833-PDB** index. DINOv2-L ft_b and mean-top-five scoring were frozen; no training or score-rule tuning was performed. Encoding and report generation finished in 67.6 seconds.

| Evaluation subset | Queries | Top-1 | Top-5 | Top-10 | Top-20 | Top-50 |
|---|---:|---:|---:|---:|---:|---:|
| Listed candidate present, exploratory | 117 | 7/117 (5.98%) | 15/117 (12.82%) | 19/117 (16.24%) | 28/117 (23.93%) | 36/117 (30.77%) |
| New crops with listed candidate present, exploratory | 97 | 6/97 (6.19%) | 12/97 (12.37%) | 15/97 (15.46%) | 21/97 (21.65%) | 29/97 (29.90%) |
| Previously mapped or multiple depicted sources, covered | 19 | 1/19 (5.26%) | 3/19 (15.79%) | 4/19 (21.05%) | 7/19 (36.84%) | 7/19 (36.84%) |
| Confirmed single-source subset, covered | 16 | 1/16 (6.25%) | 3/16 (18.75%) | 4/16 (25.00%) | 7/16 (43.75%) | 7/16 (43.75%) |

**362 queries have no listed source accession in the index.** They still received rankings; absent targets were not counted as exact-ID successes. Across all 479 queries, listed-candidate hits are 1.46% at rank 1, 3.13% at rank 5 and 3.97% at rank 10 and 5.85% at rank 20 and 7.52% at rank 50.

**407 crops retain unresolved panel-to-PDB assignments.** Bulk approval confirmed the crop set; it did not turn paper-level accession lists into verified panel labels. The broad metrics accept any listed candidate and may overstate exact-source correctness. Only 19 covered crops have the earlier mapped/multiple-depicted labels; all are from the previous crop set. New-crop results are exploratory until mappings are resolved. The RNA-only crop was also approved and scored, but is excluded from protein mapped subsets.

The results demonstrate limited source-accession retrieval on this collection and substantial index-coverage gaps. They do not score whether alternative accessions are structurally or biologically equivalent. Publication labels remain, some sources are duplicated, and multiple crops share papers. This is a development diagnostic, not a blind or paper-independent accuracy estimate.

## Review and evidence

- `gallery.html`: each query beside its top 10 matches; filter covered, absent, top-10 hit and new figures.
- `predictions.csv`: 9,580 rows, the top 20 matches for every query.
- `predictions.json`: complete source candidate ranks, source split membership, top 20 predictions and metadata.
- `query_embeddings.npz`: fresh 479 × 256 embeddings.
- `query_manifest.json`, `protocol.json`: frozen dataset snapshot, checkpoint/index hashes, preprocessing and environment.
- `validation.json`: three exact top-20 comparisons with the deployed Engine, 9,580 independent aggregation checks, unchanged checkpoint and dataset checks.
- `local_validation.json`: downloaded embedding, ranking, asset and independently recomputed metric checks.

Checkpoint: `c3ea295708f3747c696520abcc848c2fbbdc7eed4c42c32dd657d11f61e60db4`.
Index: `3fd09639ff615fbc3bd8e74f3d475f4923d197b46783f46eb863667a4c5b32cd`.
The original model/index and live service were preserved. Rerun with `bench/84_wild_approved.py` using a new output directory; the runner refuses to overwrite completed predictions.

Top-50 was derived from the saved full-gallery source ranks, independently cross-checked against every candidate source rank. No new encoder inference was needed. The visual cards show top 10 and the CSV lists top 20; top-50 here refers to the retrieval metric.
