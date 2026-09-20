# PhotoProt Wild-v1 — 20 September 2026

## Intended use

Diagnostic retrieval of candidate deposited structures from a single scientific figure crop. Not a clinical tool, blind benchmark, protein-family ground truth, or image-to-3D reconstruction task.

## Population and selection

479 user-approved crops from all 433 available source figures in the development collection. One additional source record had no downloaded image. 74 crop rectangles were reused; 405 new primary crops were added. Bulk approval selected the crops, not their scientific labels. One RNA-only diagnostic was retained in broad results and excluded from mapped-protein subsets.

117 crops have at least one listed candidate in the 38,833-PDB index. 362 do not. Panel-to-accession mapping remains uncertain for 407 crops. Confirmed single-source, indexed subset: 16 crops. Source PDBs can occur in training, validation or test reference splits. Multiple crops share papers and some source images are duplicated; use `grouping_key`, DOI, source hash and `duplicate_source_ids` when forming future splits. Embedded text is retained. Do not relabel this as a clean, independent generalization benchmark.

## Preprocessing

Original-pixel rectangle `[left, top, right, bottom]`, with exclusive right/bottom. Alpha composited onto white; aspect-preserving long side 464 using Pillow LANCZOS; centered on white RGB 512×512; Pillow BILINEAR resize to 224×224. No rotation, reflection, inpainting or text removal. ImageNet mean/std normalization occurs at inference. All source/master/model-input hashes are recorded.

## Evaluation

Frozen DINOv2-L ft_b, 256-dimensional normalized projection. Exact cosine retrieval over 836,399 images; mean of top five reference-view scores per PDB; alphabetical PDB tie-break. A broad hit accepts any listed source accession, so unresolved candidate lists can inflate success. No biological/structural-equivalence grading. Missing sources count as misses in all-query recall and MRR. Historical `median_rank` is computed only over present targets, including in the all-query summary; it is not an absent-target-inclusive median.

`predictions.json` includes all source candidate ranks and top-20 results. `predictions.csv` has 9,580 rows. `query_embeddings.npz` has 479×256 fresh query embeddings with ordered crop IDs. The 479-query cohort and denominators are frozen regardless of which pixels are redistributed.

## Distribution

`wild-v1-open-images.zip` contains 354 RGB512/model-input PNG pairs, selected by recorded CC BY/CC0 terms, plus attribution. 125 records have no redistributed pixels. Full 479-query metadata, embeddings, predictions, protocol, original validations and metrics are available. Do not calculate full-cohort fresh-image accuracy from only the 354-image subset. Saved-embedding full-index reranking covers all 479.

The original `report_image` paths refer to the internal gallery and are archival provenance, not files promised in this release. The gallery and reference-image thumbnails are not distributed. See the repository README for fresh local ranking commands.

## Citation

Cite this repository and release tag `v0.1.0`, and cite the original source article(s) listed in `attribution.json` when reusing figures. Describe it as PhotoProt Wild-v1, an exploratory development collection, and state the subset and denominator used.
