# PhotoProt: reproducibility and Wild-v1

> **Hackathon prototype.** PhotoProt was built for a hackathon. It is experimental and not yet polished or production-ready. Expect rough edges, incomplete coverage and incorrect matches; verify results against the original PDB entry.

[**Donate / support continued development →**](https://photoprot.uk/support.html) · Donations are not enabled yet. The support page will provide the payment link once it is activated.

Search deposited protein structures using a single cartoon/ribbon image. This repository provides the frozen DINOv2-L retrieval checkpoint, the actual production index, rendering/training/evaluation source, and the real-world figure benchmark. The [Codex plugin](https://github.com/CXFG01/photoprot-codex) is distributed separately.

The public demo is [photoprot.uk](https://photoprot.uk). Current deployment limits,
emergency shutoff and security-review limitations are in the
[operations guide](webapp/OPERATIONS.md). The `v0.1.0` release preserves the
original reproduction snapshot; `main` also contains subsequent service hardening.

## Reproduce the reported numbers first

Python 3.11+; no GPU or packages required for this check:

```sh
git clone https://github.com/CXFG01/photoprot-reproducibility.git
cd photoprot-reproducibility
python repro/verify.py
```

This independently recomputes all six Wild-v1 metric subsets from the 479 saved per-query source ranks, checks 9,580 ranked rows, and verifies the recorded query-embedding checksum. It does not rerun inference.

## Download exact artifacts

The [v0.1.0 release](https://github.com/CXFG01/photoprot-reproducibility/releases/tag/v0.1.0) contains the original 1.64 GB checkpoint, 656 MB production index, 118 MB render manifest, sequence-component split, index metadata and 80 MB open-image archive. Every asset has its size and SHA-256 in [artifacts/manifest.json](artifacts/manifest.json).

```sh
python repro/download.py index.npz
# For fresh image inference:
python repro/download.py stage_b_last.pt wild-v1-open-images.zip
# Or download every released asset:
python repro/download.py all
```

Downloads are hash-checked before completion. Images and model weights have separate licence terms; see [NOTICE.md](NOTICE.md).

## Recompute rankings against the full production index

Use a CUDA-capable PyTorch environment with NumPy, Pillow and Transformers. The measured environment is recorded in [configs/environment.json](configs/environment.json); [requirements-inference.txt](requirements-inference.txt) pins its direct dependencies. Install the CUDA build separately as shown below. The published environment was Linux, Python 3.11.16 and NVIDIA RTX A6000.

```sh
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
python repro/rerank_wild.py --index downloads/index.npz --output reranked-wild.json
```

This uses the saved 479 query embeddings and performs fresh exact cosine search across all **836,399 reference images / 38,833 PDBs**. It compares full-gallery source ranks and every top-20 list with the frozen predictions. Default device is CUDA; `--device cpu` is available but much slower. Different hardware/precision can change near-ties; differences are reported, not hidden.

## Encode a new image locally

```sh
python repro/search.py /path/to/protein.png --artifacts downloads
```

The model configuration is bundled, so the base DINOv2 weights are not downloaded again. The checkpoint and index are verified before use. Preprocessing is 224×224 PIL bilinear, ImageNet normalization, CLS token, learned 256-dimensional normalized head. Each PDB's score is the mean of its five best reference-view cosine similarities; ties use alphabetical PDB order. Scores are not probabilities or evidence of homology. The service does not implement calibrated rejection.

## Wild-v1 real-world benchmark

**479 crops from 433 source figures**, with original labels and varied representations retained. It is an exploratory development diagnostic, not a blind, paper-independent test.

| Subset | n | Top-1 | Top-5 | Top-10 | Top-20 |
|---|---:|---:|---:|---:|---:|
| Listed source present, exploratory | 117 | 5.98% | 12.82% | 16.24% | 23.93% |
| Confirmed single source, indexed | 16 | 6.25% | 18.75% | 25.00% | 43.75% |
| All approved crops, absent references count as misses | 479 | 1.46% | 3.13% | 3.97% | 5.85% |

362 crops have no listed source PDB in the index; 407 mappings remain unresolved at panel level. Broad candidate-hit metrics can overestimate exact-source accuracy. Read the [dataset card](benchmarks/wild-v1/DATASET_CARD.md) and [original report](benchmarks/wild-v1/RESULTS.md).

All 479 records, predictions and embeddings are included. The open-image release contains **354** paired 512/224 PNG inputs with recorded CC BY/CC0 licences. **125** crops are metadata-only because their recorded terms are restrictive or unresolved. Missing redistributed pixels do not remove those queries from the saved-results benchmark. [Attribution](benchmarks/wild-v1/attribution.json) provides authors, source links, licences, transformations and hashes for every query. To reconstruct inputs from source files you lawfully obtained, use `repro/restore_inputs.py`.

## Synthetic held-out results

| Protocol | Queries | Candidate PDBs | Top-1 | Top-5 | Top-10 | MRR | Median rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| view0_query | 5,174 | 38,833 | 29.71% | 54.10% | 65.11% | 0.4139 | 4 |
| style_matched | 5,171 | 38,833 | 18.49% | 39.49% | 49.41% | 0.2861 | 11 |

Query-cohort views are excluded from the reference images. Exact gallery counts, exclusions and hashes are in [the report](benchmarks/synthetic/full-index.json). The older 59.05% Top-1 result uses only 3,884 candidate PDBs; it is not the production-index result. [All smaller-gallery baselines](benchmarks/synthetic/test-gallery-baselines.json) are included separately.

## Code, training and scope

- `modal_pdb.py`, `modal_render_pdb.py`, `photoprot/`, `configs/render_v1.json`: acquisition and deterministic rendering source.
- `bench/`: preserved experiment, split, training and evaluation source. Historical runners expect `~/photoprot`; their original hashes are retained for audit. These are not automatically run by the quickstart.
- `webapp/server.py`: preserved service inference source for comparison. This repository's runnable inference entry point is `repro/search.py`; the full website assets are not bundled.
- `repro/`: portable metric verification, artifact download, input reconstruction and full-index inference/reranking.
- `deploy/`: persistent Brev API and named-tunnel service definitions.

The release reproduces frozen inference and benchmark scoring. It does **not** claim a bit-for-bit training rerun: the full rendered training-image archive and a complete original training command/config snapshot are not bundled. Current training source includes later validation/augmentation changes and must not be treated as the exact recipe for the preserved `ft_b` checkpoint. See [model and training notes](docs/MODEL.md).

Original PhotoProt code is MIT licensed. DINOv2-derived weights retain upstream Apache-2.0 terms. Individual paper figures retain their recorded licences; the MIT licence does not relicense third-party content.
