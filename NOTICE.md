# Third-party material and release scope

DINOv2 was developed by Meta Platforms, Inc. and affiliates. The released checkpoint is derived from `facebook/dinov2-large` and modified by PhotoProt through a projection head and fine-tuning. DINOv2 code and model weights are Apache-2.0; see `licenses/DINOv2-APACHE-2.0.txt` and https://github.com/facebookresearch/dinov2. Preserve this attribution and licence when redistributing the checkpoint. PhotoProt is not a Meta, NVIDIA or OpenAI product.

PDB structures and metadata originate from the wwPDB/RCSB PDB. Reference embeddings were computed from PhotoProt renders. Third-party libraries are not vendored; their own licences apply.

Paper figures retain their original licences and are not covered by the repository’s code licence. `benchmarks/wild-v1/attribution.json` retains the original authors, titles, DOI/source/figure links, recorded licence, crop bounds and processing changes for each query. The open-image asset includes only records marked `pixel_redistribution: true` under recorded CC BY/CC0 terms. Some records have older unversioned CC BY descriptions; follow the linked article's actual licence and credit lines. No new rights to third-party material are claimed. Check source-specific exclusions before reusing a figure.

125 crops are excluded from the redistributed pixels, including NoDerivatives, NonCommercial and unresolved/special publisher terms. Metadata, measured rankings and query feature arrays remain in the benchmark package. Source URLs and hashes support reconstruction from separately obtained originals; no permission is implied by the presence of a source link.
