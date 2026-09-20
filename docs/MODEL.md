# Frozen model and training provenance

Backbone: Meta DINOv2-L (`facebook/dinov2-large`), CLS token, hidden size 1024. Retrieval head: Linear(1024,1024), GELU, Linear(1024,256), L2 normalization. Backbone input: 224×224. Stored checkpoint includes model/head state and optimizer/scheduler state; load only a verified trusted artifact. The portable runner uses PyTorch `weights_only=True`.

The preserved run log reports four unfrozen transformer blocks, 674,178 training images from 31,065 PDB entries and 3,000 stage-B updates. Checkpoint metadata confirms step 3000, dimension 256 and cosine scheduler base learning rates 1e-5 / 1e-3. These facts are not a complete original command snapshot. Multi-positive contrastive learning groups positives by PDB ID. The corpus split uses 40%-sequence-identity connected components, not structural/fold clustering. Complete protection against every biological variant has not been demonstrated.

The current `bench/50_finetune.py` contains later augmentation/validation additions. Do not infer that the released checkpoint was trained using every current default. Exact retraining reproducibility is therefore incomplete; frozen inference and full-index reranking use the exact released checkpoint and index and are independently testable.

Acquisition source chooses representatives of 40%-identity entity clusters and prepares asymmetric units/assemblies; successful rendering further determines the final indexed collection. The index is a subset, not the full PDB archive. Original acquisition comments describing a planned full archive index are historical plans; released manifest/index files define actual coverage.

Model, index, render manifest and split hashes are in `artifacts/manifest.json` and the original benchmark protocols. The measured environment is in `configs/environment.json`. The full rendered training-image tar corpus is not redistributed here; renderer source/configuration and the exact output manifest are provided.
