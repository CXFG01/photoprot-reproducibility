# Training and scoring update

Implemented 2026-09-19. Original stage-B weights and reference caches are preserved.

## Checkpoint validation

Stage B now uses a fixed seed-selected panel of 512 validation PDB entries by default. Query view 0 is excluded from reference views 1–15. All panel embeddings are recomputed with the current checkpoint at every evaluation; embeddings from different checkpoints are never mixed.

Validation runs before training, every 500 steps, and at the final step. Selection uses top-1 exact-PDB recall, with the earlier checkpoint retained on ties. The default entry score is the average of the five highest reference-image similarities (`--val_scoring top5`). `max` and `top3` remain available. The panel is a small checkpoint-selection benchmark, not the full production gallery or a publication-image benchmark.

Each named run writes `stage_b_best.pt`, `stage_b_last.pt`, `validation_panel.json`, and `validation_history.json` under `data/ckpt/<run_name>`. Checkpoints record configuration, augmentation version, validation protocol, metrics, optimizer/scheduler/scaler state, and main-process PyTorch RNG state. Final val/test embeddings use the selected best checkpoint and a separate embedding tag.

Resume skips previously consumed sampling batches and rejects incompatible settings or a changed schedule length. Worker augmentation streams are not guaranteed bit-identical across interrupted runs. To extend training, initialize a new named run from saved weights with a fresh optimizer and schedule. Example (not launched as a full training run):

```bash
.venv-pp/bin/python bench/50_finetune.py --stage b \
  --run_name stage_b_publication_pilot_v1 \
  --init_checkpoint data/ckpt/stage_b_last.pt \
  --steps_b 1500 --lr_backbone 2e-6 --lr_head 1e-4 \
  --val_every 500 --val_entries 512 --val_scoring top5
```

## Training augmentations

Implemented with the existing torchvision/Pillow stack; Albumentations is not required.

- 75%: fit the complete image into 72–100% of the input width/height, preserve aspect ratio, and randomly position it within available margins. Padding uses median corner colour.
- 25%: mild random crop retaining 85–100% of image area, with aspect ratio 0.95–1.05.
- Mild colour jitter, 10% grayscale, 15% blur.
- 30%: downsample to 112–224 pixels then upsample (for 224px training).
- 25%: JPEG quality 35–95.

No reflections, elastic warps, or image mixing. Evaluation retains the existing deterministic resize and normalization. These augmentations have been tested for execution and shape correctness; accuracy improvement requires a training experiment.

## Cached scoring run

`61_cached_scoring.py` uses all 836,399 existing reference embeddings and the 74 saved publication-query embeddings. It performs no encoder inference or training. It checks checkpoint/manifest/legacy-cache hashes, validates complete unique manifest coverage, and reproduces every original query's top-10 max-score ranking before trusting the comparison.

It chooses among max, top-three mean, and top-five mean using validation top-1 recall (MRR breaks ties). Synthetic validation/test queries are view 0; all query rows of the evaluated split are removed from the full reference gallery to avoid self-matches. Wild queries use the entire unchanged gallery. The displayed candidate picture is its highest-scoring individual image, while the entry score averages the selected top reference similarities.

The selected rule is top-five. On the full-gallery test protocol, top-1 changes from 20.66% to 29.71%, and top-10 from 54.93% to 65.11%. These are not directly comparable to the older test-only-gallery numbers. On the 19 source-covered publication queries, exact-source top-1 changes from 0 to 1 and top-10 from 3 to 4. Top-three gives 2 and 5 respectively on that small diagnostic set; it was not used to choose the rule. Biological relevance is not graded by these counts.

Outputs: `data/results/wild_v1_cached_scoring/` on Brev; downloaded under `results/wild_v1_cached_scoring/` locally. Original max-scoring results remain in `wild_v1_full_corpus/`.

## Verification

Four tests check resumed sampling, deterministic/disjoint validation selection, augmentation shape/reproducibility, and aggregation against independently calculated variable-length examples. An isolated two-step GPU training smoke run exercises actual augmentation, validation, best/last saves, and final best-checkpoint loading. It is not an accuracy experiment or the deployed model.
