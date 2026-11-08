# StageVista furnishing recipe in mflux (FLUX.2-Klein)

Branch `furnishing-recipe` in `Stage-Vista/mflux` adds the full
3DRecon furnishing training recipe to mflux's trainer:

- `training_adapter/furnishing_adapter.py` — `FurnishingPairsAdapter`:
  - logit-normal sigma with Flux-2 shift (default 3.0; `furnishing_sigma_shift`)
    via the existing `sample_sigma` hook (continuous distribution)
  - multi-block packing: K noised unfurnished views (img_ids t=0..K-1) +
    clean reference/anchor images (t=10,20,...) in one sequence
  - Plücker channel: zero-init `Linear(6 -> inner_dim)` wrapped onto the
    DiT's `x_embedder`; per-token ray maps from K/c2w in the parquet
  - delta-residual targets (GT − anchor) in the trainer loss
- `training_adapter/furnishing_dataset.py` — parquet-backed dataset reading
  3DRecon's `parquet/pairs_index/part-*.parquet` directly (views_json has
  paths + K/c2w inline). Deterministic roster, even-stride trim to
  `furnishing_max_images`, horizon-biased target, holdout via
  `exclude_scenes_json` / `val_scenes_json`.
- `common/training/trainer.py` — `_single_example_loss` extended: velocity
  loss over noised blocks, furniture mask `|GT−anchor| > thresh`,
  `furn_boost` fixed/`auto`+cap+ramp per-token weighting, ΔFM repulsion
  (`furnishing_cfm_lambda`) vs a rolling previous-sample negative gated to
  different scenes and training mode only.
- `common/training/runner.py` + `state/training_spec.py` — new top-level
  `"furnishing"` (recipe knobs) and `"furnishing_data"` (parquet root, path
  prefix, holdout jsons) config sections.

## Run

```bash
mflux-train --config src/mflux/models/flux2/training_adapter/_example/train_furnishing.json
```

with `furnishing_data.parquet_root` pointed at the 3DRecon pairs index and
`exclude_scenes_json`/`val_scenes_json` pointed at `holdout_scenes.json`
(32 scenes / 832 rows, disjoint from the 940-scene train split).

Prompt is text-only (no pose text ever — inference has none); geometry is
Plücker-only. Text stays under the 512-token cap (brief-only, ~110 tokens).
