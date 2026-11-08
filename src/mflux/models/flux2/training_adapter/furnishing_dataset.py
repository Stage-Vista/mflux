"""Parquet-backed dataset for furnishing-pairs training (StageVista recipe).

Reads the 3DRecon datagen output directly:
  <parquet_root>/part-*.parquet    one row per (scene, set, yaw, combo)
  (image paths inside views_json are absolute from the datagen machine;
   pass path_prefix/--path_remap style prefix to re-root them, e.g. when
   the dataset is unpacked to /data/furnishing)

Each row's views_json carries per-view paths + K/c2w (the Plücker source).
Sampling mirrors klein_ft/pairs_dataset.py: deterministic 'all' roster,
even-stride trim to max_images, horizon-biased target index, every view
noised on its unfurnished lineage (text-only furnishing, no pose text).

Holdout: pass exclude_scenes=holdout for the train dataset and
include_scenes=holdout for the val dataset (mflux's iterator already has
get_validation_batch; the runner wires it when the spec has a val section).
"""

from __future__ import annotations

import glob
import json
import os
import random
import zlib
from typing import Any

import mlx.core as mx
import numpy as np

from mflux.models.common.training.dataset.batch import DataItem
from mflux.models.common.training.dataset.dataset import Dataset
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.training_adapter.furnishing_adapter import (
    FurnishingConfig, compute_plucker_grid)
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import (
    _Flux2KleinEditHelpers)
from mflux.utils.image_util import ImageUtil


def load_rows(parquet_root: str, *, exclude: set | None = None,
              include: set | None = None, max_row_views: int | None = None
              ) -> list[dict]:
    import pyarrow.parquet as pq
    paths = [p for p in sorted(glob.glob(os.path.join(parquet_root, "*.parquet")))
             if not os.path.basename(p).startswith("._")]
    if not paths:
        raise FileNotFoundError(f"no parquet parts under {parquet_root}")
    rows = []
    for p in paths:
        for row in pq.read_table(p).to_pylist():
            sk = row.get("scene_key")
            if exclude and sk in exclude:
                continue
            if include is not None and sk not in include:
                continue
            if max_row_views is not None and int(row.get("n_views") or 0) > max_row_views:
                continue
            rows.append(row)
    return rows


def infer_brief(row: dict) -> str:
    """Inference-matched brief: room type + furnishing text + INPUT-side
    density from config_json only (measured parquet label is a GT leak)."""
    try:
        cfg = json.loads(row.get("config_json") or "{}")
    except Exception:
        cfg = {}
    room = (row.get("room_type") or "room").replace("_", " ")
    text = (row.get("furnishing_text") or "").replace(
        "Make the empty room match this furnishing description: ", "")
    dens_map = {"minimal": "minimally staged, essential pieces only, no clutter",
                "moderate": "moderately furnished",
                "layered": "richly furnished with layered decor"}
    dens = dens_map.get(cfg.get("staging_density") or "", "")
    out = f"Furnish this empty {room}. {text}"
    if dens:
        out += f" Keep it {dens}."
    out += " Choose a natural, plausible furniture layout for the space."
    return out[:1200]  # well under the 512-token prompt cap


def build_sample(row: dict, fc: FurnishingConfig, path_prefix: str = "") -> dict:
    views = json.loads(row["views_json"])
    ordered = sorted(views, key=lambda v: (v.get("pos_index", 0),
                                           v["yaw_deg"], v["pitch_deg"]))
    if len(ordered) > fc.max_images:
        stride = len(ordered) / fc.max_images
        ordered = [ordered[int(i * stride)] for i in range(fc.max_images)]

    key = (f"{row['scene_key']}/{row.get('position_set','')}/"
           f"{row.get('yaw_variant','')}/{row.get('combo_id','')}")
    crc = zlib.crc32(key.encode())
    h_idx = [i for i, v in enumerate(ordered) if abs(v["pitch_deg"]) <= 35.0]
    if h_idx and ((crc >> 8) % 1000) < 650:
        t = h_idx[crc % len(h_idx)]
    else:
        t = crc % len(ordered)

    def fix(p: str) -> str:
        return (path_prefix + p) if path_prefix and p.startswith("/") else p

    return {
        "scene": row["scene_key"],
        "prompt": infer_brief(row),
        "views": ordered,
        "target_idx": t,
        "unfurnished_paths": [fix(v["unfurnished_path"]) for v in ordered],
        "gt_paths": [fix(v["furnished_clean_path"]) for v in ordered],
        "ref_paths": ([fix(v["unfurnished_path"]) for v in ordered]
                      if fc.self_anchors else []),
    }


def _encode_one(vae, tiling_config, path: str, w: int, h: int):
    """image file -> (packed_tokens (1,T,C), grid_ids (1,T,4)) at w x h.

    Goes through LatentCreator.encode_image exactly like the stock edit
    adapter (_encode_output_latents) so normalization/batch-dim/dtype match
    what the VAE expects — a hand-rolled resize+to_array+VAEUtil.encode blew
    the Metal buffer limit on the first row."""
    from mflux.models.common.latent_creator.latent_creator import LatentCreator
    encoded = LatentCreator.encode_image(
        vae=vae, image_path=path, height=h, width=w, tiling_config=tiling_config)
    encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
    encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
    encoded = Flux2LatentCreator.patchify_latents(encoded)
    encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(
        encoded, vae=vae)
    ids = Flux2LatentCreator.prepare_grid_ids(encoded, t_coord=0)
    return Flux2LatentCreator.pack_latents(encoded), ids


class FurnishingDataset(Dataset):
    """Parquet rows -> DataItems with fully materialized latents/cond.

    encode_all(adapter) runs once up front (same lifecycle as mflux's
    runner); DataItem.clean_latents = concatenated noised-block GT tokens,
    DataItem.cond carries the rest.
    """

    def __init__(self, rows: list[dict], fc: FurnishingConfig, seed: int = 0,
                 path_prefix: str = ""):
        self._rows = rows
        self._fc = fc
        self._rng = random.Random(seed)
        self._path_prefix = path_prefix
        super().__init__([])

    def encode_all(self, adapter) -> None:
        import gc
        vae = adapter._flux2.vae
        tiling = adapter._flux2.tiling_config
        cw, ch = self._fc.cond_w, self._fc.cond_h
        items: list[DataItem] = []
        for i, row in enumerate(self._rows):
            s = build_sample(row, self._fc, self._path_prefix)
            prompt_embeds, text_ids = adapter._encode_prompt(prompt=s["prompt"])
            mx.eval(prompt_embeds, text_ids)

            gt_lat, gt_ids = [], []
            for p in s["gt_paths"]:
                tok, ids = _encode_one(vae, tiling, p, cw, ch)
                mx.eval(tok, ids)
                gt_lat.append(tok)
                gt_ids.append(mx.array(ids))
            anc_lat = []
            for p in s["unfurnished_paths"]:
                tok, _ = _encode_one(vae, tiling, p, cw, ch)
                mx.eval(tok)
                anc_lat.append(tok)
            ref_lat, ref_ids = [], []
            for p in s["ref_paths"]:
                tok, ids = _encode_one(vae, tiling, p, cw, ch)
                mx.eval(tok, ids)
                ref_lat.append(tok)
                ref_ids.append(mx.array(ids))
            # free the VAE's transient buffers after each row's encodes
            gc.collect()
            mx.clear_cache()

            block_tokens = mx.concatenate(gt_lat, axis=1) if len(gt_lat) > 1 else gt_lat[0]
            # per-image T-coords: noised blocks t=0..K-1, refs t=10,20,...
            block_ids = mx.concatenate(
                [_set_t(g, float(j)) for j, g in enumerate(gt_ids)], axis=1)
            if ref_lat:
                ref_tokens = (mx.concatenate(ref_lat, axis=1)
                              if len(ref_lat) > 1 else ref_lat[0])
                ref_id_cat = mx.concatenate(
                    [_set_t(g, float(10 + 10 * j)) for j, g in enumerate(ref_ids)],
                    axis=1)
            else:
                ref_tokens = mx.zeros((1, 0, gt_lat[0].shape[2]), dtype=gt_lat[0].dtype)
                ref_id_cat = mx.zeros((1, 0, 4), dtype=mx.int32)

            plucker = None
            if self._fc.plucker:
                maps = [_plucker_for_map(v, cw, ch) for v in s["views"]]
                if s["ref_paths"]:
                    maps += [_plucker_for_map(v, cw, ch) for v in s["views"]]
                plucker = mx.array(
                    np.concatenate([m.reshape(-1, 6) for m in maps], axis=0)[None],
                    dtype=mx.bfloat16)

            cond = {
                "gt_tokens_list": gt_lat,
                "anchor_tokens_list": anc_lat,
                "block_tokens": block_tokens,
                "ref_tokens": ref_tokens,
                "block_ids": block_ids,
                "ref_ids": ref_id_cat,
                "prompt_embeds": prompt_embeds,
                "text_ids": text_ids,
                "plucker": plucker,
                "scene": s["scene"],
            }
            items.append(DataItem(
                data_id=i, prompt=s["prompt"], image_path="",
                clean_latents=block_tokens, cond=cond, width=cw, height=ch))
        self.data = items


def _set_t(ids: mx.array, t: float) -> mx.array:
    # ids: (B, T, 4) with col 0 = T coord; replace it. MLX arrays are
    # immutable — build the new column and concat, no .copy().
    return mx.concatenate([mx.full((ids.shape[0], ids.shape[1], 1), t,
                                   dtype=ids.dtype),
                           ids[..., 1:]], axis=2)


def _plucker_for_map(v: dict, cw: int, ch: int) -> np.ndarray:
    K = v["K"]
    Ks = {"fx": K["fx"] * cw / 768, "fy": K["fy"] * ch / 512,
          "cx": K["cx"] * cw / 768, "cy": K["cy"] * ch / 512}
    return compute_plucker_grid(Ks, v["c2w"], cw, ch)
