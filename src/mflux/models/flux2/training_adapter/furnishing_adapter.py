"""Furnishing-pairs training adapter for FLUX.2-Klein (StageVista recipe).

Extends Flux2EditTrainingAdapter with the qwen_edit_ft/klein_ft recipe:

  * multi-block packing: K noised unfurnished views (t=0..K-1) + N clean
    reference/anchor images (t=10,20,...) in ONE image sequence, loss over
    the noised prefix only (mirrors klein_ft build_img_ids_grids).
  * delta_residual: training target is GT-anchor per noised block.
  * Plücker channel: zero-init Linear(6 -> inner_dim) added onto the DiT's
    x_embedder; per-token ray maps from K/c2w stored in the parquet.
  * logit-normal sigma with Flux-2 shift (default 3.0) via sample_sigma.
  * furn_boost (fixed/'auto' + cap + ramp) per-token loss weighting on the
    furniture mask |GT-anchor| > thresh — implemented in the trainer's loss.
  * ΔFM repulsion handled in the trainer (needs cross-sample state).

Data comes from a parquet-backed dataset (see furnishing_dataset.py), not
from *_in/_out file pairs.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer as Flux2Transformer2DModel
from mflux.models.flux2.training_adapter.flux2_base_training_adapter import Flux2BaseTrainingAdapter
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers


def compute_plucker_grid(K: dict, c2w, width: int, height: int) -> np.ndarray:
    """Per-token Plücker map [H/16, W/16, 6] (origin, unit dir), row-major —
    same math as 3DRecon's qwen/klein_ft arms."""
    hp, wp = height // 16, width // 16
    u = (np.arange(wp, dtype=np.float64) + 0.5) * 16.0 - 0.5
    v = (np.arange(hp, dtype=np.float64) + 0.5) * 16.0 - 0.5
    uu, vv = np.meshgrid(u, v)
    rays = np.stack([(uu - K["cx"]) / K["fx"],
                     (vv - K["cy"]) / K["fy"],
                     np.ones_like(uu)], axis=-1)
    c2w = np.asarray(c2w, dtype=np.float64)
    dirs = rays @ c2w[:3, :3].T
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape).copy()
    return np.concatenate([origins, dirs], axis=-1).astype(np.float32)


def sample_sigma_logit_normal(rng: random.Random, shift: float) -> float:
    u = rng.gauss(0.0, 1.0)
    t = 1.0 / (1.0 + math.exp(-u))
    if shift != 1.0:
        t = shift * t / (1.0 + (shift - 1.0) * t)
    return min(max(t, 0.02), 0.98)


class FurnishingConfig:
    """Extra knobs carried in TrainingSpec.extra (JSON-serializable)."""

    def __init__(self, extra: dict | None):
        extra = extra or {}
        self.delta_residual: bool = bool(extra.get("furnishing_delta_residual", True))
        self.sigma_shift: float = float(extra.get("furnishing_sigma_shift", 3.0))
        self.furn_delta_thresh: float = float(extra.get("furnishing_furn_delta_thresh", 0.5))
        self.furn_boost: str = str(extra.get("furnishing_furn_boost", "1.0"))
        self.furn_boost_cap: float = float(extra.get("furnishing_furn_boost_cap", 10.0))
        self.furn_boost_ramp: int = int(extra.get("furnishing_furn_boost_ramp", 0))
        self.cfm_lambda: float = float(extra.get("furnishing_cfm_lambda", 0.05))
        self.plucker: bool = bool(extra.get("furnishing_plucker", True))
        self.max_images: int = int(extra.get("furnishing_max_images", 4))
        self.cond_w: int = int(extra.get("furnishing_cond_w", 384))
        self.cond_h: int = int(extra.get("furnishing_cond_h", 256))
        # anchor images (self-anchors) are appended as clean refs
        self.self_anchors: bool = bool(extra.get("furnishing_self_anchors", True))


class PluckerHead:
    """Zero-init Linear(6 -> inner_dim); wraps x_embedder via a module-level
    slot read at forward time (MLX graph-friendly: the map is a constant
    per sample, injected by the adapter before each forward)."""

    def __init__(self, inner_dim: int):
        from mlx import nn
        self.proj = nn.Linear(6, inner_dim, bias=False)
        self.proj.weight = mx.zeros_like(self.proj.weight)

    def __call__(self, maps: mx.array) -> mx.array:
        return self.proj(maps)


_current_plucker: list[mx.array | None] = [None]


def install_plucker_injection(transformer: Flux2Transformer2DModel, inner_dim: int):
    """Wrap transformer.x_embedder: out = x_embedder(x) + head(plucker_map)."""
    from mlx import nn

    head = PluckerHead(inner_dim)
    orig = transformer.x_embedder

    class _Wrapped(nn.Module):
        def __call__(self, x):
            y = orig(x)
            pl = _current_plucker[0]
            if pl is not None:
                y = y + head(pl.astype(y.dtype))
            return y

    transformer.x_embedder = _Wrapped()
    return head


class FurnishingPairsAdapter(Flux2BaseTrainingAdapter):
    """Adapter over the furnishing-pairs parquet (see furnishing_dataset.py).

    DataItem.cond carries:
      gt_tokens:      list of packed GT latents per noised block (K)
      anchor_tokens:  list of packed unfurnished latents per noised block
      noise_paths:    paths of the unfurnished images (for plucker/roles)
      ref_tokens:     concat of clean reference latents (roster + anchors)
      ref_ids:        img_ids for ref_tokens (t=10,20,...)
      block_ids:      img_ids for the noised prefix (t=0..K-1)
      prompt_embeds / text_ids
      plucker:        concatenated per-image plucker maps (aligned to
                      [noised..., refs...] token order) or None
      scene:          scene_key (for ΔFM scene gating)
    """

    def __init__(self, *, model_config: ModelConfig, quantize: int | None,
                 furnishing: FurnishingConfig):
        super().__init__(
            model_config=model_config,
            quantize=quantize,
            model_factory=lambda cfg, q: Flux2KleinEdit(model_config=cfg, quantize=q),
        )
        self.furn = furnishing
        self.plucker_head = None
        if furnishing.plucker:
            self.plucker_head = install_plucker_injection(
                self._flux2.transformer, self._flux2.transformer.inner_dim)

    # -- sigma: continuous logit-normal with Flux-2 shift (uses the existing
    #    sample_sigma hook — the trainer passes a continuous sigma when this
    #    returns non-None) --
    def sample_sigma(self, *, rng: random.Random, image_seq_len: int) -> float | None:
        return sample_sigma_logit_normal(rng, self.furn.sigma_shift)

    # -- data encoding is done by furnishing_dataset.py (parquet rows ->
    #    latents/cond); this adapter only provides predict_noise --
    def encode_data(self, **kw):  # not used; dataset builds DataItems
        raise NotImplementedError("use furnishing_dataset.FurnishingDataset")

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array,
                      cond: Any, config: Config) -> mx.array:
        # latents_t: concatenated noised blocks [K blocks] (B, sumT, C)
        hidden_states = mx.concatenate([latents_t, cond["ref_tokens"]], axis=1)
        img_ids = mx.concatenate([cond["block_ids"], cond["ref_ids"]], axis=1)
        _current_plucker[0] = cond.get("plucker")
        try:
            noise = self._flux2.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=cond["prompt_embeds"],
                timestep=config.scheduler.timesteps[t],
                img_ids=img_ids,
                txt_ids=cond["text_ids"],
                guidance=None,
            )
        finally:
            _current_plucker[0] = None
        return noise[:, : latents_t.shape[1]]
