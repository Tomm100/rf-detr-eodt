# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Pretrained DINOv2 weights for the baseline, resampled to its patch size as timm does (EODT_SPEC.md §8, point 4)."""

from __future__ import annotations

import math

from timm.layers import resample_abs_pos_embed, resample_patch_embed

from rfdetr.models.backbone.dinov2 import get_config
from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersConfig,
)

_PATCH_EMBED = "embeddings.patch_embeddings.projection.weight"
_POS_EMBED = "embeddings.position_embeddings"


def load_pretrained_dinov2(encoder: WindowedDinov2WithRegistersBackbone, size: str) -> None:
    """Load the pretrained DINOv2 with registers into RF-DETR's encoder, whatever its patch size and grid.

    RF-DETR loads the Hugging Face weights only at the native patch 14 and 37 x 37 grid (``DinoV2.__init__``). The
    native model is loaded here in the same way, then the patch embedding and the positional embedding are resampled
    onto the patch size and the grid of ``encoder`` with the functions and arguments of timm's
    ``checkpoint_filter_fn``, which loads the backbone of EoDT: FlexiViT resize of the patch embedding, bicubic
    antialiased interpolation of the patch positions. The position of the CLS token and every other tensor are copied
    unchanged.

    Args:
        encoder: DINOv2 encoder built by RF-DETR (``Backbone.encoder.encoder``), with registers.
        size: DINOv2 size: ``"small"``, ``"base"`` or ``"large"``.

    Raises:
        RuntimeError: If the checkpoint lacks a weight of the native model, has one that it does not expect or one
            of another shape.
    """
    name = f"facebook/dinov2-with-registers-{size}"  # the checkpoint DinoV2 loads at patch 14
    native_config = WindowedDinov2WithRegistersConfig(**get_config(size, use_registers=True))
    native, info = WindowedDinov2WithRegistersBackbone.from_pretrained(
        name, config=native_config, output_loading_info=True
    )
    problems = {key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys") if info[key]}
    if problems:
        raise RuntimeError(f"{name} does not match RF-DETR's DINOv2 encoder: {problems}")

    # both functions return the tensor unchanged when it already has the target size
    state_dict = native.state_dict()
    target = encoder.state_dict()
    patch_size = list(target[_PATCH_EMBED].shape[-2:])
    state_dict[_PATCH_EMBED] = resample_patch_embed(
        state_dict[_PATCH_EMBED], patch_size, interpolation="bicubic", antialias=True
    )
    grid = math.isqrt(target[_POS_EMBED].shape[1] - 1)  # the position of the CLS token comes first
    state_dict[_POS_EMBED] = resample_abs_pos_embed(
        state_dict[_POS_EMBED], [grid, grid], num_prefix_tokens=1, interpolation="bicubic", antialias=True
    )
    encoder.load_state_dict(state_dict)
