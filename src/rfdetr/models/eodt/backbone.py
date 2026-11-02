# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
# Backbone loading adapted from EoMT (https://github.com/tue-mps/eomt), models/vit.py
# Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. Licensed under the MIT License.
# ------------------------------------------------------------------------
"""DINOv2 backbone from timm, re-patched at load time."""

from __future__ import annotations

import timm
from torch import Tensor, nn


class EoDTBackbone(nn.Module):
    """DINOv2 ViT from timm with global attention and a configurable patch size.

    timm resamples the pretrained patch embedding (FlexiViT) and interpolates the
    positional embedding onto the ``resolution / patch_size`` grid while loading.
    Images are expected to be already normalized by the dataloader.

    Args:
        backbone_name: timm model name, e.g. ``vit_small_patch14_reg4_dinov2``.
            Register tokens are determined by the name.
        resolution: Side of the square input image, in pixels.
        patch_size: Patch size the pretrained weights are resampled to.
        pretrained: Whether to load the pretrained DINOv2 weights.
    """

    def __init__(self, backbone_name: str, resolution: int, patch_size: int, pretrained: bool) -> None:
        super().__init__()
        if resolution % patch_size != 0:
            raise ValueError(f"resolution ({resolution}) must be divisible by patch_size ({patch_size})")
        self.vit = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            img_size=(resolution, resolution),
            patch_size=patch_size,
            num_classes=0,
        )

    @property
    def embed_dim(self) -> int:
        """Hidden dimension of the ViT."""
        return self.vit.embed_dim

    @property
    def num_prefix_tokens(self) -> int:
        """Number of tokens before the patches (CLS and registers)."""
        return self.vit.num_prefix_tokens

    @property
    def grid_size(self) -> tuple[int, int]:
        """Patch grid as (height, width)."""
        return tuple(self.vit.patch_embed.grid_size)

    def forward(self, x: Tensor) -> Tensor:
        """Run the ViT and return all tokens after the final norm.

        Args:
            x: Normalized images of shape ``[B, 3, resolution, resolution]``.

        Returns:
            Tokens of shape ``[B, num_prefix_tokens + H * W, embed_dim]``, prefix tokens first.
        """
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)  # adds pos embed to patches, prepends CLS and registers
        x = self.vit.norm_pre(x)
        for block in self.vit.blocks:
            x = block(x)
        return self.vit.norm(x)
