# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
# Backbone loading and query concatenation adapted from EoMT (https://github.com/tue-mps/eomt),
# models/vit.py and models/eomt.py
# Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. Licensed under the MIT License.
# ------------------------------------------------------------------------
"""DINOv2 backbone from timm, re-patched at load time and split into L1 and L2 blocks."""

from __future__ import annotations

import timm
import torch
from torch import Tensor, nn


class EoDTBackbone(nn.Module):
    """DINOv2 ViT from timm with global attention and a configurable patch size.

    timm resamples the pretrained patch embedding (FlexiViT) and interpolates the
    positional embedding onto the ``resolution / patch_size`` grid while loading.
    Images are expected to be already normalized by the dataloader.

    The blocks are split into L1 (ViT tokens only) and the last L2 blocks, where the
    queries are prepended to the ViT tokens and share their full attention.

    Args:
        backbone_name: timm model name, e.g. ``vit_small_patch14_reg4_dinov2``.
            Register tokens are determined by the name.
        resolution: Side of the square input image, in pixels.
        patch_size: Patch size the pretrained weights are resampled to.
        num_l2_blocks: Number of final blocks that also process the queries.
        pretrained: Whether to load the pretrained DINOv2 weights.
    """

    def __init__(
        self,
        backbone_name: str,
        resolution: int,
        patch_size: int,
        num_l2_blocks: int,
        pretrained: bool,
    ) -> None:
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
        depth = len(self.vit.blocks)
        if not 1 <= num_l2_blocks <= depth:
            raise ValueError(f"num_l2_blocks ({num_l2_blocks}) must be between 1 and the ViT depth ({depth})")
        self.num_l2_blocks = num_l2_blocks
        self.num_l1_blocks = depth - num_l2_blocks

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

    def forward_l1(self, x: Tensor) -> Tensor:
        """Embed the images and run the L1 blocks on the ViT tokens only.

        Args:
            x: Normalized images of shape ``[B, 3, resolution, resolution]``.

        Returns:
            Tokens of shape ``[B, num_prefix_tokens + H * W, embed_dim]``, before the final norm.
        """
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)  # adds pos embed to patches, prepends CLS and registers
        x = self.vit.norm_pre(x)
        for block in self.vit.blocks[: self.num_l1_blocks]:
            x = block(x)
        return x

    def forward_l2(self, tokens: Tensor, queries: Tensor) -> tuple[list[Tensor], Tensor]:
        """Prepend the queries to the ViT tokens and run the L2 blocks with full attention.

        Inside L2 the sequence is ``[queries, CLS, registers, patches]``, as in EoMT.

        Args:
            tokens: Output of :meth:`forward_l1`, shape ``[B, num_prefix_tokens + H * W, embed_dim]``.
            queries: Query tokens of shape ``[B, Q, embed_dim]``; ``Q`` may be 0.

        Returns:
            The query states after each L2 block (``num_l2_blocks`` tensors of shape
            ``[B, Q, embed_dim]``) and the ViT tokens after the last block, both before the final norm.
        """
        num_queries = queries.shape[1]
        x = torch.cat((queries, tokens), dim=1)
        query_states = []
        for block in self.vit.blocks[self.num_l1_blocks :]:
            x = block(x)
            query_states.append(x[:, :num_queries])
        return query_states, x[:, num_queries:]
