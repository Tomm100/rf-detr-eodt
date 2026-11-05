# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
# Two-stage query selection adapted from RF-DETR, models/transformer.py (Transformer.forward, two_stage branch)
# and models/lwdetr.py (first-stage heads)
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""First stage of the ``selection`` queries of EoDT: the top-Q patches after L1 and their boxes (EODT_SPEC.md §3.3)."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from rfdetr.models.math import MLP
from rfdetr.models.transformer import gen_encoder_output_proposals


def reparam_boxes(delta: Tensor, reference: Tensor) -> Tensor:
    """Apply a box delta to reference boxes with RF-DETR's ``bbox_reparam`` formula.

    Args:
        delta: Deltas ``[..., 4]``: center offset in units of the reference size, log of the size ratio.
        reference: Reference boxes ``[..., 4]``, cxcywh; ``delta`` and ``reference`` broadcast with each other.

    Returns:
        Boxes ``[..., 4]``, cxcywh: center = delta_xy * wh + xy, size = exp(delta_wh) * wh.
    """
    center = delta[..., :2] * reference[..., 2:] + reference[..., :2]
    size = delta[..., 2:].exp() * reference[..., 2:]
    return torch.cat([center, size], dim=-1)


class QuerySelection(nn.Module):
    """Two-stage query selection of RF-DETR on one scale: the patches after L1.

    Every patch gets a proposal box centred on it (``gen_encoder_output_proposals``) and, after a ``Linear`` and a
    ``LayerNorm``, class logits; the Q patches with the highest class score are selected and a box head predicts a
    delta with respect to their proposals. This is the ``group_detr = 1`` case of ``Transformer.forward`` with
    ``bbox_reparam``. The heads are copies of the final ones, made after their initialization as ``LWDETR`` does:
    the class bias starts from a prior probability of 0.01 and the last box layer from zero, so that at first every
    box is its proposal.

    Args:
        hidden_dim: Width of the ViT.
        num_queries: Number of selected patches Q.
        class_embed: Final class head of EoDT, copied as the class head of this stage.
        bbox_embed: Final box head of EoDT, copied as the box head of this stage.
    """

    def __init__(self, hidden_dim: int, num_queries: int, class_embed: nn.Linear, bbox_embed: MLP) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)
        self.enc_out_class_embed = copy.deepcopy(class_embed)
        self.enc_out_bbox_embed = copy.deepcopy(bbox_embed)

    def forward(self, patches: Tensor, grid_size: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
        """Select the Q patches with the highest class score and predict their boxes.

        Args:
            patches: Patch tokens after L1 and ``vit.norm``, ``[B, H * W, D]`` in row-major order. They are not
                detached: the loss of this stage also reaches the L1 blocks.
            grid_size: Patch grid ``(H, W)``.

        Returns:
            The boxes b0 detached from the graph, ``[B, Q, 4]``, and the outputs of this stage for its loss: class
            logits ``[B, Q, C]`` and the same boxes b0 before the detach.
        """
        # no padding mask: the input is a square resize (proposals that fall outside (0.01, 0.99) are zeroed)
        memory, proposals = gen_encoder_output_proposals(patches, None, [grid_size], unsigmoid=False)
        memory = self.enc_output_norm(self.enc_output(memory))
        logits = self.enc_out_class_embed(memory)
        topk = torch.topk(logits.max(-1)[0], self.num_queries, dim=1)[1]  # [B, Q]

        def gather(values: Tensor) -> Tensor:
            return torch.gather(values, 1, topk.unsqueeze(-1).expand(-1, -1, values.shape[-1]))

        # the box head runs on the selected tokens only, as in RF-DETR
        boxes = reparam_boxes(self.enc_out_bbox_embed(gather(memory)), gather(proposals))
        return boxes.detach(), gather(logits), boxes
