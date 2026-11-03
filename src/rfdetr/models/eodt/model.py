# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
# Learned queries and final norm before the head adapted from EoMT (https://github.com/tue-mps/eomt),
# models/eomt.py
# Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. Licensed under the MIT License.
# ------------------------------------------------------------------------
# Detection head and output format adapted from RF-DETR, models/lwdetr.py
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""EoDT detector: queries in the last L2 ViT blocks and one detection head shared by all of them."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from rfdetr.models.eodt.backbone import EoDTBackbone
from rfdetr.models.math import MLP
from rfdetr.utilities.tensors import NestedTensor


class EoDT(nn.Module):
    """Encoder-only detector with learned queries (``query_mode = "learned"``).

    The queries are learned embeddings, the same for every image, prepended to the ViT tokens
    before the first L2 block. After every L2 block the query states go through the pretrained
    final norm of the ViT (``vit.norm``) and then through one head shared by all blocks: a
    ``Linear`` for the classes and a 3-layer MLP for the absolute box, ``sigmoid(MLP(query))``.

    Args:
        backbone: DINOv2 backbone split into L1 and L2 blocks.
        num_queries: Number of queries Q.
        num_classes: Number of class logits, as in ``LWDETR`` (91 for COCO: raw category ids 1..90
            used as indices, one independent sigmoid per class and no "no-object" logit).
    """

    def __init__(self, backbone: EoDTBackbone, num_queries: int, num_classes: int) -> None:
        super().__init__()
        if num_queries < 1:
            raise ValueError(f"num_queries ({num_queries}) must be at least 1")
        if num_classes < 1:
            raise ValueError(f"num_classes ({num_classes}) must be at least 1")
        self.backbone = backbone
        self.num_queries = num_queries
        hidden_dim = backbone.embed_dim

        # default init N(0, 1), as self.q in EoMT
        self.query_feat = nn.Embedding(num_queries, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # init prior_prob setting for focal loss, as in LWDETR
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # bbox_embed keeps the default nn.Linear init, as DETR: with query_mode = "learned" the box is
        # absolute and there is no reference box. With query_mode = "selection" (step 11) the last layer
        # must be zero-initialized as in LWDETR, because there the MLP predicts a delta with respect to b0.

    def forward(
        self, samples: NestedTensor | Tensor, targets: list[dict[str, Tensor]] | None = None
    ) -> dict[str, Any]:
        """Predict classes and boxes after every L2 block.

        Args:
            samples: Normalized images of shape ``[B, 3, resolution, resolution]``, alone or in the
                ``NestedTensor`` of RF-DETR's dataloader. Its padding mask is ignored: the input is a square
                resize without padding (``build_eodt_from_config`` rejects multi-scale and non-square resize).
            targets: Ignored. Accepted because ``RFDETRModelModule.training_step`` passes them to the model.

        Returns:
            ``pred_logits`` ``[B, Q, num_classes]`` and ``pred_boxes`` ``[B, Q, 4]`` (cxcywh in [0, 1])
            from the last L2 block, and ``aux_outputs`` with the same two keys for the other L2 blocks,
            first to second-to-last. No ``enc_outputs`` with learned queries.
        """
        x = samples.tensors if isinstance(samples, NestedTensor) else samples
        tokens = self.backbone.forward_l1(x)
        queries = self.query_feat.weight[None].expand(x.shape[0], -1, -1)
        query_states, _ = self.backbone.forward_l2(tokens, queries)

        hs = torch.stack([self.backbone.vit.norm(state) for state in query_states])  # [L2, B, Q, D]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "aux_outputs": [
                {"pred_logits": logits, "pred_boxes": boxes}
                for logits, boxes in zip(outputs_class[:-1], outputs_coord[:-1])
            ],
        }
