# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
# Learned queries and final norm before the head adapted from EoMT (https://github.com/tue-mps/eomt),
# models/eomt.py
# Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. Licensed under the MIT License.
# ------------------------------------------------------------------------
# Detection head, selection queries and output format adapted from RF-DETR, models/lwdetr.py and
# models/transformer.py
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
from rfdetr.models.eodt.config import QueryMode
from rfdetr.models.eodt.selection import QuerySelection, reparam_boxes
from rfdetr.models.math import MLP
from rfdetr.models.transformer import gen_sineembed_for_position
from rfdetr.utilities.tensors import NestedTensor


class EoDT(nn.Module):
    """Encoder-only detector: the queries go through the last L2 ViT blocks with the patches (EODT_SPEC.md §3).

    ``query_mode`` chooses the queries (§3.3):

    - ``"learned"`` (as EoMT): learned embeddings, the same for every image. The head predicts the absolute box,
      ``sigmoid(MLP(query))``.
    - ``"selection"`` (as RF-DETR): :class:`QuerySelection` selects the top-Q patches after L1 and predicts their
      boxes b0, detached; a learned offset per query (``refpoint_embed``, zero at first) moves b0 to the reference.
      Each query is a learned content plus an encoding of its reference (``ref_point_head``), and in every L2 block
      the head predicts a delta with respect to the reference.

    After every L2 block the query states go through the pretrained final norm of the ViT (``vit.norm``) and then
    through one head shared by all blocks: a ``Linear`` for the classes and a 3-layer MLP for the box.

    Args:
        backbone: DINOv2 backbone split into L1 and L2 blocks.
        num_queries: Number of queries Q.
        num_classes: Number of class logits, as in ``LWDETR`` (91 for COCO: raw category ids 1..90
            used as indices, one independent sigmoid per class and no "no-object" logit).
        query_mode: ``"learned"`` or ``"selection"``.
    """

    def __init__(self, backbone: EoDTBackbone, num_queries: int, num_classes: int, *, query_mode: QueryMode) -> None:
        super().__init__()
        if num_queries < 1:
            raise ValueError(f"num_queries ({num_queries}) must be at least 1")
        if num_classes < 1:
            raise ValueError(f"num_classes ({num_classes}) must be at least 1")
        if query_mode not in ("learned", "selection"):
            raise ValueError(f"query_mode must be 'learned' or 'selection', got {query_mode!r}")
        self.backbone = backbone
        self.num_queries = num_queries
        self.query_mode = query_mode
        hidden_dim = backbone.embed_dim

        # the queries of learned, the content of the queries of selection; default init N(0, 1), as self.q in EoMT
        # and query_feat in LWDETR
        self.query_feat = nn.Embedding(num_queries, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # init prior_prob setting for focal loss, as in LWDETR
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # With learned queries bbox_embed keeps the default nn.Linear init, as DETR: the box is absolute and there
        # is no reference box.
        if query_mode == "selection":
            # the box is a delta with respect to the reference: last layer at zero, as in LWDETR, so that every
            # query starts from its reference
            nn.init.constant_(self.bbox_embed.layers[-1].weight, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias, 0)
            # copies the heads after their init, as LWDETR does for enc_out_class_embed and enc_out_bbox_embed
            self.selection = QuerySelection(hidden_dim, num_queries, self.class_embed, self.bbox_embed)
            self.refpoint_embed = nn.Embedding(num_queries, 4)
            nn.init.constant_(self.refpoint_embed.weight, 0)
            self.ref_point_head = MLP(2 * hidden_dim, hidden_dim, hidden_dim, 2)

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
            ``pred_logits`` ``[B, Q, num_classes]`` and ``pred_boxes`` ``[B, Q, 4]`` (cxcywh) from the last L2 block,
            and ``aux_outputs`` with the same two keys for the other L2 blocks, first to second-to-last. With
            ``selection`` also ``enc_outputs``: class logits and boxes b0 of the first stage, for its loss. With
            ``learned`` the boxes are in [0, 1]; with ``selection`` they come from the ``bbox_reparam`` formula and
            are not bounded, as in RF-DETR.
        """
        x = samples.tensors if isinstance(samples, NestedTensor) else samples
        tokens = self.backbone.forward_l1(x)
        if self.query_mode == "learned":
            queries = self.query_feat.weight[None].expand(x.shape[0], -1, -1)
        else:
            # vit.norm only on this branch, as RF-DETR normalizes the features it exports; L2 gets the raw tokens
            patches = self.backbone.vit.norm(tokens[:, self.backbone.num_prefix_tokens :])
            boxes_b0, enc_logits, enc_boxes = self.selection(patches, self.backbone.grid_size)
            reference = reparam_boxes(self.refpoint_embed.weight[None], boxes_b0)  # [B, Q, 4]
            # as ref_point_head in RF-DETR's decoder: D / 2 frequencies per coordinate, 2D features
            query_pos = self.ref_point_head(gen_sineembed_for_position(reference, self.backbone.embed_dim // 2))
            queries = self.query_feat.weight[None] + query_pos
        query_states, _ = self.backbone.forward_l2(tokens, queries)

        hs = torch.stack([self.backbone.vit.norm(state) for state in query_states])  # [L2, B, Q, D]
        outputs_class = self.class_embed(hs)
        if self.query_mode == "learned":
            outputs_coord = self.bbox_embed(hs).sigmoid()
        else:
            # always with respect to the reference, as lite_refpoint_refine in RF-DETR
            outputs_coord = reparam_boxes(self.bbox_embed(hs), reference)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "aux_outputs": [
                {"pred_logits": logits, "pred_boxes": boxes}
                for logits, boxes in zip(outputs_class[:-1], outputs_coord[:-1])
            ],
        }
        if self.query_mode == "selection":
            out["enc_outputs"] = {"pred_logits": enc_logits, "pred_boxes": enc_boxes}
        return out
