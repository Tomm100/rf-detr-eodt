# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
# Layer-wise learning rate and weight decay adapted from RF-DETR, models/backbone/backbone.py
# (get_named_param_lr_pairs, get_dinov2_lr_decay_rate, get_dinov2_weight_decay_rate)
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Optimizer parameter groups of EoDT, with the same learning rate and weight decay as the RF-DETR baseline.

Every tensor of the timm ViT gets the value that RF-DETR gives to the same tensor of its Hugging Face DINOv2.
"""

from __future__ import annotations

from typing import Any

from rfdetr.models.eodt.model import EoDT

# timm names of the embeddings, the "embeddings" module of Hugging Face DINOv2
_EMBEDDING_KEYS = ("patch_embed", "pos_embed", "cls_token", "reg_token")


def get_eodt_lr_decay_rate(name: str, lr_decay_rate: float, num_layers: int) -> float:
    """Return the layer-wise learning rate decay of a ViT parameter, as ``get_dinov2_lr_decay_rate``.

    The embeddings are layer 0, block ``i`` is layer ``i + 1`` (the L2 blocks included) and the final norm is
    layer ``num_layers + 1``, which has no decay.

    Args:
        name: Parameter name in EoDT, e.g. ``backbone.vit.blocks.3.attn.qkv.weight``.
        lr_decay_rate: Decay per layer.
        num_layers: Number of layers that decay.

    Returns:
        ``lr_decay_rate ** (num_layers + 1 - layer_id)``.
    """
    layer_id = num_layers + 1
    if any(key in name for key in _EMBEDDING_KEYS):
        layer_id = 0
    elif ".blocks." in name:
        layer_id = int(name.split(".blocks.")[1].split(".")[0]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_eodt_weight_decay_rate(name: str) -> float:
    """Return the weight decay multiplier of a ViT parameter, as ``get_dinov2_weight_decay_rate``.

    No weight decay on embeddings, biases and norms. The layer scale (``ls1.gamma``, ``ls2.gamma``) keeps it, as
    ``lambda1`` in the baseline.

    Args:
        name: Parameter name in EoDT.

    Returns:
        0.0 or 1.0.
    """
    if any(key in name for key in _EMBEDDING_KEYS) or "bias" in name or "norm" in name:
        return 0.0
    return 1.0


def build_eodt_param_dicts(args: Any, model: EoDT) -> list[dict[str, Any]]:
    """Build one single-parameter group per trainable parameter, as RF-DETR's ``_build_param_dicts``.

    ViT parameters get ``lr_encoder``, the layer-wise decay and ``lr_component_decay ** 2``, as the DINOv2
    encoder of the baseline. Queries and head get ``lr`` and the optimizer's weight decay, as LWDETR's queries
    and heads.

    Args:
        args: Namespace supplying ``lr``, ``lr_encoder``, ``lr_vit_layer_decay``, ``lr_component_decay`` and
            ``weight_decay``.
        model: The EoDT model whose parameters the optimizer will own.

    Returns:
        One group per trainable parameter, queries and head first, then the ViT, in the order of
        ``_build_param_dicts``.
    """
    # as out_feature_indexes[-1] + 1 of the baseline Medium, whose last feature is the output of the last block
    num_layers = len(model.backbone.vit.blocks) + 1
    other_param_dicts = []
    backbone_param_dicts = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            # same order of operations as get_named_param_lr_pairs, so the values match the baseline exactly
            lr = (
                args.lr_encoder
                * get_eodt_lr_decay_rate(name, lr_decay_rate=args.lr_vit_layer_decay, num_layers=num_layers)
                * args.lr_component_decay**2
            )
            weight_decay = args.weight_decay * get_eodt_weight_decay_rate(name)
            backbone_param_dicts.append({"params": param, "lr": lr, "weight_decay": weight_decay})
        else:
            other_param_dicts.append({"params": param, "lr": args.lr})
    return other_param_dicts + backbone_param_dicts
