# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Builders that turn an ``EoDTConfig`` into EoDT, called by RF-DETR's ``build_model_from_config``."""

from __future__ import annotations

from rfdetr.config import MultiScale, TrainConfig
from rfdetr.models.eodt.backbone import EoDTBackbone
from rfdetr.models.eodt.config import EoDTConfig
from rfdetr.models.eodt.model import EoDT


def build_eodt_from_config(model_config: EoDTConfig, train_config: TrainConfig | None = None) -> EoDT:
    """Build EoDT with the pretrained DINOv2 weights.

    Args:
        model_config: EoDT configuration.
        train_config: Training configuration. When given, it must keep the input square and fixed, because
            EoDT ignores the padding mask and its ViT has a fixed patch grid.

    Returns:
        EoDT with ``num_classes + 1`` class logits, as ``build_model`` does for LWDETR.

    Raises:
        ValueError: If ``train_config`` enables multi-scale training or disables the square resize.
    """
    if train_config is not None and (
        train_config.multi_scale is not MultiScale.OFF or not train_config.square_resize_div_64
    ):
        raise ValueError(
            "EoDT needs a fixed square input: set multi_scale: off and square_resize_div_64: true "
            f"(got multi_scale: {train_config.multi_scale.value}, "
            f"square_resize_div_64: {train_config.square_resize_div_64})"
        )
    backbone = EoDTBackbone(
        model_config.backbone_name,
        model_config.resolution,
        model_config.patch_size,
        model_config.num_l2_blocks,
        pretrained=True,
    )
    # num_classes + 1 as in build_model: COCO category ids 1..90 are used directly as logit indices
    return EoDT(backbone, model_config.num_queries, model_config.num_classes + 1)
