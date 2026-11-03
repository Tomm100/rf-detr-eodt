# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Builders that turn an ``EoDTConfig`` into EoDT and its loss.

They are called by RF-DETR's ``build_model_from_config`` and ``build_criterion_from_config``.
"""

from __future__ import annotations

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import MultiScale, TrainConfig
from rfdetr.models._defaults import MODEL_DEFAULTS, ModelDefaults
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.eodt.backbone import EoDTBackbone
from rfdetr.models.eodt.config import EoDTConfig
from rfdetr.models.eodt.model import EoDT
from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.models.postprocess import PostProcess


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


def build_eodt_criterion_from_config(
    model_config: EoDTConfig,
    train_config: TrainConfig,
    defaults: ModelDefaults = MODEL_DEFAULTS,
) -> tuple[SetCriterion, PostProcess]:
    """Build RF-DETR's criterion and postprocessor for EoDT.

    ``build_criterion_and_postprocessors`` is used unchanged; only the two namespace fields that describe
    LWDETR's decoder are replaced with their EoDT meaning.

    Args:
        model_config: EoDT configuration.
        train_config: Training configuration, which supplies ``cls_loss_coef``.
        defaults: Constants of RF-DETR, which supply the other loss and matcher coefficients.

    Returns:
        A 2-tuple of ``(SetCriterion, PostProcess)``.
    """
    ns = _namespace_from_configs(model_config, train_config, defaults)
    # build_criterion_and_postprocessors weights dec_layers - 1 aux outputs: one per L2 block but the last
    ns.dec_layers = model_config.num_l2_blocks
    # learned queries have no first stage: no enc_outputs and no "_enc" weights (step 11 derives it from query_mode)
    ns.two_stage = False
    return build_criterion_and_postprocessors(ns)
