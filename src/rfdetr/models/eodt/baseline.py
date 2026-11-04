# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""The RF-DETR baseline of EoDT (EODT_SPEC.md §5): its configuration, and the changes made after RF-DETR builds it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import model_validator

from rfdetr.config import ModelConfig
from rfdetr.models.eodt.dinov2_weights import load_pretrained_dinov2
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from rfdetr.models.lwdetr import LWDETR

logger = get_logger()


class BaselineConfig(ModelConfig):
    """Architecture configuration of the baseline: an RF-DETR on DINOv2 with registers, without Group-DETR.

    RF-DETR builds it with its own code (``build_model``), then :func:`adapt_baseline` loads the pretrained DINOv2
    weights at its patch size, which RF-DETR does not do for a patch other than 14. Unlike ``RFDETRMediumConfig``, the
    architecture fields of ``ModelConfig`` keep no default here: the YAML file writes every value of the chosen
    RF-DETR configuration, so a missing one is an error instead of a hidden default.

    Attributes:
        encoder: DINOv2 with 4 registers, of size small, base or large, in RF-DETR's windowed implementation;
            ``num_windows: 1`` gives global attention.
        group_detr: One group of queries: no Group-DETR (§5).
        pretrain_weights: No RF-DETR checkpoint (and so no Objects365): the baseline starts from DINOv2 only.
    """

    encoder: Literal[
        "dinov2_registers_windowed_small", "dinov2_registers_windowed_base", "dinov2_registers_windowed_large"
    ]
    group_detr: Literal[1] = 1
    pretrain_weights: None = None

    @model_validator(mode="after")
    def _check_positional_encoding_size(self) -> BaselineConfig:
        """Require the positional embedding on the patch grid, as in EoDT: no interpolation at every forward."""
        if self.resolution % self.patch_size or self.positional_encoding_size != self.resolution // self.patch_size:
            raise ValueError(
                f"resolution ({self.resolution}) must be a multiple of patch_size ({self.patch_size}) and "
                f"positional_encoding_size ({self.positional_encoding_size}) must be resolution // patch_size"
            )
        return self


def adapt_baseline(model: LWDETR, model_config: BaselineConfig) -> None:
    """Apply the baseline's changes to the LWDETR that RF-DETR built from ``model_config``.

    Loads the pretrained DINOv2 weights resampled to the patch size of the configuration
    (:func:`~rfdetr.models.eodt.dinov2_weights.load_pretrained_dinov2`).

    Args:
        model: The model returned by ``build_model``, with randomly initialized DINOv2 weights.
        model_config: Configuration the model was built from.
    """
    size = model_config.encoder.rsplit("_", 1)[-1]
    load_pretrained_dinov2(model.backbone[0].encoder.encoder, size)
    # DinoV2 has just warned that it does not load the weights at this patch size
    logger.info(f"DINOv2 {size} with registers: pretrained weights resampled to patch {model_config.patch_size}")
