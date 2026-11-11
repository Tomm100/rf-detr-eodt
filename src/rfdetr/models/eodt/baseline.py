# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""The RF-DETR baseline of EoDT (EODT_SPEC.md §5): its configuration, and the changes made after RF-DETR builds it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import model_validator
from torch import Tensor, nn

from rfdetr.config import ModelConfig
from rfdetr.models.backbone.dinov2 import get_config
from rfdetr.models.backbone.projector import LayerNorm
from rfdetr.models.eodt.dinov2_weights import load_pretrained_dinov2
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from rfdetr.models.lwdetr import LWDETR

logger = get_logger()


class BaselineConfig(ModelConfig):
    """Architecture configuration of the baseline: an RF-DETR on DINOv2 with registers, without Group-DETR.

    RF-DETR builds it with its own code (``build_model``), then :func:`adapt_baseline` loads the pretrained DINOv2
    weights at its patch size, which RF-DETR does not do for a patch other than 14, and removes the projector if asked.
    Unlike ``RFDETRMediumConfig``, the required architecture fields of ``ModelConfig`` keep no default here: the YAML
    file writes each of them, so a missing one is an error instead of the hidden default of an RF-DETR size.

    Attributes:
        encoder: DINOv2 with 4 registers, of size small, base or large, in RF-DETR's windowed implementation;
            ``num_windows: 1`` gives global attention. Only :class:`UpperBoundConfig` is without registers.
        out_feature_indexes: Blocks (from 1) whose output goes to the projector, strictly increasing. The last one
            must be the last block of the ViT: RF-DETR runs the ViT only up to the last exported block, so the later
            pretrained blocks would never be used, and counts the layers of the learning-rate decay up to it.
        projector: Keep RF-DETR's ``MultiScaleProjector``. Without it the decoder reads the last block of the ViT
            through :class:`LinearProjector`; ``out_feature_indexes`` then holds that block only, ``projector_scale``
            is ``["P4"]``, the stride of the ViT, and ``dual_projector`` is off (its second ``MultiScaleProjector``
            would feed the decoder). ``num_windows`` must also be 1: RF-DETR puts in windows every block it does not
            export, so with one exported block no block would have global attention.
        group_detr: One group of queries: no Group-DETR (§5).
        pretrain_weights: No RF-DETR checkpoint (and so no Objects365): the baseline starts from DINOv2 only.
    """

    encoder: Literal[
        "dinov2_registers_windowed_small", "dinov2_registers_windowed_base", "dinov2_registers_windowed_large"
    ]
    projector: bool
    group_detr: Literal[1] = 1
    pretrain_weights: None = None

    @property
    def dinov2_size(self) -> str:
        """DINOv2 size of the encoder: ``"small"``, ``"base"`` or ``"large"``."""
        return self.encoder.rsplit("_", 1)[-1]

    @property
    def registers(self) -> bool:
        """Whether the encoder is DINOv2 with registers, as RF-DETR reads it from the name (``backbone.py``)."""
        return "registers" in self.encoder.split("_")

    @model_validator(mode="after")
    def _check_positional_encoding_size(self) -> BaselineConfig:
        """Require the positional embedding on the patch grid, as in EoDT: no interpolation at every forward."""
        if self.resolution % self.patch_size or self.positional_encoding_size != self.resolution // self.patch_size:
            raise ValueError(
                f"resolution ({self.resolution}) must be a multiple of patch_size ({self.patch_size}) and "
                f"positional_encoding_size ({self.positional_encoding_size}) must be resolution // patch_size"
            )
        return self

    @model_validator(mode="after")
    def _check_out_feature_indexes(self) -> BaselineConfig:
        """Require the exported blocks to end at the last block of the ViT; without projector, one block, no windows."""
        depth = get_config(self.dinov2_size, use_registers=self.registers)["num_hidden_layers"]
        indexes = self.out_feature_indexes
        if not indexes or indexes != sorted(set(indexes)) or indexes[0] < 1 or indexes[-1] != depth:
            raise ValueError(
                f"out_feature_indexes ({indexes}) must be strictly increasing blocks from 1 (0 is the embedding) and "
                f"end at the last block of DINOv2 {self.dinov2_size} ({depth})"
            )
        if not self.projector and (len(indexes) != 1 or self.projector_scale != ["P4"] or self.dual_projector):
            raise ValueError(
                f"without projector the decoder reads the last block only: out_feature_indexes must be [{depth}] "
                f"(got {indexes}), projector_scale ['P4'] (got {self.projector_scale}) and dual_projector false"
            )
        if not self.projector and self.num_windows != 1:
            raise ValueError(
                f"without projector num_windows must be 1 (got {self.num_windows}): RF-DETR puts in windows every "
                f"block that it does not export, so with only block {depth} exported all the {depth} blocks would be "
                "windowed and none would have global attention"
            )
        return self


class UpperBoundConfig(BaselineConfig):
    """Architecture configuration of the upper bound: an RF-DETR of Table 7 as published, with our training (§10).

    The model is built as the baseline's (:class:`BaselineConfig`); its own class marks the runs of the upper bound
    in the YAML file, which W&B tags ``upperbound`` (``rfdetr.models.eodt.train.wandb_tags``).

    Attributes:
        encoder: DINOv2-B without registers, as RF-DETR's published checkpoints and Table 7.
    """

    encoder: Literal["dinov2_windowed_base"]  # type: ignore[assignment]


class LinearProjector(nn.Module):
    """Projector of the baseline without RF-DETR's ``MultiScaleProjector`` (EODT_SPEC.md §5).

    Brings the output of the last ViT block, after DINOv2's final LayerNorm, to ``hidden_dim`` with a ``Linear`` and
    then the LayerNorm that also ends ``MultiScaleProjector``: removing the projector removes the multi-scale fusion,
    not the normalization of the decoder's input. It takes and returns lists of feature maps, as
    ``MultiScaleProjector`` does.

    Args:
        in_channels: Width of the ViT.
        out_channels: Hidden dimension of the decoder.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = LayerNorm(out_channels)  # what get_norm("LN", ...) gives MultiScaleProjector

    def forward(self, feature_maps: list[Tensor]) -> list[Tensor]:
        """Project the only feature map, ``[B, in_channels, H, W]``, to ``[B, out_channels, H, W]``."""
        (x,) = feature_maps
        x = self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return [self.norm(x)]


def adapt_baseline(model: LWDETR, model_config: BaselineConfig) -> None:
    """Apply the baseline's changes to the LWDETR that RF-DETR built from ``model_config``.

    Loads the pretrained DINOv2 weights resampled to the patch size of the configuration
    (:func:`~rfdetr.models.eodt.dinov2_weights.load_pretrained_dinov2`) and, with ``projector: false``, replaces
    the projector with :class:`LinearProjector`.

    Args:
        model: The model returned by ``build_model``, with randomly initialized DINOv2 weights.
        model_config: Configuration the model was built from.
    """
    backbone = model.backbone[0]
    load_pretrained_dinov2(backbone.encoder.encoder, model_config.dinov2_size)
    # DinoV2 has just warned that it does not load the weights at this patch size
    logger.info(
        f"DINOv2 {model_config.dinov2_size} {'with' if model_config.registers else 'without'} registers: pretrained "
        f"weights resampled to patch {model_config.patch_size}"
    )
    if not model_config.projector:
        backbone.projector = LinearProjector(backbone.encoder._out_feature_channels[-1], model_config.hidden_dim)
