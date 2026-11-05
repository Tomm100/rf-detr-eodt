# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Configuration of the EoDT detector, used as the ``ModelConfig`` of RF-DETR's training pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from rfdetr.config import ModelConfig

# ModelConfig fields that EoDT or RF-DETR's training pipeline read and that a configuration may change.
# Every other field must keep its default, so that a configuration cannot set a value EoDT would ignore.
_USED_FIELDS = frozenset(
    {
        # EoDT
        "backbone_name",
        "num_l2_blocks",
        "query_mode",
        "resolution",
        "patch_size",
        "num_queries",
        "num_classes",
        # loss and postprocessing of RF-DETR
        "ia_bce_loss",
        "num_select",
        # training pipeline and checkpoint metadata
        "device",
        "amp",
        "compile",
        "fused_optimizer",
        "license",
        "model_name",
    }
)


#: How the queries of EoDT are built (EODT_SPEC.md §3.3).
QueryMode = Literal["learned", "selection"]


class EoDTConfig(ModelConfig):
    """Architecture configuration of EoDT.

    It is a ``ModelConfig`` because ``RFDETRModelModule``, ``RFDETRDataModule`` and ``build_trainer`` read
    ``ModelConfig`` fields. The decoder and projector fields of LWDETR are ``None``, and every other field
    that EoDT does not use must keep its default.

    Attributes:
        backbone_name: timm name of the DINOv2 backbone. The register tokens are determined by the name.
        num_l2_blocks: Number of final ViT blocks that also process the queries (L2). No default: every
            configuration sets it, because it depends on the backbone size.
        query_mode: How the queries are built (§3.3): ``"selection"`` (as RF-DETR, the main model) or ``"learned"``
            (as EoMT, an ablation). No default: a missing key must not start one of the two in silence.
        resolution: Side of the square input image, in pixels.
        patch_size: Patch size the pretrained DINOv2 weights are resampled to.
        num_queries: Number of queries Q. With ``selection`` at most the number of patches, among which the queries
            are selected.
        group_detr: Number of query groups. Only 1 (no Group-DETR) for now.
        num_windows: Global attention, no windows. Read by the data module to pad the batch.
        positional_encoding_size: Not used by EoDT (timm interpolates its own ``pos_embed``), but read as an
            integer by ``RFDETRModelModule.on_load_checkpoint``, where it has no effect on timm keys. It must
            equal ``resolution // patch_size``; ``ModelConfig`` updates it when ``resolution`` is set.
    """

    backbone_name: str = "vit_small_patch14_reg4_dinov2"
    num_l2_blocks: int = Field(ge=1)
    query_mode: QueryMode
    resolution: int = 576
    patch_size: int = 16
    num_queries: int = 300
    group_detr: Literal[1] = 1
    num_windows: Literal[1] = 1
    positional_encoding_size: int = 576 // 16

    # Decoder and projector of LWDETR: not part of EoDT.
    encoder: None = None
    out_feature_indexes: None = None
    dec_layers: None = None
    projector_scale: None = None
    hidden_dim: None = None  # the hidden size is the one of the backbone
    sa_nheads: None = None
    ca_nheads: None = None
    dec_n_points: None = None
    two_stage: None = None
    bbox_reparam: None = None
    lite_refpoint_refine: None = None
    layer_norm: None = None

    @model_validator(mode="after")
    def _check_unused_fields(self) -> EoDTConfig:
        """Reject a value that EoDT would ignore in a field that it does not use."""
        expected = {"positional_encoding_size": self.resolution // self.patch_size}
        for name, field in type(self).model_fields.items():
            if name in _USED_FIELDS:
                continue
            value = expected.get(name, field.get_default(call_default_factory=True))
            if getattr(self, name) != value:
                raise ValueError(f"'{name}' is not used by EoDT and must be {value!r}")
        return self

    @model_validator(mode="after")
    def _check_num_queries(self) -> EoDTConfig:
        """With ``selection`` the Q queries are Q distinct patches: there must be at least Q of them."""
        num_patches = (self.resolution // self.patch_size) ** 2
        if self.query_mode == "selection" and self.num_queries > num_patches:
            raise ValueError(
                f"with query_mode 'selection' num_queries ({self.num_queries}) must be at most the number of patches "
                f"({num_patches} at resolution {self.resolution} and patch_size {self.patch_size})"
            )
        return self
