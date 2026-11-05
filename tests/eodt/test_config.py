# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for rfdetr.models.eodt.config.EoDTConfig (step 4a of EODT_SPEC.md §9).

Covers:
- Defaults taken from the specification (§3.1, §3.3, §4).
- L2 has no default and must be at least 1.
- Fields with a single allowed value for now: ``query_mode``, ``group_detr``, ``num_windows``.
- ModelConfig fields that EoDT does not use cannot leave their default.
- Fields that EoDT or the training pipeline read can be changed.
- EoDTConfig is a ModelConfig, as RF-DETR's training pipeline requires.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from rfdetr.config import ModelConfig
from rfdetr.models.eodt import EoDTConfig


def test_defaults_from_spec() -> None:
    """Backbone, resolution, patch size and queries default to the values of the specification."""
    config = EoDTConfig(num_l2_blocks=3, query_mode="learned")

    assert config.backbone_name == "vit_small_patch14_reg4_dinov2"  # §3.1
    assert config.resolution == 576  # §3.1
    assert config.patch_size == 16  # §3.1
    assert config.num_queries == 300  # §3.3
    assert config.query_mode == "learned"  # §3.3, §8 point 8
    assert config.group_detr == 1  # §4, alternative (a)
    assert config.num_windows == 1  # global attention, §3.1
    assert config.num_classes == 90  # COCO, 91 logits after the + 1 of the builder (§8 point 7)
    assert config.positional_encoding_size == 576 // 16


@pytest.mark.parametrize("kwargs", [{}, {"num_l2_blocks": 0}], ids=["missing", "zero"])
def test_num_l2_blocks_required(kwargs: dict[str, Any]) -> None:
    """L2 has no default in the code and must be at least 1."""
    with pytest.raises(ValidationError, match="num_l2_blocks"):
        EoDTConfig(query_mode="learned", **kwargs)


@pytest.mark.parametrize(
    ("name", "value"),
    [("query_mode", "mixed"), ("group_detr", 13), ("num_windows", 2)],
)
def test_single_value_fields(name: str, value: Any) -> None:
    """Only learned or selection queries, one query group and global attention for now."""
    with pytest.raises(ValidationError, match=name):
        EoDTConfig(**{"num_l2_blocks": 3, "query_mode": "learned", name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("dec_layers", 4),
        ("hidden_dim", 256),
        ("encoder", "dinov2_windowed_small"),
        ("two_stage", True),
        ("segmentation_head", True),
        ("use_grouppose_keypoints", True),
        ("num_keypoints_per_class", [17]),
        ("gradient_checkpointing", True),
        ("backbone_lora", True),
        ("pretrain_weights", "rf-detr-medium.pth"),
        ("postprocess_trace_alpha", 0.5),  # read by PostProcess only for keypoints
        ("positional_encoding_size", 99),  # must equal resolution // patch_size
    ],
)
def test_unused_fields_rejected(name: str, value: Any) -> None:
    """A value that EoDT would silently ignore is an error that names the field."""
    with pytest.raises(ValidationError, match=name):
        EoDTConfig(**{"num_l2_blocks": 3, "query_mode": "learned", name: value})


def test_unused_field_assignment_rejected() -> None:
    """The same check runs when a field is assigned after construction."""
    config = EoDTConfig(num_l2_blocks=3, query_mode="learned")
    with pytest.raises(ValidationError, match="gradient_checkpointing"):
        config.gradient_checkpointing = True


def test_used_fields_accepted() -> None:
    """Fields read by EoDT, the loss or the training pipeline can be changed."""
    config = EoDTConfig(
        num_l2_blocks=4,
        query_mode="learned",
        backbone_name="vit_large_patch14_reg4_dinov2",
        resolution=640,
        num_queries=200,
        num_classes=80,
        num_select=100,
        ia_bce_loss=False,
        compile=True,
    )

    assert (config.num_l2_blocks, config.resolution, config.num_queries) == (4, 640, 200)
    assert config.positional_encoding_size == 640 // 16  # kept in sync by ModelConfig


def test_is_model_config() -> None:
    """RFDETRModelModule, RFDETRDataModule and build_trainer take a ModelConfig."""
    assert issubclass(EoDTConfig, ModelConfig)
