# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the configuration of the baseline (step 8a of EODT_SPEC.md §9, §5).

Covers:
- RF-DETR's ``build_model_from_config`` builds an RF-DETR (``LWDETR``) for a ``BaselineConfig``.
- Every architecture field must be written: none falls back to the default of an RF-DETR size.
- DINOv2 with registers, one query group and no RF-DETR checkpoint are enforced (§5).
- The positional embedding must lie on the patch grid.
"""

from typing import Any

import pytest
import torch
from pydantic import ValidationError

from rfdetr.models.eodt import BaselineConfig
from rfdetr.models.lwdetr import LWDETR, build_model_from_config

# Medium without windows at a small resolution (14 x 14 patches keep the build fast)
CONFIG = {
    "encoder": "dinov2_registers_windowed_small",
    "out_feature_indexes": [12],
    "resolution": 224,
    "patch_size": 16,
    "positional_encoding_size": 14,
    "num_windows": 1,
    "projector_scale": ["P4"],
    "hidden_dim": 256,
    "dec_layers": 4,
    "sa_nheads": 8,
    "ca_nheads": 16,
    "dec_n_points": 2,
    "device": "cpu",
}
REQUIRED = set(CONFIG) - {"device"}


def _config(**changes: Any) -> dict[str, Any]:
    """Return the arguments of ``CONFIG`` with the given changes."""
    return {**CONFIG, **changes}


def test_builds_rfdetr() -> None:
    """A BaselineConfig builds RF-DETR's model, with the decoder of the configuration."""
    torch.manual_seed(0)
    model = build_model_from_config(BaselineConfig(**CONFIG))

    assert isinstance(model, LWDETR)
    assert len(model.transformer.decoder.layers) == 4


def test_architecture_fields_have_no_default() -> None:
    """Exactly the architecture fields are required: the YAML file writes each of them."""
    required = {name for name, field in BaselineConfig.model_fields.items() if field.is_required()}

    assert required == REQUIRED


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_missing_architecture_field(name: str) -> None:
    """A missing architecture field is an error."""
    arguments = _config()
    del arguments[name]

    with pytest.raises(ValidationError, match=name):
        BaselineConfig(**arguments)


@pytest.mark.parametrize(
    "changes",
    [
        {"encoder": "dinov2_windowed_small"},  # no registers
        {"group_detr": 13},  # Group-DETR
        {"pretrain_weights": "rf-detr-medium.pth"},  # RF-DETR checkpoint, pretrained on Objects365
    ],
    ids=["no-registers", "group-detr", "rfdetr-checkpoint"],
)
def test_rejects_values_outside_the_baseline(changes: dict[str, Any]) -> None:
    """DINOv2 with registers, one query group and no RF-DETR checkpoint (§5)."""
    with pytest.raises(ValidationError):
        BaselineConfig(**_config(**changes))


def test_baseline_defaults() -> None:
    """Without these keys the configuration has one query group and no RF-DETR checkpoint."""
    config = BaselineConfig(**CONFIG)

    assert config.group_detr == 1
    assert config.pretrain_weights is None


@pytest.mark.parametrize(
    "changes",
    [{"positional_encoding_size": 37}, {"resolution": 232}],
    ids=["grid-of-another-resolution", "resolution-not-multiple-of-patch"],
)
def test_positional_embedding_on_patch_grid(changes: dict[str, Any]) -> None:
    """The positional embedding must have the patch grid, so that it is never interpolated at forward time."""
    with pytest.raises(ValidationError, match="positional_encoding_size"):
        BaselineConfig(**_config(**changes))
