# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the configuration of the baseline (steps 8a and 8b of EODT_SPEC.md §9, §5).

Covers:
- RF-DETR's ``build_model_from_config`` builds an RF-DETR (``LWDETR``) for a ``BaselineConfig``.
- Every architecture field must be written, ``projector`` included: none falls back to the default of an RF-DETR size.
- DINOv2 with registers, one query group and no RF-DETR checkpoint are enforced (§5).
- The positional embedding must lie on the patch grid.
- The exported blocks increase and end at the last block of the ViT: a ViT-L that keeps the blocks of ViT-S is an
  error.
- Without projector the configuration exports the last block only, at the stride of the ViT, and has no windows:
  RF-DETR would put every block in windows, since it does so with every block that it does not export.
"""

from typing import Any

import pytest
import torch
from pydantic import ValidationError

from rfdetr.models.backbone.dinov2 import compute_window_block_indexes
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
    "projector": True,
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


@pytest.mark.parametrize(
    "changes",
    [
        {"encoder": "dinov2_registers_windowed_large", "out_feature_indexes": [3, 6, 9, 12]},  # ViT-S value forgotten
        {"out_feature_indexes": [3, 6, 9]},  # the last block of the ViT is not exported
        {"out_feature_indexes": [3, 6, 9, 12, 13]},  # beyond the last block
        {"out_feature_indexes": [6, 3, 12]},  # not increasing
        {"out_feature_indexes": [3, 3, 12]},  # repeated
        {"out_feature_indexes": [0, 12]},  # 0 is the embedding, not a block
        {"out_feature_indexes": []},
    ],
    ids=["vitl-with-vits-blocks", "last-missing", "beyond-depth", "not-increasing", "repeated", "embedding", "empty"],
)
def test_out_feature_indexes_end_at_last_block(changes: dict[str, Any]) -> None:
    """The exported blocks strictly increase from 1 and end at the last block: 12 for ViT-S and ViT-B, 24 for ViT-L."""
    with pytest.raises(ValidationError, match="out_feature_indexes"):
        BaselineConfig(**_config(**changes))


@pytest.mark.parametrize(
    ("encoder", "out_feature_indexes"),
    [
        ("dinov2_registers_windowed_small", [3, 6, 9, 12]),
        ("dinov2_registers_windowed_base", [3, 6, 9, 12]),
        ("dinov2_registers_windowed_large", [6, 12, 18, 24]),
        ("dinov2_registers_windowed_large", [24]),
    ],
)
def test_out_feature_indexes_of_each_size(encoder: str, out_feature_indexes: list[int]) -> None:
    """Blocks that end at the last one of each size are accepted."""
    config = BaselineConfig(**_config(encoder=encoder, out_feature_indexes=out_feature_indexes))

    assert config.out_feature_indexes == out_feature_indexes


def test_without_projector() -> None:
    """Without projector the configuration exports the last block only, at the stride of the ViT."""
    config = BaselineConfig(**_config(projector=False))

    assert (config.projector, config.out_feature_indexes, config.projector_scale) == (False, [12], ["P4"])


@pytest.mark.parametrize(
    "changes",
    [{"out_feature_indexes": [3, 6, 9, 12]}, {"projector_scale": ["P3"]}, {"dual_projector": True}],
    ids=["several-blocks", "other-scale", "dual-projector"],
)
def test_without_projector_rejects_what_it_would_ignore(changes: dict[str, Any]) -> None:
    """Linear + LayerNorm reads one block and does not resample it, and a second projector would bypass it."""
    with pytest.raises(ValidationError, match="without projector"):
        BaselineConfig(**_config(projector=False, **changes))


def test_one_exported_block_puts_every_block_in_windows() -> None:
    """Why windows are rejected without projector: RF-DETR puts in windows every block that it does not export."""
    assert compute_window_block_indexes([12]) == list(range(12))  # all the 12 blocks of ViT-S, none global
    assert compute_window_block_indexes([3, 6, 9, 12]) == [0, 1, 2, 4, 5, 7, 8, 10, 11]  # Medium: 3 global blocks


@pytest.mark.parametrize("num_windows", [2, 4])
def test_without_projector_rejects_windows(num_windows: int) -> None:
    """Without projector the ViT has no windows: with them, no block would have global attention."""
    with pytest.raises(ValidationError, match=rf"without projector num_windows must be 1 \(got {num_windows}\)"):
        BaselineConfig(**_config(projector=False, num_windows=num_windows))
