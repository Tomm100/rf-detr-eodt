# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for building EoDT from its configuration (step 4a of EODT_SPEC.md §9).

Covers:
- RF-DETR's ``build_model_from_config`` returns EoDT for an ``EoDTConfig`` (§8 point 2).
- ``num_classes + 1`` class logits: 91 for COCO (§8 point 7).
- Resolution, L2, number of queries and backbone name set in the configuration reach the model (§3.7).
- ``NestedTensor`` input, as given by RF-DETR's dataloader, with the mask ignored (§8 point 3).
- A training configuration with variable or padded input is rejected.
- Build and forward on ViT-S, ViT-B and ViT-L from the configuration only (§3.6); the backbone name
  reaches the model.

ViT-B and ViT-L cases carry the ``gpu`` marker (slow on CPU): ``-m "not gpu"`` runs ViT-S only.
"""

from typing import Any

import pytest
import timm
import torch
from torch import Tensor

from rfdetr.config import TrainConfig
from rfdetr.models.eodt import EoDT, EoDTConfig
from rfdetr.models.lwdetr import LWDETR, build_model_from_config
from rfdetr.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list

RESOLUTION = 224  # 14 x 14 grid keeps the CPU tests fast
NUM_QUERIES = 10


def _images(batch_size: int, resolution: int = RESOLUTION) -> Tensor:
    """Return a fixed batch of random images."""
    return torch.randn(batch_size, 3, resolution, resolution, generator=torch.Generator().manual_seed(0))


def _config(**overrides: Any) -> EoDTConfig:
    """Return a small ViT-S configuration, with L2 = 3 unless overridden."""
    return EoDTConfig(
        **{
            "num_l2_blocks": 3,
            "resolution": RESOLUTION,
            "num_queries": NUM_QUERIES,
            "query_mode": "learned",
            **overrides,
        }
    )


def _build(config: EoDTConfig, train_config: TrainConfig | None = None) -> EoDT:
    """Build the model through RF-DETR's entry point, with a fixed seed."""
    torch.manual_seed(0)
    return build_model_from_config(config, train_config)


def test_build_model_from_config_returns_eodt() -> None:
    """An EoDTConfig builds EoDT, not LWDETR."""
    model = _build(_config())

    assert isinstance(model, EoDT)
    assert not isinstance(model, LWDETR)


@pytest.mark.parametrize(("num_classes", "num_logits"), [(90, 91), (2, 3)])
def test_num_classes_plus_one(num_classes: int, num_logits: int) -> None:
    """As in build_model, the class head has num_classes + 1 logits: 91 for COCO."""
    model = _build(_config(num_classes=num_classes)).eval()

    with torch.no_grad():
        out = model(_images(2))

    assert model.class_embed.out_features == num_logits
    assert out["pred_logits"].shape == (2, NUM_QUERIES, num_logits)


def test_resolution_reaches_model() -> None:
    """With resolution 640 the patch grid becomes 40 x 40."""
    model = _build(_config(resolution=640))

    assert model.backbone.grid_size == (40, 40)
    assert model.backbone.vit.pos_embed.shape[1] == 40 * 40


@pytest.mark.parametrize("num_l2_blocks", [3, 4])
def test_num_l2_blocks_reaches_model(num_l2_blocks: int) -> None:
    """With L2 = 4 the queries enter one block earlier than with L2 = 3, and there is one more aux output."""
    model = _build(_config(num_l2_blocks=num_l2_blocks)).eval()
    num_tokens = []
    hooks = [
        block.register_forward_pre_hook(lambda module, args: num_tokens.append(args[0].shape[1]))
        for block in model.backbone.vit.blocks
    ]

    with torch.no_grad():
        try:
            out = model(_images(2))
        finally:
            for hook in hooks:
                hook.remove()

    depth = len(model.backbone.vit.blocks)
    vit_tokens = model.backbone.num_prefix_tokens + RESOLUTION // 16 * (RESOLUTION // 16)
    first_l2_block = depth - num_l2_blocks
    assert model.backbone.num_l1_blocks == first_l2_block
    assert num_tokens == [vit_tokens] * first_l2_block + [vit_tokens + NUM_QUERIES] * num_l2_blocks
    assert len(out["aux_outputs"]) == num_l2_blocks - 1


def test_num_queries_reaches_model() -> None:
    """The number of queries sets the learned embeddings and the number of predictions."""
    model = _build(_config(num_queries=100)).eval()

    with torch.no_grad():
        out = model(_images(2))

    assert model.query_feat.num_embeddings == 100
    assert out["pred_boxes"].shape == (2, 100, 4)


def test_backbone_name_reaches_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backbone name in the configuration is the timm model that is created.

    ``timm.create_model`` records the name it receives and then creates ViT-S, so no other weights are needed.
    """
    create_model = timm.create_model
    names = []

    def record_name(name: str, **kwargs: Any) -> torch.nn.Module:
        names.append(name)
        return create_model("vit_small_patch14_reg4_dinov2", **kwargs)

    monkeypatch.setattr(timm, "create_model", record_name)
    _build(_config(backbone_name="vit_large_patch14_reg4_dinov2"))

    assert names == ["vit_large_patch14_reg4_dinov2"]


def test_nested_tensor_input() -> None:
    """A NestedTensor gives exactly the output of the plain tensor, whatever its mask; targets are ignored."""
    model = _build(_config()).eval()
    x = _images(2)
    targets = [{"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}] * 2
    half_padded = torch.zeros(2, RESOLUTION, RESOLUTION, dtype=torch.bool)
    half_padded[:, :, RESOLUTION // 2 :] = True  # a model that used the mask would give a different output

    with torch.no_grad():
        expected = model(x)
        from_list = model(nested_tensor_from_tensor_list(list(x)), targets)  # RF-DETR's collate
        with_mask = model(NestedTensor(x, half_padded))

    for out in (from_list, with_mask):
        torch.testing.assert_close(out["pred_logits"], expected["pred_logits"], rtol=0, atol=0)
        torch.testing.assert_close(out["pred_boxes"], expected["pred_boxes"], rtol=0, atol=0)
        for aux, expected_aux in zip(out["aux_outputs"], expected["aux_outputs"]):
            torch.testing.assert_close(aux["pred_logits"], expected_aux["pred_logits"], rtol=0, atol=0)
            torch.testing.assert_close(aux["pred_boxes"], expected_aux["pred_boxes"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "train_kwargs",
    [
        {"multi_scale": "per-batch"},
        {"multi_scale": "per-sample"},
        {"multi_scale": "off", "square_resize_div_64": False},
    ],
    ids=["per-batch", "per-sample", "no-square-resize"],
)
def test_rejects_variable_input(train_kwargs: dict[str, Any]) -> None:
    """Multi-scale or non-square resize would change or pad the input, which EoDT cannot take."""
    train_config = TrainConfig(dataset_dir=".", **train_kwargs)

    with pytest.raises(ValueError, match="fixed square input"):
        _build(_config(), train_config)


def test_accepts_fixed_square_input() -> None:
    """With multi-scale off and the square resize, the model is built."""
    train_config = TrainConfig(dataset_dir=".", multi_scale="off", square_resize_div_64=True)

    assert isinstance(_build(_config(), train_config), EoDT)


@pytest.mark.parametrize(
    ("backbone_name", "num_l2_blocks", "depth", "embed_dim"),
    [
        pytest.param("vit_small_patch14_reg4_dinov2", 3, 12, 384, id="vits"),
        pytest.param("vit_base_patch14_reg4_dinov2", 3, 12, 768, id="vitb", marks=pytest.mark.gpu),
        pytest.param("vit_large_patch14_reg4_dinov2", 4, 24, 1024, id="vitl", marks=pytest.mark.gpu),
    ],
)
def test_build_and_forward_sizes(backbone_name: str, num_l2_blocks: int, depth: int, embed_dim: int) -> None:
    """Each backbone size is built from the configuration only, with its L2 as in EoMT (§3.2, §3.6)."""
    model = _build(_config(backbone_name=backbone_name, num_l2_blocks=num_l2_blocks)).eval()

    with torch.no_grad():
        out = model(_images(2))

    vit = model.backbone.vit
    assert vit.pretrained_cfg["architecture"] == backbone_name
    assert (len(vit.blocks), vit.embed_dim) == (depth, embed_dim)
    assert model.backbone.num_l1_blocks == depth - num_l2_blocks
    assert out["pred_logits"].shape == (2, NUM_QUERIES, 91)
    assert out["pred_boxes"].shape == (2, NUM_QUERIES, 4)
    assert len(out["aux_outputs"]) == num_l2_blocks - 1
