# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for rfdetr.models.eodt.backbone.EoDTBackbone (step 1 of EODT_SPEC.md §9).

Covers:
- Pretrained loading at patch 16: no missing or unexpected keys, patch embedding and
  positional embedding resampled by timm, every other tensor unchanged.
- At patch 14 the hand-written forward matches timm's ``forward_features`` exactly.
- Patch grid is ``R / 16 x R / 16``; registers add 4 prefix tokens.
- Forward and backward on ViT-S, ViT-B and ViT-L (§3.6).

ViT-B and ViT-L cases carry the ``gpu`` marker (slow on CPU): ``-m "not gpu"`` runs ViT-S only.
"""

import pytest
import timm
import torch
from timm.layers import resample_abs_pos_embed, resample_patch_embed
from torch import nn

from rfdetr.models.eodt import EoDTBackbone

BACKBONES = [
    pytest.param("vit_small_patch14_reg4_dinov2", id="vits"),
    pytest.param("vit_base_patch14_reg4_dinov2", id="vitb", marks=pytest.mark.gpu),
    pytest.param("vit_large_patch14_reg4_dinov2", id="vitl", marks=pytest.mark.gpu),
]
VITS = "vit_small_patch14_reg4_dinov2"
# Native DINOv2 setting of the pretrained checkpoints: 518 px at patch 14 (37 x 37 grid).
NATIVE_RESOLUTION = 518
NATIVE_PATCH_SIZE = 14
# Irrelevant here: these tests run all blocks without queries.
NUM_L2_BLOCKS = 3


@pytest.fixture(scope="module")
def timm_original():
    """Return a loader of the untouched pretrained timm models, cached per name."""
    cache: dict[str, nn.Module] = {}

    def load(name: str) -> nn.Module:
        if name not in cache:
            cache[name] = timm.create_model(name, pretrained=True, num_classes=0).eval()
        return cache[name]

    return load


@pytest.mark.parametrize("name", BACKBONES)
def test_load_keys_and_resampling(name: str, timm_original) -> None:
    """Patch-16 weights equal the original except for the two embeddings, which timm resampled.

    Missing or unexpected keys would already make timm's strict load fail inside ``create_model``;
    the tensor comparison below shows that every other weight was loaded unchanged.
    """
    resolution, patch_size = 576, 16
    grid = resolution // patch_size
    original = timm_original(name).state_dict()
    loaded = EoDTBackbone(name, resolution, patch_size, NUM_L2_BLOCKS, pretrained=True).vit.state_dict()

    assert set(loaded) == set(original)

    resampled = {"patch_embed.proj.weight", "pos_embed"}
    for key in set(loaded) - resampled:
        assert torch.equal(loaded[key], original[key]), key

    embed_dim = original["pos_embed"].shape[-1]
    assert loaded["patch_embed.proj.weight"].shape == (embed_dim, 3, patch_size, patch_size)
    assert torch.equal(
        loaded["patch_embed.proj.weight"],
        resample_patch_embed(original["patch_embed.proj.weight"], [patch_size, patch_size]),
    )
    # CLS and register tokens have no positional embedding (no_embed_class=True).
    assert loaded["pos_embed"].shape == (1, grid * grid, embed_dim)
    assert torch.equal(
        loaded["pos_embed"],
        resample_abs_pos_embed(original["pos_embed"], [grid, grid], num_prefix_tokens=0),
    )


@pytest.mark.parametrize("name", BACKBONES)
def test_patch14_identical_to_timm(name: str, timm_original) -> None:
    """At the native patch 14 and resolution, the output equals timm's forward_features."""
    backbone = EoDTBackbone(name, NATIVE_RESOLUTION, NATIVE_PATCH_SIZE, NUM_L2_BLOCKS, pretrained=True).eval()
    x = torch.randn(1, 3, NATIVE_RESOLUTION, NATIVE_RESOLUTION, generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        ours = backbone(x)
        reference = timm_original(name).forward_features(x)

    torch.testing.assert_close(ours, reference, rtol=0, atol=0)


@pytest.mark.parametrize("resolution", [576, 640])
def test_grid_shape(resolution: int) -> None:
    """The patch grid is R/16 x R/16 and the output has prefix + grid tokens."""
    backbone = EoDTBackbone(VITS, resolution, 16, NUM_L2_BLOCKS, pretrained=False).eval()
    grid = resolution // 16

    with torch.no_grad():
        out = backbone(torch.randn(2, 3, resolution, resolution))

    assert backbone.grid_size == (grid, grid)
    assert out.shape == (2, backbone.num_prefix_tokens + grid * grid, backbone.embed_dim)


def test_registers() -> None:
    """The reg4 backbone has CLS + 4 registers, i.e. 4 tokens more than plain DINOv2."""
    resolution = 224
    with_registers = EoDTBackbone(VITS, resolution, 16, NUM_L2_BLOCKS, pretrained=False).eval()
    without_registers = EoDTBackbone("vit_small_patch14_dinov2", resolution, 16, NUM_L2_BLOCKS, pretrained=False).eval()
    x = torch.randn(1, 3, resolution, resolution)

    with torch.no_grad():
        tokens_with = with_registers(x).shape[1]
        tokens_without = without_registers(x).shape[1]

    assert with_registers.num_prefix_tokens == 5
    assert with_registers.vit.reg_token.shape == (1, 4, with_registers.embed_dim)
    assert tokens_with - tokens_without == 4


@pytest.mark.parametrize("name", BACKBONES)
def test_forward_backward(name: str) -> None:
    """Forward and backward give a finite loss and a finite gradient on every parameter."""
    resolution = 224  # small grid keeps ViT-L backward within CPU memory
    backbone = EoDTBackbone(name, resolution, 16, NUM_L2_BLOCKS, pretrained=True).train()

    loss = backbone(torch.randn(2, 3, resolution, resolution)).mean()
    loss.backward()

    assert torch.isfinite(loss)
    for param_name, param in backbone.named_parameters():
        assert param.grad is not None, param_name
        assert torch.isfinite(param.grad).all(), param_name


def test_resolution_not_divisible() -> None:
    """A resolution that is not a multiple of the patch size is rejected."""
    with pytest.raises(ValueError, match="divisible"):
        EoDTBackbone(VITS, 500, 16, NUM_L2_BLOCKS, pretrained=False)
