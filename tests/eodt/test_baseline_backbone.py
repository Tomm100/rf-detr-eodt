# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the DINOv2 backbone of the baseline (step 8a of EODT_SPEC.md §9, §8 point 4).

Covers:
- RF-DETR's ``DinoV2`` with registers builds with several exported stages, which raised ``ValueError`` before the
  fix in ``dinov2.py``.
- The pretrained Hugging Face checkpoint is loaded at patch 16 with no missing or unexpected key: every tensor but
  the two resampled embeddings equals the checkpoint, read independently of the loader, and differs from the random
  initialization that RF-DETR alone would leave.
- The conversion 14 -> 16 gives exactly the weights that timm gives EoDT, tensor by tensor, and the baseline ViT
  without windows computes exactly the features of EoDT's ViT.
- Windows and exported stages do not change the loaded weights.
- A checkpoint that does not match the encoder is rejected.
- The upper bound of Table 7 (§9 step 11 (3), §10), built from its YAML files: DINOv2-B without registers loads at
  patch 20 (2XL) and 12 (Max) with exactly the keys of ``facebook/dinov2-base``, gives exactly the weights that timm
  gives its DINOv2-B without registers at the same patch and resolution, and has a 44 x 44 and a 69 x 69 grid.

ViT-B and ViT-L cases carry the ``gpu`` marker (slow on CPU, large downloads): ``-m "not gpu"`` runs ViT-S only.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import TrainConfig
from rfdetr.models.backbone.dinov2 import DinoV2, size_to_width
from rfdetr.models.backbone.dinov2_with_windowed_attn import WindowedDinov2WithRegistersBackbone
from rfdetr.models.eodt import BaselineConfig, EoDTBackbone, load_configs, load_pretrained_dinov2
from rfdetr.models.lwdetr import build_model, build_model_from_config
from tests.eodt.vit_names import eodt_vit_name

# DINOv2 size, timm name of EoDT's backbone, depth
SIZES = [
    pytest.param("small", "vit_small_patch14_reg4_dinov2", 12, id="vits"),
    pytest.param("base", "vit_base_patch14_reg4_dinov2", 12, id="vitb", marks=pytest.mark.gpu),
    pytest.param("large", "vit_large_patch14_reg4_dinov2", 24, id="vitl", marks=pytest.mark.gpu),
]
RESOLUTION = 576  # the baseline's: 36 x 36 patches
PATCH_SIZE = 16
GRID = RESOLUTION // PATCH_SIZE
PATCH_EMBED = "embeddings.patch_embeddings.projection.weight"
POS_EMBED = "embeddings.position_embeddings"
MASK_TOKEN = "embeddings.mask_token"
# upper bound of Table 7: YAML file of the run and its patch grid; DINOv2-B without registers in timm
BASELINE_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "baseline"
UPPER_BOUNDS = [
    pytest.param("rfdetr_table7_2xl_upperbound.yaml", 44, id="2xl", marks=pytest.mark.gpu),  # 880 px, patch 20
    pytest.param("rfdetr_table7_max_upperbound.yaml", 69, id="max", marks=pytest.mark.gpu),  # 828 px, patch 12
]
TIMM_VITB_NO_REGISTERS = "vit_base_patch14_dinov2.lvd142m"


def _config(size: str, depth: int, **overrides: Any) -> BaselineConfig:
    """Return the Medium baseline without windows on the DINOv2 of the given size and depth."""
    return BaselineConfig(
        **{
            "encoder": f"dinov2_registers_windowed_{size}",
            "out_feature_indexes": [depth],
            "resolution": RESOLUTION,
            "patch_size": PATCH_SIZE,
            "positional_encoding_size": GRID,
            "num_windows": 1,
            "projector": True,
            "projector_scale": ["P4"],
            "hidden_dim": 256,
            "dec_layers": 4,
            "sa_nheads": 8,
            "ca_nheads": 16,
            "dec_n_points": 2,
            "device": "cpu",
            **overrides,
        }
    )


@pytest.fixture(scope="module")
def baseline_encoder() -> Callable[[str, int], nn.Module]:
    """Return a builder of the baseline's DINOv2 encoder through ``build_model_from_config``, cached per size."""
    cache: dict[str, nn.Module] = {}

    def build(size: str, depth: int) -> nn.Module:
        if size not in cache:
            torch.manual_seed(0)
            cache[size] = build_model_from_config(_config(size, depth)).backbone[0].encoder.encoder.eval()
        return cache[size]

    return build


@pytest.fixture(scope="module")
def upperbound_encoder() -> Callable[[str], nn.Module]:
    """Return a builder of the DINOv2 encoder of an upper bound from its YAML file, on CPU, cached per file."""
    cache: dict[str, nn.Module] = {}

    def build(name: str) -> nn.Module:
        if name not in cache:
            torch.manual_seed(0)
            config = load_configs(BASELINE_CONFIGS / name)[0].model_copy(update={"device": "cpu"})
            cache[name] = build_model_from_config(config).backbone[0].encoder.encoder.eval()
        return cache[name]

    return build


@pytest.fixture(scope="module")
def eodt_backbone() -> Callable[[str], EoDTBackbone]:
    """Return a builder of EoDT's pretrained timm backbone at the baseline's resolution, cached per name."""
    cache: dict[str, EoDTBackbone] = {}

    def build(name: str) -> EoDTBackbone:
        if name not in cache:
            cache[name] = EoDTBackbone(name, RESOLUTION, PATCH_SIZE, num_l2_blocks=3, pretrained=True).eval()
        return cache[name]

    return build


@pytest.mark.parametrize(
    ("size", "out_feature_indexes"),
    [
        pytest.param("small", [3, 6, 9, 12], id="vits"),
        pytest.param("base", [3, 6, 9, 12], id="vitb", marks=pytest.mark.gpu),
        pytest.param("large", [6, 12, 18, 24], id="vitl", marks=pytest.mark.gpu),
    ],
)
def test_registers_with_several_stages(size: str, out_feature_indexes: list[int]) -> None:
    """DinoV2 with registers builds with four exported stages and returns four maps (§8 point 4).

    Before the fix ``out_indices: [12]`` of the JSON configuration disagreed with the four ``out_features``. At
    patch 16 RF-DETR does not load the weights, so nothing is downloaded.
    """
    dino = DinoV2(
        shape=(224, 224),
        out_feature_indexes=out_feature_indexes,
        size=size,
        use_registers=True,
        patch_size=16,
        num_windows=2,
        positional_encoding_size=14,
    ).eval()

    with torch.no_grad():
        feature_maps = dino(torch.randn(1, 3, 224, 224))

    assert dino.encoder.config.out_features == [f"stage{i}" for i in out_feature_indexes]
    assert dino.encoder.config.out_indices == out_feature_indexes
    assert dino.encoder.config.num_register_tokens == 4
    assert [tuple(feature_map.shape) for feature_map in feature_maps] == [(1, size_to_width[size], 14, 14)] * 4


@pytest.mark.parametrize(
    ("size", "depth"),
    [
        pytest.param("small", 12, id="vits"),
        pytest.param("base", 12, id="vitb", marks=pytest.mark.gpu),
        pytest.param("large", 24, id="vitl", marks=pytest.mark.gpu),
    ],
)
def test_load_matches_checkpoint(size: str, depth: int, baseline_encoder) -> None:
    """The encoder has exactly the keys of the checkpoint, and every tensor but the two resampled ones equals it.

    The checkpoint is read from its safetensors file, independently of the loader.
    """
    checkpoint = load_file(hf_hub_download(f"facebook/dinov2-with-registers-{size}", "model.safetensors"))
    loaded = baseline_encoder(size, depth).state_dict()

    assert set(loaded) == set(checkpoint)
    for key in set(loaded) - {PATCH_EMBED, POS_EMBED}:
        assert torch.equal(loaded[key], checkpoint[key]), key
    width = size_to_width[size]
    assert loaded[PATCH_EMBED].shape == (width, 3, PATCH_SIZE, PATCH_SIZE)
    assert loaded[POS_EMBED].shape == (1, 1 + GRID * GRID, width)
    assert torch.equal(loaded[POS_EMBED][:, :1], checkpoint[POS_EMBED][:, :1])  # the position of CLS is kept


def test_loaded_weights_are_not_random(baseline_encoder) -> None:
    """Every loaded tensor differs from the random initialization that RF-DETR alone leaves at patch 16.

    The exception is ``mask_token``, zero both in the checkpoint and at initialization (the detector does not use it).
    """
    torch.manual_seed(0)
    config = _config("small", 12)
    rfdetr_only = build_model(_namespace_from_configs(config, TrainConfig(dataset_dir=".", output_dir=".")))
    initialized = rfdetr_only.backbone[0].encoder.encoder.state_dict()
    loaded = baseline_encoder("small", 12).state_dict()

    assert set(loaded) == set(initialized)
    for key in set(loaded) - {MASK_TOKEN}:
        assert not torch.equal(loaded[key], initialized[key]), key
    assert not loaded[MASK_TOKEN].any() and not initialized[MASK_TOKEN].any()


@pytest.mark.parametrize(("size", "timm_name", "depth"), SIZES)
def test_same_weights_as_eodt(size: str, timm_name: str, depth: int, baseline_encoder, eodt_backbone) -> None:
    """Every DINOv2 tensor of the baseline equals the one that timm loads into EoDT at patch 16.

    timm adds the position of CLS to the CLS token and keeps no position for CLS and registers; it fuses query, key
    and value into one ``qkv``. ``mask_token`` has no counterpart.
    """
    encoder = baseline_encoder(size, depth)
    baseline = encoder.state_dict()
    eodt = eodt_backbone(timm_name).vit.state_dict()

    compared = set()
    for name, tensor in baseline.items():
        if name == MASK_TOKEN:
            continue
        eodt_name = eodt_vit_name(name)
        expected = eodt[eodt_name]
        if name == "embeddings.cls_token":
            tensor = tensor + baseline[POS_EMBED][:, :1]
        elif name == POS_EMBED:
            tensor = tensor[:, 1:]
        elif match := re.search(r"\.(query|key|value)\.", name):
            third = expected.shape[0] // 3
            index = ("query", "key", "value").index(match.group(1))
            expected = expected[index * third : (index + 1) * third]
        assert torch.equal(tensor, expected), f"{name} -> {eodt_name}"
        compared.add(eodt_name)

    assert compared == set(eodt)


@pytest.mark.parametrize(("size", "timm_name", "depth"), SIZES)
def test_same_features_as_eodt(size: str, timm_name: str, depth: int, baseline_encoder, eodt_backbone) -> None:
    """On the same image, the last block of the baseline without windows gives the features of EoDT's ViT.

    RF-DETR exports the patch tokens after DINOv2's final LayerNorm, as ``EoDTBackbone.forward`` returns them. The
    tolerance is fp32's, because Hugging Face computes query, key and value with three products and timm with one; on
    the PC of the thesis the two are equal bit for bit.
    """
    encoder = baseline_encoder(size, depth)
    backbone = eodt_backbone(timm_name)
    x = torch.randn(1, 3, RESOLUTION, RESOLUTION, generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        baseline = encoder(x)[0][-1]  # [1, C, H, W]
        eodt = backbone(x)[:, backbone.num_prefix_tokens :]  # [1, H * W, C]

    assert baseline.shape == (1, size_to_width[size], GRID, GRID)
    torch.testing.assert_close(baseline.flatten(2).transpose(1, 2), eodt)  # default fp32 tolerances


def test_windows_do_not_change_weights(baseline_encoder) -> None:
    """The Medium with windows and four exported stages loads exactly the same DINOv2 weights."""
    torch.manual_seed(0)
    config = _config("small", 12, num_windows=2, out_feature_indexes=[3, 6, 9, 12])
    with_windows = build_model_from_config(config).backbone[0].encoder.encoder.state_dict()
    without_windows = baseline_encoder("small", 12).state_dict()

    assert set(with_windows) == set(without_windows)
    for key, value in without_windows.items():
        assert torch.equal(with_windows[key], value), key


@pytest.mark.parametrize(("name", "grid"), UPPER_BOUNDS)
def test_upperbound_loads_checkpoint_without_registers(name: str, grid: int, upperbound_encoder) -> None:
    """DINOv2-B without registers at patch 20 and 12 has exactly the keys of ``facebook/dinov2-base``.

    Every tensor but the two resampled ones equals the checkpoint, read from its safetensors file independently of the
    loader; the patch embedding has the patch size of the YAML and the positional embedding its grid.
    """
    config = load_configs(BASELINE_CONFIGS / name)[0]
    checkpoint = load_file(hf_hub_download("facebook/dinov2-base", "model.safetensors"))
    encoder = upperbound_encoder(name)
    loaded = encoder.state_dict()

    assert encoder.config.num_register_tokens == 0
    assert set(loaded) == set(checkpoint)
    for key in set(loaded) - {PATCH_EMBED, POS_EMBED}:
        assert torch.equal(loaded[key], checkpoint[key]), key
    assert loaded[PATCH_EMBED].shape == (768, 3, config.patch_size, config.patch_size)
    assert loaded[POS_EMBED].shape == (1, 1 + grid * grid, 768)
    assert torch.equal(loaded[POS_EMBED][:, :1], checkpoint[POS_EMBED][:, :1])  # the position of CLS is kept


@pytest.mark.parametrize(("name", "grid"), UPPER_BOUNDS)
def test_upperbound_same_weights_as_timm(name: str, grid: int, upperbound_encoder) -> None:
    """Every DINOv2 tensor of the upper bound equals the one timm loads at the same patch size and resolution.

    timm resamples DINOv2-B without registers with its own ``checkpoint_filter_fn``. Unlike the models with registers,
    its ``pos_embed`` keeps the position of CLS, as Hugging Face's: CLS token and positions compare as they are; query,
    key and value are the thirds of ``qkv``, and ``mask_token`` has no counterpart.
    """
    config = load_configs(BASELINE_CONFIGS / name)[0]
    vit = timm.create_model(
        TIMM_VITB_NO_REGISTERS,
        pretrained=True,
        patch_size=config.patch_size,
        img_size=config.resolution,
        num_classes=0,
    )
    expected_state = vit.state_dict()

    compared = set()
    for key, tensor in upperbound_encoder(name).state_dict().items():
        if key == MASK_TOKEN:
            continue
        timm_name = eodt_vit_name(key)
        expected = expected_state[timm_name]
        if match := re.search(r"\.(query|key|value)\.", key):
            third = expected.shape[0] // 3
            index = ("query", "key", "value").index(match.group(1))
            expected = expected[index * third : (index + 1) * third]
        assert torch.equal(tensor, expected), f"{key} -> {timm_name}"
        compared.add(timm_name)

    assert compared == set(expected_state)
    assert vit.patch_embed.grid_size == (grid, grid)


@pytest.mark.parametrize(("name", "grid"), UPPER_BOUNDS)
def test_upperbound_patch_grid(name: str, grid: int, upperbound_encoder) -> None:
    """At the resolution of its YAML the encoder returns the four exported maps on a 44 x 44 or 69 x 69 grid."""
    resolution = load_configs(BASELINE_CONFIGS / name)[0].resolution
    x = torch.randn(1, 3, resolution, resolution, generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        feature_maps = upperbound_encoder(name)(x)[0]

    assert [tuple(feature_map.shape) for feature_map in feature_maps] == [(1, 768, grid, grid)] * 4


@pytest.mark.parametrize("problem", ["missing_keys", "unexpected_keys", "mismatched_keys"])
def test_rejects_checkpoint_that_does_not_match(problem: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint with a missing, unexpected or reshaped key is an error, not a silently random weight."""
    info: dict[str, Any] = {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": []}
    info[problem] = {"embeddings.cls_token"}
    monkeypatch.setattr(
        WindowedDinov2WithRegistersBackbone, "from_pretrained", classmethod(lambda cls, *args, **kwargs: (None, info))
    )
    encoder = DinoV2(
        shape=(224, 224),
        out_feature_indexes=[12],
        size="small",
        use_registers=True,
        patch_size=16,
        num_windows=1,
        positional_encoding_size=14,
    ).encoder

    with pytest.raises(RuntimeError, match=problem):
        load_pretrained_dinov2(encoder, "small")
