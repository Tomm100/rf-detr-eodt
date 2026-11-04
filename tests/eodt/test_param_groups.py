# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the optimizer parameter groups of EoDT (step 4b of EODT_SPEC.md §9, §8 point 1).

The groups are those RF-DETR's ``get_param_dict`` gives the optimizer in ``RFDETRModelModule.configure_optimizers``.

Covers:
- Every trainable parameter is in exactly one group.
- Layer-wise learning rate and weight decay of every ViT tensor, L2 blocks included; queries and head get ``lr``.
- Changing L2 does not change any learning rate.
- The learning-rate values of the training configuration reach the groups (§3.7).
- Same learning rate and weight decay as the RF-DETR Medium baseline of ``configs/baseline``, tensor by tensor,
  registers included (decision of step 4).
- ViT-L: the decay follows its 24 blocks.

The ViT-L case carries the ``gpu`` marker (slow on CPU): ``-m "not gpu"`` runs ViT-S only.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.models.eodt import EoDTConfig, load_configs
from rfdetr.models.lwdetr import build_model_from_config
from rfdetr.training.param_groups import get_param_dict
from tests.eodt.vit_names import eodt_vit_name

# TrainConfig defaults, the starting recipe of §7.1
LR = 1e-4
LR_ENCODER = 1.5e-4
LAYER_DECAY = 0.8
COMPONENT_DECAY = 0.7
WEIGHT_DECAY = 1e-4

EMBEDDINGS = {"cls_token", "reg_token", "pos_embed", "patch_embed.proj.weight", "patch_embed.proj.bias"}
FINAL_NORM = {"norm.weight", "norm.bias"}
NORMS = {"norm1", "norm2", "norm"}  # module names of the LayerNorms in the timm ViT
HEAD_AND_QUERIES = ("query_feat.", "class_embed.", "bbox_embed.")

_BASELINE_VIT_PREFIX = "backbone.0.encoder.encoder."  # Hugging Face DINOv2 of the baseline, see vit_names.py
# baseline without windows (step 8); windows and projector do not change the groups of the ViT
BASELINE_YAML = Path(__file__).resolve().parents[2] / "configs" / "baseline" / "baseline_medium_nowin.yaml"


def _train_config(**overrides: Any) -> TrainConfig:
    """Return a training configuration with a fixed square input, as EoDT requires."""
    return TrainConfig(dataset_dir=".", multi_scale="off", **overrides)


def _eodt(**overrides: Any) -> tuple[nn.Module, EoDTConfig]:
    """Build a small ViT-S EoDT with L2 = 3 unless overridden."""
    model_config = EoDTConfig(**{"num_l2_blocks": 3, "resolution": 224, "num_queries": 10, **overrides})
    torch.manual_seed(0)
    return build_model_from_config(model_config), model_config


def _groups_by_name(model: nn.Module, model_config: ModelConfig, train_config: TrainConfig) -> dict[str, dict]:
    """Return, for every parameter name, the hyperparameters of its optimizer group."""
    groups = get_param_dict(_namespace_from_configs(model_config, train_config), model)
    names = {id(param): name for name, param in model.named_parameters()}
    return {
        names[id(param)]: {key: value for key, value in group.items() if key != "params"}
        for group in groups
        for param in group["params"]
    }


def _vit_layer_id(vit_name: str, depth: int) -> int:
    """Layer of a ViT parameter: 0 for the embeddings, i + 1 for block i, depth + 2 for the final norm."""
    if vit_name in EMBEDDINGS:
        return 0
    if vit_name in FINAL_NORM:
        return depth + 2
    match = re.fullmatch(r"blocks\.(\d+)\..+", vit_name)
    assert match, f"unexpected ViT parameter {vit_name}"
    return int(match.group(1)) + 1


def test_every_parameter_in_one_group() -> None:
    """No parameter is in two groups and the groups hold exactly the trainable parameters."""
    model, model_config = _eodt()

    groups = get_param_dict(_namespace_from_configs(model_config, _train_config()), model)

    grouped = [id(param) for group in groups for param in group["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(param) for param in model.parameters() if param.requires_grad}


def test_lr_and_weight_decay_per_tensor() -> None:
    """Every ViT tensor gets lr_encoder * 0.8^(depth + 2 - layer_id) * 0.7^2; queries and head get lr."""
    model, model_config = _eodt()
    by_name = _groups_by_name(model, model_config, _train_config())
    depth = len(model.backbone.vit.blocks)

    for name, group in by_name.items():
        if name.startswith(HEAD_AND_QUERIES):
            assert group == {"lr": LR}, name  # no weight_decay of its own: the optimizer's one applies
            continue
        vit_name = name.removeprefix("backbone.vit.")
        layer_id = _vit_layer_id(vit_name, depth)
        expected_lr = LR_ENCODER * LAYER_DECAY ** (depth + 2 - layer_id) * COMPONENT_DECAY**2
        no_decay = vit_name in EMBEDDINGS or vit_name.endswith(".bias") or vit_name.split(".")[-2] in NORMS
        assert group["lr"] == pytest.approx(expected_lr, rel=1e-12), name
        assert group["weight_decay"] == (0.0 if no_decay else WEIGHT_DECAY), name


def test_lr_and_weight_decay_examples() -> None:
    """A few tensors written out: embeddings, first and last block, layer scale, final norm, head."""
    model, model_config = _eodt()
    by_name = _groups_by_name(model, model_config, _train_config())

    def vit(name: str) -> dict:
        return by_name[f"backbone.vit.{name}"]

    for name in EMBEDDINGS:
        assert vit(name) == {"lr": pytest.approx(1.5e-4 * 0.8**14 * 0.49), "weight_decay": 0.0}, name
    assert vit("blocks.0.attn.qkv.weight") == {"lr": pytest.approx(1.5e-4 * 0.8**13 * 0.49), "weight_decay": 1e-4}
    assert vit("blocks.0.attn.qkv.bias")["weight_decay"] == 0.0
    assert vit("blocks.0.norm1.weight")["weight_decay"] == 0.0
    assert vit("blocks.11.ls1.gamma") == {"lr": pytest.approx(1.5e-4 * 0.8**2 * 0.49), "weight_decay": 1e-4}
    assert vit("norm.weight") == {"lr": pytest.approx(7.35e-5), "weight_decay": 0.0}
    for name in ("query_feat.weight", "class_embed.weight", "class_embed.bias", "bbox_embed.layers.2.weight"):
        assert by_name[name] == {"lr": 1e-4}, name


def test_l2_does_not_change_lr() -> None:
    """The L2 blocks follow the same decay as the L1 blocks: L2 = 3 and L2 = 4 give the same groups."""
    l2_3, config_3 = _eodt(num_l2_blocks=3)
    l2_4, config_4 = _eodt(num_l2_blocks=4)

    assert _groups_by_name(l2_3, config_3, _train_config()) == _groups_by_name(l2_4, config_4, _train_config())


def test_train_config_reaches_groups() -> None:
    """Learning rates, decays and weight decay of the training configuration reach the groups."""
    model, model_config = _eodt()
    train_config = _train_config(
        lr=2e-4, lr_encoder=3e-4, lr_vit_layer_decay=0.9, lr_component_decay=0.5, weight_decay=0.05
    )

    by_name = _groups_by_name(model, model_config, train_config)

    assert by_name["query_feat.weight"] == {"lr": 2e-4}
    assert by_name["backbone.vit.blocks.11.attn.qkv.weight"] == {
        "lr": pytest.approx(3e-4 * 0.9**2 * 0.25),
        "weight_decay": 0.05,
    }
    assert by_name["backbone.vit.pos_embed"] == {"lr": pytest.approx(3e-4 * 0.9**14 * 0.25), "weight_decay": 0.0}


def test_same_as_baseline() -> None:
    """Every ViT tensor of the RF-DETR Medium baseline has exactly the lr and weight decay of the same EoDT tensor.

    The baseline is the one of ``configs/baseline``, with its registers; ``mask_token`` has no counterpart in timm.
    """
    train_config = _train_config()
    eodt, eodt_config = _eodt()
    baseline_config, _ = load_configs(BASELINE_YAML)
    torch.manual_seed(0)
    baseline = build_model_from_config(baseline_config, train_config)
    eodt_groups = _groups_by_name(eodt, eodt_config, train_config)
    baseline_groups = _groups_by_name(baseline, baseline_config, train_config)

    compared = set()
    for baseline_name, baseline_group in baseline_groups.items():
        if not baseline_name.startswith("backbone.0.encoder.") or baseline_name.endswith("mask_token"):
            continue  # the projector, the decoder and the heads are not part of the ViT
        assert baseline_name.startswith(_BASELINE_VIT_PREFIX), baseline_name
        eodt_name = f"backbone.vit.{eodt_vit_name(baseline_name.removeprefix(_BASELINE_VIT_PREFIX))}"
        assert eodt_groups[eodt_name] == baseline_group, f"{baseline_name} -> {eodt_name}"
        compared.add(eodt_name)

    vit_names = {name for name in eodt_groups if name.startswith("backbone.vit.")}
    assert compared == vit_names  # registers included


@pytest.mark.gpu
def test_vitl_depth() -> None:
    """ViT-L has 24 blocks, so the decay spans 26 layers: embeddings 0.8^26, block 0 0.8^25, last block 0.8^2."""
    model, model_config = _eodt(backbone_name="vit_large_patch14_reg4_dinov2", num_l2_blocks=4)
    by_name = _groups_by_name(model, model_config, _train_config())

    def lr(name: str) -> float:
        return by_name[f"backbone.vit.{name}"]["lr"]

    assert lr("pos_embed") == pytest.approx(1.5e-4 * 0.8**26 * 0.49)
    assert lr("blocks.0.attn.qkv.weight") == pytest.approx(1.5e-4 * 0.8**25 * 0.49)
    assert lr("blocks.23.attn.qkv.weight") == pytest.approx(1.5e-4 * 0.8**2 * 0.49)
    assert lr("norm.weight") == pytest.approx(1.5e-4 * 0.49)
