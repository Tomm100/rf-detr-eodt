# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the baseline built from its YAML files by RF-DETR's training pipeline (step 8b of EODT_SPEC.md §9).

Every model is built by ``RFDETRModelModule`` from a local copy of a file of ``configs/baseline`` written by the
``baseline_yaml`` fixture.

Covers:
- The three rings of the chain: RF-DETR (``LWDETR``) with or without windows, with RF-DETR's projector or with
  Linear + LayerNorm, the decoder of RF-DETR Medium and its criterion (§5).
- Every architecture value of the YAML reaches the model (§3.6, §3.7): layers, hidden dimension, heads and points of
  the decoder, queries, windows, projector, resolution and exported blocks; the DINOv2 size in the ViT-B and ViT-L
  cases.
- Without projector the decoder reads exactly LayerNorm(Linear(last ViT block)); Linear and LayerNorm get ``lr``, as
  RF-DETR's projector.
- Forward and backward with the loss on 2 coco_mini images for the three rings: finite loss and gradient, non-zero on
  every parameter but a listed few.

ViT-B and ViT-L cases carry the ``gpu`` marker (slow on CPU): ``-m "not gpu"`` runs ViT-S only.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.eodt import LinearProjector, load_configs
from rfdetr.models.lwdetr import LWDETR
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.param_groups import get_param_dict
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

SMALL = {"resolution": 224, "positional_encoding_size": 14}  # 14 x 14 grid keeps the CPU tests fast
BASE_WEIGHTS = {"loss_ce", "loss_bbox", "loss_giou"}
FOUR_BLOCKS = ["stage3", "stage6", "stage9", "stage12"]
# Parameters without a gradient at the first step, whatever the batch:
# - mask_token of DINOv2 is not used by the detector;
# - the first two layers of the box MLPs get a zero gradient because RF-DETR initializes their last layer to zero, so
#   that every query starts from its proposal.
NO_GRADIENT = {"backbone.0.encoder.encoder.embeddings.mask_token"}
ZERO_GRADIENT = {
    f"{mlp}.layers.{layer}.{kind}"
    for mlp in ("bbox_embed", "transformer.enc_out_bbox_embed.0")
    for layer in (0, 1)
    for kind in ("weight", "bias")
}

BaselineYaml = Callable[[str], Callable[..., Path]]


def _module(path: Path) -> RFDETRModelModule:
    """Build the Lightning module from a YAML file, with a fixed seed."""
    model_config, train_config = load_configs(path)
    torch.manual_seed(0)
    return RFDETRModelModule(model_config, train_config)


def _loss_weights(dec_layers: int) -> set[str]:
    """Names of the loss weights of RF-DETR: final layer, every other decoder layer and first stage."""
    aux = {f"{name}_{i}" for i in range(dec_layers - 1) for name in BASE_WEIGHTS}
    return BASE_WEIGHTS | aux | {f"{name}_enc" for name in BASE_WEIGHTS}


def _images(batch_size: int, resolution: int) -> Tensor:
    """Return a fixed batch of random images."""
    return torch.randn(batch_size, 3, resolution, resolution, generator=torch.Generator().manual_seed(0))


def _groups_by_name(module: RFDETRModelModule) -> dict[str, dict[str, Any]]:
    """Hyperparameters of the optimizer group of every parameter, as ``configure_optimizers`` builds them."""
    groups = get_param_dict(_namespace_from_configs(module.model_config, module.train_config), module.model)
    names = {id(param): name for name, param in module.model.named_parameters()}
    return {names[id(p)]: {k: v for k, v in g.items() if k != "params"} for g in groups for p in g["params"]}


def _decoder_layer(model: LWDETR) -> nn.Module:
    """First layer of the decoder."""
    return model.transformer.decoder.layers[0]


def _encoder(model: LWDETR) -> nn.Module:
    """DINOv2 encoder of the backbone."""
    return model.backbone[0].encoder.encoder


@pytest.mark.parametrize(
    ("ring", "num_windows", "projector", "blocks"),
    [
        ("win", 2, MultiScaleProjector, FOUR_BLOCKS),
        ("nowin", 1, MultiScaleProjector, FOUR_BLOCKS),
        ("nowin_noproj", 1, LinearProjector, ["stage12"]),
    ],
)
def test_module_from_yaml(
    ring: str, num_windows: int, projector: type[nn.Module], blocks: list[str], baseline_yaml: BaselineYaml
) -> None:
    """RFDETRModelModule builds each ring at 576 px, with RF-DETR Medium's decoder and criterion and one group."""
    module = _module(baseline_yaml(ring)())
    model = module.model
    encoder = _encoder(model)

    assert isinstance(model, LWDETR)
    assert encoder.config.num_register_tokens == 4
    assert encoder.config.num_windows == num_windows
    assert encoder.config.out_features == blocks
    assert encoder.embeddings.position_embeddings.shape == (1, 1 + 36 * 36, 384)
    assert type(model.backbone[0].projector) is projector
    assert len(model.transformer.decoder.layers) == 4
    assert model.query_feat.num_embeddings == 300  # one group of 300 queries
    assert model.class_embed.out_features == 91
    assert set(module.criterion.weight_dict) == _loss_weights(4)


@pytest.mark.parametrize(
    ("changes", "read", "expected"),
    [
        pytest.param({"dec_layers": 3}, lambda m: len(m.transformer.decoder.layers), 3, id="dec_layers"),
        pytest.param(
            {"hidden_dim": 384},
            lambda m: (m.transformer.d_model, m.query_feat.embedding_dim, m.class_embed.in_features),
            (384, 384, 384),
            id="hidden_dim",
        ),
        pytest.param({"sa_nheads": 4}, lambda m: _decoder_layer(m).self_attn.num_heads, 4, id="sa_nheads"),
        pytest.param({"ca_nheads": 8}, lambda m: _decoder_layer(m).cross_attn.n_heads, 8, id="ca_nheads"),
        pytest.param({"dec_n_points": 4}, lambda m: _decoder_layer(m).cross_attn.n_points, 4, id="dec_n_points"),
        pytest.param({"num_queries": 100}, lambda m: m.query_feat.num_embeddings, 100, id="num_queries"),
        pytest.param({"num_windows": 2}, lambda m: _encoder(m).config.num_windows, 2, id="num_windows"),
        pytest.param(
            {"projector": False, "out_feature_indexes": [12]},
            lambda m: type(m.backbone[0].projector),
            LinearProjector,
            id="projector",
        ),
        pytest.param(
            {"resolution": 640, "positional_encoding_size": 40},
            lambda m: _encoder(m).embeddings.position_embeddings.shape[1],
            1 + 40 * 40,
            id="resolution",
        ),
        pytest.param(
            {"out_feature_indexes": [6, 12]},
            lambda m: _encoder(m).config.out_features,
            ["stage6", "stage12"],
            id="out_feature_indexes",
        ),
    ],
)
def test_value_reaches_model(
    changes: dict[str, Any], read: Callable[[LWDETR], Any], expected: Any, baseline_yaml: BaselineYaml
) -> None:
    """A value changed in the YAML without windows reaches the model built by RFDETRModelModule (§3.7)."""
    module = _module(baseline_yaml("nowin")({**SMALL, **changes}))

    assert read(module.model) == expected


def test_dec_layers_reach_outputs_and_criterion(baseline_yaml: BaselineYaml) -> None:
    """With 3 decoder layers there are 2 aux outputs, each with its loss weights."""
    module = _module(baseline_yaml("nowin")({**SMALL, "dec_layers": 3}))
    model = module.model.eval()

    with torch.no_grad():
        outputs = model(nested_tensor_from_tensor_list(list(_images(1, 224))))

    assert len(outputs["aux_outputs"]) == 2
    assert set(module.criterion.weight_dict) == _loss_weights(3)


def test_no_projector_is_linear_and_layernorm(baseline_yaml: BaselineYaml) -> None:
    """Without projector the decoder reads LayerNorm(Linear(last ViT block)), and MultiScaleProjector is gone."""
    module = _module(baseline_yaml("nowin_noproj")(SMALL))
    backbone = module.model.backbone[0].eval()
    projector = backbone.projector
    x = _images(2, 224)

    with torch.no_grad():
        (last_block,) = backbone.encoder(x)  # [2, 384, 14, 14], after DINOv2's final LayerNorm
        features, _ = backbone(nested_tensor_from_tensor_list(list(x)))
    linear = F.linear(last_block.permute(0, 2, 3, 1), projector.linear.weight, projector.linear.bias)
    expected = F.layer_norm(linear, (256,), projector.norm.weight, projector.norm.bias, eps=1e-6).permute(0, 3, 1, 2)

    assert (projector.linear.in_features, projector.linear.out_features) == (384, 256)
    assert len(features) == 1
    torch.testing.assert_close(features[0].tensors, expected)
    projector_parameters = [n for n, _ in module.model.named_parameters() if n.startswith("backbone.0.projector.")]
    names = ("linear.weight", "linear.bias", "norm.weight", "norm.bias")
    assert projector_parameters == [f"backbone.0.projector.{name}" for name in names]


def test_no_projector_learning_rate(baseline_yaml: BaselineYaml) -> None:
    """Linear and LayerNorm get ``lr``, as every parameter of RF-DETR's projector in the ring before."""
    for ring in ("nowin", "nowin_noproj"):
        groups = _groups_by_name(_module(baseline_yaml(ring)(SMALL)))
        projector = {name: group for name, group in groups.items() if name.startswith("backbone.0.projector.")}

        assert projector, ring
        assert all(group == {"lr": 1e-4} for group in projector.values()), ring


@pytest.mark.parametrize("ring", ["win", "nowin", "nowin_noproj"])
def test_forward_backward_coco_mini(ring: str, baseline_yaml: BaselineYaml, coco_mini_batch: Callable) -> None:
    """Forward and backward with RF-DETR's criterion on 2 coco_mini images at 576 px.

    The loss is finite; the gradient is finite and non-zero on every parameter but those listed in ``NO_GRADIENT``
    and ``ZERO_GRADIENT``, which have none or a zero one at the first step for the reasons written there.
    """
    module = _module(baseline_yaml(ring)(train_changes={"batch_size": 2}))
    samples, targets = coco_mini_batch(module.model_config, module.train_config)  # data pipeline of the YAML
    model = module.model.train()

    outputs = model(samples, targets)
    loss_dict = module.criterion(outputs, targets)
    weight_dict = module.criterion.weight_dict
    loss = torch.stack([loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict]).sum()
    loss.backward()

    assert samples.tensors.shape == (2, 3, 576, 576)
    assert outputs["pred_logits"].shape == (2, 300, 91)
    assert len(outputs["aux_outputs"]) == 3
    assert "enc_outputs" in outputs
    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        assert param.requires_grad, name
        if name in NO_GRADIENT:
            assert param.grad is None, name
            continue
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert bool(param.grad.abs().sum() > 0) == (name not in ZERO_GRADIENT), name


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("size", "out_feature_indexes", "width", "depth"),
    [("base", [3, 6, 9, 12], 768, 12), ("large", [6, 12, 18, 24], 1024, 24)],
    ids=["vitb", "vitl"],
)
def test_module_from_yaml_sizes(
    size: str,
    out_feature_indexes: list[int],
    width: int,
    depth: int,
    baseline_yaml: BaselineYaml,
    coco_mini_batch: Callable,
) -> None:
    """ViT-B and ViT-L come from the YAML without windows by changing DINOv2 size and blocks; backward as for ViT-S.

    224 px keeps the ViT-B and ViT-L backward within CPU memory.
    """
    changes = {**SMALL, "encoder": f"dinov2_registers_windowed_{size}", "out_feature_indexes": out_feature_indexes}
    module = _module(baseline_yaml("nowin")(changes, {"batch_size": 2}))
    samples, targets = coco_mini_batch(module.model_config, module.train_config)

    loss_dict = module.criterion(module.model.train()(samples, targets), targets)
    weight_dict = module.criterion.weight_dict
    loss = torch.stack([loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict]).sum()
    loss.backward()

    encoder = _encoder(module.model)
    assert (encoder.config.hidden_size, encoder.config.num_hidden_layers) == (width, depth)
    assert encoder.config.out_features == [f"stage{i}" for i in out_feature_indexes]
    assert torch.isfinite(loss)
    for name, param in module.model.named_parameters():
        if name in NO_GRADIENT:
            assert param.grad is None, name
            continue
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert bool(param.grad.abs().sum() > 0) == (name not in ZERO_GRADIENT), name
