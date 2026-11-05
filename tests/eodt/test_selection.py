# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the ``selection`` queries of EoDT (step 9a of EODT_SPEC.md §9, §3.3).

Covers:
- Output in the format of §3.5 with ``enc_outputs``, in training and in evaluation; ``learned`` has neither the
  first stage nor ``enc_outputs``.
- Top-Q: the selected patches are the Q with the highest class score, and at first every b0 is the proposal centred
  on its patch, so CLS and registers are never candidates.
- The first stage gives the same boxes and logits as the two-stage branch of RF-DETR's ``Transformer`` with the
  same weights.
- b0 has no gradient: a loss on the final outputs does not reach the first stage, while the loss of the first stage
  reaches it and the L1 blocks (the detach is on b0 only), not the L2 blocks nor the queries.
- With a zero delta the box is the reference: b0 at first, when ``refpoint_embed`` is zero; b0 moved by
  ``refpoint_embed`` otherwise.
- ``query_mode`` has no default and comes from the YAML; with ``selection`` there are at most as many queries as
  patches.
- Forward and backward with RF-DETR's criterion on 2 coco_mini images (test 4 of §9): finite loss, aux and first
  stage weights, gradient on every parameter but a listed few, for ViT-S, ViT-B and ViT-L.
- The new parameters get ``lr``; ``vit.norm``, now also on the selection branch, keeps the learning rate of the ViT.

ViT-B and ViT-L cases carry the ``gpu`` marker (slow on CPU): ``-m "not gpu"`` runs ViT-S only.
"""

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from pydantic import ValidationError
from torch import Tensor, nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import TrainConfig
from rfdetr.models.eodt import EoDT, EoDTConfig, load_configs
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
from rfdetr.models.transformer import Transformer, gen_sineembed_for_position
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.param_groups import get_param_dict

RESOLUTION = 224  # 14 x 14 = 196 patches keep the CPU tests fast
GRID = RESOLUTION // 16
NUM_QUERIES = 10
NUM_L2_BLOCKS = 3
EODT_YAML = Path(__file__).resolve().parents[2] / "configs" / "eodt" / "eodt_vits_576_l2-3_learned.yaml"
BASE_WEIGHTS = {"loss_ce", "loss_bbox", "loss_giou"}
FIRST_STAGE = ("selection.",)
NEW_PARAMETERS = ("selection.", "refpoint_embed.", "ref_point_head.", "query_feat.", "class_embed.", "bbox_embed.")
# Parameters with a zero gradient at the first step: the first two layers of the box MLPs, whose last layer starts
# from zero (every box starts from its reference or proposal), as in the baseline.
ZERO_GRADIENT = {
    f"{mlp}.layers.{layer}.{kind}"
    for mlp in ("bbox_embed", "selection.enc_out_bbox_embed")
    for layer in (0, 1)
    for kind in ("weight", "bias")
}


def _config(**overrides: Any) -> EoDTConfig:
    """Return a small ViT-S configuration with selection queries."""
    return EoDTConfig(
        **{
            "num_l2_blocks": NUM_L2_BLOCKS,
            "resolution": RESOLUTION,
            "num_queries": NUM_QUERIES,
            "query_mode": "selection",
            **overrides,
        }
    )


def _build(config: EoDTConfig | None = None) -> EoDT:
    """Build the model through RF-DETR's entry point, with a fixed seed."""
    torch.manual_seed(0)
    return build_model_from_config(config or _config())


def _images(batch_size: int = 2) -> Tensor:
    """Return a fixed batch of random images."""
    return torch.randn(batch_size, 3, RESOLUTION, RESOLUTION, generator=torch.Generator().manual_seed(1))


def _patches(model: EoDT, x: Tensor) -> Tensor:
    """Patch tokens after L1 and vit.norm, the input of the first stage."""
    return model.backbone.vit.norm(model.backbone.forward_l1(x)[:, model.backbone.num_prefix_tokens :])


def _reference(boxes_b0: Tensor, offset: Tensor) -> Tensor:
    """b0 moved by the refpoint_embed offset with the bbox_reparam formula, written out."""
    centre = offset[..., :2] * boxes_b0[..., 2:] + boxes_b0[..., :2]
    return torch.cat([centre, offset[..., 2:].exp() * boxes_b0[..., 2:]], dim=-1)


def _randomize_first_stage_boxes(model: EoDT) -> None:
    """Give the box head of the first stage a non-zero last layer, so that b0 differs from its proposal."""
    nn.init.normal_(model.selection.enc_out_bbox_embed.layers[-1].weight, std=0.05)


def _loss_weights(num_l2_blocks: int) -> set[str]:
    """Final, aux (one per L2 block but the last) and first-stage loss weights."""
    aux = {f"{name}_{i}" for i in range(num_l2_blocks - 1) for name in BASE_WEIGHTS}
    return BASE_WEIGHTS | aux | {f"{name}_enc" for name in BASE_WEIGHTS}


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
def test_output_format(training: bool) -> None:
    """Final, aux and first-stage outputs, in training and in evaluation, as RF-DETR's two-stage LWDETR."""
    model = _build().train(training)

    with torch.set_grad_enabled(training):
        out = model(_images())

    assert set(out) == {"pred_logits", "pred_boxes", "aux_outputs", "enc_outputs"}
    assert out["pred_logits"].shape == (2, NUM_QUERIES, 91)
    assert out["pred_boxes"].shape == (2, NUM_QUERIES, 4)
    assert len(out["aux_outputs"]) == NUM_L2_BLOCKS - 1
    assert out["enc_outputs"]["pred_logits"].shape == (2, NUM_QUERIES, 91)
    assert out["enc_outputs"]["pred_boxes"].shape == (2, NUM_QUERIES, 4)


def test_learned_has_no_first_stage() -> None:
    """Learned queries have no first stage, no reference and no enc_outputs."""
    model = _build(_config(query_mode="learned")).eval()

    with torch.no_grad():
        out = model(_images())

    assert "enc_outputs" not in out
    assert not any(hasattr(model, name) for name in ("selection", "refpoint_embed", "ref_point_head"))


def test_top_q() -> None:
    """The first stage selects the Q patches with the highest class score; at first b0 is their proposal.

    The proposals are centred on the patches (width and height 0.05), so the centres of b0 identify the selected
    patches: they are those of the Q best patches. The forward of the model gives the first stage exactly these
    patch tokens, so CLS and registers are never candidates.
    """
    model = _build().eval()
    selection = model.selection
    x = _images()

    with torch.no_grad():
        patches = _patches(model, x)
        scores = selection.enc_out_class_embed(selection.enc_output_norm(selection.enc_output(patches))).max(-1)[0]
        _, logits, boxes = selection(patches, (GRID, GRID))
        enc_outputs = model(x)["enc_outputs"]

    best = scores.argsort(dim=1, descending=True)[:, :NUM_QUERIES]  # [B, Q], highest first
    centres = torch.stack([(best % GRID + 0.5) / GRID, (best // GRID + 0.5) / GRID], dim=-1)
    assert scores.shape == (2, GRID * GRID)
    torch.testing.assert_close(boxes[..., :2], centres)
    torch.testing.assert_close(boxes[..., 2:], torch.full_like(boxes[..., 2:], 0.05))
    torch.testing.assert_close(logits.max(-1)[0], scores.gather(1, best))
    assert torch.equal(enc_outputs["pred_boxes"], boxes)
    assert torch.equal(enc_outputs["pred_logits"], logits)


def test_matches_rfdetr_first_stage() -> None:
    """With the same weights and patches, the first stage of RF-DETR's Transformer gives the same outputs.

    The Transformer is the one of RF-DETR with the two-stage selection on one level, ``bbox_reparam`` and no
    decoder layer, so it returns the first stage only. The weights are randomized so that the box deltas are not zero.
    RF-DETR recomputes the class head on the selected tokens instead of gathering it: the logits agree within fp32.
    """
    model = _build().eval()
    selection = model.selection
    for param in selection.parameters():
        nn.init.normal_(param, std=0.05)
    hidden_dim = model.backbone.embed_dim
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=NUM_QUERIES,
        num_decoder_layers=0,
        two_stage=True,
        num_feature_levels=1,
        lite_refpoint_refine=True,
        bbox_reparam=True,
    ).eval()
    transformer.enc_output[0].load_state_dict(selection.enc_output.state_dict())
    transformer.enc_output_norm[0].load_state_dict(selection.enc_output_norm.state_dict())
    transformer.enc_out_class_embed = nn.ModuleList([copy.deepcopy(selection.enc_out_class_embed)])
    transformer.enc_out_bbox_embed = nn.ModuleList([copy.deepcopy(selection.enc_out_bbox_embed)])

    with torch.no_grad():
        patches = _patches(model, _images())
        boxes_b0, logits, boxes = selection(patches, (GRID, GRID))
        features = patches.reshape(2, GRID, GRID, -1).permute(0, 3, 1, 2)  # row-major patches, as RF-DETR flattens
        no_padding = torch.zeros(2, GRID, GRID, dtype=torch.bool)
        # positions, reference offsets and contents are read by the decoder only, which has no layer here
        no_queries = (torch.zeros(NUM_QUERIES, 4), torch.zeros(NUM_QUERIES, hidden_dim))
        outputs = transformer([features], [no_padding], [torch.zeros_like(features)], *no_queries)
        hs, _, memory_ts, boxes_ts, _ = outputs
        rfdetr_logits = transformer.enc_out_class_embed[0](memory_ts)

    assert hs is None  # no decoder: only the first stage ran
    assert not torch.equal(boxes[..., 2:], torch.full_like(boxes[..., 2:], 0.05))  # non-zero deltas
    torch.testing.assert_close(boxes, boxes_ts, rtol=0, atol=0)
    torch.testing.assert_close(boxes_b0, boxes_ts, rtol=0, atol=0)
    torch.testing.assert_close(logits, rfdetr_logits)


def test_b0_has_no_gradient() -> None:
    """b0 is detached, the boxes of the first stage for its loss are not."""
    model = _build().train()

    boxes_b0, _, boxes = model.selection(_patches(model, _images()), (GRID, GRID))

    assert not boxes_b0.requires_grad
    assert boxes.requires_grad


def test_final_loss_does_not_reach_first_stage() -> None:
    """A loss on the final and aux outputs reaches the queries and the reference, not the first stage."""
    model = _build().train()
    out = model(_images())
    outputs = [out, *out["aux_outputs"]]

    sum(output["pred_logits"].sum() + output["pred_boxes"].sum() for output in outputs).backward()

    for name, param in model.named_parameters():
        if name.startswith(FIRST_STAGE):
            assert param.grad is None, name
    for name in ("query_feat.weight", "refpoint_embed.weight", "ref_point_head.layers.0.weight"):
        assert model.get_parameter(name).grad.abs().sum() > 0, name


def test_first_stage_loss_reaches_l1() -> None:
    """The loss of the first stage reaches its heads, vit.norm and every parameter of the L1 blocks.

    The detach is on b0 only, as in RF-DETR, where this loss also reaches the backbone. It does not reach the L2
    blocks, the queries or the final head.
    """
    model = _build().train()
    enc_outputs = model(_images())["enc_outputs"]

    (enc_outputs["pred_logits"].sum() + enc_outputs["pred_boxes"].sum()).backward()

    for name in (
        "selection.enc_output.weight",
        "selection.enc_output_norm.weight",
        "selection.enc_out_class_embed.weight",
        "selection.enc_out_bbox_embed.layers.2.weight",
        "backbone.vit.norm.weight",
    ):
        assert model.get_parameter(name).grad.abs().sum() > 0, name
    backbone = model.backbone
    for block in backbone.vit.blocks[: backbone.num_l1_blocks]:
        for name, param in block.named_parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0, name
    for block in backbone.vit.blocks[backbone.num_l1_blocks :]:
        assert all(param.grad is None for param in block.parameters())
    for name in ("query_feat.weight", "refpoint_embed.weight", "class_embed.weight", "bbox_embed.layers.2.weight"):
        assert model.get_parameter(name).grad is None, name


def test_zero_delta_box_is_b0() -> None:
    """At first refpoint_embed and the last layer of bbox_embed are zero: every L2 block predicts exactly b0.

    The first stage predicts a non-zero delta, so b0 is not its proposal (width and height 0.05).
    """
    model = _build().eval()
    _randomize_first_stage_boxes(model)

    with torch.no_grad():
        out = model(_images())

    b0 = out["enc_outputs"]["pred_boxes"]
    assert not torch.allclose(b0[..., 2:], torch.full_like(b0[..., 2:], 0.05))
    for output in [out, *out["aux_outputs"]]:
        assert torch.equal(output["pred_boxes"], b0)


def test_zero_delta_box_is_reference() -> None:
    """With a non-zero refpoint_embed and a zero delta, the box is b0 moved by refpoint_embed."""
    model = _build().eval()
    _randomize_first_stage_boxes(model)
    nn.init.normal_(model.refpoint_embed.weight, std=0.1)

    with torch.no_grad():
        out = model(_images())

    b0 = out["enc_outputs"]["pred_boxes"]
    reference = _reference(b0, model.refpoint_embed.weight[None])
    assert not torch.allclose(b0[..., 2:], torch.full_like(b0[..., 2:], 0.05))
    assert not torch.equal(reference, b0)
    for output in [out, *out["aux_outputs"]]:
        torch.testing.assert_close(output["pred_boxes"], reference)


def test_l2_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """L2 gets the raw L1 tokens and, as queries, query_feat plus the encoding of the reference.

    ``vit.norm`` is on the selection branch only. The encoding is ``ref_point_head`` on the sine embedding of the
    reference with D / 2 frequencies per coordinate, as RF-DETR's decoder computes ``query_pos``.
    """
    model = _build().eval()
    _randomize_first_stage_boxes(model)
    nn.init.normal_(model.refpoint_embed.weight, std=0.1)
    calls = []
    forward_l2 = model.backbone.forward_l2

    def record(tokens: Tensor, queries: Tensor) -> tuple[list[Tensor], Tensor]:
        calls.append((tokens, queries))
        return forward_l2(tokens, queries)

    monkeypatch.setattr(model.backbone, "forward_l2", record)
    x = _images()

    with torch.no_grad():
        out = model(x)
        l1_tokens = model.backbone.forward_l1(x)
        reference = _reference(out["enc_outputs"]["pred_boxes"], model.refpoint_embed.weight[None])
        sine = gen_sineembed_for_position(reference, model.backbone.embed_dim // 2)
        expected_queries = model.query_feat.weight[None] + model.ref_point_head(sine)

    ((tokens, queries),) = calls
    assert torch.equal(tokens, l1_tokens)
    torch.testing.assert_close(queries, expected_queries)


def test_query_mode_has_no_default() -> None:
    """Without query_mode the configuration is rejected: neither mode starts in silence."""
    with pytest.raises(ValidationError, match="query_mode"):
        EoDTConfig(num_l2_blocks=3)


def test_yaml_without_query_mode_is_rejected(tmp_path: Path) -> None:
    """A YAML file without query_mode does not load."""
    content = yaml.safe_load(EODT_YAML.read_text(encoding="utf-8"))
    del content["model"]["model_config"]["init_args"]["query_mode"]
    path = tmp_path / "eodt_without_query_mode.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValidationError, match="query_mode"):
        load_configs(path)


def test_num_queries_at_most_patches() -> None:
    """With selection there are at most as many queries as patches; learned queries have no such bound."""
    num_patches = GRID * GRID

    assert _config(num_queries=num_patches).num_queries == num_patches
    assert _config(num_queries=num_patches + 1, query_mode="learned").num_queries == num_patches + 1
    with pytest.raises(ValidationError, match=rf"num_queries \({num_patches + 1}\) must be at most .* \({num_patches}"):
        _config(num_queries=num_patches + 1)


@pytest.mark.parametrize("query_mode", ["learned", "selection"])
def test_query_mode_from_yaml(query_mode: str, eodt_yaml: Callable[..., Path]) -> None:
    """query_mode in the YAML file chooses the queries of the model and the first-stage weights of the criterion."""
    model_config, train_config = load_configs(
        eodt_yaml({"resolution": RESOLUTION, "num_queries": 100, "query_mode": query_mode})
    )
    torch.manual_seed(0)
    module = RFDETRModelModule(model_config, train_config)

    has_first_stage = query_mode == "selection"
    assert module.model.query_mode == query_mode
    assert hasattr(module.model, "selection") == has_first_stage
    first_stage_weights = {f"{name}_enc" for name in BASE_WEIGHTS}
    expected = _loss_weights(NUM_L2_BLOCKS) if has_first_stage else _loss_weights(NUM_L2_BLOCKS) - first_stage_weights
    assert set(module.criterion.weight_dict) == expected


@pytest.mark.parametrize(
    ("backbone_name", "num_l2_blocks", "resolution", "num_queries"),
    [
        pytest.param("vit_small_patch14_reg4_dinov2", 3, 576, 300, id="vits"),
        # 224 px (196 patches, 100 queries) keeps the ViT-B and ViT-L backward within CPU memory
        pytest.param("vit_base_patch14_reg4_dinov2", 3, 224, 100, id="vitb", marks=pytest.mark.gpu),
        pytest.param("vit_large_patch14_reg4_dinov2", 4, 224, 100, id="vitl", marks=pytest.mark.gpu),
    ],
)
def test_forward_backward_coco_mini(
    backbone_name: str,
    num_l2_blocks: int,
    resolution: int,
    num_queries: int,
    coco_mini_train_config: Callable[..., TrainConfig],
    coco_mini_batch: Callable,
) -> None:
    """Forward and backward with RF-DETR's criterion on 2 coco_mini images (test 4 of §9 with selection).

    The loss is finite and weights the final output, one aux output per L2 block but the last and the first stage.
    Every parameter gets a finite gradient, non-zero but for ``ZERO_GRADIENT``.
    """
    model_config = EoDTConfig(
        backbone_name=backbone_name,
        num_l2_blocks=num_l2_blocks,
        resolution=resolution,
        num_queries=num_queries,
        query_mode="selection",
    )
    train_config = coco_mini_train_config()
    samples, targets = coco_mini_batch(model_config, train_config)
    torch.manual_seed(0)
    model = build_model_from_config(model_config, train_config).train()
    criterion, _ = build_criterion_from_config(model_config, train_config)

    loss_dict = criterion(model(samples, targets), targets)
    weight_dict = criterion.weight_dict
    loss = torch.stack([loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict]).sum()
    loss.backward()

    assert samples.tensors.shape == (2, 3, resolution, resolution)
    assert set(weight_dict) == _loss_weights(num_l2_blocks)
    assert {"loss_ce_enc", "loss_bbox_enc", "loss_giou_enc"} <= set(loss_dict)
    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        assert param.requires_grad, name
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert bool(param.grad.abs().sum() > 0) == (name not in ZERO_GRADIENT), name


def test_new_parameters_get_lr() -> None:
    """First stage, queries, reference and head get lr; vit.norm keeps the learning rate of the final ViT norm."""
    config = _config()
    model = _build(config)
    groups = get_param_dict(_namespace_from_configs(config, TrainConfig(dataset_dir=".", multi_scale="off")), model)
    names = {id(param): name for name, param in model.named_parameters()}
    by_name = {names[id(p)]: {k: v for k, v in g.items() if k != "params"} for g in groups for p in g["params"]}

    assert set(by_name) == set(names.values())
    for name, group in by_name.items():
        if name.startswith(NEW_PARAMETERS):
            assert group == {"lr": 1e-4}, name
    assert by_name["backbone.vit.norm.weight"] == {"lr": pytest.approx(1.5e-4 * 0.7**2), "weight_decay": 0.0}
