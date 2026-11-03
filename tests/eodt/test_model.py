# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for rfdetr.models.eodt.model.EoDT with learned queries (step 3 of EODT_SPEC.md §9).

Covers:
- Output format of §3.5: ``pred_logits``, ``pred_boxes`` and ``aux_outputs``, no ``enc_outputs``.
- RF-DETR's ``PostProcess`` accepts the output unchanged.
- ``aux_outputs`` has L2 - 1 elements.
- Boxes in [0, 1], also with the sigmoid saturated.
- Every prediction is the shared head applied to ``vit.norm`` of the query states, block by block.
- One head only, and no LayerNorm outside the ViT.
- The queries entering L2 are the learned embeddings, the same for every image.
- Gradient on queries, head and backbone.
- Initialization of queries, class bias and box head.
- Invalid arguments are rejected.
"""

import math

import pytest
import torch
from torch import Tensor, nn

from rfdetr.models.eodt import EoDT, EoDTBackbone
from rfdetr.models.postprocess import PostProcess

VITS = "vit_small_patch14_reg4_dinov2"
RESOLUTION = 224  # 14 x 14 grid keeps the CPU tests fast
PATCH_SIZE = 16
NUM_QUERIES = 10
NUM_CLASSES = 91  # COCO: raw category ids 1..90 used as logit indices


def _images(batch_size: int, resolution: int = RESOLUTION, seed: int = 0) -> Tensor:
    """Return a fixed batch of random images."""
    return torch.randn(batch_size, 3, resolution, resolution, generator=torch.Generator().manual_seed(seed))


def _model(
    num_l2_blocks: int = 3,
    resolution: int = RESOLUTION,
    num_queries: int = NUM_QUERIES,
    pretrained: bool = False,
) -> EoDT:
    """Return a ViT-S EoDT with a fixed seed."""
    torch.manual_seed(0)
    backbone = EoDTBackbone(VITS, resolution, PATCH_SIZE, num_l2_blocks, pretrained=pretrained)
    return EoDT(backbone, num_queries, NUM_CLASSES)


def _all_predictions(out: dict) -> list[dict[str, Tensor]]:
    """Return the per-block predictions, first L2 block first."""
    return [*out["aux_outputs"], {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}]


def test_output_format() -> None:
    """At 576 px with 300 queries the output has exactly the keys and shapes of §3.5."""
    model = _model(resolution=576, num_queries=300).eval()

    with torch.no_grad():
        out = model(_images(2, 576))

    assert set(out) == {"pred_logits", "pred_boxes", "aux_outputs"}
    assert out["pred_logits"].shape == (2, 300, NUM_CLASSES)
    assert out["pred_boxes"].shape == (2, 300, 4)
    assert isinstance(out["aux_outputs"], list)
    for aux in out["aux_outputs"]:
        assert set(aux) == {"pred_logits", "pred_boxes"}
        assert aux["pred_logits"].shape == (2, 300, NUM_CLASSES)
        assert aux["pred_boxes"].shape == (2, 300, 4)


def test_postprocess_accepts_output() -> None:
    """RF-DETR's PostProcess turns the output into per-image detections without changes (§3.5)."""
    model = _model().eval()
    num_select = 300
    target_sizes = torch.tensor([[480, 640], [576, 576]])

    with torch.no_grad():
        out = model(_images(2))
        results = PostProcess(num_select=num_select)(out, target_sizes)

    assert len(results) == 2
    for result, logits in zip(results, out["pred_logits"]):
        assert set(result) == {"scores", "labels", "boxes"}
        assert result["scores"].shape == result["labels"].shape == (num_select,)
        assert result["boxes"].shape == (num_select, 4)
        assert ((result["labels"] >= 0) & (result["labels"] < NUM_CLASSES)).all()
        assert torch.isfinite(result["boxes"]).all()
        # the scores are the top sigmoid scores over all query-class pairs of our logits
        torch.testing.assert_close(result["scores"], logits.sigmoid().flatten().topk(num_select).values)


@pytest.mark.parametrize("training", [True, False])
def test_no_enc_outputs(training: bool) -> None:
    """With learned queries there is no first-stage output, in training and in evaluation."""
    model = _model().train(training)

    with torch.no_grad():
        out = model(_images(2))

    assert "enc_outputs" not in out


@pytest.mark.parametrize("num_l2_blocks", [1, 3, 4])
def test_aux_outputs_length(num_l2_blocks: int) -> None:
    """One auxiliary output for every L2 block but the last."""
    model = _model(num_l2_blocks).eval()

    with torch.no_grad():
        out = model(_images(2))

    assert len(out["aux_outputs"]) == num_l2_blocks - 1


@pytest.mark.parametrize("weight_scale", [1.0, 100.0])
def test_boxes_in_unit_interval(weight_scale: float) -> None:
    """Final and auxiliary boxes are finite and in [0, 1], also with the sigmoid pushed into saturation."""
    model = _model().eval()
    with torch.no_grad():
        for layer in model.bbox_embed.layers:
            layer.weight.mul_(weight_scale)
        out = model(_images(2))

    for prediction in _all_predictions(out):
        boxes = prediction["pred_boxes"]
        assert torch.isfinite(boxes).all()
        assert (boxes >= 0).all() and (boxes <= 1).all()


def test_outputs_match_shared_head() -> None:
    """After every L2 block, the predictions are the one shared head applied to ``vit.norm`` of the query states."""
    model = _model().eval()
    x = _images(2)
    norm_inputs, norm_outputs = [], []

    def record(module: nn.Module, args: tuple[Tensor], output: Tensor) -> None:
        norm_inputs.append(args[0])
        norm_outputs.append(output)

    hook = model.backbone.vit.norm.register_forward_hook(record)

    with torch.no_grad():
        try:
            out = model(x)
        finally:
            hook.remove()
        queries = model.query_feat.weight[None].expand(2, -1, -1)
        query_states, _ = model.backbone.forward_l2(model.backbone.forward_l1(x), queries)

        # the head runs once on the stacked blocks, as in the model, so the comparison can be exact
        hs = torch.stack(norm_outputs)
        expected_logits = model.class_embed(hs)
        expected_boxes = model.bbox_embed(hs).sigmoid()

    predictions = _all_predictions(out)
    assert len(norm_inputs) == len(query_states) == len(predictions) == 3
    for i, (state, norm_in, prediction) in enumerate(zip(query_states, norm_inputs, predictions)):
        torch.testing.assert_close(norm_in, state, rtol=0, atol=0)
        torch.testing.assert_close(prediction["pred_logits"], expected_logits[i], rtol=0, atol=0)
        torch.testing.assert_close(prediction["pred_boxes"], expected_boxes[i], rtol=0, atol=0)


def test_head_is_shared() -> None:
    """One class head and one box head: no copies per L2 block."""
    model = _model()

    def num_params(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    head_params = num_params(model.class_embed) + num_params(model.bbox_embed)
    assert num_params(model) == num_params(model.backbone) + NUM_QUERIES * model.backbone.embed_dim + head_params

    backbone_modules = {id(m) for m in model.backbone.modules()}
    linears = [m for m in model.modules() if isinstance(m, nn.Linear) and id(m) not in backbone_modules]
    assert len(linears) == 1 + 3
    assert model.class_embed in linears
    assert all(layer in linears for layer in model.bbox_embed.layers)


def test_norm_is_vit_norm() -> None:
    """There is no new LayerNorm, and the head reads the ViT's own final norm."""
    model = _model().eval()
    vit_modules = {id(m) for m in model.backbone.vit.modules()}
    assert not [m for m in model.modules() if isinstance(m, nn.LayerNorm) and id(m) not in vit_modules]

    x = _images(2)
    with torch.no_grad():
        before = model(x)
        model.backbone.vit.norm.weight.mul_(2)
        after = model(x)

    for prediction_before, prediction_after in zip(_all_predictions(before), _all_predictions(after)):
        assert not torch.allclose(prediction_before["pred_logits"], prediction_after["pred_logits"])
        assert not torch.allclose(prediction_before["pred_boxes"], prediction_after["pred_boxes"])


def test_queries_are_image_independent() -> None:
    """The queries entering the first L2 block are the learned embeddings, for every image of the batch."""
    model = _model().eval()
    first_l2_block = model.backbone.vit.blocks[model.backbone.num_l1_blocks]
    inputs = []
    hook = first_l2_block.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))

    with torch.no_grad():
        try:
            model(_images(3))
        finally:
            hook.remove()

    assert len(inputs) == 1
    for queries in inputs[0][:, :NUM_QUERIES]:
        torch.testing.assert_close(queries, model.query_feat.weight, rtol=0, atol=0)


def test_gradient_reaches_queries_head_backbone() -> None:
    """A loss on all predictions gives a finite, non-zero gradient to every query and every parameter."""
    model = _model(pretrained=True).train()

    out = model(_images(2))
    loss = sum(p["pred_logits"].mean() + p["pred_boxes"].mean() for p in _all_predictions(out))
    loss.backward()

    query_grad = model.query_feat.weight.grad
    assert query_grad is not None and torch.isfinite(query_grad).all()
    assert (query_grad.abs().sum(dim=-1) > 0).all()  # every query
    groups = {"class_embed": model.class_embed, "bbox_embed": model.bbox_embed, "backbone": model.backbone}
    for group, module in groups.items():
        for name, param in module.named_parameters():
            assert param.requires_grad, f"{group}.{name}"
            assert param.grad is not None, f"{group}.{name}"
            assert torch.isfinite(param.grad).all(), f"{group}.{name}"
            assert param.grad.abs().sum() > 0, f"{group}.{name}"


def test_initialization() -> None:
    """Queries N(0, 1) as EoMT, class bias at prior 0.01 as RF-DETR, box head with the default init as DETR."""
    model = _model(num_queries=300).eval()

    weight = model.query_feat.weight
    assert abs(weight.mean().item()) < 0.05
    assert abs(weight.std().item() - 1) < 0.05
    torch.testing.assert_close(model.class_embed.bias, torch.full((NUM_CLASSES,), -math.log(99)))

    last_layer = model.bbox_embed.layers[-1]
    assert last_layer.weight.abs().sum() > 0
    with torch.no_grad():
        boxes = model(_images(1))["pred_boxes"][0]
    assert torch.unique(boxes, dim=0).shape[0] == 300  # the queries start from different boxes


@pytest.mark.parametrize(("num_queries", "num_classes"), [(0, NUM_CLASSES), (NUM_QUERIES, 0)])
def test_invalid_arguments(num_queries: int, num_classes: int) -> None:
    """At least one query and one class."""
    backbone = EoDTBackbone(VITS, RESOLUTION, PATCH_SIZE, 3, pretrained=False)
    with pytest.raises(ValueError, match="num_queries|num_classes"):
        EoDT(backbone, num_queries, num_classes)
