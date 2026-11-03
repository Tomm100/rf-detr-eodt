# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for EoDT with RF-DETR's SetCriterion (step 4b of EODT_SPEC.md §9).

Covers:
- Forward and backward on 2 coco_mini images through RF-DETR's data module: finite loss and a finite, non-zero
  gradient on every trainable parameter, for ViT-S, ViT-B and ViT-L (§3.6).
- One aux loss weight per L2 block but the last, with RF-DETR's coefficients, and no first-stage weights
  (§8 point 6): every loss the criterion computes is weighted.
- ``cls_loss_coef`` set in the training configuration reaches the criterion (§3.7).

ViT-B and ViT-L cases carry the ``gpu`` marker (slow on CPU): ``-m "not gpu"`` runs ViT-S only.
"""

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from rfdetr.config import TrainConfig
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.eodt import EoDT, EoDTConfig
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config

# RF-DETR's coefficients: cls_loss_coef from TrainConfig, bbox and GIoU from ModelDefaults
BASE_WEIGHTS = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}


def _model_and_criterion(model_config: EoDTConfig, train_config: TrainConfig) -> tuple[EoDT, SetCriterion]:
    """Build model and criterion through RF-DETR's entry points, as RFDETRModelModule does."""
    torch.manual_seed(0)
    model = build_model_from_config(model_config, train_config)
    criterion, _ = build_criterion_from_config(model_config, train_config)
    return model, criterion


def _total_loss(loss_dict: dict[str, Tensor], weight_dict: dict[str, float]) -> Tensor:
    """Weighted sum of the losses, as RFDETRModelModule.training_step computes it."""
    return torch.stack([loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict]).sum()


@pytest.mark.parametrize(
    ("backbone_name", "num_l2_blocks", "resolution"),
    [
        pytest.param("vit_small_patch14_reg4_dinov2", 3, 576, id="vits"),
        # 224 px keeps the ViT-B and ViT-L backward within CPU memory
        pytest.param("vit_base_patch14_reg4_dinov2", 3, 224, id="vitb", marks=pytest.mark.gpu),
        pytest.param("vit_large_patch14_reg4_dinov2", 4, 224, id="vitl", marks=pytest.mark.gpu),
    ],
)
def test_forward_backward_coco_mini(
    backbone_name: str,
    num_l2_blocks: int,
    resolution: int,
    coco_mini_train_config: Callable[..., TrainConfig],
    coco_mini_batch: Callable,
) -> None:
    """Forward and backward with SetCriterion on 2 coco_mini images: finite loss, gradient on every parameter."""
    model_config = EoDTConfig(backbone_name=backbone_name, num_l2_blocks=num_l2_blocks, resolution=resolution)
    train_config = coco_mini_train_config()
    samples, targets = coco_mini_batch(model_config, train_config)
    model, criterion = _model_and_criterion(model_config, train_config)
    model.train()

    loss_dict = criterion(model(samples, targets), targets)
    loss = _total_loss(loss_dict, criterion.weight_dict)
    loss.backward()

    assert samples.tensors.shape == (2, 3, resolution, resolution)
    assert len(targets) == 2 and all(len(target["boxes"]) > 0 for target in targets)
    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        assert param.requires_grad, name
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


@pytest.mark.parametrize("num_l2_blocks", [1, 3, 4])
def test_aux_loss_weights(
    num_l2_blocks: int, coco_mini_train_config: Callable[..., TrainConfig], coco_mini_batch: Callable
) -> None:
    """One weight per L2 block but the last, no "_enc" weight, and every computed loss has a weight."""
    model_config = EoDTConfig(num_l2_blocks=num_l2_blocks, resolution=224)
    train_config = coco_mini_train_config()
    samples, targets = coco_mini_batch(model_config, train_config)
    model, criterion = _model_and_criterion(model_config, train_config)

    with torch.no_grad():
        loss_dict = criterion(model.eval()(samples), targets)

    aux_weights = {f"{name}_{i}": weight for i in range(num_l2_blocks - 1) for name, weight in BASE_WEIGHTS.items()}
    assert criterion.weight_dict == {**BASE_WEIGHTS, **aux_weights}
    # the losses of the criterion are exactly the weighted ones: none is dropped by training_step's sum
    assert {name for name in loss_dict if name.startswith("loss_")} == set(criterion.weight_dict)
    assert criterion.group_detr == 1


def test_cls_loss_coef_reaches_criterion() -> None:
    """The classification coefficient of the training configuration weights the final and the aux losses."""
    model_config = EoDTConfig(num_l2_blocks=3)
    train_config = TrainConfig(dataset_dir=".", multi_scale="off", cls_loss_coef=2.0)

    criterion, _ = build_criterion_from_config(model_config, train_config)

    ce_weights = {name: weight for name, weight in criterion.weight_dict.items() if name.startswith("loss_ce")}
    assert ce_weights == {"loss_ce": 2.0, "loss_ce_0": 2.0, "loss_ce_1": 2.0}
    assert criterion.weight_dict["loss_bbox"] == BASE_WEIGHTS["loss_bbox"]
