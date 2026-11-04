# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for training EoDT from its YAML file with RF-DETR's pipeline (step 4c of EODT_SPEC.md §9).

Every configuration is read from a copy of ``configs/eodt/eodt_vits_576_l2-3_learned.yaml`` written by the
``eodt_yaml`` fixture, which runs it on this PC with coco_mini.

Covers:
- ``RFDETRModelModule`` builds EoDT and its criterion from the YAML file (test of §9, step 4).
- A value changed in the YAML reaches the model, the criterion, the optimizer groups, the dataloader or the
  trainer (§3.7).
- Two epochs of two steps with ``build_trainer``, ``RFDETRModelModule`` and ``RFDETRDataModule``: finite loss,
  COCO evaluation, warm-up and cosine schedule, checkpoint. This is not the overfit test of step 5.
- ViT-B and ViT-L from the same YAML, changing only backbone and L2 (§3.6).

ViT-B and ViT-L cases carry the ``gpu`` marker (slow on CPU): ``-m "not gpu"`` runs ViT-S only.
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import timm
import torch
from torch import nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.models.eodt import EoDT, load_configs
from rfdetr.training.callbacks.best_model import BestModelCallback
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.callbacks.ema import RFDETREMACallback
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.param_groups import get_param_dict
from rfdetr.training.trainer import build_trainer

SMALL = {"resolution": 224}  # 14 x 14 grid keeps the CPU tests fast where the resolution is not under test
BASE_WEIGHTS = {"loss_ce", "loss_bbox", "loss_giou"}


def _module(path: Path) -> RFDETRModelModule:
    """Build the Lightning module from a YAML file, with a fixed seed."""
    model_config, train_config = load_configs(path)
    torch.manual_seed(0)
    return RFDETRModelModule(model_config, train_config)


def _aux_weights(num_aux: int) -> set[str]:
    """Names of the loss weights of the first ``num_aux`` aux outputs."""
    return {f"{name}_{i}" for i in range(num_aux) for name in BASE_WEIGHTS}


def test_module_from_yaml(eodt_yaml: Callable[..., Path]) -> None:
    """RFDETRModelModule builds EoDT ViT-S at 576 px with L2 = 3 and its criterion from the YAML file."""
    module = _module(eodt_yaml())
    model = module.model

    assert isinstance(model, EoDT)
    assert module.model_config.pretrain_weights is None  # DINOv2 from timm only, no RF-DETR checkpoint
    assert model.backbone.grid_size == (36, 36)
    assert model.backbone.num_l1_blocks == 9
    assert model.query_feat.num_embeddings == 300
    assert model.class_embed.out_features == 91
    assert set(module.criterion.weight_dict) == BASE_WEIGHTS | _aux_weights(2)


def test_resolution_reaches_model(eodt_yaml: Callable[..., Path]) -> None:
    """With resolution 640 in the YAML the patch grid becomes 40 x 40."""
    module = _module(eodt_yaml({"resolution": 640}))

    assert module.model.backbone.grid_size == (40, 40)


def test_num_l2_blocks_reaches_model(eodt_yaml: Callable[..., Path]) -> None:
    """With L2 = 4 in the YAML the queries enter block 8 instead of 9, and there is one more aux loss."""
    module = _module(eodt_yaml({**SMALL, "num_l2_blocks": 4}))
    model = module.model.eval()
    num_tokens = []
    hooks = [
        block.register_forward_pre_hook(lambda module, args: num_tokens.append(args[0].shape[1]))
        for block in model.backbone.vit.blocks
    ]

    with torch.no_grad():
        try:
            model(torch.randn(1, 3, 224, 224))
        finally:
            for hook in hooks:
                hook.remove()

    first_with_queries = next(i for i, n in enumerate(num_tokens) if n > num_tokens[0])
    assert first_with_queries == 8
    assert set(module.criterion.weight_dict) == BASE_WEIGHTS | _aux_weights(3)


def test_queries_and_classes_reach_model(eodt_yaml: Callable[..., Path]) -> None:
    """Number of queries and of classes in the YAML set the queries and the class logits."""
    module = _module(eodt_yaml({**SMALL, "num_queries": 100, "num_classes": 80}))

    assert module.model.query_feat.num_embeddings == 100
    assert module.model.class_embed.out_features == 81


def test_backbone_name_reaches_model(eodt_yaml: Callable[..., Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """The backbone in the YAML is the timm model that is created; ViT-S is created in its place, no download."""
    create_model = timm.create_model
    names = []

    def record_name(name: str, **kwargs: Any) -> nn.Module:
        names.append(name)
        return create_model("vit_small_patch14_reg4_dinov2", **kwargs)

    monkeypatch.setattr(timm, "create_model", record_name)
    _module(eodt_yaml({**SMALL, "backbone_name": "vit_large_patch14_reg4_dinov2", "num_l2_blocks": 4}))

    assert names == ["vit_large_patch14_reg4_dinov2"]


def test_learning_rates_reach_groups(eodt_yaml: Callable[..., Path]) -> None:
    """Learning rates, decays and weight decay of the YAML reach the optimizer groups."""
    train_changes = {
        "lr": 2.0e-4,
        "lr_encoder": 3.0e-4,
        "lr_vit_layer_decay": 0.9,
        "lr_component_decay": 0.5,
        "weight_decay": 0.05,
    }
    module = _module(eodt_yaml(SMALL, train_changes))
    groups = get_param_dict(_namespace_from_configs(module.model_config, module.train_config), module.model)
    names = {id(param): name for name, param in module.model.named_parameters()}
    by_name = {names[id(p)]: {k: v for k, v in g.items() if k != "params"} for g in groups for p in g["params"]}

    assert by_name["query_feat.weight"] == {"lr": 2e-4}
    assert by_name["backbone.vit.blocks.11.attn.qkv.weight"] == {
        "lr": pytest.approx(3e-4 * 0.9**2 * 0.5**2),
        "weight_decay": 0.05,
    }


def test_cls_loss_coef_reaches_criterion(eodt_yaml: Callable[..., Path]) -> None:
    """The classification coefficient of the YAML weights the final and the aux classification losses."""
    module = _module(eodt_yaml(SMALL, {"cls_loss_coef": 2.0}))

    assert module.criterion.weight_dict["loss_ce"] == 2.0
    assert module.criterion.weight_dict["loss_ce_1"] == 2.0


def test_batch_size_reaches_dataloader(eodt_yaml: Callable[..., Path]) -> None:
    """The batch size of the YAML is the number of images of a training batch."""
    model_config, train_config = load_configs(eodt_yaml(SMALL, {"batch_size": 3}))
    datamodule = RFDETRDataModule(model_config, train_config)
    datamodule.setup("fit")

    samples, targets = next(iter(datamodule.train_dataloader()))

    assert samples.tensors.shape == (3, 3, 224, 224)
    assert len(targets) == 3


@pytest.mark.parametrize(
    "train_changes",
    [{}, {"epochs": 3, "clip_max_norm": 0.5, "grad_accum_steps": 2, "ema_decay": 0.99}],
    ids=["recipe", "changed"],
)
def test_trainer_from_yaml(train_changes: dict[str, Any], eodt_yaml: Callable[..., Path]) -> None:
    """build_trainer takes epochs, clipping, accumulation and EMA from the YAML, with COCO evaluation."""
    model_config, train_config = load_configs(eodt_yaml(SMALL, train_changes))
    expected = {"epochs": 12, "clip_max_norm": 0.1, "grad_accum_steps": 1, "ema_decay": 0.993, **train_changes}

    trainer = build_trainer(train_config, model_config, accelerator="cpu")

    assert trainer.max_epochs == expected["epochs"]
    assert trainer.gradient_clip_val == expected["clip_max_norm"]
    assert trainer.accumulate_grad_batches == expected["grad_accum_steps"]
    ema = [callback for callback in trainer.callbacks if isinstance(callback, RFDETREMACallback)]
    assert len(ema) == 1 and ema[0]._decay == expected["ema_decay"]
    assert any(isinstance(callback, COCOEvalCallback) for callback in trainer.callbacks)
    assert any(isinstance(callback, BestModelCallback) for callback in trainer.callbacks)


def test_fit_from_yaml(eodt_yaml: Callable[..., Path]) -> None:
    """Two epochs of two steps run with the full RF-DETR training stack, from the YAML file.

    The warm-up of one epoch is two optimizer steps and the cosine takes the learning rate to 0 after four. The
    optimizer holds the layer-wise groups of step 4b, with the weight decay of the YAML for queries and head.
    """
    model_config, train_config = load_configs(
        eodt_yaml(SMALL, {"batch_size": 2, "epochs": 2, "tensorboard": False})
    )
    trainer = build_trainer(
        train_config, model_config, accelerator="cpu", limit_train_batches=2, limit_val_batches=1
    )
    torch.manual_seed(0)
    module = RFDETRModelModule(model_config, train_config)

    trainer.fit(module, RFDETRDataModule(model_config, train_config))

    assert trainer.state.finished
    assert trainer.global_step == 4
    assert math.isfinite(trainer.callback_metrics["train/loss"].item())
    assert "val/mAP_50_95" in trainer.callback_metrics
    scheduler_config = trainer.lr_scheduler_configs[0]
    lr_lambda = scheduler_config.scheduler.lr_lambdas[0]
    assert [lr_lambda(step) for step in range(5)] == pytest.approx([0.0, 0.5, 1.0, 0.5, 0.0], abs=1e-12)
    assert scheduler_config.interval == "step"
    assert scheduler_config.scheduler.last_epoch == 4  # stepped after each of the 4 optimizer steps
    optimizer = trainer.optimizers[0]
    names = {id(param): name for name, param in module.model.named_parameters()}
    groups = {names[id(param)]: group for group in optimizer.param_groups for param in group["params"]}
    assert (groups["query_feat.weight"]["initial_lr"], groups["query_feat.weight"]["weight_decay"]) == (1e-4, 1e-4)
    qkv = groups["backbone.vit.blocks.11.attn.qkv.weight"]
    assert qkv["initial_lr"] == pytest.approx(1.5e-4 * 0.8**2 * 0.7**2)
    assert groups["backbone.vit.pos_embed"]["weight_decay"] == 0.0
    assert all(group["lr"] == pytest.approx(0.0, abs=1e-12) for group in optimizer.param_groups)  # end of cosine
    assert (Path(train_config.output_dir) / "last.ckpt").exists()


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("backbone_name", "num_l2_blocks"),
    [("vit_base_patch14_reg4_dinov2", 3), ("vit_large_patch14_reg4_dinov2", 4)],
    ids=["vitb", "vitl"],
)
def test_module_from_yaml_sizes(
    backbone_name: str, num_l2_blocks: int, eodt_yaml: Callable[..., Path], coco_mini_batch: Callable
) -> None:
    """ViT-B and ViT-L come from the same YAML, changing only backbone and L2; forward and backward run."""
    path = eodt_yaml({**SMALL, "backbone_name": backbone_name, "num_l2_blocks": num_l2_blocks}, {"batch_size": 2})
    module = _module(path)
    samples, targets = coco_mini_batch(module.model_config, module.train_config)

    loss_dict = module.criterion(module.model.train()(samples, targets), targets)
    weight_dict = module.criterion.weight_dict
    loss = torch.stack([loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict]).sum()
    loss.backward()

    assert len(module.model.backbone.vit.blocks) - module.model.backbone.num_l1_blocks == num_l2_blocks
    assert torch.isfinite(loss)
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in module.model.parameters())
