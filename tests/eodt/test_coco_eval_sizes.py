# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the AP per object size logged by ``COCOEvalCallback`` (step 6a of EODT_SPEC.md §9, W&B of §7.3).

The validation runs on coco_mini with a model that returns the ground truth itself: every AP must be 1.
"""

from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from rfdetr.models.eodt import load_configs
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.trainer import build_trainer


class GroundTruthModel(nn.Module):
    """Return, batch after batch, the ground truth as predictions: query j is object j, with logit +10.

    Args:
        targets: Targets of the validation batches, in the order of the sequential validation loader.
        num_queries: Number of predictions per image.
        num_classes: Number of class logits.
    """

    def __init__(self, targets: list[list[dict[str, Tensor]]], num_queries: int, num_classes: int) -> None:
        super().__init__()
        self.outputs: deque[dict[str, Tensor]] = deque()
        for batch in targets:
            logits = torch.full((len(batch), num_queries, num_classes), -10.0)
            boxes = torch.tensor([0.5, 0.5, 0.1, 0.1]).repeat(len(batch), num_queries, 1)
            for image, target in enumerate(batch):
                num_objects = len(target["labels"])
                logits[image, torch.arange(num_objects), target["labels"]] = 10.0
                boxes[image, :num_objects] = target["boxes"]
            self.outputs.append({"pred_logits": logits, "pred_boxes": boxes})

    def forward(self, samples: Any, targets: Any = None) -> dict[str, Tensor]:
        return self.outputs.popleft()


def test_ap_per_size_is_logged(eodt_yaml: Callable[..., Path]) -> None:
    """Validation logs AP small, medium and large next to mAP 50:95; with the ground truth all of them are 1."""
    model_config, train_config = load_configs(eodt_yaml({"resolution": 224}, {"use_ema": False}))
    datamodule = RFDETRDataModule(model_config, train_config)
    datamodule.setup("validate")
    targets = [
        datamodule.transfer_batch_to_device(batch, torch.device("cpu"), 0)[1] for batch in datamodule.val_dataloader()
    ]
    module = RFDETRModelModule(model_config, train_config)
    module.model = GroundTruthModel(targets, model_config.num_queries, module.model.class_embed.out_features)
    trainer = build_trainer(train_config, model_config, accelerator="cpu", include_training_callbacks=False)

    metrics = trainer.validate(module, datamodule, verbose=False)[0]

    for key in ("val/mAP_50_95", "val/mAP_small", "val/mAP_medium", "val/mAP_large"):
        assert metrics[key] == 1.0, key
