# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Fixtures shared by the EoDT tests: coco_mini read through RF-DETR's data module."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from rfdetr.config import TrainConfig
from rfdetr.models.eodt import EoDTConfig
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.utilities.tensors import NestedTensor

# 20 val2017 images with 2-10 objects each, the same in train and val (see CLAUDE.md)
COCO_MINI = Path("C:/dev/tesi/data/coco_mini")


@pytest.fixture
def coco_mini_train_config(tmp_path: Path) -> Callable[..., TrainConfig]:
    """Return a factory of training configurations that read coco_mini with a fixed square input."""
    if not COCO_MINI.exists():
        pytest.skip(f"{COCO_MINI} not found")

    def make(**overrides: Any) -> TrainConfig:
        return TrainConfig(
            **{
                "dataset_dir": str(COCO_MINI),
                "dataset_file": "coco",
                "output_dir": str(tmp_path),
                "multi_scale": "off",
                "augmentation_backend": "torchvision",
                "eval_batch_size": 2,
                "num_workers": 0,
                **overrides,
            }
        )

    return make


@pytest.fixture
def coco_mini_batch(
    coco_mini_train_config: Callable[..., TrainConfig],
) -> Callable[..., tuple[NestedTensor, list[dict[str, Tensor]]]]:
    """Return a loader of the first two coco_mini val images, through RF-DETR's data module and collate.

    The val split has no random augmentation. The batch is built again at every call, so no test sees targets
    modified by another one.
    """

    def load(
        model_config: EoDTConfig, train_config: TrainConfig | None = None
    ) -> tuple[NestedTensor, list[dict[str, Tensor]]]:
        datamodule = RFDETRDataModule(model_config, train_config or coco_mini_train_config())
        datamodule.setup("validate")
        batch = next(iter(datamodule.val_dataloader()))
        # as Lightning does before training_step: the collated PackedTargets become a list of dicts
        return datamodule.transfer_batch_to_device(batch, torch.device("cpu"), 0)

    return load
