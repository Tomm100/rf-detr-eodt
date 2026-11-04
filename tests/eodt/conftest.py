# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Fixtures shared by the EoDT tests: coco_mini read through RF-DETR's data module, and copies of the EoDT YAML."""

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import Tensor

from rfdetr.config import TrainConfig
from rfdetr.models.eodt import EoDTConfig
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.utilities.tensors import NestedTensor

# 20 val2017 images with 2-10 objects each, the same in train and val (see CLAUDE.md)
COCO_MINI = Path("C:/dev/tesi/data/coco_mini")
# configuration of the first run on Leonardo (step 6)
EODT_YAML = Path(__file__).resolve().parents[2] / "configs" / "eodt" / "eodt_vits_576_l2-3_learned.yaml"


@pytest.fixture
def eodt_yaml(tmp_path: Path) -> Callable[..., Path]:
    """Return a writer of copies of the EoDT YAML that run on this PC, with the given changes.

    Paths and hardware of Leonardo are replaced by coco_mini, ``tmp_path``, one device and no worker process, and
    W&B is off unless a test turns it on; the changes of the test go into ``init_args``. Every change thus goes
    through a file and the loader.
    """
    if not COCO_MINI.exists():
        pytest.skip(f"{COCO_MINI} not found")
    counter = itertools.count()

    def write(model_changes: dict[str, Any] | None = None, train_changes: dict[str, Any] | None = None) -> Path:
        content = yaml.safe_load(EODT_YAML.read_text(encoding="utf-8"))
        content["model"]["model_config"]["init_args"].update(model_changes or {})
        content["model"]["train_config"]["init_args"].update(
            {
                "dataset_dir": str(COCO_MINI),
                "output_dir": str(tmp_path / "output"),
                "devices": 1,
                "strategy": "auto",
                "num_workers": 0,
                "wandb": False,
                **(train_changes or {}),
            }
        )
        path = tmp_path / f"config_{next(counter)}.yaml"
        path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
        return path

    return write


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
