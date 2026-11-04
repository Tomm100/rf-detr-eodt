# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the YAML of the overfit on coco_mini (step 5 of EODT_SPEC.md §9).

The overfit itself runs with ``scripts/eodt/overfit.py`` on the GPU and is not part of the suite. These tests check
its configuration on CPU:
- it differs from the YAML of the first run only in the keys needed to run locally and without augmentation;
- train and val read the 20 images in exactly the same way, so the model is evaluated on what it memorized;
- one training epoch sees every image exactly once.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from rfdetr.models.eodt import load_configs
from rfdetr.training.module_data import RFDETRDataModule

COCO_MINI = Path("C:/dev/tesi/data/coco_mini")
CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "eodt"
RUN_YAML = CONFIGS / "eodt_vits_576_l2-3_learned.yaml"  # first run on Leonardo (step 6)
OVERFIT_YAML = CONFIGS / "eodt_vits_576_l2-3_learned_overfit.yaml"
# train_config values of the overfit that differ from, or are absent in, the YAML of the first run
OVERFIT_CHANGES = {
    "dataset_dir": "C:/dev/tesi/data/coco_mini",
    "output_dir": "C:/dev/tesi/rf-detr-eodt/output/eodt_vits_576_l2-3_learned_overfit_2400steps",
    "aug_config": {},
    "scale_jitter": False,
    "warmup_epochs": 40,
    "epochs": 480,
    "amp_dtype": None,
    "eval_base_model": True,
    "eval_interval": 4,
    "checkpoint_interval": 480,
    "wandb": False,
    "run": "eodt_vits_576_l2-3_learned_overfit",
    "devices": 1,
    "strategy": "auto",
    "num_workers": 0,
}
NUM_IMAGES = 20


def _read(path: Path) -> dict[str, Any]:
    """Read a YAML file as plain data."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def datamodule() -> RFDETRDataModule:
    """Data module of the overfit YAML, with the train and val datasets built."""
    if not COCO_MINI.exists():
        pytest.skip(f"{COCO_MINI} not found")
    datamodule = RFDETRDataModule(*load_configs(OVERFIT_YAML))
    datamodule.setup("fit")
    return datamodule


def test_overfit_yaml_changes_only_listed_keys() -> None:
    """Model and recipe are those of the first run: only data, output, hardware, length and augmentation change."""
    run, overfit = _read(RUN_YAML), _read(OVERFIT_YAML)
    run_train = run["model"]["train_config"].pop("init_args")
    overfit_train = overfit["model"]["train_config"].pop("init_args")

    assert overfit == run  # class paths and the whole model_config
    assert set(run_train) <= set(overfit_train)
    assert {k: v for k, v in overfit_train.items() if run_train.get(k, object()) != v} == OVERFIT_CHANGES

    model_config, train_config = load_configs(OVERFIT_YAML)
    assert model_config.resolution == 576
    assert train_config.amp_dtype is None  # fp32
    assert train_config.aug_config == {}


def test_train_and_val_see_same_images(datamodule: RFDETRDataModule) -> None:
    """Every training sample equals the validation sample of the same image: same pixels, same targets."""
    train_dataset = datamodule.train_dataloader().dataset
    val_dataset = datamodule.val_dataloader().dataset

    assert len(train_dataset) == len(val_dataset) == NUM_IMAGES
    for index in range(NUM_IMAGES):
        train_image, train_target = train_dataset[index]
        val_image, val_target = val_dataset[index]
        assert train_image.shape == (3, 576, 576)
        assert torch.equal(train_image, val_image)
        assert train_target.keys() == val_target.keys()
        for key in train_target:
            assert torch.equal(torch.as_tensor(train_target[key]), torch.as_tensor(val_target[key])), key


def test_train_loader_sees_each_image_once(datamodule: RFDETRDataModule) -> None:
    """One epoch is 5 batches of 4 that hold the 20 images once each, sampled without replacement."""
    image_ids = []
    for batch in datamodule.train_dataloader():
        samples, targets = datamodule.transfer_batch_to_device(batch, torch.device("cpu"), 0)
        assert samples.tensors.shape == (4, 3, 576, 576)
        image_ids += [int(target["image_id"]) for target in targets]

    assert len(image_ids) == NUM_IMAGES
    assert sorted(image_ids) == sorted(datamodule.val_dataloader().dataset.ids)
