# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for rfdetr.models.eodt.load_configs and the EoDT YAML of the first run (step 4c of EODT_SPEC.md §9).

Covers:
- The YAML in ``configs/eodt/`` loads, and every value it sets reaches the configurations unchanged.
- It holds the defaults of §3 and the starting recipe of §7.0 and §7.1.
- Its hardware section, as written for Leonardo, gives DDP over 4 devices with 16 images per step.
- RF-DETR's own YAML files load with the same loader (the baseline will use it, §3.7).
- Nothing in a file is ignored: unknown, missing or duplicate keys, wrong classes and invalid values are errors.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from pytorch_lightning.strategies import DDPStrategy

from rfdetr.config import RFDETRMediumConfig, TrainConfig
from rfdetr.models.eodt import EoDTConfig, load_configs
from rfdetr.training.trainer import build_trainer

EODT_YAML = Path(__file__).resolve().parents[2] / "configs" / "eodt" / "eodt_vits_576_l2-3_learned.yaml"
RFDETR_YAML = EODT_YAML.parents[1] / "rfdetr_medium.yaml"

# paths of the sections in the loaded YAML
MODEL = ["model", "model_config"]
TRAIN = ["model", "train_config"]
MODEL_ARGS = [*MODEL, "init_args"]
TRAIN_ARGS = [*TRAIN, "init_args"]


def _load_yaml() -> dict[str, Any]:
    """Return the content of the EoDT YAML file."""
    return yaml.safe_load(EODT_YAML.read_text(encoding="utf-8"))


def _set(path: list[str], value: Any) -> Callable[[dict], None]:
    """Return a change that sets ``value`` at ``path`` in the loaded YAML."""

    def change(content: dict) -> None:
        node = content
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return change


def _delete(path: list[str]) -> Callable[[dict], None]:
    """Return a change that deletes the key at ``path`` in the loaded YAML."""

    def change(content: dict) -> None:
        node = content
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return change


def test_load_eodt_yaml() -> None:
    """Every value written in the file is the value of the configurations."""
    model_config, train_config = load_configs(EODT_YAML)

    assert type(model_config) is EoDTConfig
    assert type(train_config) is TrainConfig
    content = _load_yaml()
    model_args = content["model"]["model_config"]["init_args"]
    train_args = content["model"]["train_config"]["init_args"]
    # TrainConfig normalizes the two directories with os.path.realpath (on Windows it adds the drive letter)
    for name in ("dataset_dir", "output_dir"):
        train_args[name] = os.path.realpath(train_args[name])
    for config, args in ((model_config, model_args), (train_config, train_args)):
        for name, value in args.items():
            assert getattr(config, name) == value, name


def test_starting_recipe() -> None:
    """The first run uses the defaults of §3 and the starting recipe of §7.0 and §7.1."""
    model_config, train_config = load_configs(EODT_YAML)

    assert model_config.backbone_name == "vit_small_patch14_reg4_dinov2"
    assert (model_config.resolution, model_config.patch_size, model_config.num_l2_blocks) == (576, 16, 3)
    assert (model_config.num_queries, model_config.query_mode, model_config.group_detr) == (300, "learned", 1)
    assert model_config.num_classes == 90

    assert train_config.dataset_file == "coco"
    assert (train_config.multi_scale, train_config.square_resize_div_64) == ("off", True)
    assert train_config.augmentation_backend == "torchvision"
    assert (train_config.lr, train_config.lr_encoder, train_config.weight_decay) == (1e-4, 1.5e-4, 1e-4)
    assert (train_config.lr_vit_layer_decay, train_config.lr_component_decay) == (0.8, 0.7)
    assert (train_config.clip_max_norm, train_config.cls_loss_coef) == (0.1, 1.0)
    assert (train_config.lr_scheduler, train_config.warmup_epochs, train_config.epochs) == ("cosine", 1, 12)
    assert (train_config.use_ema, train_config.ema_decay, train_config.amp_dtype) == (True, 0.993, "bf16")
    assert (train_config.seed, train_config.eval_max_dets) == (0, 100)
    total_batch = train_config.batch_size * train_config.devices * train_config.num_nodes
    assert total_batch * train_config.grad_accum_steps == 16


def test_leonardo_hardware() -> None:
    """The hardware section as written for Leonardo gives DDP over 4 devices and 16 images per optimizer step."""
    model_config, train_config = load_configs(EODT_YAML)

    # no training callbacks: nothing is written to the Leonardo output directory
    trainer = build_trainer(train_config, model_config, accelerator="cpu", include_training_callbacks=False)

    assert isinstance(trainer.strategy, DDPStrategy)
    assert (trainer.num_devices, trainer.num_nodes, trainer.world_size) == (4, 1, 4)
    assert trainer.world_size * train_config.batch_size * trainer.accumulate_grad_batches == 16


def test_load_rfdetr_yaml() -> None:
    """RF-DETR's own YAML files use the same format and load with the same loader."""
    model_config, train_config = load_configs(RFDETR_YAML)

    assert type(model_config) is RFDETRMediumConfig
    assert type(train_config) is TrainConfig


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        pytest.param(_set(["trainer"], {"max_epochs": 3}), ValueError, r"\['trainer'\]", id="unknown-section"),
        pytest.param(_set(["model", "data_config"], {}), ValueError, r"\['data_config'\]", id="unknown-model-key"),
        pytest.param(_set([*TRAIN, "init_arg"], {}), ValueError, r"\['init_arg'\]", id="unknown-section-key"),
        pytest.param(_delete([*MODEL, "class_path"]), ValueError, r"missing .*\['class_path'\]", id="no-class-path"),
        pytest.param(
            _set([*MODEL, "class_path"], "rfdetr.config.TrainConfig"), TypeError, "ModelConfig", id="wrong-class"
        ),
        pytest.param(
            _set([*MODEL, "class_path"], "rfdetr.models.eodt.NoSuchConfig"), AttributeError, "NoSuchConfig",
            id="missing-class",
        ),
        pytest.param(_set([*TRAIN_ARGS, "epoch"], 12), ValidationError, r"Unknown parameter\(s\): 'epoch'", id="typo"),
        pytest.param(_delete([*MODEL_ARGS, "num_l2_blocks"]), ValidationError, "num_l2_blocks", id="missing-l2"),
        pytest.param(_set([*MODEL_ARGS, "query_mode"], "selection"), ValidationError, "query_mode", id="selection"),
        pytest.param(_set([*MODEL_ARGS, "group_detr"], 2), ValidationError, "group_detr", id="groups"),
    ],
)
def test_rejects_invalid_yaml(
    change: Callable[[dict], None], error: type[Exception], match: str, tmp_path: Path
) -> None:
    """Nothing in the file is silently ignored or replaced."""
    content = _load_yaml()
    change(content)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")

    with pytest.raises(error, match=match):
        load_configs(path)


def test_rejects_duplicate_key(tmp_path: Path) -> None:
    """A key written twice is an error, not the last value silently winning."""
    text = EODT_YAML.read_text(encoding="utf-8")
    assert text.count("      epochs: 12\n") == 1
    path = tmp_path / "duplicate.yaml"
    path.write_text(text.replace("      epochs: 12\n", "      epochs: 12\n      epochs: 24\n"), encoding="utf-8")

    with pytest.raises(ValueError, match=r"duplicate key\(s\) \['epochs'\]"):
        load_configs(path)
