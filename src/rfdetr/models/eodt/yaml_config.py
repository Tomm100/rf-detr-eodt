# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Read an experiment YAML file in the format of RF-DETR's ``configs/`` (EODT_SPEC.md §3.7).

The file holds the model and the training configuration, each as a ``class_path`` and its ``init_args``::

    model:
      model_config:
        class_path: rfdetr.models.eodt.EoDTConfig
        init_args: {...}
      train_config:
        class_path: rfdetr.config.TrainConfig
        init_args: {...}
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, TypeVar

import yaml

from rfdetr.config import BaseConfig, ModelConfig, TrainConfig

_ConfigT = TypeVar("_ConfigT", bound=BaseConfig)


class _UniqueKeyLoader(yaml.SafeLoader):
    """``SafeLoader`` that rejects a key written twice in a mapping, instead of keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        keys = [self.construct_object(key_node, deep=deep) for key_node, _ in node.value]
        duplicates = sorted({str(key) for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate key(s) {duplicates} in the mapping at line {node.start_mark.line + 1}")
        return super().construct_mapping(node, deep=deep)


def _check_keys(mapping: Any, expected: set[str], where: str, required: set[str] | None = None) -> None:
    """Reject a missing key or one that nobody would read."""
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a mapping, got {type(mapping).__name__}")
    unknown = set(mapping) - expected
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {sorted(unknown)}; expected {sorted(expected)}")
    missing = (expected if required is None else required) - set(mapping)
    if missing:
        raise ValueError(f"missing key(s) in {where}: {sorted(missing)}")


def _build_config(section: Any, base: type[_ConfigT], where: str) -> _ConfigT:
    """Import ``class_path``, check that it is a ``base`` subclass and build it from ``init_args``."""
    _check_keys(section, {"class_path", "init_args"}, where, required={"class_path"})
    module_name, _, class_name = section["class_path"].rpartition(".")
    config_class = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(config_class, type) and issubclass(config_class, base)):
        raise TypeError(f"{where}.class_path must be a {base.__name__}, got {section['class_path']}")
    return config_class(**(section.get("init_args") or {}))


def load_configs(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    """Read a YAML file in the format of ``configs/`` and build its model and training configurations.

    Args:
        path: The YAML file.

    Returns:
        The model configuration (e.g. ``EoDTConfig``) and the training configuration.

    Raises:
        ValueError: If a section or a key is missing, unknown or written twice, so that nothing in the file is
            ignored.
        TypeError: If a ``class_path`` is not a ``ModelConfig`` or ``TrainConfig`` subclass.
    """
    content = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)  # a SafeLoader
    _check_keys(content, {"model"}, "the file")
    _check_keys(content["model"], {"model_config", "train_config"}, "model")
    model_config = _build_config(content["model"]["model_config"], ModelConfig, "model.model_config")
    train_config = _build_config(content["model"]["train_config"], TrainConfig, "model.train_config")
    return model_config, train_config
