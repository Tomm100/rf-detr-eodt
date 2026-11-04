# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the YAML of the overfit of the baseline on coco_mini (step 8c of EODT_SPEC.md §9).

The overfit itself runs with ``scripts/eodt/overfit.py`` on the GPU and is not part of the suite. These tests check
that it is the overfit of step 5 applied to the baseline without windows:
- the model is the one of ``baseline_medium_nowin.yaml``;
- the training configuration is the one of the overfit of EoDT but output directory and run, so the overfit keys
  differ from the baseline's recipe exactly as they differ from EoDT's.

The data pipeline (train and val see the same 20 images, one epoch sees each once) is the one of the overfit of EoDT,
checked by ``test_overfit_yaml.py``: it depends only on these training values and on the resolution, 576 px in both.
"""

from pathlib import Path
from typing import Any

import yaml

from rfdetr.models.eodt import BaselineConfig, load_configs

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
NOWIN_YAML = CONFIGS / "baseline" / "baseline_medium_nowin.yaml"
OVERFIT_YAML = CONFIGS / "baseline" / "baseline_medium_nowin_overfit.yaml"
EODT_YAML = CONFIGS / "eodt" / "eodt_vits_576_l2-3_learned.yaml"
EODT_OVERFIT_YAML = CONFIGS / "eodt" / "eodt_vits_576_l2-3_learned_overfit.yaml"
NAMES = {
    "output_dir": "C:/dev/tesi/rf-detr-eodt/output/baseline_medium_nowin_overfit",
    "run": "baseline_medium_nowin_overfit",
}


def _read(path: Path) -> dict[str, Any]:
    """Read a YAML file as plain data."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _changed(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Values of ``after`` that differ from, or are absent in, ``before``."""
    return {key: value for key, value in after.items() if before.get(key, object()) != value}


def test_model_is_the_baseline_without_windows() -> None:
    """The model of the overfit is the one of the run without windows, with 576 px and Group-DETR off."""
    assert _read(OVERFIT_YAML)["model"]["model_config"] == _read(NOWIN_YAML)["model"]["model_config"]

    model_config, train_config = load_configs(OVERFIT_YAML)
    assert type(model_config) is BaselineConfig
    assert (model_config.resolution, model_config.num_windows, model_config.group_detr) == (576, 1, 1)
    assert train_config.amp_dtype is None  # fp32 on the GTX 1650


def test_training_is_the_overfit_of_eodt() -> None:
    """The training configuration is the one of the overfit of EoDT, but output directory and run."""
    overfit = _read(OVERFIT_YAML)["model"]["train_config"]
    eodt_overfit = _read(EODT_OVERFIT_YAML)["model"]["train_config"]
    overfit_args, eodt_overfit_args = overfit.pop("init_args"), eodt_overfit.pop("init_args")

    assert overfit == eodt_overfit  # class path
    assert set(overfit_args) == set(eodt_overfit_args)
    assert _changed(eodt_overfit_args, overfit_args) == NAMES


def test_overfit_keys_are_those_of_eodt() -> None:
    """The overfit changes the baseline's recipe in the same keys and values in which it changes EoDT's."""
    baseline_run = _read(NOWIN_YAML)["model"]["train_config"]["init_args"]
    baseline_overfit = _read(OVERFIT_YAML)["model"]["train_config"]["init_args"]
    eodt_run = _read(EODT_YAML)["model"]["train_config"]["init_args"]
    eodt_overfit = _read(EODT_OVERFIT_YAML)["model"]["train_config"]["init_args"]

    assert _changed(baseline_run, baseline_overfit) == {**_changed(eodt_run, eodt_overfit), **NAMES}
