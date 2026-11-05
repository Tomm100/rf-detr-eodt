# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the YAML files of the overfit of the baseline on coco_mini (steps 8c and 8d of EODT_SPEC.md §9).

The overfits themselves run with ``scripts/eodt/overfit.py`` on the GPU and are not part of the suite. These tests
check that each is the overfit of step 5 applied to a ring of the baseline: without windows (step 8c) and with
windows, the baseline of step 10 (step 8d):
- the model is the one of the run of the same ring (``baseline_medium_nowin.yaml``, ``baseline_medium_win.yaml``);
- the training configuration is the one of the overfit of EoDT but output directory and run, so the overfit keys
  differ from the baseline's recipe exactly as they differ from EoDT's.

The data pipeline (train and val see the same 20 images, one epoch sees each once) is the one of the overfit of EoDT,
checked by ``test_overfit_yaml.py``: it depends only on these training values and on the resolution, 576 px in all.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from rfdetr.models.eodt import BaselineConfig, load_configs

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
EODT_YAML = CONFIGS / "eodt" / "eodt_vits_576_l2-3_learned.yaml"
EODT_OVERFIT_YAML = CONFIGS / "eodt" / "eodt_vits_576_l2-3_learned_overfit.yaml"
# ring of the chain and its number of windows
RINGS = [pytest.param("nowin", 1, id="nowin"), pytest.param("win", 2, id="win")]


def _run_yaml(ring: str) -> Path:
    """YAML file of the run of a ring of the chain."""
    return CONFIGS / "baseline" / f"baseline_medium_{ring}.yaml"


def _overfit_yaml(ring: str) -> Path:
    """YAML file of the overfit of a ring of the chain."""
    return CONFIGS / "baseline" / f"baseline_medium_{ring}_overfit.yaml"


def _names(ring: str) -> dict[str, str]:
    """Output directory and W&B run of the overfit of a ring, the only keys that differ from the overfit of EoDT."""
    name = _overfit_yaml(ring).stem
    return {"output_dir": f"C:/dev/tesi/rf-detr-eodt/output/{name}", "run": name}


def _read(path: Path) -> dict[str, Any]:
    """Read a YAML file as plain data."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _changed(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Values of ``after`` that differ from, or are absent in, ``before``."""
    return {key: value for key, value in after.items() if before.get(key, object()) != value}


@pytest.mark.parametrize(("ring", "num_windows"), RINGS)
def test_model_is_the_baseline_of_the_ring(ring: str, num_windows: int) -> None:
    """The model of the overfit is the one of the run of the same ring, with 576 px and Group-DETR off."""
    assert _read(_overfit_yaml(ring))["model"]["model_config"] == _read(_run_yaml(ring))["model"]["model_config"]

    model_config, train_config = load_configs(_overfit_yaml(ring))
    assert type(model_config) is BaselineConfig
    assert (model_config.resolution, model_config.num_windows, model_config.group_detr) == (576, num_windows, 1)
    assert train_config.amp_dtype is None  # fp32 on the GTX 1650


@pytest.mark.parametrize(("ring", "num_windows"), RINGS)
def test_training_is_the_overfit_of_eodt(ring: str, num_windows: int) -> None:
    """The training configuration is the one of the overfit of EoDT, but output directory and run."""
    overfit = _read(_overfit_yaml(ring))["model"]["train_config"]
    eodt_overfit = _read(EODT_OVERFIT_YAML)["model"]["train_config"]
    overfit_args, eodt_overfit_args = overfit.pop("init_args"), eodt_overfit.pop("init_args")

    assert overfit == eodt_overfit  # class path
    assert set(overfit_args) == set(eodt_overfit_args)
    assert _changed(eodt_overfit_args, overfit_args) == _names(ring)


@pytest.mark.parametrize(("ring", "num_windows"), RINGS)
def test_overfit_keys_are_those_of_eodt(ring: str, num_windows: int) -> None:
    """The overfit changes the baseline's recipe in the same keys and values in which it changes EoDT's."""
    baseline_run = _read(_run_yaml(ring))["model"]["train_config"]["init_args"]
    baseline_overfit = _read(_overfit_yaml(ring))["model"]["train_config"]["init_args"]
    eodt_run = _read(EODT_YAML)["model"]["train_config"]["init_args"]
    eodt_overfit = _read(EODT_OVERFIT_YAML)["model"]["train_config"]["init_args"]

    assert _changed(baseline_run, baseline_overfit) == {**_changed(eodt_run, eodt_overfit), **_names(ring)}
