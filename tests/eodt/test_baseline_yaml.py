# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the YAML files of the baseline in ``configs/baseline`` (step 8b of EODT_SPEC.md §9, §5, §3.7).

Covers:
- Every value written in a file reaches the configurations unchanged.
- The three rings of the chain differ only where they must: ``num_windows`` (without windows), then ``projector`` and
  ``out_feature_indexes`` (without projector).
- The ring with windows is RF-DETR Medium in every architecture field but the registers, Group-DETR and the RF-DETR
  checkpoint.
- The recipe is EoDT's: the ``train_config`` of every file is the one of the first EoDT run but ``output_dir`` and
  ``run``, both named after the file (the W&B run id of ``scripts/leonardo/train.sh``).
- ``write_config`` -> ``load_configs`` gives back the same configurations, as when a job resumes a run.
- A ViT-L that keeps the blocks of ViT-S is rejected when the file is read.
"""

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from rfdetr.config import RFDETRMediumConfig, TrainConfig
from rfdetr.models.eodt import BaselineConfig, load_configs
from rfdetr.models.eodt.train import write_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
EODT_YAML = CONFIGS / "eodt" / "eodt_vits_576_l2-3_learned.yaml"
RINGS = ["win", "nowin", "nowin_noproj"]
RUNS = "/leonardo_work/IscrC_EoDT/tcentonz/runs"


def _path(ring: str) -> Path:
    """YAML file of a ring of the chain."""
    return CONFIGS / "baseline" / f"baseline_medium_{ring}.yaml"


def _read(path: Path) -> dict[str, Any]:
    """Read a YAML file as plain data."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("ring", RINGS)
def test_values_reach_configurations(ring: str) -> None:
    """Every value written in the file is the value of the configurations."""
    model_config, train_config = load_configs(_path(ring))

    assert type(model_config) is BaselineConfig
    assert type(train_config) is TrainConfig
    content = _read(_path(ring))
    model_args = content["model"]["model_config"]["init_args"]
    train_args = content["model"]["train_config"]["init_args"]
    # TrainConfig normalizes the two directories with os.path.realpath (on Windows it adds the drive letter)
    for name in ("dataset_dir", "output_dir"):
        train_args[name] = os.path.realpath(train_args[name])
    for config, args in ((model_config, model_args), (train_config, train_args)):
        for name, value in args.items():
            assert getattr(config, name) == value, name


@pytest.mark.parametrize(
    ("before", "after", "changes"),
    [
        ("win", "nowin", {"num_windows": (2, 1)}),
        ("nowin", "nowin_noproj", {"projector": (True, False), "out_feature_indexes": ([3, 6, 9, 12], [12])}),
    ],
    ids=["without-windows", "without-projector"],
)
def test_chain_changes_only_listed_keys(before: str, after: str, changes: dict[str, tuple[Any, Any]]) -> None:
    """Each ring of the chain removes one component of the ring before, and changes nothing else in the model."""
    before_model = _read(_path(before))["model"]["model_config"]
    after_model = _read(_path(after))["model"]["model_config"]
    before_args, after_args = before_model.pop("init_args"), after_model.pop("init_args")

    assert after_model == before_model  # class path
    assert set(after_args) == set(before_args)
    assert {key: (before_args[key], value) for key, value in after_args.items() if before_args[key] != value} == changes


def test_with_windows_is_rfdetr_medium() -> None:
    """The ring with windows has every field of RF-DETR Medium but the registers, Group-DETR and the checkpoint."""
    baseline, _ = load_configs(_path("win"))
    medium = RFDETRMediumConfig()

    differences = {
        name: (getattr(baseline, name), getattr(medium, name))
        for name in RFDETRMediumConfig.model_fields
        if getattr(baseline, name) != getattr(medium, name)
    }

    assert differences == {
        "encoder": ("dinov2_registers_windowed_small", "dinov2_windowed_small"),
        "group_detr": (1, 13),
        "pretrain_weights": (None, medium.pretrain_weights),
    }
    assert set(BaselineConfig.model_fields) - set(RFDETRMediumConfig.model_fields) == {"projector"}
    assert baseline.projector


@pytest.mark.parametrize("ring", RINGS)
def test_recipe_is_eodt(ring: str) -> None:
    """The training configuration is the one of the first EoDT run, but output directory and W&B run."""
    eodt_train = _read(EODT_YAML)["model"]["train_config"]
    train = _read(_path(ring))["model"]["train_config"]
    eodt_args, args = eodt_train.pop("init_args"), train.pop("init_args")

    assert train == eodt_train  # class path
    assert set(args) == set(eodt_args)
    name = _path(ring).stem
    assert {key: value for key, value in args.items() if eodt_args[key] != value} == {
        "output_dir": f"{RUNS}/{name}",
        "run": name,
    }


@pytest.mark.parametrize("ring", RINGS)
def test_write_config_round_trip(ring: str, tmp_path: Path) -> None:
    """The full configuration that a run saves in ``run_info/`` loads back identical, as when a job resumes."""
    model_config, train_config = load_configs(_path(ring))
    saved = tmp_path / "config_resolved.yaml"

    write_config(saved, model_config, train_config)
    saved_model_config, saved_train_config = load_configs(saved)

    assert type(saved_model_config) is BaselineConfig
    assert saved_model_config.model_dump() == model_config.model_dump()
    assert saved_train_config.model_dump() == train_config.model_dump()


def test_vitl_with_vits_blocks_is_rejected(tmp_path: Path) -> None:
    """A YAML moved to ViT-L that keeps the blocks [3, 6, 9, 12] of ViT-S does not load (ViT-L has 24 blocks)."""
    content = _read(_path("nowin"))
    content["model"]["model_config"]["init_args"]["encoder"] = "dinov2_registers_windowed_large"
    path = tmp_path / "baseline_vitl.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValidationError, match=r"out_feature_indexes \(\[3, 6, 9, 12\]\).*DINOv2 large \(24\)"):
        load_configs(path)
