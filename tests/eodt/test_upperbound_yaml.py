# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the YAML files of the upper bound: RF-DETR 2XL and Max of Table 7 with our training (§9, step 11 (3)).

Each run keeps the architecture of the latency benchmark's YAML of Table 7 and takes the training of the baseline
(``baseline_medium_win_lrcd1.yaml``): these tests check that the model fields, defaults included, are those of the
Table 7 file, with the projector written out and pretrained DINOv2 instead of random weights, and that the training
differs from the baseline's only in output directory and W&B run. The test jobs are checked by
``test_leonardo_files.py``, and the W&B run name of every YAML too.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from rfdetr.models.eodt import UpperBoundConfig, load_configs

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "baseline"
RUNS = "/leonardo_work/IscrC_EoDT/tcentonz/runs"
RECIPE = "baseline_medium_win_lrcd1.yaml"
SIZES = ["2xl", "max"]


def _read(name: str) -> dict[str, Any]:
    """Read a YAML file of ``configs/baseline`` as plain data."""
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("size", SIZES)
def test_architecture_of_table7(size: str) -> None:
    """The model is the one of the Table 7 file: same fields and defaults, plus the projector, minus random weights."""
    table7, upperbound = f"rfdetr_table7_{size}.yaml", f"rfdetr_table7_{size}_upperbound.yaml"
    table7_args = _read(table7)["model"]["model_config"]["init_args"]
    upperbound_model = _read(upperbound)["model"]["model_config"]

    assert upperbound_model["class_path"] == "rfdetr.models.eodt.UpperBoundConfig"
    assert table7_args.pop("pretrain_weights") is None  # random weights in the benchmark, DINOv2 here
    assert upperbound_model["init_args"] == {**table7_args, "projector": True}

    table7_config, _ = load_configs(CONFIGS / table7)
    upperbound_config, _ = load_configs(CONFIGS / upperbound)
    assert type(upperbound_config) is UpperBoundConfig
    assert not upperbound_config.registers and upperbound_config.group_detr == 1
    benchmark_fields = table7_config.model_dump()
    assert {name: upperbound_config.model_dump()[name] for name in benchmark_fields} == benchmark_fields


@pytest.mark.parametrize("encoder", ["dinov2_registers_windowed_small", "dinov2_registers_windowed_base"])
def test_rejects_encoder_with_registers(encoder: str) -> None:
    """The upper bound takes only DINOv2-B without registers, the encoder of Table 7: the W&B tag cannot mark another.

    The same arguments with the encoder of the YAML file are accepted, so the error is the encoder's.
    """
    arguments = {**_read("rfdetr_table7_2xl_upperbound.yaml")["model"]["model_config"]["init_args"], "device": "cpu"}
    UpperBoundConfig(**arguments)

    with pytest.raises(ValidationError, match="encoder"):
        UpperBoundConfig(**{**arguments, "encoder": encoder})


@pytest.mark.parametrize("size", SIZES)
def test_training_of_the_baseline(size: str) -> None:
    """The training is the baseline's of step 11 (1): only the output directory and the W&B run differ."""
    name = f"rfdetr_table7_{size}_upperbound.yaml"
    recipe = _read(RECIPE)["model"]["train_config"]
    train = _read(name)["model"]["train_config"]
    run = Path(name).stem

    assert train["class_path"] == recipe["class_path"]
    assert set(train["init_args"]) == set(recipe["init_args"])
    changes = {key: value for key, value in train["init_args"].items() if recipe["init_args"][key] != value}
    assert changes == {"output_dir": f"{RUNS}/{run}", "run": run}
