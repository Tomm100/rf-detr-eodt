# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the YAML files with ``lr_component_decay: 1.0`` (step 11 (1) of EODT_SPEC.md §9).

The two runs of step 10 (EoDT selection and the baseline with windows) are repeated on ViT-S without the
``lr_component_decay ** 2`` factor on top of the layer-wise decay of the ViT. Each YAML is a copy of the one of step 10:
these tests check that only ``lr_component_decay``, the output directory and the W&B run differ, so that model, recipe,
data and evaluation are those of the runs of 06/11/2026, whose output and W&B run are not touched. The W&B run is the
name of the file (checked for every YAML by ``test_leonardo_files.py``).
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from rfdetr.models.eodt import load_configs

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
RUNS = "/leonardo_work/IscrC_EoDT/tcentonz/runs"


def _read(name: str) -> dict[str, Any]:
    """Read a YAML file of ``configs`` (``eodt/...`` or ``baseline/...``) as plain data."""
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("base", "derived"),
    [
        ("eodt/eodt_vits_576_l2-3_selection.yaml", "eodt/eodt_vits_576_l2-3_selection_lrcd1.yaml"),
        ("baseline/baseline_medium_win.yaml", "baseline/baseline_medium_win_lrcd1.yaml"),
    ],
    ids=["eodt_selection", "baseline_medium_win"],
)
def test_only_lr_component_decay_and_names_differ(base: str, derived: str) -> None:
    """The new YAML is the one of step 10 with lr_component_decay 1.0, its own output directory and W&B run."""
    base_content, derived_content = _read(base), _read(derived)
    base_train = base_content["model"]["train_config"].pop("init_args")
    derived_train = derived_content["model"]["train_config"].pop("init_args")
    run = Path(derived).stem

    assert derived_content == base_content  # class paths and the whole model_config
    assert set(derived_train) == set(base_train)
    changes = {key: value for key, value in derived_train.items() if base_train[key] != value}
    assert changes == {"lr_component_decay": 1.0, "output_dir": f"{RUNS}/{run}", "run": run}

    base_model_config, _ = load_configs(CONFIGS / base)
    model_config, train_config = load_configs(CONFIGS / derived)
    assert model_config == base_model_config
    assert train_config.lr_component_decay == 1.0
