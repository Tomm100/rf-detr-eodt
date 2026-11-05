# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the YAML files of EoDT with selection queries (step 9b of EODT_SPEC.md §9).

Each is a copy of the YAML of the same run with learned queries: the run on Leonardo for the comparison with the
baseline (step 10) and the overfit on coco_mini (test 5 of §9 repeated with selection). These tests check that only
``query_mode``, the output directory and the W&B run differ, so that model, recipe, data and evaluation are those
already tested with learned queries; the W&B run is the name of the file (checked for every YAML by
``test_leonardo_files.py``).
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from rfdetr.models.eodt import EoDTConfig, load_configs

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "eodt"


def _read(name: str) -> dict[str, Any]:
    """Read a YAML file of ``configs/eodt`` as plain data."""
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def _changed(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Values of ``after`` that differ from those of ``before``, which has the same keys."""
    assert set(after) == set(before)
    return {key: value for key, value in after.items() if before[key] != value}


@pytest.mark.parametrize(
    ("learned", "selection", "output_dir"),
    [
        (
            "eodt_vits_576_l2-3_learned.yaml",
            "eodt_vits_576_l2-3_selection.yaml",
            "/leonardo_work/IscrC_EoDT/tcentonz/runs/eodt_vits_576_l2-3_selection",
        ),
        (
            "eodt_vits_576_l2-3_learned_overfit.yaml",
            "eodt_vits_576_l2-3_selection_overfit.yaml",
            "C:/dev/tesi/rf-detr-eodt/output/eodt_vits_576_l2-3_selection_overfit",
        ),
    ],
    ids=["run", "overfit"],
)
def test_only_query_mode_and_names_differ(learned: str, selection: str, output_dir: str) -> None:
    """The selection YAML is the learned one with query_mode selection, its own output directory and W&B run."""
    learned_content, selection_content = _read(learned), _read(selection)
    learned_model = learned_content["model"]["model_config"]
    selection_model = selection_content["model"]["model_config"]
    learned_train = learned_content["model"]["train_config"]
    selection_train = selection_content["model"]["train_config"]

    assert selection_model["class_path"] == learned_model["class_path"]
    assert _changed(learned_model["init_args"], selection_model["init_args"]) == {"query_mode": "selection"}
    assert selection_train["class_path"] == learned_train["class_path"]
    assert _changed(learned_train["init_args"], selection_train["init_args"]) == {
        "output_dir": output_dir,
        "run": Path(selection).stem,
    }

    model_config, _ = load_configs(CONFIGS / selection)
    assert type(model_config) is EoDTConfig
    assert model_config.query_mode == "selection"
    assert (model_config.resolution, model_config.num_queries, model_config.num_l2_blocks) == (576, 300, 3)
