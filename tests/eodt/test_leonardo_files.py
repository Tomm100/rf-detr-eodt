# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the files that run EoDT on Leonardo (step 6c of EODT_SPEC.md §9).

The jobs run only on Leonardo; on the PC these tests check that:
- the YAML of the test job and the one of the overfit on Leonardo differ from the YAML they copy only in the keys
  of their role, so that the recipe they test is the real one;
- every W&B run is named after its YAML file, the WANDB_RUN_ID that ``scripts/leonardo/train.sh`` sets;
- the shell scripts have Unix line endings.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from rfdetr.models.eodt import load_configs

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "eodt"
RUNS = "/leonardo_work/IscrC_EoDT/tcentonz/runs"


def _read(name: str) -> dict[str, Any]:
    """Read a YAML file of ``configs/eodt`` as plain data."""
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("base", "derived", "changes"),
    [
        (
            "eodt_vits_576_l2-3_learned.yaml",
            "eodt_vits_576_l2-3_learned_testjob.yaml",
            {
                "output_dir": f"{RUNS}/eodt_vits_576_l2-3_learned_testjob",
                "epochs": 2,
                "run": "eodt_vits_576_l2-3_learned_testjob",
            },
        ),
        (
            "eodt_vits_576_l2-3_learned_overfit.yaml",
            "eodt_vits_576_l2-3_learned_overfit_leonardo.yaml",
            {
                "dataset_dir": "/leonardo_work/IscrC_EoDT/tcentonz/data/coco_mini",
                "output_dir": f"{RUNS}/eodt_vits_576_l2-3_learned_overfit_leonardo",
                "amp_dtype": "bf16",
                "wandb": True,
                "run": "eodt_vits_576_l2-3_learned_overfit_leonardo",
            },
        ),
    ],
    ids=["testjob", "overfit_leonardo"],
)
def test_derived_yaml_changes_only_listed_keys(base: str, derived: str, changes: dict[str, Any]) -> None:
    """Model and recipe are those of the copied YAML: only the keys of the test job or of Leonardo change."""
    base_content, derived_content = _read(base), _read(derived)
    base_train = base_content["model"]["train_config"].pop("init_args")
    derived_train = derived_content["model"]["train_config"].pop("init_args")

    assert derived_content == base_content  # class paths and the whole model_config
    assert set(derived_train) == set(base_train)
    assert {key: value for key, value in derived_train.items() if base_train[key] != value} == changes
    load_configs(CONFIGS / derived)


def test_wandb_run_is_named_after_the_file() -> None:
    """The W&B run of every YAML is the name of the file, as the WANDB_RUN_ID set by train.sh."""
    paths = sorted(CONFIGS.glob("*.yaml"))
    for path in paths:
        assert load_configs(path)[1].run == path.stem, path.name


def test_scripts_have_unix_line_endings() -> None:
    """The scripts for Leonardo keep Unix line endings even when written on Windows."""
    scripts = sorted((ROOT / "scripts" / "leonardo").glob("*.sh"))

    assert {script.name for script in scripts} == {"env.sh", "overfit.sh", "prepare_offline.sh", "train.sh"}
    for script in scripts:
        assert b"\r\n" not in script.read_bytes(), script.name
