# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the files that run EoDT and the baseline on Leonardo (steps 6c and 8b of EODT_SPEC.md §9).

The jobs run only on Leonardo; on the PC these tests check that:
- the YAML files of the test jobs and of the overfits on Leonardo, which verify the infrastructure of step 6 on the
  two models of step 10 (EoDT selection and baseline_medium_win), differ from the YAML they copy only in the keys of
  their role, so that the recipe they test is the real one;
- every W&B run, of EoDT and of the baseline, is named after its YAML file, the WANDB_RUN_ID that
  ``scripts/leonardo/train.sh`` sets;
- ``train.sh`` and ``overfit.sh`` take the YAML file as argument, the files named in their usage comments exist, and
  srun gives each process the cores of sbatch;
- ``prepare_offline.sh`` puts in the cache the DINOv2 weights of the three sizes for EoDT (timm) and for the baseline
  (Hugging Face), and its Python code downloads them and then loads them without internet;
- ``benchmark.sh`` (step 13) measures EoDT, the rings of the baseline and RF-DETR's published checkpoints with the
  A100 protocol on a reserved node, and ``prepare_offline.sh`` downloads those checkpoints;
- the shell scripts have Unix line endings.
"""

import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from rfdetr.models.eodt import load_configs
from rfdetr.models.eodt.benchmark import PUBLISHED

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
RUNS = "/leonardo_work/IscrC_EoDT/tcentonz/runs"


def _read(name: str) -> dict[str, Any]:
    """Read a YAML file of ``configs`` (``eodt/...`` or ``baseline/...``) as plain data."""
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def _test_job(run: str) -> dict[str, Any]:
    """Keys of a test job: 2 epochs, its own output_dir and W&B run, named after the file."""
    return {"output_dir": f"{RUNS}/{run}", "epochs": 2, "run": run}


def _overfit_on_leonardo(run: str) -> dict[str, Any]:
    """Keys of an overfit on Leonardo: coco_mini on $WORK, bf16, W&B, its own output_dir and W&B run."""
    return {
        "dataset_dir": "/leonardo_work/IscrC_EoDT/tcentonz/data/coco_mini",
        "output_dir": f"{RUNS}/{run}",
        "amp_dtype": "bf16",
        "wandb": True,
        "run": run,
    }


@pytest.mark.parametrize(
    ("base", "derived", "role"),
    [
        ("eodt/eodt_vits_576_l2-3_learned.yaml", "eodt/eodt_vits_576_l2-3_learned_testjob.yaml", _test_job),
        (
            "eodt/eodt_vits_576_l2-3_learned_overfit.yaml",
            "eodt/eodt_vits_576_l2-3_learned_overfit_leonardo.yaml",
            _overfit_on_leonardo,
        ),
        ("eodt/eodt_vits_576_l2-3_selection.yaml", "eodt/eodt_vits_576_l2-3_selection_testjob.yaml", _test_job),
        (
            "eodt/eodt_vits_576_l2-3_selection_overfit.yaml",
            "eodt/eodt_vits_576_l2-3_selection_overfit_leonardo.yaml",
            _overfit_on_leonardo,
        ),
        ("baseline/baseline_medium_win.yaml", "baseline/baseline_medium_win_testjob.yaml", _test_job),
        (
            "baseline/baseline_medium_win_overfit.yaml",
            "baseline/baseline_medium_win_overfit_leonardo.yaml",
            _overfit_on_leonardo,
        ),
    ],
    ids=[
        "learned_testjob",
        "learned_overfit_leonardo",
        "selection_testjob",
        "selection_overfit_leonardo",
        "baseline_medium_win_testjob",
        "baseline_medium_win_overfit_leonardo",
    ],
)
def test_derived_yaml_changes_only_listed_keys(
    base: str, derived: str, role: Callable[[str], dict[str, Any]]
) -> None:
    """Model and recipe are those of the copied YAML: only the keys of the test job or of Leonardo change."""
    base_content, derived_content = _read(base), _read(derived)
    base_train = base_content["model"]["train_config"].pop("init_args")
    derived_train = derived_content["model"]["train_config"].pop("init_args")

    assert derived_content == base_content  # class paths and the whole model_config
    assert set(derived_train) == set(base_train)
    changes = {key: value for key, value in derived_train.items() if base_train[key] != value}
    assert changes == role(Path(derived).stem)
    load_configs(CONFIGS / derived)


def test_wandb_run_is_named_after_the_file() -> None:
    """The W&B run of every YAML, of EoDT and of the baseline, is the name of the file: the WANDB_RUN_ID of train.sh."""
    paths = sorted((CONFIGS / "eodt").glob("*.yaml")) + sorted((CONFIGS / "baseline").glob("*.yaml"))

    assert any(path.parent.name == "baseline" for path in paths)
    for path in paths:
        assert load_configs(path)[1].run == path.stem, path.name


@pytest.mark.parametrize("name", ["train.sh", "overfit.sh"])
def test_train_and_overfit_scripts(name: str) -> None:
    """train.sh and overfit.sh take the YAML file as argument and srun gives each process the cores of sbatch.

    Their usage comments are copied as they are on the login node: every YAML file they name exists and loads.
    """
    lines = (ROOT / "scripts" / "leonardo" / name).read_text(encoding="utf-8").splitlines()
    comments = "\n".join(line for line in lines if line.startswith("#"))
    code = "\n".join(line for line in lines if not line.startswith("#"))
    paths = re.findall(r"configs/\S+\.yaml", comments)

    assert {Path(path).parent.name for path in paths} == {"eodt", "baseline"}
    for path in paths:
        load_configs(ROOT / path)
    assert "configs/" not in code  # no YAML file written in the script
    assert re.search(r'^srun --cpus-per-task="\$SLURM_CPUS_PER_TASK" python \S+\.py "\$@"$', code, flags=re.M)


def test_prepare_offline_caches_dinov2() -> None:
    """prepare_offline.sh downloads, then loads offline, the timm and the Hugging Face DINOv2 of the three sizes.

    Its Python code runs here as on the login node, for ViT-S only: first with internet, then with
    ``HF_HUB_OFFLINE=1`` as on a compute node. The Hugging Face weights go through ``load_native_dinov2``, the loader of
    the baseline.
    """
    script = (ROOT / "scripts" / "leonardo" / "prepare_offline.sh").read_text(encoding="utf-8")
    # the loop over the sizes and its Python code, between <<'EOF' and EOF
    loop = re.search(
        r"SIZES=\"(?P<sizes>[^\"]*)\"\nfor offline in 0 1; do.*?\n"
        r"    HF_HUB_OFFLINE=\$offline python - \$SIZES <<'EOF'\n(?P<code>.*?)\nEOF\ndone\n",
        script,
        flags=re.DOTALL,
    )
    assert loop is not None
    assert loop["sizes"].split() == ["small", "base", "large"]
    assert "load_native_dinov2(size)" in loop["code"]

    for offline, state in (("0", "downloaded"), ("1", "loaded offline")):
        run = subprocess.run(
            [sys.executable, "-", "small"],
            input=loop["code"],
            env={**os.environ, "HF_HUB_OFFLINE": offline},
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr
        lines = run.stdout.splitlines()
        assert f"vit_small_patch14_reg4_dinov2: {state}" in lines
        assert f"facebook/dinov2-with-registers-small: {state}" in lines


def test_benchmark_job() -> None:
    """benchmark.sh measures every model of §6 with the A100 protocol, on a reserved node.

    EoDT and the three rings of the baseline from their YAML files, every published checkpoint of RF-DETR, whose
    weights prepare_offline.sh downloads into RF_HOME with their MD5, and the models without trained weights: EoDT on
    ViT-B and ViT-L and the architectures of RF-DETR's Table 7.
    """
    script = (ROOT / "scripts" / "leonardo" / "benchmark.sh").read_text(encoding="utf-8")
    models = re.search(r'MODELS="\n(?P<paths>.*?)\n"', script, flags=re.DOTALL)
    published = re.search(r'PUBLISHED="(?P<names>[^"]*)"', script)
    untrained = re.search(r'UNTRAINED="\n(?P<paths>.*?)\n"', script, flags=re.DOTALL)
    prepare = (ROOT / "scripts" / "leonardo" / "prepare_offline.sh").read_text(encoding="utf-8")

    assert "#SBATCH --exclusive" in script
    assert "PROTOCOL=configs/benchmark/a100.yaml" in script
    assert models is not None and published is not None and untrained is not None
    paths = models["paths"].split()
    assert {Path(path).parent.name for path in paths} == {"eodt", "baseline"}
    assert {Path(path).name for path in paths if "baseline" in path} == {
        f"baseline_medium_{ring}.yaml" for ring in ("win", "nowin", "nowin_noproj")
    }
    assert {Path(path).name for path in untrained["paths"].split()} == {
        "eodt_vitb_576_l2-3_selection.yaml",
        "eodt_vitl_576_l2-4_selection.yaml",
        *(f"rfdetr_table7_{size}.yaml" for size in ("xl", "2xl", "max")),
    }
    for path in paths + untrained["paths"].split():
        load_configs(ROOT / path)
    assert published["names"].split() == list(PUBLISHED)
    assert "for name in PUBLISHED:" in prepare and ".md5_hash" in prepare
    env = (ROOT / "scripts" / "leonardo" / "env.sh").read_text(encoding="utf-8")
    assert "export RF_HOME=$WORK_DIR/rfdetr_cache" in env


def test_scripts_have_unix_line_endings() -> None:
    """The scripts for Leonardo keep Unix line endings even when written on Windows."""
    scripts = sorted((ROOT / "scripts" / "leonardo").glob("*.sh"))

    assert {script.name for script in scripts} == {
        "benchmark.sh",
        "env.sh",
        "overfit.sh",
        "prepare_offline.sh",
        "train.sh",
    }
    for script in scripts:
        assert b"\r\n" not in script.read_bytes(), script.name
