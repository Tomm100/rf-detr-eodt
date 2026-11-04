# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the training entry point used on Leonardo (step 6b of EODT_SPEC.md §9).

Every run reads a copy of ``configs/eodt/eodt_vits_576_l2-3_learned.yaml`` written by the ``eodt_yaml`` fixture,
on CPU and in fp32. Covers:
- a run never overwrites the ``output_dir`` of another one, and resumes only the same configuration (point 6);
- a job stopped after its first epoch resumes in a second job: schedule, EMA, ``metrics.csv`` and the W&B run
  continue (24-hour limit of Leonardo, §7.3);
- ``run_info/`` holds the full configuration, the git commit and the versions of every job (§3.7);
- a bf16 run does not fall back to another precision (point 7), and the W&B tags of a run.
"""

import csv
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import timm
import wandb

from rfdetr.config import RFDETRMediumConfig
from rfdetr.models.eodt import EoDTConfig, load_configs
from rfdetr.models.eodt.train import train, wandb_tags
from rfdetr.training.callbacks.ema import RFDETREMACallback

SMALL = {"resolution": 224}  # 14 x 14 grid keeps the CPU runs fast
LOCAL = {"amp_dtype": None, "accelerator": "cpu", "batch_size": 2, "epochs": 2, "tensorboard": False}
LIMITS = {"limit_train_batches": 2, "limit_val_batches": 1}  # 2 epochs of 2 steps


def test_refuses_output_dir_of_another_run(eodt_yaml: Callable[..., Path]) -> None:
    """A new run stops before training when its output_dir already holds a run."""
    path = eodt_yaml(SMALL, LOCAL)
    output_dir = Path(load_configs(path)[1].output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.csv").touch()

    with pytest.raises(FileExistsError, match="--resume"):
        train(path)


def test_resume_needs_a_run(eodt_yaml: Callable[..., Path]) -> None:
    """Resuming an output_dir without a checkpoint is an error, not a new run."""
    with pytest.raises(FileNotFoundError, match="last.ckpt"):
        train(eodt_yaml(SMALL, LOCAL), resume=True)


def test_bf16_must_be_resolved(eodt_yaml: Callable[..., Path]) -> None:
    """A YAML that asks for bf16 fails where the trainer resolves another precision (fp32 on CPU)."""
    with pytest.raises(RuntimeError, match="bf16"):
        train(eodt_yaml(SMALL, {**LOCAL, "amp_dtype": "bf16"}))


def test_wandb_tags() -> None:
    """The tags are the type of run and the model: eodt for EoDTConfig, baseline for RF-DETR's configurations."""
    assert wandb_tags(EoDTConfig(num_l2_blocks=3), "run") == ["run", "eodt"]
    assert wandb_tags(RFDETRMediumConfig(pretrain_weights=None), "test") == ["test", "baseline"]
    with pytest.raises(ValueError, match="run_type"):
        wandb_tags(EoDTConfig(num_l2_blocks=3), "debug")


def test_stop_and_resume(eodt_yaml: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A job stopped after epoch 1 and a second job that resumes it make one run, on disk and on W&B."""
    for name, value in {
        "WANDB_MODE": "offline",
        "WANDB_RUN_ID": "eodt-resume-test",
        "WANDB_SILENT": "true",
        "WANDB_CONFIG_DIR": str(tmp_path / "wandb_config"),
        "WANDB_CACHE_DIR": str(tmp_path / "wandb_cache"),
        "WANDB_DATA_DIR": str(tmp_path / "wandb_data"),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("WANDB_TAGS", raising=False)  # set by train, restored at teardown
    logged = {**LOCAL, "wandb": True, "project": "eodt-test", "run": "resume"}
    path = eodt_yaml(SMALL, logged)
    model_config, train_config = load_configs(path)
    output_dir = Path(train_config.output_dir)

    first = train(path, run_type="test", stop_after_epoch=1, **LIMITS)
    wandb.finish()  # each job is a new process on Leonardo
    second = train(path, resume=True, run_type="test", **LIMITS)
    wandb.finish()

    # the second job continues the first: steps, schedule, EMA and the history of metrics.csv
    assert (first.global_step, second.global_step) == (2, 4)
    assert all(group["lr"] == pytest.approx(0.0, abs=1e-12) for group in second.optimizers[0].param_groups)
    ema = next(callback for callback in second.callbacks if isinstance(callback, RFDETREMACallback))
    assert int(ema._average_model.n_averaged) == 4
    with (output_dir / "metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert {int(row["epoch"]) for row in rows if row.get("val/ema_mAP_small")} == {0, 1}
    assert "train/images_per_s" in second.callback_metrics

    # one W&B run: both offline runs carry the id of the experiment
    offline_runs = sorted((output_dir / "wandb").glob("offline-run-*"))
    assert len(offline_runs) == 2
    assert all(run.name.endswith("-eodt-resume-test") for run in offline_runs)

    # run_info: the full configuration, reloadable, and one file per job with commit and versions
    run_info = output_dir / "run_info"
    saved_model_config, saved_train_config = load_configs(run_info / "config_resolved.yaml")
    assert saved_model_config.model_dump() == model_config.model_dump()
    assert saved_train_config.model_dump() == train_config.model_dump()
    assert (run_info / path.name).read_text(encoding="utf-8") == path.read_text(encoding="utf-8")
    jobs = [json.loads(job.read_text(encoding="utf-8")) for job in sorted(run_info.glob("job_*.json"))]
    assert [job["resume"] for job in jobs] == [False, True]
    assert re.fullmatch("[0-9a-f]{40}", jobs[0]["git_commit"])
    assert jobs[0]["packages"]["timm"] == timm.__version__
    assert jobs[0]["wandb"] == {"run_id": "eodt-resume-test", "mode": "offline", "tags": ["test", "eodt"]}

    # a third job cannot resume with another configuration or other batch limits
    with pytest.raises(ValueError, match=r"\blr\b"):
        train(eodt_yaml(SMALL, {**logged, "lr": 2.0e-4}), resume=True, **LIMITS)
    with pytest.raises(ValueError, match="batch limits"):
        train(path, resume=True, limit_train_batches=3, limit_val_batches=1)
