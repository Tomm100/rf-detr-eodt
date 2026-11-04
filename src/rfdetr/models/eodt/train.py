# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Train a model from its YAML file with RF-DETR's training stack, as on Leonardo (EODT_SPEC.md §9, step 6).

``train`` serves EoDT and the baseline alike: the model is the ``class_path`` of the YAML file. It runs
``build_trainer`` + ``RFDETRModelModule`` + ``RFDETRDataModule``, the path of ``RFDETR.train()``, and adds what a run
on Leonardo needs:
- an ``output_dir`` already used is never overwritten, and a job can resume from its ``last.ckpt``;
- the seed is set before the model is built, so that queries and head are reproducible;
- a ``bf16`` run fails instead of falling back to fp16 or running on a GPU that only emulates bf16;
- ``run_info/`` keeps the full configuration, the git commit and the versions of every job (§3.7), which also go to
  the loggers, W&B included (§7.3);
- time per batch, images per second and GPU memory are logged (§7.3).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything

from rfdetr.config import BaseConfig, ModelConfig, TrainConfig
from rfdetr.models.eodt.config import EoDTConfig
from rfdetr.models.eodt.yaml_config import load_configs
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.trainer import build_trainer

RunType = Literal["run", "test"]
# files and folders that only a started run writes in its output_dir
_RUN_MARKERS = ("last.ckpt", "metrics.csv", "run_info")
_CONFIG_FILE = "config_resolved.yaml"
_PACKAGES = ("rfdetr", "torch", "torchvision", "timm", "transformers", "pytorch-lightning", "torchmetrics", "wandb")


def wandb_tags(model_config: ModelConfig, run_type: RunType) -> list[str]:
    """Return the W&B tags of a run: its type (``"run"`` or ``"test"``) and its model (``"eodt"`` or ``"baseline"``).

    Args:
        model_config: Model configuration; EoDT is an ``EoDTConfig``, the baseline any other ``ModelConfig``.
        run_type: ``"run"`` for the runs of the thesis, ``"test"`` for test jobs and overfits.

    Raises:
        ValueError: If ``run_type`` is not ``"run"`` or ``"test"``.
    """
    if run_type not in ("run", "test"):
        raise ValueError(f"run_type must be 'run' or 'test', got {run_type!r}")
    return [run_type, "eodt" if isinstance(model_config, EoDTConfig) else "baseline"]


def write_config(path: Path, model_config: ModelConfig, train_config: TrainConfig) -> None:
    """Write both configurations with all their defaults, in the format that ``load_configs`` reads."""

    def section(config: BaseConfig) -> dict[str, Any]:
        config_class = type(config)
        return {
            "class_path": f"{config_class.__module__}.{config_class.__qualname__}",
            "init_args": config.model_dump(mode="json"),
        }

    content = {"model": {"model_config": section(model_config), "train_config": section(train_config)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def _differences(saved: BaseConfig, current: BaseConfig) -> list[str]:
    """Names of the fields whose value differs between two configurations."""
    saved_values, current_values = saved.model_dump(), current.model_dump()
    return sorted(name for name in current_values if saved_values.get(name) != current_values[name])


def _check_output_dir(
    output_dir: Path, model_config: ModelConfig, train_config: TrainConfig, limits: dict[str, int | None], resume: bool
) -> None:
    """Refuse to overwrite a run, and to resume one with another configuration or other batch limits.

    Raises:
        FileExistsError: If a new run would write in an ``output_dir`` that already holds one.
        FileNotFoundError: If there is no ``last.ckpt`` or no saved configuration to resume from.
        ValueError: If the configuration or the batch limits differ from those of the run being resumed.
    """
    if not resume:
        used = [name for name in _RUN_MARKERS if (output_dir / name).exists()]
        if used:
            raise FileExistsError(
                f"{output_dir} already holds a run ({', '.join(used)}): resume it with --resume or change output_dir"
            )
        return
    run_info = output_dir / "run_info"
    for path in (output_dir / "last.ckpt", run_info / _CONFIG_FILE, run_info / "limits.json"):
        if not path.exists():
            raise FileNotFoundError(f"cannot resume the run in {output_dir}: {path} does not exist")
    saved_model_config, saved_train_config = load_configs(run_info / _CONFIG_FILE)
    changed = _differences(saved_model_config, model_config) + _differences(saved_train_config, train_config)
    if json.loads((run_info / "limits.json").read_text(encoding="utf-8")) != limits:
        changed.append("batch limits")
    if changed:
        raise ValueError(f"cannot resume the run in {output_dir} with a different {', '.join(changed)}")


def _native_bf16() -> bool:
    """Whether every visible GPU computes in bf16 natively (compute capability 8.0 or newer, e.g. A100).

    ``torch.cuda.is_bf16_supported()``, which ``build_trainer`` checks, is also true on older GPUs that only emulate
    bf16, such as the GTX 1650.
    """
    devices = range(torch.cuda.device_count()) if torch.cuda.is_available() else []
    return bool(devices) and all(torch.cuda.get_device_capability(index) >= (8, 0) for index in devices)


def _git(*args: str) -> str:
    """Run git in the repository of this file. Raises if git fails: the commit of a run must be known (§3.7)."""
    command = ["git", *args]
    return subprocess.run(command, cwd=Path(__file__).parent, check=True, capture_output=True, text=True).stdout


class RunInfo(Callback):
    """Save what identifies a job in ``output_dir/run_info`` and send it to the loggers (EODT_SPEC.md §3.7, §7.3).

    It acts in ``setup``, after the processes of a DDP run have joined, so that rank 0 cannot write ``run_info/``
    while another rank is still checking that ``output_dir`` is free.

    Args:
        config_path: The YAML file of the run, copied into ``run_info/``.
        model_config: Model configuration of the run.
        train_config: Training configuration of the run.
        limits: Batch limits per epoch passed to the trainer, ``None`` when not limited.
        resume: Whether this job resumes the run: the configuration is then already saved.
        tags: W&B tags of the run.
    """

    def __init__(
        self,
        config_path: Path,
        model_config: ModelConfig,
        train_config: TrainConfig,
        limits: dict[str, int | None],
        resume: bool,
        tags: list[str],
    ) -> None:
        self.config_path = config_path
        self.model_config = model_config
        self.train_config = train_config
        self.limits = limits
        self.resume = resume
        self.tags = tags

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Write the files of rank 0 and log the configuration with the job information."""
        if not trainer.is_global_zero:
            return
        run_info = Path(self.train_config.output_dir) / "run_info"
        if not self.resume:
            write_config(run_info / _CONFIG_FILE, self.model_config, self.train_config)
            shutil.copy(self.config_path, run_info / self.config_path.name)
            (run_info / "limits.json").write_text(json.dumps(self.limits), encoding="utf-8")

        job = os.getenv("SLURM_JOB_ID") or datetime.now().strftime("local-%Y%m%d-%H%M%S")
        changes = _git("diff", "HEAD")
        info = {
            "job": job,
            "start": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "command": sys.argv,
            "config": str(self.config_path),
            "resume": self.resume,
            "limits": self.limits,
            "git_commit": _git("rev-parse", "HEAD").strip(),
            "git_status": _git("status", "--porcelain").splitlines(),  # uncommitted and untracked files
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "packages": {name: version(name) for name in _PACKAGES},
            "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "world_size": trainer.world_size,
            "precision": trainer.precision,
            "wandb": {"run_id": os.getenv("WANDB_RUN_ID"), "mode": os.getenv("WANDB_MODE"), "tags": self.tags},
        }
        (run_info / f"job_{job}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        if changes:
            (run_info / f"job_{job}.patch").write_text(changes, encoding="utf-8")

        hyperparameters = {
            "model_config": self.model_config.model_dump(mode="json"),
            "train_config": self.train_config.model_dump(mode="json"),
            "job": info,
        }
        for logger in trainer.loggers:
            logger.log_hyperparams(hyperparameters)


class TrainingStats(Callback):
    """Log time per batch, images per second over all processes and peak GPU memory (EODT_SPEC.md §7.3).

    The time is the mean over the last ``log_every_n_steps`` batches of the epoch, so that it does not include the
    validation between two epochs.
    """

    def __init__(self) -> None:
        self._times: deque[float] = deque()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start a new time window, without the validation of the previous epoch."""
        self._times.clear()

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Log the statistics of the window that ends with this batch."""
        self._times.append(time.perf_counter())
        if len(self._times) > trainer.log_every_n_steps + 1:
            self._times.popleft()
        if len(self._times) < 2:
            return
        batch_size = batch[0].tensors.shape[0]
        batch_time = (self._times[-1] - self._times[0]) / (len(self._times) - 1)
        stats = {"train/batch_time_s": batch_time, "train/images_per_s": batch_size * trainer.world_size / batch_time}
        if torch.cuda.is_available():
            stats["train/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        pl_module.log_dict(stats, on_step=True, on_epoch=False, batch_size=batch_size)


class StopAfterEpoch(Callback):
    """Stop the job at the end of an epoch, to test a resume (EODT_SPEC.md §9, step 6).

    Unlike fewer ``epochs`` in the YAML, it leaves the length of the learning-rate schedule unchanged. The epoch must
    be validated: ``last.ckpt`` is written at the end of the validation, which runs before this hook.

    Args:
        epoch: Last epoch of the job, counted from 1.
    """

    def __init__(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError(f"epoch must be at least 1, got {epoch}")
        self.epoch = epoch

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Ask the trainer to stop once the epoch is reached."""
        if trainer.current_epoch + 1 >= self.epoch:
            trainer.should_stop = True


def train(
    config_path: str | Path,
    *,
    resume: bool = False,
    run_type: RunType = "run",
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    stop_after_epoch: int | None = None,
) -> Trainer:
    """Train the model of a YAML file, or resume its run from ``output_dir/last.ckpt``.

    Args:
        config_path: YAML file in the format read by ``load_configs``.
        resume: Continue the run in ``output_dir`` instead of starting a new one.
        run_type: ``"run"`` or ``"test"``, the W&B tag of the run.
        limit_train_batches: Training batches per epoch, for test jobs only; ``None`` uses the whole dataset.
        limit_val_batches: Validation batches per epoch, for test jobs only; ``None`` uses the whole dataset.
        stop_after_epoch: Last epoch of this job (from 1), for test jobs of a resume; ``None`` trains to the end.

    Returns:
        The trainer after ``fit``.

    Raises:
        FileExistsError: If a new run would overwrite the run in ``output_dir``.
        FileNotFoundError: If ``resume`` finds no run to resume.
        ValueError: If ``resume`` finds a run with another configuration or other batch limits.
        RuntimeError: If the YAML asks for bf16 and the trainer resolved another precision, or the GPU only
            emulates bf16.
    """
    config_path = Path(config_path)
    model_config, train_config = load_configs(config_path)
    output_dir = Path(train_config.output_dir)
    limits = {"limit_train_batches": limit_train_batches, "limit_val_batches": limit_val_batches}
    _check_output_dir(output_dir, model_config, train_config, limits, resume)
    if resume:
        # set before build_trainer, which then keeps the history of metrics.csv
        train_config.resume = str(output_dir / "last.ckpt")
    tags = wandb_tags(model_config, run_type)
    os.environ["WANDB_TAGS"] = ",".join(tags)  # read by wandb.init: RF-DETR's WandbLogger takes no tags

    trainer = build_trainer(train_config, model_config, **{k: v for k, v in limits.items() if v is not None})
    if train_config.amp_dtype == "bf16" and (trainer.precision != "bf16-mixed" or not _native_bf16()):
        raise RuntimeError(
            f"the YAML asks for bf16, but the trainer resolved precision {trainer.precision!r} on a device without "
            "native bf16 (it needs a GPU of compute capability 8.0 or newer)"
        )
    # callbacks appended after build_trainer: passing callbacks= to it would replace RF-DETR's own
    trainer.callbacks.append(RunInfo(config_path, model_config, train_config, limits, resume, tags))
    trainer.callbacks.append(TrainingStats())
    if stop_after_epoch is not None:
        trainer.callbacks.append(StopAfterEpoch(stop_after_epoch))

    # RFDETRModelModule seeds only in on_fit_start: seed before building it, so that queries and head are reproducible
    seed_everything(train_config.seed)
    module = RFDETRModelModule(model_config, train_config)
    datamodule = RFDETRDataModule(model_config, train_config)
    trainer.fit(module, datamodule, ckpt_path=train_config.resume)
    return trainer
