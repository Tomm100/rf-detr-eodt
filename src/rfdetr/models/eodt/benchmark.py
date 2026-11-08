# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Latency, peak memory, parameters and MACs of EoDT, of the baseline and of RF-DETR's published checkpoints.

The protocol of EODT_SPEC.md §6, the same for every model, as in EoMT (Appendix A.3):

- the inference forward only: from a preprocessed image already on the GPU to the boxes and logits of the last
  L2 block or decoder layer (``export()``), without the outputs used only by the loss; preprocessing, copy to the
  GPU and ``PostProcess`` are not timed;
- batch 1, every image of COCO val2017 through the evaluation pipeline (square resize and ImageNet normalization),
  after a warm-up; ``torch.compile`` and bf16 autocast, measured without them (and recorded) if they fail;
- MACs counted with PyTorch's ``FlopCounterMode`` on one image, per component of the model.

The settings live in a YAML file of ``configs/benchmark/`` (:class:`BenchmarkConfig`); ``scripts/eodt/benchmark.py``
runs one model per process and compares two jobs.
"""

from __future__ import annotations

import os
import platform
import statistics
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Subset
from torch.utils.flop_counter import FlopCounterMode

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import (
    BaseConfig,
    ModelConfig,
    PretrainWeightsCompatibilityWarning,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    TrainConfig,
)
from rfdetr.datasets import build_dataset
from rfdetr.models.eodt.model import EoDT
from rfdetr.models.eodt.train import _PACKAGES, _git
from rfdetr.models.eodt.yaml_config import _UniqueKeyLoader, load_configs
from rfdetr.models.lwdetr import build_model_from_config
from rfdetr.models.weights import load_pretrain_weights
from rfdetr.utilities.io import _safe_torch_load

# RF-DETR's published detection checkpoints (§6): architecture and weight file come from RF-DETR's own configurations
PUBLISHED: dict[str, type[ModelConfig]] = {
    "nano": RFDETRNanoConfig,
    "small": RFDETRSmallConfig,
    "medium": RFDETRMediumConfig,
    "large": RFDETRLargeConfig,
}

# Step 13 (EODT_SPEC.md §9), fixed before the jobs: the mean latencies of two jobs differ by at most 2%
MAX_LATENCY_DIFF = 0.02

MACS_NOTE = (
    "MACs = FLOPs / 2 of torch.utils.flop_counter.FlopCounterMode, on one image in fp32 with the math backend of "
    "SDPA. It counts matrix multiplications, convolutions and attention; it does not count grid_sample (deformable "
    "attention of RF-DETR's decoder), softmax, normalizations and element-wise operations."
)


class BenchmarkConfig(BaseConfig):
    """Settings of a benchmark run, read from a YAML file of ``configs/benchmark/``; every field is required.

    Attributes:
        dataset_dir: COCO folder with ``val2017/`` and ``annotations/``.
        num_images: Number of val images timed, from the first; ``None`` for all of them.
        num_workers: Worker processes of the image loader.
        warmup_iters: Forward passes after the first call and before the timed ones.
        device: Torch device of the model, e.g. ``"cuda"``.
        amp_dtype: ``"bf16"`` for bf16 autocast, ``None`` for fp32.
        compile: Whether to ``torch.compile`` the model.
        compile_mode: Mode of ``torch.compile``; the modes with CUDA graphs are excluded (EODT_SPEC.md §10).
        compile_backend: ``"inductor"`` (A100) or ``"eager"`` (graph capture only, where triton is missing).
        fullgraph: Whether a graph break is an error instead of a hidden slowdown.
    """

    dataset_dir: str
    num_images: int | None
    num_workers: int
    warmup_iters: int
    device: str
    amp_dtype: Literal["bf16"] | None
    compile: bool
    compile_mode: Literal["default", "max-autotune-no-cudagraphs"]
    compile_backend: Literal["inductor", "eager"]
    fullgraph: bool


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Read a YAML file of ``configs/benchmark/``; a key written twice, unknown or missing is an error."""
    return BenchmarkConfig(**yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader))


def published_config(name: str) -> ModelConfig:
    """Configuration of a published checkpoint, with one group of queries: the parameters used in inference.

    The weight file is passed explicitly, so that the ``expand_path`` validator puts it in RF-DETR's cache folder
    (``RF_HOME``, by default ``~/.roboflow/models``).

    Args:
        name: A key of :data:`PUBLISHED`.
    """
    config_class = PUBLISHED[name]
    with warnings.catch_warnings():
        # group_detr = 1 is not the training value of the checkpoint, but inference reads only the first group
        warnings.simplefilter("ignore", PretrainWeightsCompatibilityWarning)
        return config_class(group_detr=1, pretrain_weights=config_class.model_fields["pretrain_weights"].default)


def load_checkpoint(model: nn.Module, path: str | Path) -> None:
    """Load trained weights saved by RF-DETR's ``BestModelCallback`` (``.pth`` with a ``model`` key), strictly."""
    model.load_state_dict(_safe_torch_load(path)["model"], strict=True)


def build_inference_model(
    *, model_yaml: str | Path | None = None, checkpoint: str | Path | None = None, published: str | None = None
) -> tuple[nn.Module, ModelConfig]:
    """Build a model for the benchmark: weights loaded, eval mode, ``export()`` and no gradients.

    Args:
        model_yaml: An experiment YAML of ``configs/eodt`` or ``configs/baseline``; only its model is read.
        checkpoint: Trained weights for ``model_yaml``. Without them the model keeps its initialization (pretrained
            DINOv2): the latency does not depend on the weights.
        published: A key of :data:`PUBLISHED`, instead of ``model_yaml``.

    Returns:
        The model, whose forward maps ``[1, 3, R, R]`` images to ``(boxes, logits)`` of the last block or layer, and
        its configuration.
    """
    if (model_yaml is None) == (published is None):
        raise ValueError("give either model_yaml or published")
    if published is not None:
        if checkpoint is not None:
            raise ValueError("a published model reads its own weights")
        model_config = published_config(published)
        model = build_model_from_config(model_config)
        load_pretrain_weights(model, model_config)  # RF-DETR's loader: keeps the first group of queries
    else:
        model_config = load_configs(model_yaml)[0]
        model = build_model_from_config(model_config)
        if checkpoint is not None:
            load_checkpoint(model, checkpoint)
    model.eval()
    model.export()
    model.requires_grad_(False)
    return model, model_config


def count_params(model: nn.Module) -> int:
    """Number of parameters of the model as built for inference."""
    return sum(parameter.numel() for parameter in model.parameters())


def _components(model: nn.Module) -> dict[str, list[str]]:
    """Paths of the submodules that make each component of the model, for the MACs per component.

    Each path is a module that the forward calls: ``FlopCounterMode`` counts a module with all its submodules.
    """
    if isinstance(model, EoDT):
        blocks = range(len(model.backbone.vit.blocks))
        l1 = model.backbone.num_l1_blocks
        return {
            "vit_l1": ["backbone.vit.patch_embed", *(f"backbone.vit.blocks.{i}" for i in blocks[:l1])],
            "vit_l2": [f"backbone.vit.blocks.{i}" for i in blocks[l1:]],
            "first_stage": ["selection", "ref_point_head"],  # absent with learned queries
            "heads": ["class_embed", "bbox_embed"],
        }
    return {  # LWDETR; the first stage has one module per group in a ModuleList, and inference runs the first one
        "vit": ["backbone.0.encoder"],
        "projector": ["backbone.0.projector"],
        "first_stage": [
            "transformer.enc_output.0",
            "transformer.enc_output_norm.0",
            "transformer.enc_out_class_embed.0",
            "transformer.enc_out_bbox_embed.0",
        ],
        "decoder": ["transformer.decoder"],
        "heads": ["class_embed", "bbox_embed"],
    }


def count_macs(model: nn.Module, resolution: int) -> dict[str, int]:
    """Count the MACs of one forward of the model, in total and per component.

    SDPA runs with its math backend: ``FlopCounterMode`` does not count the fused attention of the CPU, and the math
    backend gives the same count on CPU and GPU.

    Args:
        model: A model from :func:`build_inference_model`.
        resolution: Side of the square input.

    Returns:
        ``total`` and one entry per component of :func:`_components`, plus ``other``: the MACs outside every
        component, which should be about zero.
    """
    device = next(model.parameters()).device
    image = torch.zeros(1, 3, resolution, resolution, device=device)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]), FlopCounterMode(display=False) as counter:
        model(image)
    root = type(model).__name__ + "."
    per_module = {
        name.removeprefix(root): sum(ops.values())
        for name, ops in counter.get_flop_counts().items()
        if name.startswith(root)
    }
    macs = {
        component: sum(per_module.get(path, 0) for path in paths) // 2
        for component, paths in _components(model).items()
    }
    total = counter.get_total_flops() // 2
    return {"total": total, **macs, "other": total - sum(macs.values())}


def _image_only(batch: list[tuple[Tensor, Any]]) -> Tensor:
    """Collate a batch of one ``(image, target)`` pair into an image batch ``[1, 3, R, R]``."""
    return batch[0][0][None]


def val_images(config: BenchmarkConfig, model_config: ModelConfig) -> DataLoader:
    """Images of COCO val2017, one per batch, through the evaluation pipeline at the resolution of the model."""
    train_config = TrainConfig(  # the input of §7.0: square resize, no multi-scale
        dataset_file="coco",
        dataset_dir=config.dataset_dir,
        multi_scale="off",
        square_resize_div_64=True,
        augmentation_backend="torchvision",
    )
    dataset = build_dataset("val", _namespace_from_configs(model_config, train_config), model_config.resolution)
    if config.num_images is not None:
        dataset = Subset(dataset, range(config.num_images))
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, collate_fn=_image_only)


def _synchronize(device: torch.device) -> None:
    """Wait for the GPU, so that the clock measures finished work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _inference_context(device: torch.device, amp_dtype: str | None) -> Iterator[None]:
    """No gradients and, with ``"bf16"``, bf16 autocast."""
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_dtype == "bf16"):
        yield


def _compile(model: Callable[..., Any], config: BenchmarkConfig) -> Callable[..., Any]:
    """Compile a model or a function with the settings of the protocol; the work happens at the first call."""
    return torch.compile(
        model,
        mode=config.compile_mode,
        backend=config.compile_backend,
        fullgraph=config.fullgraph,
        dynamic=False,
    )


@contextmanager
def _patched_topk(imposed: Tensor | None = None) -> Iterator[list[Tensor]]:
    """Inside the context ``torch.topk`` records its indices or, with ``imposed``, returns those instead.

    The models of the benchmark call it once per forward, to select the top-Q patches of the first stage, and read
    only the indices. Dynamo inlines this function, so ``imposed`` reaches a compiled model too.

    Args:
        imposed: Indices ``[B, Q]`` to return instead of computing the ranking.

    Yields:
        The list where the indices computed by the real ``torch.topk`` are recorded.
    """
    original = torch.topk
    recorded: list[Tensor] = []

    def topk(*args: Any, **kwargs: Any) -> tuple[Tensor | None, Tensor]:
        if imposed is not None:
            return None, imposed
        values, indices = original(*args, **kwargs)
        recorded.append(indices)
        return values, indices

    torch.topk = topk  # type: ignore[assignment]
    try:
        yield recorded
    finally:
        torch.topk = original


def _max_abs_diff(outputs: tuple[Tensor, Tensor], reference: tuple[Tensor, Tensor]) -> dict[str, float]:
    """Largest difference of scores and boxes between two ``(boxes, logits)``, query by query."""
    return {
        "scores": (outputs[1].float().sigmoid() - reference[1].float().sigmoid()).abs().max().item(),
        "boxes": (outputs[0].float() - reference[0].float()).abs().max().item(),
    }


def _same_top_q_check(
    model: nn.Module,
    image: Tensor,
    reference: tuple[Tensor, Tensor],
    top_q: Tensor,
    config: BenchmarkConfig,
    compiled: bool,
    amp_dtype: str | None,
) -> dict[str, Any]:
    """Compare the model with the eager fp32 reference with the same patches selected, in the same order.

    A small numerical difference can change the top-Q ranking of the first stage. The query in position i gets the
    i-th ``query_feat`` and ``refpoint_embed``, so a different ranking is not a permutation of the outputs, and the
    query-by-query difference is about 1 even when the computation is right. Imposing the ranking of the reference
    leaves only the numerical differences.

    Args:
        model: A model from :func:`build_inference_model`, not compiled.
        image: Input ``[1, 3, R, R]``, already on the device of the model.
        reference: ``(boxes, logits)`` of the eager fp32 model on ``image``.
        top_q: Top-Q indices ``[1, Q]`` of the eager fp32 model on ``image``.
        config: The settings of the run.
        compiled: Whether the run was compiled.
        amp_dtype: The precision of the run.

    Returns:
        How many positions of the ranking and how many selected patches change in eager mode in the precision of the
        run, and, with the ranking imposed, the largest differences of eager from eager fp32 (the precision alone) and
        of compiled from eager (the compilation alone; ``None`` if the run was not compiled, or the error if this
        second compilation fails).
    """
    with _inference_context(image.device, amp_dtype), _patched_topk() as ranking:
        model(image)
    with _inference_context(image.device, amp_dtype), _patched_topk(top_q):
        eager = model(image)
        compiled_vs_eager = None
        if compiled:
            torch._dynamo.reset()
            try:
                compiled_vs_eager = _max_abs_diff(_compile(model, config)(image), eager)
            except Exception as error:  # noqa: BLE001 - the timings of the run are already measured
                compiled_vs_eager = {"error": repr(error)}
    return {
        "positions_changed": (ranking[0] != top_q).sum().item(),
        "patches_changed": top_q.numel() - torch.isin(top_q, ranking[0]).sum().item(),
        "eager_vs_eager_fp32": _max_abs_diff(eager, reference),
        "compiled_vs_eager": compiled_vs_eager,
    }


def measure_latency(model: nn.Module, images: DataLoader, config: BenchmarkConfig) -> dict[str, Any]:
    """Time the forward of the model on every image, after compilation and warm-up.

    The first call, which compiles, is timed apart. If compilation or bf16 fail, the model is measured without them,
    in this order of preference: compiled in bf16, eager in bf16, compiled in fp32, eager in fp32; the errors are
    recorded. Every forward is timed alone, between two GPU synchronizations, with the image already on the GPU.

    Args:
        model: A model from :func:`build_inference_model`.
        images: Images from :func:`val_images`.
        config: The settings of the run.

    Returns:
        Latency in ms (mean, median, standard deviation, 95th percentile), FPS, time of the first call, peak memory,
        what was actually compiled and in which precision, the errors that forced a fallback, the largest difference
        of scores and boxes from the eager fp32 model on the first image, query by query, and the check with the same
        top-Q of :func:`_same_top_q_check` (``None`` for a model without top-Q selection).
    """
    device = torch.device(config.device)
    model.to(device)
    first = next(iter(images)).to(device)
    with torch.no_grad(), _patched_topk() as ranking:
        reference = model(first)
    top_q = ranking[0] if ranking else None

    attempts = [(config.compile, config.amp_dtype), (False, config.amp_dtype), (config.compile, None), (False, None)]
    errors: list[str] = []
    for compiled, amp_dtype in dict.fromkeys(attempts):  # in order, without repetitions
        torch._dynamo.reset()
        run = _compile(model, config) if compiled else model
        start = time.perf_counter()
        try:
            with _inference_context(device, amp_dtype):
                boxes, logits = run(first)
            _synchronize(device)
        except Exception as error:  # noqa: BLE001 - the model is measured without what failed (§6)
            errors.append(f"compile={compiled}, amp_dtype={amp_dtype}: {error!r}")
            continue
        first_call_s = time.perf_counter() - start
        break
    else:
        raise RuntimeError("the model does not run: " + " | ".join(errors))

    with _inference_context(device, amp_dtype):
        done = 0
        while done < config.warmup_iters:  # cycles over the images when they are fewer than the warm-up
            for image in images:
                run(image.to(device))
                done += 1
                if done == config.warmup_iters:
                    break
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        times_ms = []
        for image in images:
            image = image.to(device)
            _synchronize(device)
            start = time.perf_counter()
            run(image)
            _synchronize(device)
            times_ms.append((time.perf_counter() - start) * 1000)

    # after the timing, and after reading the peak memory: this check compiles the model a second time
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    same_top_q = None
    if top_q is not None:
        same_top_q = _same_top_q_check(model, first, reference, top_q, config, compiled, amp_dtype)

    mean = statistics.fmean(times_ms)
    return {
        "latency_ms": {
            "mean": mean,
            "median": statistics.median(times_ms),
            "std": statistics.pstdev(times_ms),
            "p95": statistics.quantiles(times_ms, n=20, method="inclusive")[18],
        },
        "fps": 1000 / mean,
        "num_images": len(times_ms),
        "first_call_s": first_call_s,
        "peak_memory_mib": peak_memory_mib,
        "compiled": compiled,
        "amp_dtype": amp_dtype,
        "fallback_errors": errors,
        "max_abs_diff_from_eager_fp32": _max_abs_diff((boxes, logits), reference),
        "same_top_q": same_top_q,
    }


def _attention(model: nn.Module) -> str:
    """Attention implementation of the ViT: SDPA (fused when available) or eager."""
    if isinstance(model, EoDT):
        return "sdpa" if model.backbone.vit.blocks[0].attn.fused_attn else "eager"
    return model.backbone[0].encoder.encoder.config._attn_implementation


def environment_info(device: str) -> dict[str, Any]:
    """GPU, versions, git commit and SLURM job of the run (EODT_SPEC.md §3.7)."""
    return {
        "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": {name: version(name) for name in _PACKAGES},
        "git_commit": _git("rev-parse", "HEAD").strip(),
        "git_status": _git("status", "--porcelain").splitlines(),  # uncommitted and untracked files
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    model_yaml: str | Path | None = None,
    checkpoint: str | Path | None = None,
    published: str | None = None,
) -> dict[str, Any]:
    """Measure one model: parameters, MACs, latency and peak memory, with the protocol and the environment.

    Args:
        config: The settings of the run.
        model_yaml: As in :func:`build_inference_model`.
        checkpoint: As in :func:`build_inference_model`.
        published: As in :func:`build_inference_model`.

    Returns:
        A JSON-serializable record of the run.
    """
    model, model_config = build_inference_model(model_yaml=model_yaml, checkpoint=checkpoint, published=published)
    model.to(config.device)
    return {
        "model": {
            "name": f"rfdetr_{published}" if published else Path(model_yaml).stem,
            "yaml": str(model_yaml) if model_yaml else None,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "published_weights": str(model_config.pretrain_weights) if published else None,
            "resolution": model_config.resolution,
            "attention": _attention(model),
        },
        "protocol": config.model_dump(),
        "params": count_params(model),
        "macs": count_macs(model, model_config.resolution),
        "macs_note": MACS_NOTE,
        **measure_latency(model, val_images(config, model_config), config),
        "environment": environment_info(config.device),
    }


def _jobs(in_first: bool, in_second: bool) -> str:
    """Name the jobs where a problem occurs; at least one of the two flags is true."""
    return "both jobs" if in_first and in_second else f"job {1 if in_first else 2}"


def compare_results(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    """Check that two jobs measured the same model as the protocol asks (step 13).

    A model measured without compilation or in another precision has a record (§6), but its latency cannot be compared
    with the others, even when the two jobs agree.

    Args:
        first: Record of the model in the first job.
        second: Record of the same model in the second job.

    Returns:
        What fails the check: the model is not compiled or not in the precision of the protocol in a job, parameters
        or MACs differ, or the mean latencies differ by more than :data:`MAX_LATENCY_DIFF`. Empty when all is right.
    """
    failures = [
        f"{key} differs: {first[key]} vs {second[key]}" for key in ("params", "macs") if first[key] != second[key]
    ]
    not_compiled = not first["compiled"], not second["compiled"]
    if any(not_compiled):
        failures.append(f"not compiled in {_jobs(*not_compiled)}")
    wrong_precision = [record["amp_dtype"] != record["protocol"]["amp_dtype"] for record in (first, second)]
    if any(wrong_precision):
        wrong = first if wrong_precision[0] else second
        failures.append(
            f"amp_dtype {wrong['amp_dtype']} instead of {wrong['protocol']['amp_dtype']} in {_jobs(*wrong_precision)}"
        )
    elif first["amp_dtype"] != second["amp_dtype"]:  # two jobs with different protocols
        failures.append(f"amp_dtype differs: {first['amp_dtype']} vs {second['amp_dtype']}")
    means = first["latency_ms"]["mean"], second["latency_ms"]["mean"]
    diff = max(means) / min(means) - 1
    if diff > MAX_LATENCY_DIFF:
        failures.append(f"mean latency differs by {diff:.1%}: {means[0]:.3f} vs {means[1]:.3f} ms")
    return failures
