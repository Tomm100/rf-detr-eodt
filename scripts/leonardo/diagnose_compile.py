# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
# ruff: noqa: E402 - TORCHDYNAMO_VERBOSE must be set before torch is imported
"""Find what fails when Inductor compiles the models with two-stage queries (EODT_SPEC.md §9, step 13).

On the A100 every model with the two-stage query selection fell back to eager mode with ``BackendCompilerFailed``,
while EoDT ``learned`` compiled. This script compiles those models piece by piece, from the first stage alone to the
whole models, with the settings of a YAML file of ``configs/benchmark/``, in bf16 and in fp32, and prints the full
traceback of every failure. A case that fails in bf16 is tried again without Inductor's index propagation and without
its unrolled reductions, to see whether the failure goes through them: a diagnosis, not a fix.

The first job showed that ``gen_encoder_output_proposals`` fails alone. The cases named ``<case>_<rewrite>`` run the six
cases that contain it with a rewrite of that function, on the real models and without changing their files, to find
out which rewrite compiles. On a GPU node, from the code folder::

    python scripts/leonardo/diagnose_compile.py configs/benchmark/a100.yaml
    python scripts/leonardo/diagnose_compile.py configs/benchmark/a100.yaml --cases first_stage first_stage_stack_xywh

Every attempt prints its outcome and its time (compilation and first call) as soon as it ends, so that a job stopped
by its time limit keeps the results obtained so far; the table at the end summarizes them.
"""

from __future__ import annotations

import os

os.environ["TORCHDYNAMO_VERBOSE"] = "1"  # read when torch is imported

import argparse
import json
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import cache, partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch._inductor import config as inductor_config

import rfdetr.models.eodt.selection as query_selection
import rfdetr.models.transformer as rfdetr_transformer
from rfdetr.models.eodt.benchmark import (
    BenchmarkConfig,
    _compile,
    _inference_context,
    build_inference_model,
    environment_info,
    load_benchmark_config,
)
from rfdetr.models.eodt.selection import reparam_boxes
from rfdetr.models.transformer import gen_sineembed_for_position

ROOT = Path(__file__).resolve().parents[2]
EODT_SELECTION = ROOT / "configs" / "eodt" / "eodt_vits_576_l2-3_selection.yaml"
EODT_LEARNED = ROOT / "configs" / "eodt" / "eodt_vits_576_l2-3_learned.yaml"
BASELINE = ROOT / "configs" / "baseline" / "baseline_medium_win.yaml"

# Shapes of EoDT ViT-S at 576 px, for the cases that do not need a model
GRID = (36, 36)
PATCHES = GRID[0] * GRID[1]
QUERIES = 300
WIDTH = 384
CLASSES = 91

# Attempts of every case: label, precision, Inductor's settings. An attempt with settings runs only after a bf16 failure
ATTEMPTS = (
    ("bf16", "bf16", {}),
    ("fp32", None, {}),
    ("bf16, no index propagation", "bf16", {"constant_and_index_propagation": False}),
    ("bf16, no unrolled reductions", "bf16", {"unroll_reductions_threshold": 1}),
)


class Run(NamedTuple):
    """Where and in which precision a case runs."""

    device: torch.device
    amp_dtype: str | None

    def randn(self, *shape: int) -> Tensor:
        """Random features in the precision of the run, as the model produces them under autocast."""
        dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float32
        return torch.randn(*shape, device=self.device, dtype=dtype)


class Case(NamedTuple):
    """What a case compiles, given as a function of the run, and the rewrite of the proposals it runs with, if any."""

    build: Callable[[Run], tuple[Callable[..., Any], tuple[Any, ...]]]
    rewrite: Callable[..., tuple[Tensor, Tensor]] | None = None


def _is_valid(boxes: Tensor) -> Tensor:
    """Whether all the coordinates of a proposal are in (0.01, 0.99), as ``gen_encoder_output_proposals`` checks."""
    return ((boxes > 0.01) & (boxes < 0.99)).all(-1, keepdim=True)


def rewritten_proposals(
    memory: Tensor,
    memory_padding_mask: Tensor | None = None,
    spatial_shapes: Sequence[tuple[int, int]] | Tensor | None = None,
    unsigmoid: bool = True,
    *,
    skip_single_cat: bool = False,
    stack_xywh: bool = False,
    valid_before_cat: bool = False,
) -> tuple[Tensor, Tensor]:
    """``gen_encoder_output_proposals`` of ``models/transformer.py`` with a rewrite, to find one that Inductor compiles.

    Signature and branches for the padding mask and ``unsigmoid`` are the original ones, and the outputs in eager mode
    are identical bit by bit (checked by ``tests/eodt/test_benchmark.py``).

    Args:
        memory: As in the original.
        memory_padding_mask: As in the original.
        spatial_shapes: As in the original.
        unsigmoid: As in the original.
        skip_single_cat: Skip the final ``cat`` when there is only one level.
        stack_xywh: Build each level with one ``stack`` of cx, cy, w, h: no ``cat`` inside another.
        valid_before_cat: Keep every ``cat``, but compute the validity before the outer one, so that the reduction over
            the four coordinates does not read through it.

    Returns:
        As in the original: the memory and the proposals, with the invalid proposals zeroed or set to infinity.
    """
    proposals = []
    valid_parts = []
    _cur = 0
    batch_size, _, _ = memory.shape
    assert spatial_shapes is not None
    for lvl, (height, width) in enumerate(spatial_shapes):
        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + height * width)].reshape(batch_size, height, width, 1)

            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            valid_height = torch.zeros_like(memory[:, 0, 0], dtype=torch.long) + height
            valid_width = torch.zeros_like(memory[:, 0, 0], dtype=torch.long) + width

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, height - 1, height, dtype=torch.float32, device=memory.device),
            torch.linspace(0, width - 1, width, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        if stack_xywh:
            center_x = (grid_x.unsqueeze(0) + 0.5) / valid_width.float().reshape(-1, 1, 1)
            center_y = (grid_y.unsqueeze(0) + 0.5) / valid_height.float().reshape(-1, 1, 1)
            size = torch.ones_like(center_x) * 0.05 * (2.0**lvl)
            proposal = torch.stack([center_x, center_y, size, size], -1).reshape(batch_size, -1, 4)
        else:
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # height, width, 2
            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).reshape(-1, 1, 1, 2)
            proposals_grid = (grid.unsqueeze(0) + 0.5) / scale.float()
            wh = torch.ones_like(proposals_grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((proposals_grid, wh), -1).reshape(batch_size, -1, 4)
            if valid_before_cat:
                valid_parts.append((_is_valid(proposals_grid) & _is_valid(wh)).reshape(batch_size, -1, 1))
        proposals.append(proposal)
        _cur += height * width

    output_proposals = proposals[0] if skip_single_cat and len(proposals) == 1 else torch.cat(proposals, 1)
    output_proposals_valid = torch.cat(valid_parts, 1) if valid_before_cat else _is_valid(output_proposals)

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


REWRITES = {
    "no_single_cat": partial(rewritten_proposals, skip_single_cat=True),
    "stack_xywh": partial(rewritten_proposals, stack_xywh=True),
    "valid_before_cat": partial(rewritten_proposals, valid_before_cat=True),
}


@contextmanager
def _using_rewrite(rewrite: Callable[..., tuple[Tensor, Tensor]] | None) -> Iterator[None]:
    """Inside the context RF-DETR and EoDT call ``rewrite``, if given, instead of ``gen_encoder_output_proposals``."""
    original = rfdetr_transformer.gen_encoder_output_proposals
    if rewrite is not None:
        rfdetr_transformer.gen_encoder_output_proposals = query_selection.gen_encoder_output_proposals = rewrite
    try:
        yield
    finally:
        rfdetr_transformer.gen_encoder_output_proposals = query_selection.gen_encoder_output_proposals = original


@cache
def _model(path: Path, device: torch.device) -> tuple[nn.Module, int]:
    """Build a model of the benchmark once, only if a chosen case uses it; return it with its resolution."""
    model, model_config = build_inference_model(model_yaml=path)
    return model.to(device), model_config.resolution


def _image(resolution: int, device: torch.device) -> Tensor:
    """Random image in fp32, as the benchmark gives it to the model also under autocast."""
    return torch.randn(1, 3, resolution, resolution, device=device)


def _boxes(run: Run, count: int) -> tuple[Tensor, ...]:
    """Random box tensors ``[1, Q, 4]``."""
    return tuple(run.randn(1, QUERIES, 4) for _ in range(count))


def first_stage(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The ``QuerySelection`` of EoDT ``selection``, alone."""
    model, _ = _model(EODT_SELECTION, run.device)
    return lambda patches: model.selection(patches, model.backbone.grid_size), (run.randn(1, PATCHES, WIDTH),)


def proposals(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """``gen_encoder_output_proposals``: ``linspace``, ``meshgrid`` and the ``cat`` of the grid and of the boxes."""
    return (
        lambda memory: rfdetr_transformer.gen_encoder_output_proposals(memory, None, [GRID], unsigmoid=False),
        (run.randn(1, PATCHES, WIDTH),),
    )


def topk_gather(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """``torch.topk`` and ``gather``, as in the selection of the first stage."""

    def select(logits: Tensor, values: Tensor) -> Tensor:
        top_q = torch.topk(logits.max(-1)[0], QUERIES, dim=1)[1]
        return torch.gather(values, 1, top_q.unsqueeze(-1).expand(-1, -1, values.shape[-1]))

    return select, (run.randn(1, PATCHES, CLASSES), run.randn(1, PATCHES, 4))


def reparam(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """``reparam_boxes``: one ``cat`` of element-wise inputs."""
    return reparam_boxes, _boxes(run, 2)


def reparam_select(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """One ``cat``, then a constant index along its dimension and an element-wise use."""
    return lambda delta, reference: reparam_boxes(delta, reference)[:, :, 0] * 2.0, _boxes(run, 2)


def reparam_chain(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """A ``cat`` that reads another ``cat`` along the same dimension."""
    return lambda delta, offset, proposal: reparam_boxes(delta, reparam_boxes(offset, proposal)), _boxes(run, 3)


def reparam_chain_select(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Two ``cat`` in a chain, then a constant index along their dimension and an element-wise use."""
    return (
        lambda delta, offset, proposal: reparam_boxes(delta, reparam_boxes(offset, proposal))[:, :, 0] * 2.0,
        _boxes(run, 3),
    )


def sine_embedding(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """``gen_sineembed_for_position`` on a plain tensor, without a ``cat`` before it."""
    return lambda boxes: gen_sineembed_for_position(boxes, WIDTH // 2), _boxes(run, 1)


def eodt_queries(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """L1, first stage, reference boxes and query embedding of EoDT ``selection``, without L2 and the head."""
    model, resolution = _model(EODT_SELECTION, run.device)
    return lambda image: model._queries(image)[:3], (_image(resolution, run.device),)


def eodt_selection(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The whole inference forward of EoDT ``selection``."""
    model, resolution = _model(EODT_SELECTION, run.device)
    return model, (_image(resolution, run.device),)


def eodt_learned(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The whole inference forward of EoDT ``learned``, the control: on the A100 it compiled."""
    model, resolution = _model(EODT_LEARNED, run.device)
    return model, (_image(resolution, run.device),)


def baseline_transformer(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Only the transformer (first stage and decoder) of the baseline, on features computed in eager mode."""
    model, resolution = _model(BASELINE, run.device)
    with _inference_context(run.device, run.amp_dtype):
        srcs, _, poss, cross_attn_srcs = model.backbone(_image(resolution, run.device))
    refpoints, queries = model.refpoint_embed.weight[: model.num_queries], model.query_feat.weight[: model.num_queries]

    def transformer(srcs: list[Tensor], poss: list[Tensor]) -> tuple[Tensor, Tensor]:
        return model.transformer(srcs, None, poss, refpoints, queries, cross_attn_srcs=cross_attn_srcs)[:2]

    return transformer, (srcs, poss)


def baseline_medium_win(run: Run) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The whole inference forward of the baseline with windows."""
    model, resolution = _model(BASELINE, run.device)
    return model, (_image(resolution, run.device),)


CASES: dict[str, Case] = {
    "first_stage": Case(first_stage),
    "proposals": Case(proposals),
    "topk_gather": Case(topk_gather),
    "reparam": Case(reparam),
    "reparam_select": Case(reparam_select),
    "reparam_chain": Case(reparam_chain),
    "reparam_chain_select": Case(reparam_chain_select),
    "sine_embedding": Case(sine_embedding),
    "eodt_queries": Case(eodt_queries),
    "eodt_selection": Case(eodt_selection),
    "eodt_learned": Case(eodt_learned),
    "baseline_transformer": Case(baseline_transformer),
    "baseline_medium_win": Case(baseline_medium_win),
}
# Every rewrite on the six cases that contain gen_encoder_output_proposals: a rewrite that compiles alone is not enough
CASES |= {
    f"{base}_{name}": Case(CASES[base].build, rewrite)
    for base in (
        "proposals",
        "first_stage",
        "eodt_queries",
        "eodt_selection",
        "baseline_transformer",
        "baseline_medium_win",
    )
    for name, rewrite in REWRITES.items()
}


def attempt(
    case: Case, run: Run, config: BenchmarkConfig, inductor_settings: Mapping[str, Any] | None = None
) -> tuple[float, str | None]:
    """Compile and run one case, and print the full traceback if it fails.

    Args:
        case: The case.
        run: Device and precision.
        config: The settings of ``torch.compile``.
        inductor_settings: Values of ``torch._inductor.config`` for this attempt only.

    Returns:
        The time of the compilation and of the first call in seconds, and the error in one line (``None`` if it works).
    """
    function, inputs = case.build(run)
    torch._dynamo.reset()
    error = None
    start = time.perf_counter()
    try:
        with (
            _using_rewrite(case.rewrite),
            inductor_config.patch(**(inductor_settings or {})),
            _inference_context(run.device, run.amp_dtype),
        ):
            _compile(function, config)(*inputs)
    except Exception as exception:  # noqa: BLE001 - a failure is the result of the attempt
        print(traceback.format_exc())
        inner = getattr(exception, "inner_exception", exception)  # BackendCompilerFailed wraps the error of Inductor
        error = f"{type(inner).__name__}: {' '.join(str(inner).splitlines()[:1])}"
    return time.perf_counter() - start, error


def run_cases(cases: Mapping[str, Case], config: BenchmarkConfig) -> dict[str, dict[str, tuple[float, str | None]]]:
    """Run every case in bf16 and fp32 and, if it fails in bf16, in bf16 with the other settings of :data:`ATTEMPTS`.

    Args:
        cases: The cases to run, by name.
        config: The settings of ``torch.compile``.

    Returns:
        For every case and attempt that ran, its time in seconds and its error in one line (``None`` if it works).
    """
    device = torch.device(config.device)
    results: dict[str, dict[str, tuple[float, str | None]]] = {}
    for name, case in cases.items():
        results[name] = {}
        for label, amp_dtype, inductor_settings in ATTEMPTS:
            if inductor_settings and results[name]["bf16"][1] is None:
                continue
            print(f"\n=== {name} | {label}", flush=True)
            seconds, error = attempt(case, Run(device, amp_dtype), config, inductor_settings)
            results[name][label] = seconds, error
            print(f"--- {name} | {label}: {'FAIL' if error else 'ok'} in {seconds:.1f} s {error or ''}", flush=True)
    return results


def print_summary(results: Mapping[str, Mapping[str, tuple[float, str | None]]]) -> None:
    """Print one row per case with the outcome and time of each attempt, then the errors."""
    labels = [label for label, _, _ in ATTEMPTS]
    print("\nSummary\n" + f"{'case':<40}" + "".join(f"{label:<30}" for label in labels))
    for name, row in results.items():
        cells = (
            f"{'FAIL' if row[label][1] else 'ok'} {row[label][0]:.1f} s" if label in row else "-" for label in labels
        )
        print(f"{name:<40}" + "".join(f"{cell:<30}" for cell in cells))
    print("\nErrors")
    for name, row in results.items():
        for label, (_, error) in row.items():
            if error:
                print(f"{name} | {label}: {error}")


def _version(package: str) -> str:
    """Version of an installed package; triton is missing on Windows."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    """Parse the command line, print the environment and run the chosen cases."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("protocol", type=Path, help="YAML file of configs/benchmark/")
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES), help="cases to run")
    args = parser.parse_args()

    config = load_benchmark_config(args.protocol)
    print(json.dumps(environment_info(config.device), indent=2))
    print(f"triton {_version('triton')}, sympy {_version('sympy')}")
    print_summary(run_cases({name: CASES[name] for name in args.cases}, config))


if __name__ == "__main__":
    main()
