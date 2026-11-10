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

The cause was ``gen_encoder_output_proposals``, rewritten since then (EODT_SPEC.md §8, point 10): every case must now
compile. On a GPU node, from the code folder::

    python scripts/leonardo/diagnose_compile.py configs/benchmark/a100.yaml
    python scripts/leonardo/diagnose_compile.py configs/benchmark/a100.yaml --cases first_stage reparam_select

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
from collections.abc import Callable, Mapping
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch._inductor import config as inductor_config

from rfdetr.models.eodt.benchmark import (
    BenchmarkConfig,
    _compile,
    _inference_context,
    build_inference_model,
    environment_info,
    load_benchmark_config,
)
from rfdetr.models.eodt.selection import reparam_boxes
from rfdetr.models.transformer import gen_encoder_output_proposals, gen_sineembed_for_position

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


Case = Callable[[Run], tuple[Callable[..., Any], tuple[Any, ...]]]  # gives the function to compile and its inputs


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
    """``gen_encoder_output_proposals``: ``linspace``, ``meshgrid``, the ``stack`` of the boxes and their validity."""
    return (
        lambda memory: gen_encoder_output_proposals(memory, None, [GRID], unsigmoid=False),
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
    "first_stage": first_stage,
    "proposals": proposals,
    "topk_gather": topk_gather,
    "reparam": reparam,
    "reparam_select": reparam_select,
    "reparam_chain": reparam_chain,
    "reparam_chain_select": reparam_chain_select,
    "sine_embedding": sine_embedding,
    "eodt_queries": eodt_queries,
    "eodt_selection": eodt_selection,
    "eodt_learned": eodt_learned,
    "baseline_transformer": baseline_transformer,
    "baseline_medium_win": baseline_medium_win,
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
    function, inputs = case(run)
    torch._dynamo.reset()
    error = None
    start = time.perf_counter()
    try:
        with inductor_config.patch(**(inductor_settings or {})), _inference_context(run.device, run.amp_dtype):
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
    print("\nSummary\n" + f"{'case':<22}" + "".join(f"{label:<30}" for label in labels))
    for name, row in results.items():
        cells = (
            f"{'FAIL' if row[label][1] else 'ok'} {row[label][0]:.1f} s" if label in row else "-" for label in labels
        )
        print(f"{name:<22}" + "".join(f"{cell:<30}" for cell in cells))
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
