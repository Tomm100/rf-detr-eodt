# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the latency benchmark (step 13 of EODT_SPEC.md §9, protocol of §6).

Covers:
- ``EoDT.export()``: the inference forward gives the final outputs of the training forward.
- Every model of the benchmark, as built for inference: the export forward gives the outputs of the eval forward,
  ``torch.compile`` (graph capture with the ``eager`` backend, ``fullgraph``) gives exactly the same outputs, and the
  MACs per component cover the total. The last check found the generator in RF-DETR's ``C2f.forward`` that Dynamo
  evaluated before extending the list.
- The published checkpoint with one group of queries gives the outputs of the one with 13 groups.
- ``count_macs`` counts the attention also on the CPU, where ``FlopCounterMode`` misses the fused SDPA.
- Trained weights load strictly; the YAML of the protocol is the one decided; ``compare_results`` and the
  ``compare`` command apply the 2% criterion and fail a model not compiled or not in the precision of the protocol.
- A run on coco_mini writes a complete record, and a model that does not compile is measured without compiling.
- ``torch.topk`` can be imposed, also on a compiled model: the check of the compiled model with the same top-Q.
- ``scripts/leonardo/diagnose_compile.py`` runs its cases and reports a failure without stopping, and its rewrites of
  ``gen_encoder_output_proposals`` give the bits of the original and reach the real models. Windows has no triton, so
  only the graph capture is tested here: what Inductor does is seen only on Leonardo.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from torch import Tensor, nn

from rfdetr.config import RFDETRMediumConfig
from rfdetr.models.eodt import EoDT, EoDTBackbone, load_configs
from rfdetr.models.eodt.benchmark import (
    MAX_LATENCY_DIFF,
    PUBLISHED,
    BenchmarkConfig,
    _patched_topk,
    build_inference_model,
    compare_results,
    count_macs,
    count_params,
    load_benchmark_config,
    load_checkpoint,
    measure_latency,
    published_config,
    run_benchmark,
)
from rfdetr.models.lwdetr import build_model_from_config
from rfdetr.models.transformer import gen_encoder_output_proposals
from rfdetr.models.weights import load_pretrain_weights

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
COCO_MINI = Path("C:/dev/tesi/data/coco_mini")  # 20 val2017 images (see CLAUDE.md)
BENCHMARK_SCRIPT = ROOT / "scripts" / "eodt" / "benchmark.py"
DIAGNOSE_SCRIPT = ROOT / "scripts" / "leonardo" / "diagnose_compile.py"
VITS = "vit_small_patch14_reg4_dinov2"
MEDIUM_WEIGHTS = Path(str(published_config("medium").pretrain_weights))
needs_medium = pytest.mark.skipif(not MEDIUM_WEIGHTS.exists(), reason=f"{MEDIUM_WEIGHTS} not found")

MODEL_YAMLS = {
    "eodt_selection": CONFIGS / "eodt" / "eodt_vits_576_l2-3_selection.yaml",
    "eodt_learned": CONFIGS / "eodt" / "eodt_vits_576_l2-3_learned.yaml",
    "baseline_win": CONFIGS / "baseline" / "baseline_medium_win.yaml",
    "baseline_nowin": CONFIGS / "baseline" / "baseline_medium_nowin.yaml",
    "baseline_nowin_noproj": CONFIGS / "baseline" / "baseline_medium_nowin_noproj.yaml",
}
_REWRITES = ("no_single_cat", "stack_xywh", "valid_before_cat")
_CASES_WITH_PROPOSALS = (
    "proposals",
    "first_stage",
    "eodt_queries",
    "eodt_selection",
    "baseline_transformer",
    "baseline_medium_win",
)
_CASE_NAMES = (
    *_CASES_WITH_PROPOSALS,
    "topk_gather",
    "reparam",
    "reparam_select",
    "reparam_chain",
    "reparam_chain_select",
    "sine_embedding",
    "eodt_learned",
    *(f"{case}_{rewrite}" for case in _CASES_WITH_PROPOSALS for rewrite in _REWRITES),
)
EODT_COMPONENTS = {"vit_l1", "vit_l2", "first_stage", "heads"}
LWDETR_COMPONENTS = {"vit", "projector", "first_stage", "decoder", "heads"}


def _image(resolution: int, seed: int = 0) -> Tensor:
    """Return a fixed random image ``[1, 3, R, R]``."""
    return torch.randn(1, 3, resolution, resolution, generator=torch.Generator().manual_seed(seed))


def _small_eodt(query_mode: str, seed: int = 0) -> EoDT:
    """Return a ViT-S EoDT at 224 px with 10 queries and a box head that is not zero."""
    torch.manual_seed(seed)
    model = EoDT(EoDTBackbone(VITS, 224, 16, 3, pretrained=False), 10, 91, query_mode=query_mode)
    nn.init.normal_(model.bbox_embed.layers[-1].weight, std=0.01)  # zero with selection: boxes would not depend on hs
    return model.eval()


@pytest.mark.parametrize("query_mode", ["learned", "selection"])
def test_eodt_forward_export_matches_forward(query_mode: str) -> None:
    """After ``export()`` EoDT returns ``(boxes, logits)`` of the last L2 block, as its training forward."""
    model = _small_eodt(query_mode)
    image = _image(224)

    with torch.no_grad():
        out = model(image)
        model.export()
        boxes, logits = model(image)

    torch.testing.assert_close(boxes, out["pred_boxes"])
    torch.testing.assert_close(logits, out["pred_logits"])


@pytest.mark.parametrize("name", list(MODEL_YAMLS))
def test_inference_model(name: str) -> None:
    """The benchmarked model computes the eval forward, also when compiled, and its MACs per component add up.

    Built twice with the same seed: as for training (eval forward) and for the benchmark (``export()``).
    """
    path = MODEL_YAMLS[name]
    torch.manual_seed(0)
    model, model_config = build_inference_model(model_yaml=path)
    torch.manual_seed(0)
    reference = build_model_from_config(load_configs(path)[0]).eval()
    image = _image(model_config.resolution)

    with torch.no_grad():
        out = reference(image)
        boxes, logits = model(image)
        torch._dynamo.reset()
        compiled_boxes, compiled_logits = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)(image)

    torch.testing.assert_close(boxes, out["pred_boxes"])
    torch.testing.assert_close(logits, out["pred_logits"])
    assert torch.equal(compiled_boxes, boxes)
    assert torch.equal(compiled_logits, logits)

    macs = count_macs(model, model_config.resolution)
    components = EODT_COMPONENTS if isinstance(model, EoDT) else LWDETR_COMPONENTS
    assert set(macs) == {"total", "other", *components}
    assert macs["other"] == 0
    first_stage_absent = name == "eodt_learned"
    assert all(macs[key] > 0 for key in components - ({"first_stage"} if first_stage_absent else set()))
    assert macs["first_stage"] == 0 if first_stage_absent else macs["first_stage"] > 0


@needs_medium
def test_published_one_group_matches_thirteen() -> None:
    """With one group of queries the published Medium gives the outputs of the checkpoint's 13 groups, compiled too.

    Inference reads only the first group, so the other 12 are left out of the model and of its parameters.
    """
    model, model_config = build_inference_model(published="medium")
    full_config = RFDETRMediumConfig(pretrain_weights=str(MEDIUM_WEIGHTS))
    full = build_model_from_config(full_config)
    load_pretrain_weights(full, full_config)
    full.eval()
    full.export()
    image = _image(model_config.resolution)

    with torch.no_grad():
        boxes, logits = model(image)
        full_boxes, full_logits = full(image)
        torch._dynamo.reset()
        compiled_boxes, compiled_logits = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)(image)

    assert full_config.group_detr == 13 and model_config.group_detr == 1
    torch.testing.assert_close(boxes, full_boxes)
    torch.testing.assert_close(logits, full_logits)
    assert torch.equal(compiled_boxes, boxes)
    assert torch.equal(compiled_logits, logits)
    assert count_params(model) < count_params(full)


def test_published_configs() -> None:
    """Every published model keeps RF-DETR's architecture and weight file, with one group of queries."""
    for name, config_class in PUBLISHED.items():
        config = published_config(name)
        default = config_class()
        assert config.group_detr == 1
        assert Path(str(config.pretrain_weights)).name == config_class.model_fields["pretrain_weights"].default
        for field in ("resolution", "patch_size", "num_windows", "dec_layers", "num_queries", "hidden_dim", "encoder"):
            assert getattr(config, field) == getattr(default, field), (name, field)


class _Attention(nn.Module):
    """One linear layer and one SDPA on the pixels of the image as tokens: 2 heads of dimension 4."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(3, 24)

    def forward(self, image: Tensor) -> Tensor:
        tokens = image.flatten(2).transpose(1, 2)  # [1, N, 3]
        q, k, v = self.qkv(tokens).unflatten(-1, (3, 2, 4)).permute(2, 0, 3, 1, 4)  # each [1, 2, N, 4]
        return F.scaled_dot_product_attention(q, k, v)


def test_count_macs_counts_sdpa_on_cpu() -> None:
    """On the CPU the attention is counted too: QK^T and AV, N * N * d MACs each per head."""
    resolution = 4
    tokens = resolution * resolution
    expected = tokens * 3 * 24 + 2 * (2 * tokens * tokens * 4)  # linear + two heads x (QK^T + AV)

    assert count_macs(_Attention(), resolution)["total"] == expected


@pytest.mark.gpu
def test_count_macs_same_on_cpu_and_gpu() -> None:
    """The math backend of SDPA gives the same count on CPU and GPU."""
    model = _small_eodt("selection")
    model.export()
    model.requires_grad_(False)

    on_cpu = count_macs(model, 224)
    on_gpu = count_macs(model.cuda(), 224)

    assert on_cpu == on_gpu


def test_load_checkpoint_is_strict(tmp_path: Path) -> None:
    """A ``.pth`` of RF-DETR's ``BestModelCallback`` loads all the weights; a missing tensor is an error."""
    model = _small_eodt("selection")
    path = tmp_path / "last_ema.pth"
    torch.save({"model": model.state_dict(), "args": {"epochs": 12}}, path)
    other = _small_eodt("selection", seed=1)

    load_checkpoint(other, path)

    for (name, tensor), (other_name, other_tensor) in zip(model.state_dict().items(), other.state_dict().items()):
        assert name == other_name and torch.equal(tensor, other_tensor), name
    state = model.state_dict()
    del state["query_feat.weight"]
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(other, path)


def test_protocol_files(tmp_path: Path) -> None:
    """``a100.yaml`` holds the protocol of §6 as decided for step 13; a wrong file is rejected."""
    a100 = load_benchmark_config(CONFIGS / "benchmark" / "a100.yaml")
    local = load_benchmark_config(CONFIGS / "benchmark" / "local.yaml")

    assert a100.model_dump() == {
        "dataset_dir": "/leonardo_scratch/fast/IscrC_EoDT/coco",
        "num_images": None,
        "num_workers": 7,
        "warmup_iters": 500,
        "device": "cuda",
        "amp_dtype": "bf16",
        "compile": True,
        "compile_mode": "default",
        "compile_backend": "inductor",
        "fullgraph": True,
    }
    # the local trial differs only where the PC cannot do the same
    differences = {key for key, value in local.model_dump().items() if getattr(a100, key) != value}
    assert differences == {"dataset_dir", "num_workers", "warmup_iters", "amp_dtype", "compile_backend"}

    content = yaml.safe_load((CONFIGS / "benchmark" / "a100.yaml").read_text(encoding="utf-8"))
    for broken in ({**content, "batch_size": 1}, {key: value for key, value in content.items() if key != "fullgraph"}):
        path = tmp_path / "broken.yaml"
        path.write_text(yaml.safe_dump(broken), encoding="utf-8")
        with pytest.raises(ValueError):
            load_benchmark_config(path)
    path.write_text((CONFIGS / "benchmark" / "a100.yaml").read_text(encoding="utf-8") + "fullgraph: false\n", "utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_benchmark_config(path)


def _record(mean_ms: float, **changes: Any) -> dict[str, Any]:
    """Return a minimal benchmark record with the given mean latency."""
    return {
        "model": {"name": "m"},
        "latency_ms": {"mean": mean_ms},
        "fps": 1000 / mean_ms,
        "peak_memory_mib": 100.0,
        "params": 1000,
        "macs": {"total": 50, "vit": 50, "other": 0},
        "compiled": True,
        "amp_dtype": "bf16",
        "protocol": {"amp_dtype": "bf16"},
        **changes,
    }


def test_compare_results() -> None:
    """Same parameters and MACs and mean latencies within 2%."""
    assert MAX_LATENCY_DIFF == 0.02
    assert compare_results(_record(10.0), _record(10.15)) == []
    assert len(compare_results(_record(10.0), _record(10.3))) == 1
    assert len(compare_results(_record(10.3), _record(10.0))) == 1  # symmetric
    assert len(compare_results(_record(10.0), _record(10.0, params=1001))) == 1


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ({"compiled": False}, {}, "not compiled in job 1"),
        ({}, {"compiled": False}, "not compiled in job 2"),
        ({"compiled": False}, {"compiled": False}, "not compiled in both jobs"),
        ({"amp_dtype": None}, {}, "amp_dtype None instead of bf16 in job 1"),
        ({}, {"amp_dtype": None}, "amp_dtype None instead of bf16 in job 2"),
        ({"amp_dtype": None}, {"amp_dtype": None}, "amp_dtype None instead of bf16 in both jobs"),
    ],
    ids=["eager_1", "eager_2", "eager_both", "fp32_1", "fp32_2", "fp32_both"],
)
def test_compare_results_fails_outside_the_protocol(
    first: dict[str, Any], second: dict[str, Any], expected: str
) -> None:
    """A model not compiled or not in the precision of the protocol fails, even when the two jobs agree."""
    assert compare_results(_record(10.0, **first), _record(10.0, **second)) == [expected]


def test_compare_results_follows_the_precision_of_the_protocol() -> None:
    """fp32 is right when the protocol asks for it (GTX 1650), and two jobs with different protocols still differ."""
    fp32 = {"amp_dtype": None, "protocol": {"amp_dtype": None}}

    assert compare_results(_record(10.0, **fp32), _record(10.0, **fp32)) == []
    assert compare_results(_record(10.0), _record(10.0, **fp32)) == ["amp_dtype differs: bf16 vs None"]


def test_compare_command(tmp_path: Path) -> None:
    """``benchmark.py compare`` prints a verdict per model and returns 1 when one of them fails."""
    means = {"job1": {"stable": 10.0, "unstable": 10.0}, "job2": {"stable": 10.15, "unstable": 10.3}}
    for job, models in means.items():
        (tmp_path / job).mkdir()
        for name, mean in models.items():
            (tmp_path / job / f"{name}.json").write_text(json.dumps(_record(mean)), encoding="utf-8")
        (tmp_path / job / "eager.json").write_text(json.dumps(_record(10.0, compiled=False)), encoding="utf-8")
        (tmp_path / job / "fp32.json").write_text(json.dumps(_record(10.0, amp_dtype=None)), encoding="utf-8")

    run = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "compare", str(tmp_path / "job1"), str(tmp_path / "job2")],
        capture_output=True,
        text=True,
    )

    assert run.returncode == 1, run.stderr
    lines = run.stdout.splitlines()
    names = ("stable", "unstable", "eager", "fp32")
    rows = {line.split()[0]: line for line in lines if line.startswith(names) and ("PASS" in line or "FAIL" in line)}
    assert "compiled" in lines[0] and "amp" in lines[0]
    assert rows["stable"].endswith("PASS") and "yes/yes" in rows["stable"] and "bf16/bf16" in rows["stable"]
    assert "FAIL: mean latency differs by 3.0%" in rows["unstable"]
    assert "no/no" in rows["eager"] and "FAIL: not compiled in both jobs" in rows["eager"]
    assert "fp32/fp32" in rows["fp32"] and "FAIL: amp_dtype None instead of bf16 in both jobs" in rows["fp32"]
    assert run.stdout.strip().endswith("step 13: FAIL")


def _coco_mini_config(**changes: Any) -> BenchmarkConfig:
    """Return a short run on the CPU over coco_mini, with graph capture and fullgraph."""
    if not COCO_MINI.exists():
        pytest.skip(f"{COCO_MINI} not found")
    return BenchmarkConfig(
        **{
            "dataset_dir": str(COCO_MINI),
            "num_images": 3,
            "num_workers": 0,
            "warmup_iters": 2,
            "device": "cpu",
            "amp_dtype": None,
            "compile": True,
            "compile_mode": "default",
            "compile_backend": "eager",
            "fullgraph": True,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "source",
    [
        {"model_yaml": MODEL_YAMLS["eodt_selection"]},
        {"model_yaml": MODEL_YAMLS["baseline_win"]},
        pytest.param({"published": "medium"}, marks=needs_medium),
    ],
    ids=["eodt_selection", "baseline_win", "rfdetr_medium"],
)
def test_run_on_coco_mini(source: dict[str, Any]) -> None:
    """A run writes a complete JSON record: compiled, one time per image, same outputs as the eager model."""
    record = run_benchmark(_coco_mini_config(), **source)

    json.dumps(record)  # serializable
    assert record["compiled"] is True and record["amp_dtype"] is None and record["fallback_errors"] == []
    assert record["num_images"] == 3
    no_difference = {"scores": 0.0, "boxes": 0.0}
    assert record["max_abs_diff_from_eager_fp32"] == no_difference
    assert record["same_top_q"] == {
        "positions_changed": 0,
        "patches_changed": 0,
        "eager_vs_eager_fp32": no_difference,
        "compiled_vs_eager": no_difference,
    }
    assert 0 < record["latency_ms"]["median"] <= record["latency_ms"]["p95"]
    assert record["fps"] == pytest.approx(1000 / record["latency_ms"]["mean"])
    assert record["macs"]["other"] == 0 and record["params"] > 0
    assert record["model"]["attention"] == "sdpa" and record["model"]["resolution"] == 576
    assert record["protocol"]["compile_backend"] == "eager"
    assert record["environment"]["git_commit"]


class _GraphBreak(nn.Module):
    """A model with a graph break: ``torch.compile`` with ``fullgraph`` fails on it."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        torch._dynamo.graph_break()
        value = image.mean() * self.scale
        return value.expand(1, 2, 4), value.expand(1, 2, 3)


def test_measure_without_compile_when_it_fails() -> None:
    """A model that does not compile is measured in eager mode and the error is recorded (§6)."""
    images = [torch.ones(1, 3, 4, 4)] * 3

    result = measure_latency(_GraphBreak(), images, _coco_mini_config(num_images=None))

    assert result["compiled"] is False
    assert len(result["fallback_errors"]) == 1 and "compile=True" in result["fallback_errors"][0]
    assert result["num_images"] == 3
    assert result["same_top_q"] is None  # no top-Q selection in this model


@pytest.mark.parametrize("name", ["eodt_selection", "baseline_win"])
def test_imposed_top_q(name: str) -> None:
    """The recorded ranking changes nothing; another ranking is not a permutation, and reaches a compiled model.

    Reversing the ranking does not just reverse the outputs: the query in position i gets the i-th ``query_feat``.
    """
    model, model_config = build_inference_model(model_yaml=MODEL_YAMLS[name])
    image = _image(model_config.resolution)

    with torch.no_grad(), _patched_topk() as ranking:
        free_boxes, free_logits = model(image)
    with torch.no_grad(), _patched_topk(ranking[0]):
        same_boxes, same_logits = model(image)
    with torch.no_grad(), _patched_topk(ranking[0].flip(1)):
        other_boxes, other_logits = model(image)
        torch._dynamo.reset()
        compiled_boxes, compiled_logits = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)(image)
    torch._dynamo.reset()

    assert len(ranking) == 1 and ranking[0].shape == (1, model_config.num_queries)
    assert torch.equal(same_boxes, free_boxes) and torch.equal(same_logits, free_logits)
    assert not torch.equal(other_boxes, free_boxes)
    assert not torch.equal(other_logits.flip(1), free_logits)
    assert torch.equal(compiled_boxes, other_boxes) and torch.equal(compiled_logits, other_logits)


def _diagnose_script(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the diagnosis script; the variable it sets is restored at the end of the test."""
    monkeypatch.setenv("TORCHDYNAMO_VERBOSE", "1")
    spec = importlib.util.spec_from_file_location("diagnose_compile", DIAGNOSE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_only_protocol(tmp_path: Path) -> Path:
    """Write a protocol on the CPU with graph capture only, the one thing possible without triton."""
    protocol = {
        "dataset_dir": "unused",
        "num_images": None,
        "num_workers": 0,
        "warmup_iters": 1,
        "device": "cpu",
        "amp_dtype": None,
        "compile": True,
        "compile_mode": "default",
        "compile_backend": "eager",
        "fullgraph": True,
    }
    path = tmp_path / "capture_only.yaml"
    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    return path


def _run_diagnosis(tmp_path: Path, *cases: str) -> subprocess.CompletedProcess[str]:
    """Run the diagnosis script in its own process, as on Leonardo."""
    return subprocess.run(
        [sys.executable, str(DIAGNOSE_SCRIPT), str(_capture_only_protocol(tmp_path)), *cases],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_diagnosis_runs_only_the_chosen_cases(tmp_path: Path) -> None:
    """``--cases`` runs only the named cases, in bf16 and fp32, each with its time, and builds no model for them."""
    run = _run_diagnosis(tmp_path, "--cases", "reparam_select", "reparam_chain_select")

    assert run.returncode == 0, run.stderr
    assert re.search(r"--- reparam_select \| bf16: ok in \d+\.\d s", run.stdout)
    assert re.search(r"--- reparam_chain_select \| fp32: ok in \d+\.\d s", run.stdout)
    assert "=== first_stage" not in run.stdout and "no index propagation: " not in run.stdout


def test_diagnosis_rejects_an_unknown_case(tmp_path: Path) -> None:
    """An unknown case is an error that lists the valid ones."""
    run = _run_diagnosis(tmp_path, "--cases", "nonexistent")

    assert run.returncode == 2
    assert "invalid choice: 'nonexistent'" in run.stderr and "reparam_chain_select" in run.stderr


@pytest.mark.gpu
def test_diagnosis_runs_every_case(tmp_path: Path) -> None:
    """All the cases compile with graph capture: a mistake in the script must not waste a job on Leonardo."""
    run = _run_diagnosis(tmp_path)

    assert run.returncode == 0, run.stderr
    summary = run.stdout[run.stdout.index("\nSummary\n") :]
    assert "FAIL" not in summary
    for name in _CASE_NAMES:
        assert re.search(rf"^{name} +ok \d+\.\d s +ok \d+\.\d s +-", summary, flags=re.M), name


def test_diagnosis_reports_a_failure_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    """A case that fails prints its traceback, is retried without index propagation, and does not stop the others."""
    diagnose = _diagnose_script(monkeypatch)
    config = load_benchmark_config(_capture_only_protocol(tmp_path))

    def broken(run: Any) -> tuple[nn.Module, tuple[Tensor]]:
        return _GraphBreak(), (run.randn(1, 3, 4, 4),)

    results = diagnose.run_cases({"broken": diagnose.Case(broken), "reparam": diagnose.CASES["reparam"]}, config)
    diagnose.print_summary(results)
    out = capsys.readouterr().out

    assert set(results["broken"]) == {"bf16", "fp32", "bf16, no index propagation", "bf16, no unrolled reductions"}
    assert all(error is not None and seconds > 0 for seconds, error in results["broken"].values())
    assert set(results["reparam"]) == {"bf16", "fp32"} and results["reparam"]["bf16"][1] is None
    assert "Traceback (most recent call last)" in out
    assert "--- broken | bf16: FAIL in" in out and "--- reparam | bf16: ok in" in out
    assert "broken | fp32: Unsupported" in out[out.index("\nErrors\n") :]


def _same_bits(first: Tensor, second: Tensor) -> bool:
    """Whether two tensors have the same type, shape and bytes: ``torch.equal`` does not tell 0.0 from -0.0."""
    return (
        first.dtype == second.dtype
        and first.shape == second.shape
        and first.float().cpu().numpy().tobytes() == second.float().cpu().numpy().tobytes()
    )


def _padding_mask(shapes: list[tuple[int, int]], batch: int, device: str) -> Tensor:
    """Return a padding mask with images of different sizes and, with more images, one that is all padding."""
    levels = []
    for height, width in shapes:
        level = torch.zeros(batch, height, width, dtype=torch.bool, device=device)
        level[0, -1:, :] = True
        level[0, :, -1:] = True
        if batch > 1:
            level[-1] = True  # no valid row or column: a division by zero, then overwritten
        levels.append(level.flatten(1))
    return torch.cat(levels, 1)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("rewrite", _REWRITES)
def test_rewritten_proposals_are_identical_bit_by_bit(
    monkeypatch: pytest.MonkeyPatch, rewrite: str, device: str
) -> None:
    """In eager mode every rewrite gives the bits of the original: padding mask, unsigmoid, levels, batch, precision."""
    function = _diagnose_script(monkeypatch).REWRITES[rewrite]
    generator = torch.Generator().manual_seed(0)
    assert not _same_bits(torch.tensor([0.0]), torch.tensor([-0.0]))  # the comparison sees the sign of a zero

    for shapes in ([(4, 6)], [(4, 6), (2, 3)]):
        for batch in (1, 3):
            size = sum(height * width for height, width in shapes)
            noise = torch.randn(batch, size, 8, generator=generator).to(device)
            for dtype in (torch.float32, torch.bfloat16):
                memory = noise.to(dtype)
                for mask in (None, _padding_mask(shapes, batch, device)):
                    for unsigmoid in (True, False):
                        original = gen_encoder_output_proposals(memory, mask, shapes, unsigmoid=unsigmoid)
                        rewritten = function(memory, mask, shapes, unsigmoid=unsigmoid)
                        where = (shapes, batch, dtype, mask is None, unsigmoid)
                        assert _same_bits(rewritten[0], original[0]), where
                        assert _same_bits(rewritten[1], original[1]), where


@pytest.mark.parametrize("case", ["first_stage", "baseline_transformer"])
def test_rewrite_reaches_the_real_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    """During an attempt the model calls the rewrite, also when compiled, and afterwards the original is back."""
    diagnose = _diagnose_script(monkeypatch)
    config = load_benchmark_config(_capture_only_protocol(tmp_path))
    calls: list[int] = []

    def counting(*args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        calls.append(1)
        return gen_encoder_output_proposals(*args, **kwargs)

    run = diagnose.Run(torch.device("cpu"), None)
    _, error = diagnose.attempt(diagnose.Case(diagnose.CASES[case].build, counting), run, config)

    assert error is None and calls
    assert diagnose.rfdetr_transformer.gen_encoder_output_proposals is gen_encoder_output_proposals
    assert diagnose.query_selection.gen_encoder_output_proposals is gen_encoder_output_proposals
