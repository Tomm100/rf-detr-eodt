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
  ``compare`` command apply the 2% criterion.
- A run on coco_mini writes a complete record, and a model that does not compile is measured without compiling.
"""

import json
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
from rfdetr.models.weights import load_pretrain_weights

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
COCO_MINI = Path("C:/dev/tesi/data/coco_mini")  # 20 val2017 images (see CLAUDE.md)
BENCHMARK_SCRIPT = ROOT / "scripts" / "eodt" / "benchmark.py"
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
        **changes,
    }


def test_compare_results() -> None:
    """Same parameters, MACs, compilation and precision, and mean latencies within 2%."""
    assert MAX_LATENCY_DIFF == 0.02
    assert compare_results(_record(10.0), _record(10.15)) == []
    assert len(compare_results(_record(10.0), _record(10.3))) == 1
    assert len(compare_results(_record(10.3), _record(10.0))) == 1  # symmetric
    assert len(compare_results(_record(10.0), _record(10.0, params=1001))) == 1
    assert len(compare_results(_record(10.0), _record(10.0, compiled=False))) == 1


def test_compare_command(tmp_path: Path) -> None:
    """``benchmark.py compare`` prints a verdict per model and returns 1 when one of them fails."""
    means = {"job1": {"stable": 10.0, "unstable": 10.0}, "job2": {"stable": 10.15, "unstable": 10.3}}
    for job, models in means.items():
        (tmp_path / job).mkdir()
        for name, mean in models.items():
            (tmp_path / job / f"{name}.json").write_text(json.dumps(_record(mean)), encoding="utf-8")

    run = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "compare", str(tmp_path / "job1"), str(tmp_path / "job2")],
        capture_output=True,
        text=True,
    )

    assert run.returncode == 1, run.stderr
    lines = run.stdout.splitlines()
    rows = {line.split()[0]: line for line in lines if line.startswith(("stable", "unstable")) and "PASS" in line}
    rows |= {line.split()[0]: line for line in lines if line.startswith(("stable", "unstable")) and "FAIL" in line}
    assert rows["stable"].endswith("PASS")
    assert "FAIL: mean latency differs by 3.0%" in rows["unstable"]
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
    assert record["max_abs_diff_from_eager_fp32"] == {"scores": 0.0, "boxes": 0.0}
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
