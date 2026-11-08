# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Latency benchmark of EoDT, of the baseline and of RF-DETR's published checkpoints (EODT_SPEC.md §6, step 13).

Measure one model per process, with the protocol of a YAML file of ``configs/benchmark/``::

    python scripts/eodt/benchmark.py run configs/benchmark/a100.yaml \
        --model configs/eodt/eodt_vits_576_l2-3_selection.yaml [--checkpoint <run>/last_ema.pth] --output <job>/x.json
    python scripts/eodt/benchmark.py run configs/benchmark/a100.yaml --published medium --output <job>/rfdetr_medium.json

On Leonardo ``scripts/leonardo/benchmark.sh`` measures every model in one job. Then compare two jobs, the test of
step 13 (same parameters and MACs, mean latencies within 2%)::

    python scripts/eodt/benchmark.py compare <job 1> <job 2>

It prints, for every model of the first job, latency, FPS, peak memory, parameters, compilation and precision in the
two jobs, MACs per component, and the verdict of the comparison; the exit code is 1 if a model fails or is missing from
the second job. A model not compiled, or not in the precision of the protocol, fails even when the two jobs agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rfdetr.models.eodt.benchmark import MACS_NOTE, PUBLISHED, compare_results, load_benchmark_config, run_benchmark

COMPILED = {True: "yes", False: "no"}
PRECISION = {"bf16": "bf16", None: "fp32"}


def run(args: argparse.Namespace) -> None:
    """Measure one model and write its record as JSON."""
    record = run_benchmark(
        load_benchmark_config(args.protocol),
        model_yaml=args.model,
        checkpoint=args.checkpoint,
        published=args.published,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    latency = record["latency_ms"]
    print(
        f"{record['model']['name']}: {latency['mean']:.3f} ms (median {latency['median']:.3f}, std {latency['std']:.3f})"
        f", {record['fps']:.1f} FPS, compiled {record['compiled']}, amp {record['amp_dtype']}, "
        f"{record['macs']['total'] / 1e9:.2f} GMAC, {record['params'] / 1e6:.2f} M parameters -> {args.output}"
    )
    print(f"  max_abs_diff_from_eager_fp32: {json.dumps(record['max_abs_diff_from_eager_fp32'])}")
    print(f"  same_top_q: {json.dumps(record['same_top_q'])}")


def _read_job(folder: Path) -> dict[str, dict[str, Any]]:
    """Records of a job folder, by model name."""
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))}


def _per_job(record: dict[str, Any], other: dict[str, Any] | None, key: str, names: dict[Any, str]) -> str:
    """Value of ``key`` in the two jobs, as ``first/second``, with ``-`` for a model missing from the second."""
    return f"{names[record[key]]}/{names[other[key]] if other else '-'}"


def compare(args: argparse.Namespace) -> int:
    """Print the results of the first job with the latency of the second, and the verdict of step 13."""
    first, second = _read_job(args.first), _read_job(args.second)
    if not first:
        raise SystemExit(f"no JSON record in {args.first}")
    failed = False
    print(
        f"{'model':<34} {'ms (1)':>8} {'ms (2)':>8} {'diff':>6} {'FPS':>7} {'MiB':>7} {'M par':>6} {'GMAC':>6} "
        f"{'compiled':>8} {'amp':>9}  verdict"
    )
    for name, record in first.items():
        other = second.get(name)
        failures = compare_results(record, other) if other else [f"missing from {args.second}"]
        failed |= bool(failures)
        mean = record["latency_ms"]["mean"]
        mean_2 = f"{other['latency_ms']['mean']:8.3f} {other['latency_ms']['mean'] / mean - 1:+6.1%}" if other else ""
        memory = record["peak_memory_mib"]
        print(
            f"{name:<34} {mean:8.3f} {mean_2:>15} {record['fps']:7.1f} {memory or float('nan'):7.0f} "
            f"{record['params'] / 1e6:6.2f} {record['macs']['total'] / 1e9:6.2f} "
            f"{_per_job(record, other, 'compiled', COMPILED):>8} {_per_job(record, other, 'amp_dtype', PRECISION):>9}  "
            + ("PASS" if not failures else "FAIL: " + "; ".join(failures))
        )
    print("\nGMAC per component (first job)")
    for name, record in first.items():
        components = ", ".join(f"{key} {value / 1e9:.3f}" for key, value in record["macs"].items() if key != "total")
        print(f"{name:<34} {components}")
    print(f"\n{MACS_NOTE}")
    print("\nstep 13:", "FAIL" if failed else "PASS")
    return int(failed)


def main() -> None:
    """Parse the command line and run ``run`` or ``compare``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="measure one model")
    run_parser.add_argument("protocol", type=Path, help="YAML file of configs/benchmark/")
    source = run_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="experiment YAML of configs/eodt or configs/baseline")
    source.add_argument("--published", choices=list(PUBLISHED), help="published RF-DETR checkpoint")
    run_parser.add_argument("--checkpoint", type=Path, help="trained weights (.pth with a 'model' key) for --model")
    run_parser.add_argument("--output", type=Path, required=True, help="JSON file of the record")

    compare_parser = commands.add_parser("compare", help="compare two jobs (test of step 13)")
    compare_parser.add_argument("first", type=Path, help="folder of the JSON records of the first job")
    compare_parser.add_argument("second", type=Path, help="folder of the JSON records of the second job")

    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        sys.exit(compare(args))


if __name__ == "__main__":
    main()
