# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Train EoDT or the baseline from its YAML file (EODT_SPEC.md §9, step 6).

On Leonardo it runs once per GPU, launched by ``srun`` in ``scripts/leonardo/train.sh``; on the PC it also runs on
its own::

    python scripts/leonardo/train.py configs/eodt/eodt_vits_576_l2-3_learned.yaml
    python scripts/leonardo/train.py configs/eodt/eodt_vits_576_l2-3_learned.yaml --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr.models.eodt.train import train


def main() -> None:
    """Parse the command line and train."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="YAML file read by rfdetr.models.eodt.load_configs")
    parser.add_argument("--resume", action="store_true", help="continue the run from last.ckpt in its output_dir")
    parser.add_argument("--run-type", choices=("run", "test"), default="run", help="W&B tag of the run")
    parser.add_argument("--limit-train-batches", type=int, help="training batches per epoch, test jobs only")
    parser.add_argument("--limit-val-batches", type=int, help="validation batches per epoch, test jobs only")
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="last epoch of this job, counted from 1: test jobs, and runs split into jobs of whole epochs",
    )
    args = parser.parse_args()
    train(
        args.config,
        resume=args.resume,
        run_type=args.run_type,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        stop_after_epoch=args.stop_after_epoch,
    )


if __name__ == "__main__":
    main()
