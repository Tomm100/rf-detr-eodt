# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Overfit EoDT on the 20 images of coco_mini with RF-DETR's training stack (step 5 of EODT_SPEC.md §9).

Usage::

    .venv/Scripts/python.exe scripts/eodt/overfit.py configs/eodt/eodt_vits_576_l2-3_learned_overfit.yaml

Trains with ``build_trainer`` + ``RFDETRModelModule`` + ``RFDETRDataModule``, as ``RFDETR.train()`` does, then writes
to the ``output_dir`` of the YAML file:
- ``curves.png``: loss and mAP per epoch, read from Lightning's ``metrics.csv``;
- ``predictions.png``: ground truth and the predictions of the trained model and of its EMA on three images;
and prints the final metrics against the success criteria of step 5.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from pytorch_lightning import seed_everything

from rfdetr.datasets.coco import CocoDetection
from rfdetr.models.eodt import load_configs
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.trainer import build_trainer

# Success criteria of step 5, fixed before the run; all are read at the last epoch (see the plan of step 5)
MIN_MAP_50 = 0.95
MIN_MAP_50_95 = 0.85
MAX_LOSS_GIOU = 0.10  # mean (1 - GIoU) of the matched boxes of the last L2 block

# Categorical slots 1-3 of the dataviz reference palette, in order, and its neutral inks
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MUTED, GRID, SURFACE = "#898781", "#e1e0d9", "#fcfcfb"


def read_metrics(path: Path) -> dict[int, dict[str, float]]:
    """Read Lightning's ``metrics.csv`` and merge its rows into one dict of metrics per epoch."""
    metrics: dict[int, dict[str, float]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            epoch = metrics.setdefault(int(row["epoch"]), {})
            epoch.update({key: float(value) for key, value in row.items() if value and key not in ("epoch", "step")})
    return metrics


def plot_curves(metrics: dict[int, dict[str, float]], path: Path) -> None:
    """Plot total loss, loss terms and mAP of the trained model and of the EMA per epoch."""
    epochs = sorted(metrics)
    x = [epoch + 1 for epoch in epochs]

    def series(key: str) -> list[float]:
        return [metrics[epoch].get(key, math.nan) for epoch in epochs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("epoch (5 steps each)")

    axes[0].semilogy(x, series("train/loss"), color=SERIES[0], linewidth=2)
    axes[0].set_title("Training loss (weighted sum over the 3 L2 blocks)")

    for color, name in zip(SERIES, ("loss_ce", "loss_bbox", "loss_giou")):
        axes[1].semilogy(x, series(f"train/{name}"), color=color, linewidth=2, label=name)
        axes[1].semilogy(x, series(f"train/{name}_aux"), color=color, linewidth=1, linestyle="--")
    axes[1].axhline(MAX_LOSS_GIOU, color=MUTED, linewidth=1, linestyle=":")
    axes[1].set_title("Loss terms (solid: last L2 block, dashed: sum of the 2 others)")
    axes[1].legend(frameon=False)

    for color, prefix, label in ((SERIES[0], "val/", "model"), (SERIES[1], "val/ema_", "EMA")):
        axes[2].plot(x, series(f"{prefix}mAP_50_95"), color=color, linewidth=2, label=f"{label} mAP 50:95")
        axes[2].plot(x, series(f"{prefix}mAP_50"), color=color, linewidth=1.5, linestyle="--", label=f"{label} AP50")
    for threshold in (MIN_MAP_50_95, MIN_MAP_50):
        axes[2].axhline(threshold, color=MUTED, linewidth=1, linestyle=":")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("COCO AP on the 20 images (dotted: thresholds)")
    axes[2].legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


@torch.no_grad()
def predict(module: RFDETRModelModule, images: torch.Tensor, orig_sizes: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """Run the model in eval mode and return the detections in pixels of the original images."""
    module.model.eval()
    return module.postprocess(module.model(images), orig_sizes)


def draw_predictions(
    module: RFDETRModelModule,
    base_state_dict: dict[str, torch.Tensor],
    dataset: CocoDetection,
    path: Path,
    score_threshold: float,
) -> None:
    """Draw ground truth, model and EMA predictions on the 2 images with most objects and the one with the smallest.

    Args:
        module: The module after ``fit``, which holds the EMA weights (``RFDETREMACallback.on_train_end``).
        base_state_dict: Weights of the trained model, from the last checkpoint.
        dataset: The validation ``CocoDetection``, identical to the training one.
        path: Output PNG.
        score_threshold: Minimum score of a drawn prediction.
    """
    coco = dataset.coco
    annotations = {image_id: coco.imgToAnns[image_id] for image_id in dataset.ids}
    image_ids = sorted(dataset.ids, key=lambda i: len(annotations[i]), reverse=True)[:2]
    image_ids.append(
        min((i for i in dataset.ids if i not in image_ids), key=lambda i: min(a["area"] for a in annotations[i]))
    )
    samples = [dataset[dataset.ids.index(image_id)] for image_id in image_ids]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)
    images = torch.stack([image for image, _ in samples]).to(device)
    orig_sizes = torch.stack([target["orig_size"] for _, target in samples]).to(device)

    detections = {"EMA": predict(module, images, orig_sizes)}
    module.load_state_dict(base_state_dict)
    detections["model"] = predict(module, images, orig_sizes)

    fig, axes = plt.subplots(len(image_ids), 3, figsize=(15, 4.6 * len(image_ids)), facecolor=SURFACE)
    for row, image_id in enumerate(image_ids):
        info = coco.loadImgs(image_id)[0]
        picture = Image.open(Path(dataset.root) / info["file_name"]).convert("RGB")
        ground_truth = [(a["bbox"], coco.cats[a["category_id"]]["name"]) for a in annotations[image_id]]
        columns = [(f"ground truth: {len(ground_truth)} objects", SERIES[0], ground_truth)]
        for name in ("model", "EMA"):
            result = detections[name][row]
            keep = result["scores"] >= score_threshold
            boxes = [
                ((x0, y0, x1 - x0, y1 - y0), f"{coco.cats[int(label)]['name']} {score:.2f}")
                for (x0, y0, x1, y1), label, score in zip(
                    result["boxes"][keep].tolist(), result["labels"][keep].tolist(), result["scores"][keep].tolist()
                )
            ]
            columns.append((f"{name}: {len(boxes)} detections with score >= {score_threshold}", SERIES[1], boxes))
        for ax, (title, color, boxes) in zip(axes[row], columns):
            ax.imshow(picture)
            ax.set_title(f"image {image_id} - {title}", fontsize=9)
            ax.axis("off")
            for (x, y, w, h), text in boxes:
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.5))
                ax.text(x, y, text, fontsize=6, color="white", va="top", bbox={"facecolor": color, "pad": 1, "lw": 0})
    handles = [Line2D([], [], color=SERIES[0], label="ground truth"), Line2D([], [], color=SERIES[1], label="prediction")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def check(metrics: dict[int, dict[str, float]]) -> bool:
    """Print the final metrics and the success criteria of step 5; return whether all of them hold."""
    final = metrics[max(metrics)]
    print(f"\nLast epoch: {max(metrics) + 1}")
    for name in ("mAP_50_95", "mAP_50", "mAP_75", "mAR"):
        print(f"  {name:10s} model {final[f'val/{name}']:.4f}   EMA {final[f'val/ema_{name}']:.4f}")
    for name in ("loss", "loss_ce", "loss_bbox", "loss_giou", "loss_ce_aux", "loss_bbox_aux", "loss_giou_aux"):
        print(f"  train/{name:14s} {final[f'train/{name}']:.4f}")

    finite = all(math.isfinite(epoch.get("train/loss", math.nan)) for epoch in metrics.values())
    criteria = [
        (f"model AP50 >= {MIN_MAP_50}", final["val/mAP_50"] >= MIN_MAP_50),
        (f"EMA AP50 >= {MIN_MAP_50}", final["val/ema_mAP_50"] >= MIN_MAP_50),
        (f"model mAP 50:95 >= {MIN_MAP_50_95}", final["val/mAP_50_95"] >= MIN_MAP_50_95),
        (f"EMA mAP 50:95 >= {MIN_MAP_50_95}", final["val/ema_mAP_50_95"] >= MIN_MAP_50_95),
        ("train/loss finite in every epoch", finite),
        (f"train/loss_giou <= {MAX_LOSS_GIOU}", final["train/loss_giou"] <= MAX_LOSS_GIOU),
    ]
    print("\nSuccess criteria of step 5:")
    for name, passed in criteria:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return all(passed for _, passed in criteria)


def main() -> None:
    """Train on coco_mini from the YAML file, then write the curves, the predictions and the verdict."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="YAML file read by rfdetr.models.eodt.load_configs")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="minimum score of a drawn prediction")
    args = parser.parse_args()
    plt.switch_backend("Agg")  # figures are only saved to files

    model_config, train_config = load_configs(args.config)
    output_dir = Path(train_config.output_dir)
    if (output_dir / "metrics.csv").exists():
        raise SystemExit(f"{output_dir} already holds a run: delete it or change output_dir")

    # RFDETRModelModule seeds only in on_fit_start: seed before building it, so that queries and head are reproducible
    seed_everything(train_config.seed)
    trainer = build_trainer(train_config, model_config)
    module = RFDETRModelModule(model_config, train_config)
    datamodule = RFDETRDataModule(model_config, train_config)
    start = time.perf_counter()
    trainer.fit(module, datamodule)
    minutes = (time.perf_counter() - start) / 60
    peak = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"\nfit: {minutes:.1f} min, {trainer.global_step} steps, peak memory reserved by PyTorch {peak:.2f} GiB")

    metrics = read_metrics(output_dir / "metrics.csv")
    plot_curves(metrics, output_dir / "curves.png")
    base_state_dict = torch.load(output_dir / "last.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    dataset = datamodule.val_dataloader().dataset
    draw_predictions(module, base_state_dict, dataset, output_dir / "predictions.png", args.score_threshold)
    passed = check(metrics)
    print(f"\nStep 5: {'PASS' if passed else 'FAIL'}. Curves and predictions in {output_dir}")


if __name__ == "__main__":
    main()
