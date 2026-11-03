# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Dump an RF-DETR Medium built from its configuration, to compare the fork with ref-rfdetr.

Run by ``test_rfdetr_unchanged.py`` in a subprocess whose ``PYTHONPATH`` points to the ``src`` folder of one
of the two source trees. It uses only RF-DETR's own API, so it runs on both.

Usage::

    python rfdetr_reference.py OUTPUT_FILE
"""

import sys
import warnings

import torch

import rfdetr
from rfdetr.config import RFDETRMediumConfig, TrainConfig
from rfdetr.models.lwdetr import build_model_from_config


def main(output_file: str) -> None:
    """Build RF-DETR Medium with random weights and save its state dict and its output on a fixed image."""
    warnings.simplefilter("ignore")  # pretrain_weights=None warns that the model starts from scratch
    torch.manual_seed(0)
    # patch 16: the DINOv2 weights are not loaded, so nothing is downloaded and the weights are random
    model_config = RFDETRMediumConfig(pretrain_weights=None, device="cpu")
    train_config = TrainConfig(dataset_dir=".", output_dir=".")
    model = build_model_from_config(model_config, train_config).eval()

    resolution = model_config.resolution
    x = torch.randn(1, 3, resolution, resolution, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        outputs = model(x)

    torch.save({"rfdetr_file": rfdetr.__file__, "state_dict": model.state_dict(), "outputs": outputs}, output_file)


if __name__ == "__main__":
    main(sys.argv[1])
