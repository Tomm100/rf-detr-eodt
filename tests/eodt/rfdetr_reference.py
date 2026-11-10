# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Dump an RF-DETR model built from one of its original configurations, to compare the fork with ref-rfdetr.

It also dumps the outputs of ``gen_encoder_output_proposals``, which the fork rewrites, bit by bit.

Run by ``test_rfdetr_unchanged.py`` in a subprocess whose ``PYTHONPATH`` points to the ``src`` folder of one
of the two source trees. It uses only RF-DETR's own API, so it runs on both.

Usage::

    python rfdetr_reference.py {medium,base} OUTPUT_FILE
"""

import itertools
import sys
import warnings

import torch

import rfdetr
from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import ModelConfig, RFDETRBaseConfig, RFDETRMediumConfig, TrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
from rfdetr.models.transformer import gen_encoder_output_proposals
from rfdetr.training.param_groups import get_param_dict

# Original configurations without an RF-DETR checkpoint:
# - medium: patch 16, so RF-DETR does not load DINOv2 and the weights are random (nothing is downloaded);
# - base: patch 14 and DINOv2's 37 x 37 grid, so RF-DETR loads facebook/dinov2-small from Hugging Face.
CONFIGS: dict[str, type[ModelConfig]] = {"medium": RFDETRMediumConfig, "base": RFDETRBaseConfig}


def _padding_mask(shapes: list[tuple[int, int]], batch: int, device: str) -> torch.Tensor:
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


def dump_proposals() -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """Dump the outputs of ``gen_encoder_output_proposals`` on every branch of the function, as bytes.

    Bytes and not tensors: ``torch.equal`` does not tell 0.0 from -0.0. The fork rewrites how the proposals are
    built (EODT_SPEC.md §8, point 10), so its outputs must keep every bit of RF-DETR's.
    """
    dump = {}
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    for device, shapes, batch, dtype, masked, unsigmoid in itertools.product(
        devices, ([(4, 6)], [(4, 6), (2, 3)]), (1, 3), (torch.float32, torch.bfloat16), (False, True), (True, False)
    ):
        size = sum(height * width for height, width in shapes)
        memory = torch.randn(batch, size, 8, generator=torch.Generator().manual_seed(3)).to(device, dtype)
        mask = _padding_mask(shapes, batch, device) if masked else None
        outputs = gen_encoder_output_proposals(memory, mask, shapes, unsigmoid=unsigmoid)
        key = f"{device} {shapes} batch {batch} {dtype} masked {masked} unsigmoid {unsigmoid}"
        for name, output in zip(("memory", "proposals"), outputs):
            dump[f"{key} {name}"] = (str(output.dtype), tuple(output.shape), output.float().cpu().numpy().tobytes())
    return dump


def main(config_name: str, output_file: str) -> None:
    """Build an original RF-DETR configuration and save model, outputs, losses and optimizer groups."""
    warnings.simplefilter("ignore")  # pretrain_weights=None warns that the model starts from scratch
    torch.manual_seed(0)
    model_config = CONFIGS[config_name](pretrain_weights=None, device="cpu")
    train_config = TrainConfig(dataset_dir=".", output_dir=".")
    model = build_model_from_config(model_config, train_config).eval()
    criterion, _ = build_criterion_from_config(model_config, train_config)

    resolution = model_config.resolution
    x = torch.randn(1, 3, resolution, resolution, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        outputs = model(x)

    # losses on fixed targets, in training mode (all the Group-DETR queries)
    targets = [{"labels": torch.tensor([1, 5]), "boxes": torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.4, 0.2]])}]
    model.train()
    torch.manual_seed(2)
    with torch.no_grad():
        loss_dict = criterion(model(x, targets), targets)

    # optimizer groups as (hyperparameters, parameter names)
    names = {id(param): name for name, param in model.named_parameters()}
    param_groups = []
    for group in get_param_dict(_namespace_from_configs(model_config, train_config), model):
        hyperparameters = {key: value for key, value in group.items() if key != "params"}
        param_groups.append((hyperparameters, [names[id(param)] for param in group["params"]]))

    torch.save(
        {
            "rfdetr_file": rfdetr.__file__,
            "state_dict": model.state_dict(),
            "outputs": outputs,
            "weight_dict": criterion.weight_dict,
            "loss_dict": loss_dict,
            "param_groups": param_groups,
            "proposals": dump_proposals(),
        },
        output_file,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
