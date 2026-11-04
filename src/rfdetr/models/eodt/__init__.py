# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""EoDT model subpackage."""

from rfdetr.models.eodt.backbone import EoDTBackbone
from rfdetr.models.eodt.baseline import BaselineConfig, LinearProjector, adapt_baseline
from rfdetr.models.eodt.build import build_eodt_criterion_from_config, build_eodt_from_config
from rfdetr.models.eodt.config import EoDTConfig
from rfdetr.models.eodt.dinov2_weights import load_native_dinov2, load_pretrained_dinov2
from rfdetr.models.eodt.model import EoDT
from rfdetr.models.eodt.param_groups import build_eodt_param_dicts
from rfdetr.models.eodt.yaml_config import load_configs

__all__ = [
    "BaselineConfig",
    "EoDT",
    "EoDTBackbone",
    "EoDTConfig",
    "LinearProjector",
    "adapt_baseline",
    "build_eodt_criterion_from_config",
    "build_eodt_from_config",
    "build_eodt_param_dicts",
    "load_configs",
    "load_native_dinov2",
    "load_pretrained_dinov2",
]
