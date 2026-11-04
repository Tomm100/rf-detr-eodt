# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""EoDT model subpackage."""

from rfdetr.models.eodt.backbone import EoDTBackbone
from rfdetr.models.eodt.build import build_eodt_criterion_from_config, build_eodt_from_config
from rfdetr.models.eodt.config import EoDTConfig
from rfdetr.models.eodt.model import EoDT
from rfdetr.models.eodt.param_groups import build_eodt_param_dicts
from rfdetr.models.eodt.yaml_config import load_configs

__all__ = [
    "EoDT",
    "EoDTBackbone",
    "EoDTConfig",
    "build_eodt_criterion_from_config",
    "build_eodt_from_config",
    "build_eodt_param_dicts",
    "load_configs",
]
