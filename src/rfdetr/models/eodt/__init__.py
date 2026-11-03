# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""EoDT model subpackage."""

from rfdetr.models.eodt.backbone import EoDTBackbone
from rfdetr.models.eodt.build import build_eodt_from_config
from rfdetr.models.eodt.config import EoDTConfig
from rfdetr.models.eodt.model import EoDT

__all__ = ["EoDT", "EoDTBackbone", "EoDTConfig", "build_eodt_from_config"]
