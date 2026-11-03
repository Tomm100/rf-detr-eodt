# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""EoDT model subpackage."""

from rfdetr.models.eodt.backbone import EoDTBackbone
from rfdetr.models.eodt.model import EoDT

__all__ = ["EoDT", "EoDTBackbone"]
