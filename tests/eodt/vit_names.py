# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Names of the same DINOv2 tensor in the baseline (Hugging Face, through RF-DETR) and in EoDT (timm)."""

import re

# Hugging Face names of the baseline encoder (after "backbone.0.encoder.encoder.") -> timm names of EoDT, first match
# wins. query, key and value are the three thirds of timm's qkv; mask_token has no counterpart.
BASELINE_TO_EODT = [
    (r"^embeddings\.cls_token$", "cls_token"),
    (r"^embeddings\.register_tokens$", "reg_token"),
    (r"^embeddings\.position_embeddings$", "pos_embed"),
    (r"^embeddings\.patch_embeddings\.projection\.", "patch_embed.proj."),
    (r"^encoder\.layer\.(\d+)\.attention\.attention\.(query|key|value)\.", r"blocks.\1.attn.qkv."),
    (r"^encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"blocks.\1.attn.proj."),
    (r"^encoder\.layer\.(\d+)\.layer_scale(\d)\.lambda1$", r"blocks.\1.ls\2.gamma"),
    (r"^encoder\.layer\.(\d+)\.", r"blocks.\1."),  # norm1, norm2, mlp.fc1, mlp.fc2
    (r"^layernorm\.", "norm."),
]


def eodt_vit_name(baseline_name: str) -> str:
    """Return the timm name in EoDT of a tensor of the baseline encoder, given by its Hugging Face name.

    Args:
        baseline_name: Name in the baseline encoder, without the ``backbone.0.encoder.encoder.`` prefix.

    Returns:
        Name in EoDT's timm ViT, without the ``backbone.vit.`` prefix.

    Raises:
        KeyError: If the tensor has no counterpart in timm.
    """
    for pattern, replacement in BASELINE_TO_EODT:
        name, matched = re.subn(pattern, replacement, baseline_name)
        if matched:
            return name
    raise KeyError(f"no EoDT counterpart for {baseline_name}")
