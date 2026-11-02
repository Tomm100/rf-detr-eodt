# ------------------------------------------------------------------------
# EoDT: encoder-only detection on top of RF-DETR
# ------------------------------------------------------------------------
"""Tests for the L1 / L2 split of rfdetr.models.eodt.backbone.EoDTBackbone (step 2 of EODT_SPEC.md §9).

Covers:
- Shapes of ``forward_l1`` and ``forward_l2`` at the default 576 px and 300 queries.
- Layout inside L2: ``[queries, ViT tokens]``, and the outputs are cut at the same positions.
- The tokens out of L1 equal those of the timm ViT without queries.
- With 0 queries the output equals the step-1 forward and timm's ``forward_features``, for ViT-S,
  ViT-B and ViT-L with the default L2.
- The queries receive a gradient.
- The queries change the ViT tokens from the first L2 block on, and nothing before it
  (in L2 the patches attend to the queries).
- The image changes every query state after every L2 block (the queries attend to the patches,
  which replaces the decoder's cross-attention).
- Out-of-range L2 is rejected.

ViT-B and ViT-L cases carry the ``gpu`` marker (slow on CPU): ``-m "not gpu"`` runs ViT-S only.
"""

import pytest
import torch
from torch import Tensor

from rfdetr.models.eodt import EoDTBackbone

VITS = "vit_small_patch14_reg4_dinov2"
# Default L2 of §3.2: 3 for ViT-S and ViT-B, 4 for ViT-L.
BACKBONES = [
    pytest.param("vit_small_patch14_reg4_dinov2", 3, id="vits"),
    pytest.param("vit_base_patch14_reg4_dinov2", 3, id="vitb", marks=pytest.mark.gpu),
    pytest.param("vit_large_patch14_reg4_dinov2", 4, id="vitl", marks=pytest.mark.gpu),
]
RESOLUTION = 224  # 14 x 14 grid keeps the CPU tests fast
PATCH_SIZE = 16
NUM_QUERIES = 10


def _images(batch_size: int, resolution: int = RESOLUTION, seed: int = 0) -> Tensor:
    """Return a fixed batch of random images."""
    return torch.randn(batch_size, 3, resolution, resolution, generator=torch.Generator().manual_seed(seed))


def _queries(backbone: EoDTBackbone, batch_size: int, num_queries: int, seed: int = 1) -> Tensor:
    """Return random queries of shape ``[batch_size, num_queries, embed_dim]``."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, num_queries, backbone.embed_dim, generator=generator)


def test_shapes() -> None:
    """At 576 px with 300 queries: 5 + 36 x 36 ViT tokens, and one query state per L2 block."""
    backbone = EoDTBackbone(VITS, 576, PATCH_SIZE, 3, pretrained=False).eval()

    with torch.no_grad():
        tokens = backbone.forward_l1(_images(2, 576))
        query_states, vit_tokens = backbone.forward_l2(tokens, _queries(backbone, 2, 300))

    assert backbone.num_l1_blocks == 9
    assert tokens.shape == (2, 5 + 36 * 36, 384)
    assert len(query_states) == 3
    for state in query_states:
        assert state.shape == (2, 300, 384)
    assert vit_tokens.shape == (2, 5 + 36 * 36, 384)


def test_l2_layout() -> None:
    """The first L2 block receives ``[queries, ViT tokens]``; the outputs are cut at the same positions."""
    backbone = EoDTBackbone(VITS, RESOLUTION, PATCH_SIZE, 3, pretrained=False).eval()
    l2_blocks = backbone.vit.blocks[backbone.num_l1_blocks :]
    queries = _queries(backbone, 2, NUM_QUERIES)
    inputs, outputs = [], []
    hooks = [l2_blocks[0].register_forward_pre_hook(lambda module, args: inputs.append(args[0]))]
    hooks += [block.register_forward_hook(lambda module, args, output: outputs.append(output)) for block in l2_blocks]

    with torch.no_grad():
        tokens = backbone.forward_l1(_images(2))
        try:
            query_states, vit_tokens = backbone.forward_l2(tokens, queries)
        finally:
            for hook in hooks:
                hook.remove()

    assert len(inputs) == 1
    torch.testing.assert_close(inputs[0][:, :NUM_QUERIES], queries, rtol=0, atol=0)
    torch.testing.assert_close(inputs[0][:, NUM_QUERIES:], tokens, rtol=0, atol=0)
    for state, output in zip(query_states, outputs, strict=True):
        torch.testing.assert_close(state, output[:, :NUM_QUERIES], rtol=0, atol=0)
    torch.testing.assert_close(vit_tokens, outputs[-1][:, NUM_QUERIES:], rtol=0, atol=0)


@pytest.mark.parametrize("num_l2_blocks", [3, 4])
def test_l1_tokens_match_vit_without_queries(num_l2_blocks: int) -> None:
    """The tokens out of L1 equal the output of the same block in timm's plain forward_features."""
    backbone = EoDTBackbone(VITS, RESOLUTION, PATCH_SIZE, num_l2_blocks, pretrained=True).eval()
    x = _images(2)
    captured = []
    hook = backbone.vit.blocks[backbone.num_l1_blocks - 1].register_forward_hook(
        lambda module, args, output: captured.append(output)
    )

    with torch.no_grad():
        try:
            backbone.vit.forward_features(x)
        finally:
            hook.remove()
        ours = backbone.forward_l1(x)

    assert backbone.num_l1_blocks == 12 - num_l2_blocks
    assert len(captured) == 1
    torch.testing.assert_close(ours, captured[0], rtol=0, atol=0)


@pytest.mark.parametrize(("name", "num_l2_blocks"), BACKBONES)
def test_zero_queries_matches_step1(name: str, num_l2_blocks: int) -> None:
    """With 0 queries, L1 + L2 followed by the final norm equal the step-1 forward and timm's ViT."""
    backbone = EoDTBackbone(name, RESOLUTION, PATCH_SIZE, num_l2_blocks, pretrained=True).eval()
    x = _images(2)

    with torch.no_grad():
        query_states, vit_tokens = backbone.forward_l2(backbone.forward_l1(x), _queries(backbone, 2, 0))
        ours = backbone.vit.norm(vit_tokens)
        step1 = backbone(x)
        timm_vit = backbone.vit.forward_features(x)

    assert len(query_states) == num_l2_blocks
    for state in query_states:
        assert state.shape == (2, 0, backbone.embed_dim)
    torch.testing.assert_close(ours, step1, rtol=0, atol=0)
    torch.testing.assert_close(ours, timm_vit, rtol=0, atol=0)


def test_queries_receive_gradient() -> None:
    """A loss on the query states gives a finite, non-zero gradient to every query of every image."""
    backbone = EoDTBackbone(VITS, RESOLUTION, PATCH_SIZE, 3, pretrained=True).train()
    queries = _queries(backbone, 2, NUM_QUERIES).requires_grad_()

    query_states, _ = backbone.forward_l2(backbone.forward_l1(_images(2)), queries)
    loss = sum(state.pow(2).mean() for state in query_states)
    loss.backward()

    assert queries.grad is not None
    assert torch.isfinite(queries.grad).all()
    assert (queries.grad.abs().sum(dim=-1) > 0).all()


@pytest.mark.parametrize("num_l2_blocks", [3, 4, 12])
def test_queries_affect_vit_tokens_only_in_l2(num_l2_blocks: int) -> None:
    """Different queries leave every L1 block unchanged and change the patches in every L2 block.

    With L2 = 4 the queries enter one block earlier than with L2 = 3; with L2 = 12 (no L1) at the first block.
    """
    backbone = EoDTBackbone(VITS, RESOLUTION, PATCH_SIZE, num_l2_blocks, pretrained=True).eval()
    x = _images(2)
    num_vit_tokens = backbone.num_prefix_tokens + (RESOLUTION // PATCH_SIZE) ** 2
    first_patch = NUM_QUERIES + backbone.num_prefix_tokens

    def block_outputs(queries: Tensor) -> list[Tensor]:
        outputs = []
        hooks = [
            block.register_forward_hook(lambda module, args, output: outputs.append(output))
            for block in backbone.vit.blocks
        ]
        try:
            with torch.no_grad():
                backbone.forward_l2(backbone.forward_l1(x), queries)
        finally:
            for hook in hooks:
                hook.remove()
        return outputs

    outputs_a = block_outputs(_queries(backbone, 2, NUM_QUERIES, seed=1))
    outputs_b = block_outputs(_queries(backbone, 2, NUM_QUERIES, seed=2))

    assert backbone.num_l1_blocks == 12 - num_l2_blocks
    assert len(outputs_a) == len(outputs_b) == 12
    for i, (out_a, out_b) in enumerate(zip(outputs_a, outputs_b)):
        if i < backbone.num_l1_blocks:
            assert out_a.shape[1] == num_vit_tokens, i
            torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
        else:
            assert out_a.shape[1] == NUM_QUERIES + num_vit_tokens, i
            assert not torch.allclose(out_a[:, first_patch:], out_b[:, first_patch:]), i


def test_image_affects_queries_in_every_l2_block() -> None:
    """With the same queries, two different images change every query state after every L2 block.

    The check is per query, so it would also catch an attention mask that cut only some queries off the patches.
    """
    backbone = EoDTBackbone(VITS, RESOLUTION, PATCH_SIZE, 3, pretrained=True).eval()
    queries = _queries(backbone, 2, NUM_QUERIES)

    with torch.no_grad():
        states_a, _ = backbone.forward_l2(backbone.forward_l1(_images(2, seed=0)), queries)
        states_b, _ = backbone.forward_l2(backbone.forward_l1(_images(2, seed=1)), queries)

    assert len(states_a) == len(states_b) == 3
    for i, (state_a, state_b) in enumerate(zip(states_a, states_b)):
        unchanged = torch.isclose(state_a, state_b).all(dim=-1)  # [B, Q]: query state identical
        assert not unchanged.any(), i


@pytest.mark.parametrize("num_l2_blocks", [0, 13])
def test_invalid_num_l2_blocks(num_l2_blocks: int) -> None:
    """L2 must be between 1 and the ViT depth (12 for ViT-S)."""
    with pytest.raises(ValueError, match="num_l2_blocks"):
        EoDTBackbone(VITS, RESOLUTION, PATCH_SIZE, num_l2_blocks, pretrained=False)
