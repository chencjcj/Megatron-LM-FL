# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Numerical parity tests for the fused Triton DSA sparse-attention forward kernel.

``megatron.plugin.dsa_kernel.triton_sparse_attention.fused_dsa_attention`` is a
drop-in replacement for ``dsa.unfused_dsa_fn``: it computes the same top-k sparse
attention with a flash-style online softmax over only the top-k keys instead of
materialising the dense ``[b, np, sq, skv]`` score matrix. These tests pin the
forward output (and the autograd gradients) against the unfused reference.

Run with::

    pytest tests/unit_tests/plugin/dsa_kernel/test_triton_sparse_attention.py -v
"""

from __future__ import annotations

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa import unfused_dsa_fn

pytest.importorskip("triton", reason="Triton is required for the fused kernels")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _sm90_or_newer() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


requires_sm90 = pytest.mark.skipif(
    not _sm90_or_newer(), reason="fused sparse-attention kernel requires SM90+"
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.detach().float().reshape(1, -1), b.detach().float().reshape(1, -1)
    ).item()


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.detach().float().abs().max().clamp(min=1e-12)
    return ((got.detach().float() - ref.detach().float()).abs().max() / denom).item()


def _make_inputs(sq, skv, b, np, hn, hnv, topk, mask, seed=7, dtype=torch.bfloat16):
    """query/key/value in SBHD + top-k indices selected from masked random scores.

    Selecting the top-k from ``random_scores + mask`` mirrors the real indexer, so
    the top-k always includes at least one causally-valid key (the diagonal) — no
    fully-masked rows, where the reference softmax would produce NaN.
    """
    torch.manual_seed(seed)
    query = torch.randn(sq, b, np, hn, dtype=dtype, device="cuda")
    key = torch.randn(skv, b, np, hn, dtype=dtype, device="cuda")
    value = torch.randn(skv, b, np, hnv, dtype=dtype, device="cuda")
    scores = torch.randn(b, sq, skv, device="cuda")
    if mask is not None:
        scores = scores + (mask if mask.dim() == 3 else mask.unsqueeze(0))
    else:
        causal = torch.triu(
            torch.full((sq, skv), float("-inf"), device="cuda"), diagonal=1
        )
        scores = scores + causal
    topk_indices = scores.topk(min(topk, skv), dim=-1).indices.contiguous()  # [b, sq, topk]
    return query, key, value, topk_indices


def _block_diag_mask(cu_seqlens, sq, skv):
    """[sq, skv] float additive document-boundary causal mask (THD-style)."""
    device = "cuda"
    q_pos = torch.arange(sq, device=device)
    k_pos = torch.arange(skv, device=device)
    cs = cu_seqlens.to(device)
    q_doc = torch.searchsorted(cs[1:], q_pos, right=True)
    k_doc = torch.searchsorted(cs[1:], k_pos, right=True)
    causal = k_pos[None, :] <= q_pos[:, None]
    same_doc = q_doc[:, None] == k_doc[None, :]
    valid = causal & same_doc
    return torch.where(
        valid, torch.zeros((), device=device), torch.full((), float("-inf"), device=device)
    ).float()


@requires_sm90
class TestSparseAttentionForward:
    """``fused_dsa_attention`` forward vs ``unfused_dsa_fn``."""

    @pytest.mark.parametrize("seqlen", [64, 128, 256])
    @pytest.mark.parametrize("topk", [16, 64])
    @pytest.mark.parametrize("hn,hnv", [(64, 64), (192, 256)])
    def test_default_causal(self, seqlen, topk, hn, hnv):
        from megatron.plugin.dsa_kernel.triton_sparse_attention import fused_dsa_attention

        b, np = 2, 4
        scale = hn**-0.5
        query, key, value, idx = _make_inputs(seqlen, seqlen, b, np, hn, hnv, topk, None)

        ref = unfused_dsa_fn(query, key, value, idx, scale, causal_mask=None)
        got = fused_dsa_attention(query, key, value, idx, scale, causal_mask=None)

        assert got.shape == ref.shape
        assert _cos(got, ref) > 0.999, f"cos={_cos(got, ref)}"
        assert _rel(got, ref) < 2e-2, f"rel={_rel(got, ref)}"

    @pytest.mark.parametrize("topk", [16, 64])
    def test_thd_block_diagonal_mask(self, topk):
        from megatron.plugin.dsa_kernel.triton_sparse_attention import fused_dsa_attention

        sq = skv = 96
        b, np, hn, hnv = 1, 4, 64, 64
        scale = hn**-0.5
        cu = torch.tensor([0, 40, 96], dtype=torch.int32)
        mask = _block_diag_mask(cu, sq, skv)  # [sq, skv]
        query, key, value, idx = _make_inputs(sq, skv, b, np, hn, hnv, topk, mask)

        ref = unfused_dsa_fn(query, key, value, idx, scale, causal_mask=mask)
        got = fused_dsa_attention(query, key, value, idx, scale, causal_mask=mask)

        assert _cos(got, ref) > 0.999, f"cos={_cos(got, ref)}"
        assert _rel(got, ref) < 2e-2, f"rel={_rel(got, ref)}"

    def test_cp_style_nonsquare_mask(self):
        """Local queries (sq) vs full keys (skv > sq) with a global-position mask."""
        from megatron.plugin.dsa_kernel.triton_sparse_attention import fused_dsa_attention

        sq, skv = 32, 64
        b, np, hn, hnv = 1, 4, 64, 64
        scale = hn**-0.5
        # Query j maps to global positions [16..47] (a zigzag-like slice); causal
        # against all skv keys by global position.
        q_pos = torch.arange(16, 16 + sq, device="cuda")
        k_pos = torch.arange(skv, device="cuda")
        valid = k_pos[None, :] <= q_pos[:, None]
        mask = torch.where(
            valid, torch.zeros((), device="cuda"), torch.full((), float("-inf"), device="cuda")
        ).float()
        query, key, value, idx = _make_inputs(sq, skv, b, np, hn, hnv, 16, mask)

        ref = unfused_dsa_fn(query, key, value, idx, scale, causal_mask=mask)
        got = fused_dsa_attention(query, key, value, idx, scale, causal_mask=mask)

        assert got.shape == ref.shape
        assert _cos(got, ref) > 0.999, f"cos={_cos(got, ref)}"
        assert _rel(got, ref) < 2e-2, f"rel={_rel(got, ref)}"

    def test_backward_matches_reference(self):
        from megatron.plugin.dsa_kernel.triton_sparse_attention import fused_dsa_attention

        sq = skv = 128
        b, np, hn, hnv = 2, 4, 64, 64
        scale = hn**-0.5
        query, key, value, idx = _make_inputs(sq, skv, b, np, hn, hnv, 32, None)

        def run(fn, mask):
            q = query.detach().clone().requires_grad_(True)
            k = key.detach().clone().requires_grad_(True)
            v = value.detach().clone().requires_grad_(True)
            out = fn(q, k, v, idx, scale, causal_mask=mask)
            out.float().pow(2).sum().backward()
            return out, q.grad, k.grad, v.grad

        o_ref, dq_ref, dk_ref, dv_ref = run(unfused_dsa_fn, None)
        o_got, dq_got, dk_got, dv_got = run(fused_dsa_attention, None)

        assert _cos(o_got, o_ref) > 0.999
        for got, ref, name in [
            (dq_got, dq_ref, "dq"), (dk_got, dk_ref, "dk"), (dv_got, dv_ref, "dv")
        ]:
            assert _cos(got, ref) > 0.999, f"{name} cos={_cos(got, ref)}"
            assert _rel(got, ref) < 2e-2, f"{name} rel={_rel(got, ref)}"

    def test_backward_with_mask(self):
        """Backward through the HAS_MASK path (THD block-diagonal document mask)."""
        from megatron.plugin.dsa_kernel.triton_sparse_attention import fused_dsa_attention

        sq = skv = 96
        b, np, hn, hnv = 1, 4, 64, 64
        scale = hn**-0.5
        cu = torch.tensor([0, 40, 96], dtype=torch.int32)
        mask = _block_diag_mask(cu, sq, skv)
        query, key, value, idx = _make_inputs(sq, skv, b, np, hn, hnv, 32, mask)

        def run(fn):
            q = query.detach().clone().requires_grad_(True)
            k = key.detach().clone().requires_grad_(True)
            v = value.detach().clone().requires_grad_(True)
            out = fn(q, k, v, idx, scale, causal_mask=mask)
            out.float().pow(2).sum().backward()
            return q.grad, k.grad, v.grad

        ref = run(unfused_dsa_fn)
        got = run(fused_dsa_attention)
        for g, r, name in zip(got, ref, ("dq", "dk", "dv")):
            assert _cos(g, r) > 0.999, f"{name} cos={_cos(g, r)}"
            assert _rel(g, r) < 2e-2, f"{name} rel={_rel(g, r)}"
