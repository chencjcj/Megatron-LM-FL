# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Triton fused DSA sparse-attention kernels (non-absorbed variant), forward + backward.

The unfused reference (``dsa.py``: :func:`unfused_dsa_fn`) materialises the full
dense ``[b, np, sq, skv]`` score matrix in fp32 for every DSA layer, applies the
top-k sparsity + causal mask, softmaxes over the whole row and contracts with the
values. That is ``O(sq * skv)`` work/memory regardless of the ``index_topk``
sparsity, and at the GLM5.2 shape (``sq = skv = 4096``) it dominates the forward.

These kernels instead do a **flash-style online softmax over only the top-k
gathered keys** (``O(sq * topk)``): each query attends to its own ``topk`` key
positions, tracked with a running max / denominator so the full score row is
never materialised. The maths matches the reference exactly (softmax over the
non-top-k positions is a no-op because they carry ``-inf``), so loss curves stay
comparable; this is an implementation optimisation only.

* Forward stores the log-sum-exp (``lse``) per (query, head) so the backward can
  recover the softmax probabilities without re-tracking the running max.
* Backward is also flash-style: ``dq`` accumulates per query (no atomics), while
  ``dk``/``dv`` are scattered into fp32 accumulators with ``tl.atomic_add``
  (multiple queries share the same key), then cast back to the input dtype.

Layout mirrors :func:`unfused_dsa_fn`:

* ``query``  ``[sq, b, np, hn]``
* ``key``    ``[skv, b, np, hn]``
* ``value``  ``[skv, b, np, hnv]``
* ``topk_indices`` ``[b, sq, topk]`` (int; shared across heads, indexes the KV
  sequence dim), selected from the already-masked indexer scores.
* ``causal_mask`` optional ``[sq, skv]`` or ``[b, sq, skv]`` fp32 additive mask
  (``-inf`` masked, ``0`` valid); ``None`` means the default upper-triangular
  causal mask (valid iff ``key_pos <= query_pos``). THD document-boundary and CP
  global-position masks are passed through this argument, as the reference does.

Output: ``[sq, b, np * hnv]`` (same as the reference).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

import triton
import triton.language as tl


NEG_INF = float("-inf")
EPS = 1e-10


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (>= 16)."""
    return max(16, 1 << (n - 1).bit_length())


def sm90_or_newer() -> bool:
    """True on Hopper (SM90) or newer, where these kernels are supported."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _mask_strides(mask: Optional[Tensor], b: int, sq: int, skv: int) -> Tuple[int, int, int]:
    """(batch, query, key) strides for the optional additive mask; zeros if absent."""
    if mask is None:
        return 0, 0, 0
    if mask.dim() == 2:
        assert mask.shape == (sq, skv), f"mask {tuple(mask.shape)} != {(sq, skv)}"
        return 0, mask.stride(0), mask.stride(1)
    assert mask.dim() == 3 and mask.shape[1:] == (sq, skv)
    sm_b = mask.stride(0) if mask.shape[0] == b else 0
    return sm_b, mask.stride(1), mask.stride(2)


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, IDX_ptr, M_ptr, OUT_ptr, LSE_ptr,
    SQ, SKV, TOPK, HN, HNV,
    sq_s, sq_b, sq_h, sq_d,
    sk_s, sk_b, sk_h, sk_d,
    sv_s, sv_b, sv_h, sv_d,
    si_b, si_q, si_k,
    sm_b, sm_q, sm_k,
    so_s, so_b, so_h, so_d,
    sl_s, sl_b, sl_h,
    softmax_scale,
    HAS_MASK: tl.constexpr,
    HN_PAD: tl.constexpr,
    HV_PAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per (batch, query, head); online softmax over the top-k keys."""
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_d = tl.arange(0, HN_PAD)
    offs_v = tl.arange(0, HV_PAD)
    d_mask = offs_d < HN
    v_mask = offs_v < HNV

    q_ptrs = Q_ptr + pid_q * sq_s + pid_b * sq_b + pid_h * sq_h + offs_d * sq_d
    q_vec = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HV_PAD], dtype=tl.float32)

    for k0 in range(0, TOPK, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        valid_k = offs_k < TOPK
        idx = tl.load(IDX_ptr + pid_b * si_b + pid_q * si_q + offs_k * si_k, mask=valid_k, other=0)
        idx = idx.to(tl.int32)

        k_ptrs = K_ptr + idx[:, None] * sk_s + pid_b * sk_b + pid_h * sk_h + offs_d[None, :] * sk_d
        k_tile = tl.load(k_ptrs, mask=valid_k[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(q_vec[None, :] * k_tile, axis=1) * softmax_scale  # [BLOCK_K]

        if HAS_MASK:
            bias = tl.load(
                M_ptr + pid_b * sm_b + pid_q * sm_q + idx * sm_k, mask=valid_k, other=float("-inf")
            )
            scores = scores + bias
        else:
            scores = tl.where(idx <= pid_q, scores, float("-inf"))
        scores = tl.where(valid_k, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v_ptrs = V_ptr + idx[:, None] * sv_s + pid_b * sv_b + pid_h * sv_h + offs_v[None, :] * sv_d
        v_tile = tl.load(v_ptrs, mask=valid_k[:, None] & v_mask[None, :], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        m_i = m_new

    l_safe = tl.maximum(l_i, 1e-10)
    out = acc / l_safe
    o_ptrs = OUT_ptr + pid_q * so_s + pid_b * so_b + pid_h * so_h + offs_v * so_d
    tl.store(o_ptrs, out.to(OUT_ptr.dtype.element_ty), mask=v_mask)
    # lse s.t. exp(score - lse) is the normalised probability (matches out above).
    tl.store(LSE_ptr + pid_q * sl_s + pid_b * sl_b + pid_h * sl_h, m_i + tl.log(l_safe))


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, IDX_ptr, M_ptr, O_ptr, LSE_ptr, DO_ptr,
    DQ_ptr, DK_ptr, DV_ptr,
    SQ, SKV, TOPK, HN, HNV,
    sq_s, sq_b, sq_h, sq_d,
    sk_s, sk_b, sk_h, sk_d,
    sv_s, sv_b, sv_h, sv_d,
    si_b, si_q, si_k,
    sm_b, sm_q, sm_k,
    so_s, so_b, so_h, so_d,
    sl_s, sl_b, sl_h,
    sdo_s, sdo_b, sdo_h, sdo_d,
    sdq_s, sdq_b, sdq_h, sdq_d,
    softmax_scale,
    HAS_MASK: tl.constexpr,
    HN_PAD: tl.constexpr,
    HV_PAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per (batch, query, head). dq accumulates locally; dk/dv scatter."""
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_d = tl.arange(0, HN_PAD)
    offs_v = tl.arange(0, HV_PAD)
    d_mask = offs_d < HN
    v_mask = offs_v < HNV

    q_vec = tl.load(
        Q_ptr + pid_q * sq_s + pid_b * sq_b + pid_h * sq_h + offs_d * sq_d, mask=d_mask, other=0.0
    ).to(tl.float32)
    o_vec = tl.load(
        O_ptr + pid_q * so_s + pid_b * so_b + pid_h * so_h + offs_v * so_d, mask=v_mask, other=0.0
    ).to(tl.float32)
    do_vec = tl.load(
        DO_ptr + pid_q * sdo_s + pid_b * sdo_b + pid_h * sdo_h + offs_v * sdo_d,
        mask=v_mask, other=0.0,
    ).to(tl.float32)
    lse = tl.load(LSE_ptr + pid_q * sl_s + pid_b * sl_b + pid_h * sl_h)

    # D = dO . o  (== sum_j p_j * (dO . v_j)); the softmax-Jacobian correction term.
    D = tl.sum(do_vec * o_vec, axis=0)

    dq_acc = tl.zeros([HN_PAD], dtype=tl.float32)

    for k0 in range(0, TOPK, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        valid_k = offs_k < TOPK
        idx = tl.load(IDX_ptr + pid_b * si_b + pid_q * si_q + offs_k * si_k, mask=valid_k, other=0)
        idx = idx.to(tl.int32)

        k_ptrs = K_ptr + idx[:, None] * sk_s + pid_b * sk_b + pid_h * sk_h + offs_d[None, :] * sk_d
        k_tile = tl.load(k_ptrs, mask=valid_k[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_ptrs = V_ptr + idx[:, None] * sv_s + pid_b * sv_b + pid_h * sv_h + offs_v[None, :] * sv_d
        v_tile = tl.load(v_ptrs, mask=valid_k[:, None] & v_mask[None, :], other=0.0).to(tl.float32)

        scores = tl.sum(q_vec[None, :] * k_tile, axis=1) * softmax_scale
        if HAS_MASK:
            bias = tl.load(
                M_ptr + pid_b * sm_b + pid_q * sm_q + idx * sm_k, mask=valid_k, other=float("-inf")
            )
            scores = scores + bias
        else:
            scores = tl.where(idx <= pid_q, scores, float("-inf"))
        scores = tl.where(valid_k, scores, float("-inf"))

        p = tl.exp(scores - lse)  # [BLOCK_K] normalised probabilities (0 for masked)
        dp = tl.sum(do_vec[None, :] * v_tile, axis=1)  # [BLOCK_K]
        ds = p * (dp - D) * softmax_scale  # [BLOCK_K]

        dq_acc += tl.sum(ds[:, None] * k_tile, axis=0)  # [HN_PAD]

        # dv_j = p_j * dO ; dk_j = ds_j * q  -> scatter-add into the shared keys.
        dv_contrib = p[:, None] * do_vec[None, :]  # [BLOCK_K, HV_PAD]
        dk_contrib = ds[:, None] * q_vec[None, :]  # [BLOCK_K, HN_PAD]
        dv_ptrs = DV_ptr + idx[:, None] * sv_s + pid_b * sv_b + pid_h * sv_h + offs_v[None, :] * sv_d
        dk_ptrs = DK_ptr + idx[:, None] * sk_s + pid_b * sk_b + pid_h * sk_h + offs_d[None, :] * sk_d
        tl.atomic_add(dv_ptrs, dv_contrib, mask=valid_k[:, None] & v_mask[None, :])
        tl.atomic_add(dk_ptrs, dk_contrib, mask=valid_k[:, None] & d_mask[None, :])

    dq_ptrs = DQ_ptr + pid_q * sdq_s + pid_b * sdq_b + pid_h * sdq_h + offs_d * sdq_d
    tl.store(dq_ptrs, dq_acc, mask=d_mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _fwd_sparse_attention_triton(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    causal_mask: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Triton forward. Returns (out ``[sq, b, np, hnv]``, lse ``[sq, b, np]``, idx int32)."""
    sq, b, np, hn = query.shape
    skv = key.shape[0]
    hnv = value.shape[3]
    topk = topk_indices.shape[-1]

    assert key.shape[1] == b and key.shape[2] == np and key.shape[3] == hn
    assert value.shape[0] == skv and value.shape[1] == b and value.shape[2] == np
    assert topk_indices.shape[0] == b and topk_indices.shape[1] == sq

    idx = topk_indices.to(torch.int32).contiguous()
    out = torch.empty((sq, b, np, hnv), dtype=query.dtype, device=query.device)
    lse = torch.empty((sq, b, np), dtype=torch.float32, device=query.device)

    has_mask = causal_mask is not None
    mask_arg = causal_mask.to(torch.float32) if has_mask else query
    sm_b, sm_q, sm_k = _mask_strides(causal_mask, b, sq, skv)

    _sparse_attn_fwd_kernel[(b, sq, np)](
        query, key, value, idx, mask_arg, out, lse,
        sq, skv, topk, hn, hnv,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2),
        sm_b, sm_q, sm_k,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        softmax_scale,
        HAS_MASK=has_mask,
        HN_PAD=_next_pow2(hn),
        HV_PAD=_next_pow2(hnv),
        BLOCK_K=64,
        num_warps=4,
    )
    return out, lse, idx


def _bwd_sparse_attention_triton(
    query, key, value, idx, softmax_scale, causal_mask, out, lse, grad_out
) -> Tuple[Tensor, Tensor, Tensor]:
    """Triton backward. Returns (dq, dk, dv) in the input dtype."""
    sq, b, np, hn = query.shape
    skv = key.shape[0]
    hnv = value.shape[3]
    topk = idx.shape[-1]

    grad_out = grad_out.reshape(sq, b, np, hnv).contiguous()
    dq = torch.empty_like(query)
    # dk/dv are scattered across queries -> fp32 accumulators for atomic_add.
    dk_f = torch.zeros((skv, b, np, hn), dtype=torch.float32, device=query.device)
    dv_f = torch.zeros((skv, b, np, hnv), dtype=torch.float32, device=query.device)

    has_mask = causal_mask is not None
    mask_arg = causal_mask.to(torch.float32) if has_mask else query
    sm_b, sm_q, sm_k = _mask_strides(causal_mask, b, sq, skv)

    _sparse_attn_bwd_kernel[(b, sq, np)](
        query, key, value, idx, mask_arg, out, lse, grad_out,
        dq, dk_f, dv_f,
        sq, skv, topk, hn, hnv,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2),
        sm_b, sm_q, sm_k,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        softmax_scale,
        HAS_MASK=has_mask,
        HN_PAD=_next_pow2(hn),
        HV_PAD=_next_pow2(hnv),
        BLOCK_K=64,
        num_warps=4,
    )
    return dq, dk_f.to(key.dtype), dv_f.to(value.dtype)


class _TritonSparseAttention(torch.autograd.Function):
    """Fully fused Triton sparse attention (forward + flash-style backward)."""

    @staticmethod
    def forward(ctx, query, key, value, topk_indices, softmax_scale, causal_mask):
        out, lse, idx = _fwd_sparse_attention_triton(
            query, key, value, topk_indices, softmax_scale, causal_mask
        )
        ctx.save_for_backward(query, key, value, idx, out, lse, causal_mask)
        ctx.softmax_scale = softmax_scale
        return out.reshape(query.shape[0], query.shape[1], query.shape[2] * value.shape[3])

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value, idx, out, lse, causal_mask = ctx.saved_tensors
        dq, dk, dv = _bwd_sparse_attention_triton(
            query, key, value, idx, ctx.softmax_scale, causal_mask, out, lse, grad_out
        )
        needs = ctx.needs_input_grad
        return (
            dq if needs[0] else None,
            dk if needs[1] else None,
            dv if needs[2] else None,
            None,
            None,
            None,
        )


def fused_dsa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    causal_mask: Optional[Tensor] = None,
) -> Tensor:
    """Drop-in replacement for ``unfused_dsa_fn`` backed by the Triton fwd+bwd kernels.

    Same signature and output ``[sq, b, np * hnv]``; numerically equivalent to the
    dense reference (see the parity unit tests).
    """
    return _TritonSparseAttention.apply(
        query, key, value, topk_indices, softmax_scale, causal_mask
    )
