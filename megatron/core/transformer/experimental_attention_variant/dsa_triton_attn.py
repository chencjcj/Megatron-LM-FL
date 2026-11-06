# Copyright (c) 2026, FlagScale contributors. All rights reserved.
"""FlagScale addition: Triton + torch kernels for DSA sparse attention (MUSA-tuned).

Drop-in replacement for ``unfused_dsa_fn`` in ``dsa.py``:

    mask = build_dsa_mask(topk_indices, skv)          # cache per topk_indices (shared indexer)
    out = sparse_attention(query, key, value, mask, softmax_scale)

Position (i, j) is attended iff j is in topk_indices[b, i, :] and j <= i.

Design. On MUSA, torch bf16 GEMMs run ~2x faster than Triton tl.dot and a single-pass flash
kernel with head dim 256 is register bound (~2.2 ms fwd), so the work is split:
  forward
    * _lse_partial/_lse_combine: online max/sum over causal column tiles of S = Q K^T (bf16
      tl.dot inputs, exact fp32 accumulation; eager's fp32/TF32 bmm is also exact for bf16
      inputs) -> lse, split over column chunks for load balance.
    * _probs_kernel: recompute S per tile, P = mask * exp(S - lse) -> bf16 [b*np, sq, skv]
      (zeros above the diagonal; same rounding point as eager's ``.to(value.dtype)``).
    * O = P V: dense torch bf16 bmm. Saved for backward: packed Q/K/V, O, lse [b*np, sq]
      fp32 and (SAVE_PROBS) the bf16 P; no fp32 [sq, skv] activations are kept.
  backward
    * _bwd_tile_kernel recomputes S and dP = dO V^T in fp32 per causal tile, P in fp32,
      dS = P * (dP - delta) * scale with delta = rowsum(dO * O), and writes dS (and P if it
      was not saved) as bf16.
      (Keeping P and dP in fp32 makes dQ/dK at least as accurate as eager, whose dP is bf16.)
    * dV = P^T dO, dQ = dS K, dK = dS^T Q: dense torch bf16 bmms.

Rows without any allowed position produce output 0 and zero gradients (no NaN).
query/key/value may be arbitrary strided views; they are packed into contiguous
[b*np, s, d] buffers (cost included in the timings). MUSA bmm cannot write strided outputs,
so GEMM results are copied into the [s, b, np, d] layouts.
"""

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634

# Tile config of the Triton GEMM kernels; num_stages=1 (stages>1 is slower/unsafe on MUSA).
# NOTE: the MUSA Triton backend miscompiles _bwd_tile_kernel for some tiles (e.g. BN=64 with
# BK=64 gives sporadic NaN/garbage); re-verify correctness (tests in the FlagScale perf
# notes) before changing these.
FWD_TILE = dict(BM=64, BN=128, BK=64, num_warps=8)
BWD_TILE = dict(BM=64, BN=128, BK=64, num_warps=8)
# Column split of the lse pass (load balance across the causal triangle).
LSE_SPLIT = 512
# Save the forward bf16 P [b*np, sq, skv] for backward (bit-identical to the backward
# recompute): -0.13 ms bwd at the reference shape for +256 MiB of saved activations.
SAVE_PROBS = True


# --------------------------------------------------------------------------------------
# Mask
# --------------------------------------------------------------------------------------
@triton.jit
def _mask_scatter_kernel(idx_ptr, mask_ptr, SQ, SKV, TOPK, si_b, si_s, si_t,
                         BLOCK: tl.constexpr):
    """mask[b, i, idx[b, i, t]] = 1 for valid (0 <= idx <= i, idx < SKV) indices."""
    i = tl.program_id(0)
    bid = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    inb = offs < TOPK
    j = tl.load(idx_ptr + bid * si_b + i * si_s + offs * si_t, mask=inb, other=-1)
    ok = inb & (j >= 0) & (j <= i) & (j < SKV)
    one = tl.full((BLOCK,), 1, dtype=tl.uint8)
    tl.store(mask_ptr + (bid * SQ + i).to(tl.int64) * SKV + j, one, mask=ok)


def build_dsa_mask(topk_indices: torch.Tensor, skv: int) -> torch.Tensor:
    """Build the combined topk & causal mask.

    Args:
        topk_indices: [b, sq, topk] integer indices into the key dimension. Entries
            with j > i (or out of range, e.g. -1 padding) are ignored.
        skv: key sequence length.

    Returns:
        uint8 tensor [b, sq, skv], 1 where attention is allowed. Cache it and reuse it
        for every layer that shares ``topk_indices``.
    """
    b, sq, topk = topk_indices.shape
    mask = torch.zeros((b, sq, skv), dtype=torch.uint8, device=topk_indices.device)
    _mask_scatter_kernel[(sq, b)](
        topk_indices, mask, sq, skv, topk, *topk_indices.stride(),
        BLOCK=triton.next_power_of_2(topk), num_warps=4,
    )
    return mask


# --------------------------------------------------------------------------------------
# Triton kernels (all GEMM operands are contiguous [b*np, s, D] buffers)
# --------------------------------------------------------------------------------------
@triton.jit
def _lse_partial_kernel(
    q_ptr, k_ptr, m_ptr, pm_ptr, pl_ptr, S, NP, NSPLIT, qk_scale,
    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, CH: tl.constexpr,
):
    """Online (max, sum) of mask * exp2(q.k * c) over causal columns [sp*CH, (sp+1)*CH) of
    row block pid_m (split over columns for load balance; no S materialization)."""
    pid_m = tl.program_id(0)
    sp = tl.program_id(1)
    bh = tl.program_id(2)
    bid = bh // NP
    offs_r = pid_m * BM + tl.arange(0, BM)
    m_i = tl.full((BM,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BM,), dtype=tl.float32)
    c0 = sp * CH
    c1 = tl.minimum(c0 + CH, pid_m * BM + BM)
    if c0 < c1:
        offs_k = tl.arange(0, BK)
        q_base = q_ptr + bh * S * D + offs_r[:, None] * D
        for n0 in range(c0, c1, BN):
            offs_c = n0 + tl.arange(0, BN)
            k_base = k_ptr + bh * S * D + offs_c[:, None] * D
            s = tl.zeros((BM, BN), dtype=tl.float32)
            for d0 in range(0, D, BK):
                s += tl.dot(tl.load(q_base + (d0 + offs_k)[None, :]),
                            tl.trans(tl.load(k_base + (d0 + offs_k)[None, :])))
            msk = tl.load(m_ptr + bid * S * S + offs_r[:, None] * S + offs_c[None, :])
            s = tl.where(msk != 0, s * qk_scale, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            l_i = l_i * tl.math.exp2(m_i - m_safe) + tl.sum(tl.math.exp2(s - m_safe[:, None]), 1)
            m_i = m_new
    off = (bh * NSPLIT + sp) * S + offs_r
    tl.store(pm_ptr + off, m_i)
    tl.store(pl_ptr + off, l_i)


@triton.jit
def _lse_combine_kernel(pm_ptr, pl_ptr, lse_ptr, S, NSPLIT, BLOCK: tl.constexpr):
    """lse = log2 sum_sp l_sp * exp2(m_sp - m) + m; +inf for rows with no allowed entry."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    for sp in range(0, NSPLIT):
        m = tl.maximum(m, tl.load(pm_ptr + (bh * NSPLIT + sp) * S + offs))
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    l = tl.zeros((BLOCK,), dtype=tl.float32)
    for sp in range(0, NSPLIT):
        pm = tl.load(pm_ptr + (bh * NSPLIT + sp) * S + offs)
        l += tl.load(pl_ptr + (bh * NSPLIT + sp) * S + offs) * tl.math.exp2(pm - m_safe)
    has = l > 0.0
    tl.store(lse_ptr + bh * S + offs,
             tl.where(has, m + tl.math.log2(tl.where(has, l, 1.0)), float("inf")))


@triton.jit
def _probs_kernel(
    q_ptr, k_ptr, m_ptr, lse_ptr, p_ptr, S, NP, qk_scale,
    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """P[bh, tile] = mask * exp2(q.k * c - lse) as bf16; tiles above the diagonal -> zeros."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bh = tl.program_id(2)
    offs_r = pid_m * BM + tl.arange(0, BM)
    offs_c = pid_n * BN + tl.arange(0, BN)
    out_off = bh * S * S + offs_r[:, None] * S + offs_c[None, :]
    if pid_n * BN > pid_m * BM + BM - 1:
        tl.store(p_ptr + out_off, tl.zeros((BM, BN), dtype=p_ptr.dtype.element_ty))
        return
    offs_k = tl.arange(0, BK)
    q_base = q_ptr + bh * S * D + offs_r[:, None] * D
    k_base = k_ptr + bh * S * D + offs_c[:, None] * D
    s = tl.zeros((BM, BN), dtype=tl.float32)
    for d0 in range(0, D, BK):
        s += tl.dot(tl.load(q_base + (d0 + offs_k)[None, :]),
                    tl.trans(tl.load(k_base + (d0 + offs_k)[None, :])))
    bid = bh // NP
    msk = tl.load(m_ptr + bid * S * S + offs_r[:, None] * S + offs_c[None, :])
    lse = tl.load(lse_ptr + bh * S + offs_r)
    p = tl.where(msk != 0, tl.math.exp2(s * qk_scale - lse[:, None]), 0.0)
    tl.store(p_ptr + out_off, p.to(p_ptr.dtype.element_ty))


@triton.jit
def _delta_kernel(
    o_ptr, do_ptr, delta_ptr, S, NP,
    so_s, so_b, so_h, sdo_s, sdo_b, sdo_h,
    DV: tl.constexpr, BM: tl.constexpr,
):
    """delta[bh, i] = sum_d o[i, b, h, d] * do[i, b, h, d] (fp32)."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NP
    hid = bh % NP
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, DV)
    o = tl.load(o_ptr + bid * so_b + hid * so_h + offs_m[:, None] * so_s + offs_d[None, :])
    do = tl.load(do_ptr + bid * sdo_b + hid * sdo_h + offs_m[:, None] * sdo_s + offs_d[None, :])
    tl.store(delta_ptr + bh * S + offs_m, tl.sum(o.to(tl.float32) * do.to(tl.float32), 1))


@triton.jit
def _bwd_tile_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, m_ptr, lse_ptr, delta_ptr, p_ptr, ds_ptr,
    S, NP, qk_scale, sm_scale,
    D: tl.constexpr, DV: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    STORE_P: tl.constexpr,
):
    """Per tile: P = exp2(q.k * c - lse) (masked), dP = do.v, dS = P * (dP - delta) * scale.
    Writes dS (and P if STORE_P) as bf16 [bh, S, S]; tiles above the diagonal -> zeros."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bh = tl.program_id(2)
    offs_r = pid_m * BM + tl.arange(0, BM)
    offs_c = pid_n * BN + tl.arange(0, BN)
    out_off = bh * S * S + offs_r[:, None] * S + offs_c[None, :]
    if pid_n * BN > pid_m * BM + BM - 1:
        z = tl.zeros((BM, BN), dtype=ds_ptr.dtype.element_ty)
        if STORE_P:
            tl.store(p_ptr + out_off, z)
        tl.store(ds_ptr + out_off, z)
        return
    offs_k = tl.arange(0, BK)
    q_base = q_ptr + bh * S * D + offs_r[:, None] * D
    k_base = k_ptr + bh * S * D + offs_c[:, None] * D
    s = tl.zeros((BM, BN), dtype=tl.float32)
    for d0 in range(0, D, BK):
        s += tl.dot(tl.load(q_base + (d0 + offs_k)[None, :]),
                    tl.trans(tl.load(k_base + (d0 + offs_k)[None, :])))
    do_base = do_ptr + bh * S * DV + offs_r[:, None] * DV
    v_base = v_ptr + bh * S * DV + offs_c[:, None] * DV
    dp = tl.zeros((BM, BN), dtype=tl.float32)
    for d0 in range(0, DV, BK):
        dp += tl.dot(tl.load(do_base + (d0 + offs_k)[None, :]),
                     tl.trans(tl.load(v_base + (d0 + offs_k)[None, :])))
    bid = bh // NP
    msk = tl.load(m_ptr + bid * S * S + offs_r[:, None] * S + offs_c[None, :])
    lse = tl.load(lse_ptr + bh * S + offs_r)
    delta = tl.load(delta_ptr + bh * S + offs_r)
    p = tl.where(msk != 0, tl.math.exp2(s * qk_scale - lse[:, None]), 0.0)
    ds = p * (dp - delta[:, None]) * sm_scale
    if STORE_P:
        tl.store(p_ptr + out_off, p.to(p_ptr.dtype.element_ty))
    tl.store(ds_ptr + out_off, ds.to(ds_ptr.dtype.element_ty))


# --------------------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------------------
def _pack(x):
    """[s, b, np, d] (any strides) -> contiguous [b * np, s, d]."""
    s, b, np_, d = x.shape
    return x.permute(1, 2, 0, 3).reshape(b * np_, s, d).contiguous()


def _unpack_into(dst, x):
    """Copy x [b * np, s, d] into dst [s, b, np, d]."""
    s, b, np_, d = dst.shape
    dst.copy_(x.view(b, np_, s, d).permute(2, 0, 1, 3))
    return dst


def _fwd(query, key, value, mask, softmax_scale):
    sq, b, np_, hn = query.shape
    hnv = value.shape[3]
    bh = b * np_
    dev = query.device
    t = FWD_TILE
    q_p, k_p, v_p = _pack(query), _pack(key), _pack(value)
    qk_scale = softmax_scale * _LOG2E
    nsplit = triton.cdiv(sq, LSE_SPLIT)
    pm = torch.empty((bh, nsplit, sq), dtype=torch.float32, device=dev)
    pl = torch.empty_like(pm)
    _lse_partial_kernel[(sq // t["BM"], nsplit, bh)](
        q_p, k_p, mask, pm, pl, sq, np_, nsplit, qk_scale,
        D=hn, BM=t["BM"], BN=t["BN"], BK=t["BK"], CH=LSE_SPLIT,
        num_warps=t["num_warps"], num_stages=1,
    )
    lse = torch.empty((bh, sq), dtype=torch.float32, device=dev)
    _lse_combine_kernel[(sq // 128, bh)](pm, pl, lse, sq, nsplit, BLOCK=128, num_warps=4)
    del pm, pl
    probs = torch.empty((bh, sq, sq), dtype=value.dtype, device=dev)
    _probs_kernel[(sq // t["BM"], sq // t["BN"], bh)](
        q_p, k_p, mask, lse, probs, sq, np_, qk_scale,
        D=hn, BM=t["BM"], BN=t["BN"], BK=t["BK"], num_warps=t["num_warps"], num_stages=1,
    )
    o = torch.bmm(probs, v_p)
    if not SAVE_PROBS:
        probs = None
    out = _unpack_into(torch.empty((sq, b, np_, hnv), dtype=value.dtype, device=dev), o)
    return out, lse, q_p, k_p, v_p, probs


def _bwd(q_p, k_p, v_p, mask, out, lse, probs, grad_out, softmax_scale, like):
    sq, b, np_, hnv = out.shape
    hn = q_p.shape[2]
    bh = b * np_
    dev = q_p.device
    t = BWD_TILE
    do4 = grad_out.reshape(sq, b, np_, hnv)
    if do4.stride(3) != 1:
        do4 = do4.contiguous()
    delta = torch.empty((bh, sq), dtype=torch.float32, device=dev)
    _delta_kernel[(sq // 64, bh)](
        out, do4, delta, sq, np_, *out.stride()[:3], *do4.stride()[:3], DV=hnv, BM=64,
        num_warps=4,
    )
    do_p = _pack(do4)
    store_p = probs is None
    if store_p:
        probs = torch.empty((bh, sq, sq), dtype=v_p.dtype, device=dev)
    ds = torch.empty((bh, sq, sq), dtype=q_p.dtype, device=dev)
    _bwd_tile_kernel[(sq // t["BM"], sq // t["BN"], bh)](
        q_p, k_p, v_p, do_p, mask, lse, delta, probs, ds,
        sq, np_, softmax_scale * _LOG2E, softmax_scale,
        D=hn, DV=hnv, BM=t["BM"], BN=t["BN"], BK=t["BK"], STORE_P=store_p,
        num_warps=t["num_warps"], num_stages=1,
    )
    q_like, k_like, v_like = like
    dv = _unpack_into(torch.empty(v_like, dtype=v_p.dtype, device=dev),
                      torch.bmm(probs.transpose(1, 2), do_p))
    del probs
    dq = _unpack_into(torch.empty(q_like, dtype=q_p.dtype, device=dev), torch.bmm(ds, k_p))
    dk = _unpack_into(torch.empty(k_like, dtype=k_p.dtype, device=dev),
                      torch.bmm(ds.transpose(1, 2), q_p))
    return dq, dk, dv


class SparseAttentionFunction(torch.autograd.Function):
    """Autograd wrapper; see ``sparse_attention``."""

    @staticmethod
    def forward(ctx, query, key, value, mask, softmax_scale):
        out, lse, q_p, k_p, v_p, probs = _fwd(query, key, value, mask, softmax_scale)
        # Packed [b*np, s, d] copies are saved instead of the (possibly strided) inputs.
        ctx.save_for_backward(q_p, k_p, v_p, mask, out, lse, probs)
        ctx.softmax_scale = softmax_scale
        ctx.like = (query.shape, key.shape, value.shape)
        sq, b, np_, hnv = out.shape
        return out.view(sq, b, np_ * hnv)

    @staticmethod
    def backward(ctx, grad_out):
        q_p, k_p, v_p, mask, out, lse, probs = ctx.saved_tensors
        dq, dk, dv = _bwd(q_p, k_p, v_p, mask, out, lse, probs, grad_out, ctx.softmax_scale,
                          ctx.like)
        return dq, dk, dv, None, None


def sparse_attention(query, key, value, mask, softmax_scale):
    """DSA sparse attention (drop-in for ``unfused_dsa_fn`` given a prebuilt mask).

    Args:
        query: [sq, b, np, hn] bf16 (any strides).
        key: [skv, b, np, hn] bf16 (any strides).
        value: [skv, b, np, hnv] bf16 (any strides).
        mask: uint8 [b, sq, skv] from ``build_dsa_mask`` (topk & causal), contiguous.
        softmax_scale: float.

    Returns:
        [sq, b, np * hnv] tensor with value's dtype. Gradients dq/dk/dv are contiguous
        tensors with the input shapes/dtypes.

    Requirements: sq == skv (causal self-attention), sq a multiple of 128, hn and hnv
    multiples of 64, hnv a power of two (GLM5/DSA: 2048/256/256).
    """
    sq, skv = query.shape[0], key.shape[0]
    hn, hnv = query.shape[3], value.shape[3]
    assert sq == skv, "DSA Triton attention assumes sq == skv (causal self-attention)"
    assert sq % 128 == 0, "sq must be a multiple of 128"
    assert hn % 64 == 0 and hnv % 64 == 0 and (hnv & (hnv - 1)) == 0
    assert mask.dtype == torch.uint8 and tuple(mask.shape) == (query.shape[1], sq, skv)
    assert mask.is_contiguous()
    return SparseAttentionFunction.apply(query, key, value, mask, float(softmax_scale))
