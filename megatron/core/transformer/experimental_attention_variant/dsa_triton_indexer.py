# Copyright (c) 2026, FlagScale contributors. All rights reserved.
"""FlagScale addition: fused Triton kernels for the DSA lightning-indexer scores.

index_scores[b, i, j] = sum_h w[i, b, h] * relu(q[i, b, h, :] . k[j, b, :])   (j <= i)
                      = -inf                                               (j >  i)

The eager implementation materializes the per-head scores as a [sq, b, h, sk] fp32
tensor (2 GiB at sq=2048, b=4, h=32) several times in forward and backward. These kernels
recompute per-head tiles on the fly (bf16 inputs, fp32 accumulation, which is exact for
the bf16 products) and skip tiles that lie entirely above the causal diagonal.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _indexer_fwd_kernel(
    q_ptr, w_ptr, k_ptr, out_ptr,
    SQ, SK,
    sq_s, sq_b, sq_h, sq_d,
    sw_s, sw_b, sw_h,
    sk_s, sk_b, sk_d,
    so_b, so_s, so_t,
    H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bid = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < SQ
    n_mask = offs_n < SK
    out_ptrs = out_ptr + bid * so_b + offs_m[:, None] * so_s + offs_n[None, :] * so_t
    io_mask = m_mask[:, None] & n_mask[None, :]
    if pid_n * BN > pid_m * BM + BM - 1:
        # Tile entirely above the causal diagonal.
        tl.store(out_ptrs, tl.full((BM, BN), float("-inf"), tl.float32), mask=io_mask)
        return
    k_tile = tl.load(
        k_ptr + offs_n[:, None] * sk_s + bid * sk_b + offs_d[None, :] * sk_d,
        mask=n_mask[:, None], other=0.0,
    )
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for h in range(H):
        q_tile = tl.load(
            q_ptr + offs_m[:, None] * sq_s + bid * sq_b + h * sq_h + offs_d[None, :] * sq_d,
            mask=m_mask[:, None], other=0.0,
        )
        w_h = tl.load(w_ptr + offs_m * sw_s + bid * sw_b + h * sw_h, mask=m_mask, other=0.0)
        s = tl.dot(q_tile, tl.trans(k_tile))
        acc += tl.maximum(s, 0.0) * w_h.to(tl.float32)[:, None]
    acc = tl.where(offs_n[None, :] > offs_m[:, None], float("-inf"), acc)
    tl.store(out_ptrs, acc, mask=io_mask)


@triton.jit
def _indexer_bwd_dq_kernel(
    q_ptr, w_ptr, k_ptr, g_ptr, dq_ptr, dw_ptr,
    SQ, SK,
    sq_s, sq_b, sq_h, sq_d,
    sw_s, sw_b, sw_h,
    sk_s, sk_b, sk_d,
    sg_b, sg_s, sg_t,
    sdq_s, sdq_b, sdq_h, sdq_d,
    sdw_s, sdw_b, sdw_h,
    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bid = tl.program_id(1)
    h = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < SQ
    q_tile = tl.load(
        q_ptr + offs_m[:, None] * sq_s + bid * sq_b + h * sq_h + offs_d[None, :] * sq_d,
        mask=m_mask[:, None], other=0.0,
    )
    w_h = tl.load(w_ptr + offs_m * sw_s + bid * sw_b + h * sw_h, mask=m_mask, other=0.0).to(tl.float32)
    dq = tl.zeros((BM, D), dtype=tl.float32)
    dw = tl.zeros((BM,), dtype=tl.float32)
    hi = tl.minimum(pid_m * BM + BM, SK)
    for start_n in range(0, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        n_mask = offs_n < SK
        k_tile = tl.load(
            k_ptr + offs_n[:, None] * sk_s + bid * sk_b + offs_d[None, :] * sk_d,
            mask=n_mask[:, None], other=0.0,
        )
        # G is zero at causally-masked positions (the caller applies the valid mask).
        g = tl.load(
            g_ptr + bid * sg_b + offs_m[:, None] * sg_s + offs_n[None, :] * sg_t,
            mask=m_mask[:, None] & n_mask[None, :], other=0.0,
        )
        s = tl.dot(q_tile, tl.trans(k_tile))
        dw += tl.sum(g * tl.maximum(s, 0.0), axis=1)
        gs = tl.where(s > 0.0, g * w_h[:, None], 0.0)
        if SPLIT:
            # Two bf16 GEMMs on a hi/lo split of the fp32 grad keep ~16 mantissa bits
            # (finer than the TF32 the eager einsum uses) at bf16 tensor-core speed.
            gs_hi = gs.to(tl.bfloat16)
            gs_lo = (gs - gs_hi.to(tl.float32)).to(tl.bfloat16)
            dq += tl.dot(gs_hi, k_tile)
            dq += tl.dot(gs_lo, k_tile)
        else:
            dq += tl.dot(gs, k_tile.to(tl.float32))
    tl.store(
        dq_ptr + offs_m[:, None] * sdq_s + bid * sdq_b + h * sdq_h + offs_d[None, :] * sdq_d,
        dq.to(dq_ptr.dtype.element_ty), mask=m_mask[:, None],
    )
    tl.store(dw_ptr + offs_m * sdw_s + bid * sdw_b + h * sdw_h, dw.to(dw_ptr.dtype.element_ty), mask=m_mask)


@triton.jit
def _indexer_bwd_dk_kernel(
    q_ptr, w_ptr, k_ptr, g_ptr, dk_ptr,
    SQ, SK,
    sq_s, sq_b, sq_h, sq_d,
    sw_s, sw_b, sw_h,
    sk_s, sk_b, sk_d,
    sg_b, sg_s, sg_t,
    sdk_g, sdk_s, sdk_b, sdk_d,
    HPG: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    bid = tl.program_id(1)
    hg = tl.program_id(2)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    n_mask = offs_n < SK
    k_tile = tl.load(
        k_ptr + offs_n[:, None] * sk_s + bid * sk_b + offs_d[None, :] * sk_d,
        mask=n_mask[:, None], other=0.0,
    )
    dk = tl.zeros((BN, D), dtype=tl.float32)
    lo = (pid_n * BN // BM) * BM
    for start_m in range(lo, SQ, BM):
        offs_m = start_m + tl.arange(0, BM)
        m_mask = offs_m < SQ
        g = tl.load(
            g_ptr + bid * sg_b + offs_m[:, None] * sg_s + offs_n[None, :] * sg_t,
            mask=m_mask[:, None] & n_mask[None, :], other=0.0,
        )
        for hh in range(HPG):
            h = hg * HPG + hh
            q_tile = tl.load(
                q_ptr + offs_m[:, None] * sq_s + bid * sq_b + h * sq_h + offs_d[None, :] * sq_d,
                mask=m_mask[:, None], other=0.0,
            )
            w_h = tl.load(w_ptr + offs_m * sw_s + bid * sw_b + h * sw_h, mask=m_mask, other=0.0).to(tl.float32)
            s = tl.dot(q_tile, tl.trans(k_tile))
            gs = tl.where(s > 0.0, g * w_h[:, None], 0.0)
            if SPLIT:
                gs_hi = gs.to(tl.bfloat16)
                gs_lo = (gs - gs_hi.to(tl.float32)).to(tl.bfloat16)
                dk += tl.dot(tl.trans(gs_hi), q_tile)
                dk += tl.dot(tl.trans(gs_lo), q_tile)
            else:
                dk += tl.dot(tl.trans(gs), q_tile.to(tl.float32))
    tl.store(
        dk_ptr + hg * sdk_g + offs_n[:, None] * sdk_s + bid * sdk_b + offs_d[None, :] * sdk_d,
        dk, mask=n_mask[:, None],
    )


# Tuned on MTT S5000 at sq=2048, b=4, h=32, d=128. Timings are very sensitive to these
# (register spills): e.g. dq BM64/BN64 is 5x slower than BM32/BN32, num_warps=2 is 7x slower.
FWD_CFG = dict(BM=64, BN=128, num_warps=8, num_stages=1)
BWD_DQ_CFG = dict(BM=32, BN=32, num_warps=4, num_stages=1)
BWD_DK_CFG = dict(BM=32, BN=32, HPG=2, num_warps=4, num_stages=1)
# fp32 grad operand as hi/lo bf16 pair: as accurate as the eager TF32 einsum; a plain fp32
# tl.dot on MUSA was both slower and less accurate here.
SPLIT_BF16 = True


def indexer_scores_fwd(q: torch.Tensor, w: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """q [sq, b, h, d] bf16, w [sq, b, h], k [sk, b, d] bf16 -> [b, sq, sk] fp32 causal-masked."""
    sq, b, h, d = q.shape
    sk = k.size(0)
    out = torch.empty((b, sq, sk), dtype=torch.float32, device=q.device)
    BM, BN = FWD_CFG["BM"], FWD_CFG["BN"]
    grid = (triton.cdiv(sq, BM), triton.cdiv(sk, BN), b)
    _indexer_fwd_kernel[grid](
        q, w, k, out, sq, sk,
        *q.stride(), *w.stride(), *k.stride(), *out.stride(),
        H=h, D=d, BM=BM, BN=BN,
        num_warps=FWD_CFG["num_warps"], num_stages=FWD_CFG["num_stages"],
    )
    return out


def indexer_scores_bwd(q, w, k, grad_scores):
    """Grads of sum(grad_scores * index_scores) wrt q, w, k. grad_scores [b, sq, sk] fp32
    must be zero above the causal diagonal."""
    sq, b, h, d = q.shape
    sk = k.size(0)
    dq = torch.empty_like(q)
    dw = torch.empty_like(w)
    c = BWD_DQ_CFG
    _indexer_bwd_dq_kernel[(triton.cdiv(sq, c["BM"]), b, h)](
        q, w, k, grad_scores, dq, dw, sq, sk,
        *q.stride(), *w.stride(), *k.stride(), *grad_scores.stride(), *dq.stride(), *dw.stride(),
        D=d, BM=c["BM"], BN=c["BN"], SPLIT=SPLIT_BF16,
        num_warps=c["num_warps"], num_stages=c["num_stages"],
    )
    c = BWD_DK_CFG
    hpg = c["HPG"]
    assert h % hpg == 0
    dk_part = torch.empty((h // hpg, sk, b, d), dtype=torch.float32, device=q.device)
    _indexer_bwd_dk_kernel[(triton.cdiv(sk, c["BN"]), b, h // hpg)](
        q, w, k, grad_scores, dk_part, sq, sk,
        *q.stride(), *w.stride(), *k.stride(), *grad_scores.stride(), *dk_part.stride(),
        HPG=hpg, D=d, BM=c["BM"], BN=c["BN"], SPLIT=SPLIT_BF16,
        num_warps=c["num_warps"], num_stages=c["num_stages"],
    )
    dk = dk_part.sum(0).to(k.dtype)
    return dq, dw, dk
