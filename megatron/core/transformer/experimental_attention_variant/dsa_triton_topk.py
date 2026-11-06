# Copyright (c) 2026, FlagScale contributors. All rights reserved.
"""FlagScale addition: row-wise top-k *index set* via radix select (Triton).

Replaces ``scores.topk(k, dim=-1)[1]`` for the DSA indexer, where only the set of selected
key positions matters (it is scattered into a mask), not their order. One program per row:
fp32 scores are mapped to order-preserving uint32 keys, the k-th largest key is found with a
32-step bitwise search, then all keys above it plus the lowest-index ties are written out.

For distinct finite scores the selected set equals torch.topk's. Among exactly-tied values
(in practice only the -inf entries of causally-masked positions in early rows) the choice can
differ from torch.topk, which is itself unspecified there; those positions are masked out by
the causal mask downstream either way.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_index_kernel(x_ptr, out_ptr, N, K, sx_r, so_r, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    valid = offs < N
    x = tl.load(x_ptr + row * sx_r + offs, mask=valid, other=float("-inf"))
    bits = x.to(tl.uint32, bitcast=True)
    # Order-preserving map: negatives -> flip all bits, non-negatives -> flip sign bit.
    neg = (bits >> 31) != 0
    key = tl.where(neg, ~bits, bits | 0x80000000)
    # Padding lanes must never be selected: give them key 0 (below every real key incl. -inf,
    # whose key is 0x007FFFFF) and exclude them from counts.
    key = tl.where(valid, key, 0)
    thr = tl.zeros((), dtype=tl.uint32)
    for i in range(32):
        cand = thr | (1 << (31 - i)).to(tl.uint32)
        cnt = tl.sum(((key >= cand) & valid).to(tl.int32), axis=0)
        thr = tl.where(cnt >= K, cand, thr)
    gt = (key > thr) & valid
    eq = (key == thr) & valid
    n_gt = tl.sum(gt.to(tl.int32), axis=0)
    eq_rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1
    take_eq = eq & (eq_rank < K - n_gt)
    sel = gt | take_eq
    pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
    tl.store(out_ptr + row * so_r + pos, offs.to(tl.int64), mask=sel)


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Unordered top-k indices along the last dim. scores [..., N] fp32 contiguous rows."""
    n = scores.size(-1)
    assert scores.dtype == torch.float32 and k <= n
    x = scores.reshape(-1, n)
    if x.stride(-1) != 1:
        x = x.contiguous()
    out = torch.empty((x.size(0), k), dtype=torch.int64, device=scores.device)
    block = triton.next_power_of_2(n)
    _topk_index_kernel[(x.size(0),)](
        x, out, n, k, x.stride(0), out.stride(0), BLOCK=block, num_warps=8 if block >= 2048 else 4
    )
    return out.view(*scores.shape[:-1], k)
