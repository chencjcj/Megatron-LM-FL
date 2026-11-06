##### FlagScale Begin #####
# Fused Triton cross entropy (forward + backward) for the tensor-parallel-size-1,
# no-label-smoothing case. Opt-in via TRITON_CROSS_ENTROPY=1 (see
# cross_entropy.vocab_parallel_cross_entropy).
#
# Semantics match _VocabParallelCrossEntropy exactly (TP=1, label_smoothing=0):
#   loss[r] = log(sum_v exp(x[r, v] - max_r)) - (x[r, t] - max_r)   if t in [start, end)
#   loss[r] = log(sum_v exp(x[r, v] - max_r))                        otherwise
#   grad[r, v] = (softmax[r, v] - [v == t and t valid]) * grad_output[r]
# The forward reads the (bf16) logits once and keeps only a per-row fp32
# logsumexp; the backward recomputes the softmax from it and writes the grad in
# the input dtype directly. No [s, b, V] fp32 tensor is ever materialized.

from typing import Optional

import torch
import triton
import triton.language as tl

# Tuned on MTT S5000 for logits [2048, 4, 154880] bf16.
_FWD_BLOCK_V = 8192
_FWD_NUM_WARPS = 8
_BWD_BLOCK_V = 4096
_BWD_BLOCKS_PER_PROG = 4
_BWD_NUM_WARPS = 16


@triton.jit
def _ce_fwd_kernel(
    logits_ptr,
    target_ptr,
    loss_ptr,
    lse_ptr,
    stride_row,
    V,
    vocab_start,
    vocab_end,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride_row
    offs = tl.arange(0, BLOCK_V)

    # Online max / sum-exp in fp32 (base-2 domain) with one scalar rescale per
    # block, i.e. a single exp2 per element. Every block has >= 1 in-range
    # element, so the running max is finite after the first block.
    m = float("-inf")
    s = 0.0
    for start in range(0, V, BLOCK_V):
        idx = start + offs
        x = tl.load(row_ptr + idx, mask=idx < V, other=float("-inf")).to(tl.float32)
        x = x * 1.4426950408889634
        m_new = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp2(m - m_new) + tl.sum(tl.exp2(x - m_new), axis=0)
        m = m_new

    # log(sum exp(x - max)) in natural-log units.
    log_sum = tl.log2(s) * 0.6931471805599453
    max_row = m * 0.6931471805599453

    target = tl.load(target_ptr + row)
    valid = (target >= vocab_start) & (target < vocab_end)
    local_t = tl.where(valid, target - vocab_start, 0)
    x_t = tl.load(row_ptr + local_t).to(tl.float32)
    predicted = tl.where(valid, x_t - max_row, 0.0)

    tl.store(loss_ptr + row, log_sum - predicted)
    tl.store(lse_ptr + row, max_row + log_sum)


@triton.jit
def _ce_bwd_kernel(
    logits_ptr,
    grad_in_ptr,
    target_ptr,
    lse_ptr,
    grad_out_ptr,
    stride_row,
    stride_grad_row,
    V,
    vocab_start,
    vocab_end,
    BLOCK_V: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_prog = tl.program_id(1)
    row_ptr = logits_ptr + row * stride_row
    grad_row_ptr = grad_in_ptr + row * stride_grad_row
    offs = tl.arange(0, BLOCK_V)

    lse2 = tl.load(lse_ptr + row) * 1.4426950408889634
    g = tl.load(grad_out_ptr + row).to(tl.float32)
    target = tl.load(target_ptr + row)
    valid = (target >= vocab_start) & (target < vocab_end)
    local_t = tl.where(valid, target - vocab_start, -1)

    for i in range(BLOCKS_PER_PROG):
        idx = (col_prog * BLOCKS_PER_PROG + i) * BLOCK_V + offs
        mask = idx < V
        x = tl.load(row_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        p = tl.exp2(x * 1.4426950408889634 - lse2)
        p = tl.where(idx == local_t, p - 1.0, p)
        tl.store(grad_row_ptr + idx, (p * g).to(grad_in_ptr.dtype.element_ty), mask=mask)


class _TritonCrossEntropy(torch.autograd.Function):
    """Fused cross entropy over the full (non-split) vocab dimension."""

    @staticmethod
    def forward(ctx, logits, target, vocab_start_index=0, vocab_end_index=None):
        """Triton cross entropy forward function."""
        V = logits.size(-1)
        if vocab_end_index is None:
            vocab_end_index = vocab_start_index + V
        logits_2d = logits.reshape(-1, V)
        if logits_2d.stride(-1) != 1:
            logits_2d = logits_2d.contiguous()
        target_1d = target.reshape(-1)
        N = logits_2d.size(0)

        loss = torch.empty(N, dtype=torch.float32, device=logits.device)
        lse = torch.empty(N, dtype=torch.float32, device=logits.device)
        if N > 0:
            _ce_fwd_kernel[(N,)](
                logits_2d,
                target_1d,
                loss,
                lse,
                logits_2d.stride(0),
                V,
                vocab_start_index,
                vocab_end_index,
                BLOCK_V=_FWD_BLOCK_V,
                num_warps=_FWD_NUM_WARPS,
            )

        ctx.save_for_backward(logits_2d, target_1d, lse)
        ctx.vocab_start_index = vocab_start_index
        ctx.vocab_end_index = vocab_end_index
        ctx.logits_shape = logits.shape
        return loss.view(target.shape)

    @staticmethod
    def backward(ctx, grad_output):
        """Triton cross entropy backward function."""
        logits_2d, target_1d, lse = ctx.saved_tensors
        N, V = logits_2d.shape
        grad_out_1d = grad_output.reshape(-1).contiguous()
        grad_input = torch.empty_like(logits_2d)
        if N > 0:
            grid = (N, triton.cdiv(V, _BWD_BLOCK_V * _BWD_BLOCKS_PER_PROG))
            _ce_bwd_kernel[grid](
                logits_2d,
                grad_input,
                target_1d,
                lse,
                grad_out_1d,
                logits_2d.stride(0),
                grad_input.stride(0),
                V,
                ctx.vocab_start_index,
                ctx.vocab_end_index,
                BLOCK_V=_BWD_BLOCK_V,
                BLOCKS_PER_PROG=_BWD_BLOCKS_PER_PROG,
                num_warps=_BWD_NUM_WARPS,
            )
        return grad_input.view(ctx.logits_shape), None, None, None


def triton_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int = 0,
    vocab_end_index: Optional[int] = None,
) -> torch.Tensor:
    """Fused Triton cross entropy. logits [..., V], target [...] -> fp32 loss [...]."""
    return _TritonCrossEntropy.apply(logits, target, vocab_start_index, vocab_end_index)


##### FlagScale End #####
