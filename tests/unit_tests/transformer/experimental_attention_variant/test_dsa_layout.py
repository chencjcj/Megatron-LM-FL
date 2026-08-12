# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Identity/consistency tests for the packed + zigzag CP layout helpers in
``dsa_layout.py`` (ported from NVIDIA/Megatron-LM).

These are pure index-math properties that must hold for the CP sharding to be
correct, and they run on CPU (no distributed / GPU needed):

* **Permutation**: concatenating every CP rank's local token positions covers
  the full sequence exactly once (each token owned by exactly one rank).
* **Reorder identity**: the gathered-KV reorder index restores the rank-by-rank
  gathered tensors to global (packed) order — ``gathered[reorder] == arange``.
* **Per-doc balance** (packed): each rank owns ``doc_len / cp_size`` tokens of
  every document (front + mirrored back), matching the per-document zigzag that
  ``_apply_rotary_pos_emb_thd`` assumes.

Run with::

    pytest tests/unit_tests/transformer/experimental_attention_variant/test_dsa_layout.py -v
"""

from __future__ import annotations

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_local_positions,
    build_packed_allgather_cp_query_positions_and_key_reorder,
    build_zigzag_allgather_cp_key_reorder,
    build_zigzag_cp_local_positions,
    get_cp_positions_from_layout,
)

DEV = torch.device("cpu")


def _cu(doc_lens):
    return torch.tensor([0, *torch.tensor(doc_lens).cumsum(0).tolist()], dtype=torch.int32)


class TestZigzagLayout:
    """Non-packed whole-sequence zigzag (what the plain-CP path uses)."""

    @pytest.mark.parametrize("cp_size", [2, 4])
    @pytest.mark.parametrize("seq_len", [64, 96])
    def test_permutation_and_reorder(self, cp_size, seq_len):
        if seq_len % (2 * cp_size) != 0:
            pytest.skip("seq_len must be divisible by 2*cp_size")
        gathered = torch.cat(
            [build_zigzag_cp_local_positions(seq_len, cp_size, r, DEV) for r in range(cp_size)]
        )
        # Every position exactly once.
        assert torch.equal(torch.sort(gathered).values, torch.arange(seq_len))
        # Reorder restores global order.
        reorder = build_zigzag_allgather_cp_key_reorder(seq_len // cp_size, cp_size, DEV)
        assert torch.equal(gathered[reorder], torch.arange(seq_len))

    def test_get_cp_positions_uniform(self):
        sq, cp = 16, 2
        skv = sq * cp
        qpos, kpos = get_cp_positions_from_layout(sq, skv, cp, 1, "allgather", DEV)
        assert torch.equal(qpos, build_zigzag_cp_local_positions(skv, cp, 1, DEV))
        assert torch.equal(kpos, torch.arange(skv))


class TestPackedLayout:
    """Per-document zigzag THD sharding (what THD+CP needs)."""

    # Each doc length must be divisible by 2*cp_size.
    CASES = [
        (2, [8, 4, 12]),
        (2, [16]),
        (4, [8, 16]),
        (4, [24]),
    ]

    @pytest.mark.parametrize("cp_size,doc_lens", CASES)
    def test_permutation(self, cp_size, doc_lens):
        cu = _cu(doc_lens)
        total = int(cu[-1])
        gathered = torch.cat(
            [
                build_packed_allgather_cp_local_positions(
                    cu, cp_size, r, DEV, cu_seqlens_cover_output=True
                )
                for r in range(cp_size)
            ]
        )
        assert gathered.numel() == total, f"{gathered.numel()} != {total}"
        # Every packed position exactly once across ranks.
        assert torch.equal(torch.sort(gathered).values, torch.arange(total))

    @pytest.mark.parametrize("cp_size,doc_lens", CASES)
    def test_reorder_restores_global(self, cp_size, doc_lens):
        cu = _cu(doc_lens)
        total = int(cu[-1])
        _, reorder = build_packed_allgather_cp_query_positions_and_key_reorder(
            cu, cu, cp_size, 0, DEV,
            query_cu_seqlens_cover_output=True,
            key_cu_seqlens_cover_output=True,
        )
        gathered = torch.cat(
            [
                build_packed_allgather_cp_local_positions(
                    cu, cp_size, r, DEV, cu_seqlens_cover_output=True
                )
                for r in range(cp_size)
            ]
        )
        assert torch.equal(gathered[reorder], torch.arange(total))

    @pytest.mark.parametrize("cp_size,doc_lens", CASES)
    def test_per_doc_balance(self, cp_size, doc_lens):
        """Each rank owns doc_len/cp_size tokens of every document, all in-range."""
        cu = _cu(doc_lens)
        for r in range(cp_size):
            pos = build_packed_allgather_cp_local_positions(
                cu, cp_size, r, DEV, cu_seqlens_cover_output=True
            )
            for d, L in enumerate(doc_lens):
                s, e = int(cu[d]), int(cu[d + 1])
                in_doc = ((pos >= s) & (pos < e)).sum().item()
                assert in_doc == L // cp_size, (
                    f"[cp={cp_size} rank={r} doc={d}] {in_doc} != {L // cp_size}"
                )
