"""Verify the conv-state composition in the AR patch.

If we split a sequence into block_a + block_b and apply causal conv in two stages
(prefill block_a → store conv_state → use it as ctx for block_b), the output must
match applying the conv to the full concatenated sequence in one shot."""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.parametrize("kernel_size,len_a,len_b", [(4, 7, 5), (4, 1, 8), (4, 10, 1), (3, 6, 3)])
def test_conv_composition_matches_single_shot(kernel_size, len_a, len_b):
    torch.manual_seed(0)
    dim = 8
    conv = torch.nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size - 1, groups=dim, bias=False)
    # bias=False so no activation/silu term to worry about for the linearity check

    # Full sequence single-shot
    full = torch.randn(1, dim, len_a + len_b)
    out_full = conv(full)[..., :len_a + len_b]
    out_b_ref = out_full[..., len_a:]

    # Two-stage: prefill block_a, save conv_state, then run block_b with ctx
    block_a = full[..., :len_a]
    block_b = full[..., len_a:]

    # Mimic the patch: new_conv_state stores the last kernel_size columns of block_a, padded if needed
    if block_a.shape[-1] >= kernel_size:
        conv_state = block_a[..., -kernel_size:]
    else:
        conv_state = F.pad(block_a, (kernel_size - block_a.shape[-1], 0))

    # block_b path with ctx
    ctx = conv_state[..., -(kernel_size - 1):]
    full_input = torch.cat([ctx, block_b], dim=-1)
    out_with_ctx = conv(full_input)
    out_b_patched = out_with_ctx[..., kernel_size - 1: kernel_size - 1 + len_b]

    assert torch.allclose(out_b_ref, out_b_patched, atol=1e-5), (
        f"conv composition diverged. kernel={kernel_size}, len_a={len_a}, len_b={len_b}\n"
        f"max abs diff = {(out_b_ref - out_b_patched).abs().max().item()}"
    )


def test_conv_state_carries_correctly_after_block_a():
    """After processing block_a + block_b in two stages, the conv_state stored after block_b
    should let a third 1-token decode produce the same output as running 3 blocks in one shot."""
    torch.manual_seed(0)
    kernel_size = 4
    dim = 8
    len_a, len_b = 5, 3

    conv = torch.nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size - 1, groups=dim, bias=False)

    full = torch.randn(1, dim, len_a + len_b + 1)  # +1 single decode token
    out_full = conv(full)[..., :len_a + len_b + 1]
    out_decode_ref = out_full[..., -1:]

    block_a = full[..., :len_a]
    block_b = full[..., len_a:len_a + len_b]
    decode_token = full[..., -1:]

    # Stage 1: block_a alone
    conv_state_a = block_a[..., -kernel_size:] if block_a.shape[-1] >= kernel_size else F.pad(
        block_a, (kernel_size - block_a.shape[-1], 0)
    )

    # Stage 2: block_b with ctx
    ctx_b = conv_state_a[..., -(kernel_size - 1):]
    full_input_b = torch.cat([ctx_b, block_b], dim=-1)
    # New conv_state after block_b
    new_conv_state_b = full_input_b[..., -kernel_size:]

    # Stage 3: decode 1 token using new_conv_state_b
    ctx_decode = new_conv_state_b[..., -(kernel_size - 1):]
    full_input_decode = torch.cat([ctx_decode, decode_token], dim=-1)
    out_decode = conv(full_input_decode)
    out_decode_patched = out_decode[..., kernel_size - 1: kernel_size]

    assert torch.allclose(out_decode_ref, out_decode_patched, atol=1e-5), (
        f"3-stage conv composition diverged. "
        f"max abs diff = {(out_decode_ref - out_decode_patched).abs().max().item()}"
    )
