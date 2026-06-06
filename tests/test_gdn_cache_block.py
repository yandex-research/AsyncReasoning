"""Tests for GDN affine cache blocks.

Run from repo root with:

    pytest tests/test_gdn_cache_block.py -q

These tests intentionally use small tensor sizes and float64 for tight numerical
checks. They validate the algebra before any Qwen/HuggingFace integration.
"""

import pytest
import torch

from shared_cache.gdn_cache_block import (
    GDNCacheBlock,
    apply_gdn_affine,
    compose_gdn_affines,
    gdn_recurrent_update,
    init_gdn_affine,
    update_affine_summary,
    )



def _make_token_params(
    *,
    batch_size: int = 2,
    num_heads: int = 3,
    d_k: int = 5,
    d_v: int = 5,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
):
    """Generate numerically stable toy GDN token parameters."""

    k = torch.randn(batch_size, num_heads, d_k, dtype=dtype, device=device) * 0.2
    v = torch.randn(batch_size, num_heads, d_v, dtype=dtype, device=device) * 0.2

    # Keep gates in a stable range. These are not meant to exactly mirror a
    # production GDN parameterization, just valid affine-update scalars.
    alpha = torch.rand(batch_size, num_heads, dtype=dtype, device=device) * 0.3 + 0.65
    beta = torch.rand(batch_size, num_heads, dtype=dtype, device=device) * 0.3 + 0.05

    return k, v, alpha, beta


def test_init_affine_is_identity():
    batch_size, num_heads, d_k, d_v = 2, 3, 5, 7
    dtype = torch.float64

    A_hat, B_hat = init_gdn_affine(
        batch_size=batch_size,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        dtype=dtype,
        device="cpu",
    )

    state = torch.randn(batch_size, num_heads, d_v, d_k, dtype=dtype)
    out = apply_gdn_affine(state, A_hat, B_hat)

    assert A_hat.shape == (batch_size, num_heads, d_k, d_k)
    assert B_hat.shape == (batch_size, num_heads, d_v, d_k)
    torch.testing.assert_close(out, state)


def test_one_token_affine_matches_direct_recurrent_update():
    batch_size, num_heads, d_k, d_v = 2, 3, 5, 7
    dtype = torch.float64

    state0 = torch.randn(batch_size, num_heads, d_v, d_k, dtype=dtype) * 0.2
    A_hat, B_hat = init_gdn_affine(
        batch_size=batch_size,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        dtype=dtype,
        device="cpu",
    )

    k, v, alpha, beta = _make_token_params(
        batch_size=batch_size,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        dtype=dtype,
    )

    state_direct = gdn_recurrent_update(
        state=state0,
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
    )

    A_hat, B_hat = update_affine_summary(
        A_hat=A_hat,
        B_hat=B_hat,
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
    )
    state_affine = apply_gdn_affine(state0, A_hat, B_hat)

    torch.testing.assert_close(state_affine, state_direct, rtol=1e-10, atol=1e-10)


def test_many_token_affine_matches_direct_recurrent_loop():
    batch_size, num_heads, d_k, d_v = 2, 3, 5, 7
    seq_len = 11
    dtype = torch.float64

    state0 = torch.randn(batch_size, num_heads, d_v, d_k, dtype=dtype) * 0.2
    state_direct = state0.clone()

    A_hat, B_hat = init_gdn_affine(
        batch_size=batch_size,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        dtype=dtype,
        device="cpu",
    )

    for _ in range(seq_len):
        k, v, alpha, beta = _make_token_params(
            batch_size=batch_size,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            dtype=dtype,
        )

        state_direct = gdn_recurrent_update(
            state=state_direct,
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
        )

        A_hat, B_hat = update_affine_summary(
            A_hat=A_hat,
            B_hat=B_hat,
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
        )

    state_affine = apply_gdn_affine(state0, A_hat, B_hat)
    torch.testing.assert_close(state_affine, state_direct, rtol=1e-10, atol=1e-10)


def test_cache_block_append_updates_state_and_affine_consistently():
    batch_size, num_heads, d_k, d_v = 2, 3, 5, 7
    seq_len = 8
    dtype = torch.float64

    state0 = torch.randn(batch_size, num_heads, d_v, d_k, dtype=dtype) * 0.2

    block = GDNCacheBlock.from_state(
        state0.clone(),
        with_affine=True,
        start_pos=10,
        end_pos=10,
    )

    state_direct = state0.clone()

    for _ in range(seq_len):
        k, v, alpha, beta = _make_token_params(
            batch_size=batch_size,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            dtype=dtype,
        )

        state_direct = gdn_recurrent_update(
            state=state_direct,
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
        )

        block.append_token_update(k=k, v=v, alpha=alpha, beta=beta)

    assert block.start_pos == 10
    assert block.end_pos == 10 + seq_len
    assert block.length == seq_len

    torch.testing.assert_close(block.recurrent_state, state_direct, rtol=1e-10, atol=1e-10)

    state_from_affine = block.apply_affine(state0)
    torch.testing.assert_close(state_from_affine, state_direct, rtol=1e-10, atol=1e-10)


def test_cache_block_affine_can_be_applied_to_different_input_state():
    batch_size, num_heads, d_k, d_v = 2, 3, 5, 7
    seq_len = 6
    dtype = torch.float64

    state_a = torch.randn(batch_size, num_heads, d_v, d_k, dtype=dtype) * 0.2
    state_b = torch.randn(batch_size, num_heads, d_v, d_k, dtype=dtype) * 0.2

    block = GDNCacheBlock.empty(
        batch_size=batch_size,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        dtype=dtype,
        device="cpu",
    )

    params = []
    for _ in range(seq_len):
        token_params = _make_token_params(
            batch_size=batch_size,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            dtype=dtype,
        )
        params.append(token_params)
        k, v, alpha, beta = token_params
        block.append_token_update(k=k, v=v, alpha=alpha, beta=beta)

    direct_a = state_a.clone()
    direct_b = state_b.clone()
    for k, v, alpha, beta in params:
        direct_a = gdn_recurrent_update(state=direct_a, k=k, v=v, alpha=alpha, beta=beta)
        direct_b = gdn_recurrent_update(state=direct_b, k=k, v=v, alpha=alpha, beta=beta)

    affine_a = block.apply_affine(state_a)
    affine_b = block.apply_affine(state_b)

    torch.testing.assert_close(affine_a, direct_a, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(affine_b, direct_b, rtol=1e-10, atol=1e-10)


def test_compose_affines_matches_sequential_application():
    batch_size, num_heads, d_k, d_v = 2, 3, 5, 7
    first_len = 4
    second_len = 5
    dtype = torch.float64

    state0 = torch.randn(batch_size, num_heads, d_v, d_k, dtype=dtype) * 0.2

    A_first, B_first = init_gdn_affine(
        batch_size=batch_size,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        dtype=dtype,
        device="cpu",
    )
    A_second, B_second = init_gdn_affine(
        batch_size=batch_size,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        dtype=dtype,
        device="cpu",
    )

    state_direct = state0.clone()

    for _ in range(first_len):
        k, v, alpha, beta = _make_token_params(
            batch_size=batch_size,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            dtype=dtype,
        )
        state_direct = gdn_recurrent_update(state=state_direct, k=k, v=v, alpha=alpha, beta=beta)
        A_first, B_first = update_affine_summary(
            A_hat=A_first,
            B_hat=B_first,
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
        )

    state_after_first = state_direct.clone()

    for _ in range(second_len):
        k, v, alpha, beta = _make_token_params(
            batch_size=batch_size,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            dtype=dtype,
        )
        state_direct = gdn_recurrent_update(state=state_direct, k=k, v=v, alpha=alpha, beta=beta)
        A_second, B_second = update_affine_summary(
            A_hat=A_second,
            B_hat=B_second,
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
        )

    A_composed, B_composed = compose_gdn_affines(
        A_first=A_first,
        B_first=B_first,
        A_second=A_second,
        B_second=B_second,
    )

    sequential_affine = apply_gdn_affine(
        apply_gdn_affine(state0, A_first, B_first),
        A_second,
        B_second,
    )
    composed_affine = apply_gdn_affine(state0, A_composed, B_composed)
    second_only = apply_gdn_affine(state_after_first, A_second, B_second)

    torch.testing.assert_close(sequential_affine, state_direct, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(composed_affine, state_direct, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(second_only, state_direct, rtol=1e-10, atol=1e-10)


def test_append_without_affine_initializes_affine():
    batch_size, num_heads, d_k, d_v = 2, 3, 5, 7
    dtype = torch.float64

    state0 = torch.randn(batch_size, num_heads, d_v, d_k, dtype=dtype) * 0.2
    block = GDNCacheBlock.from_state(state0.clone(), with_affine=False)

    assert not block.has_affine

    k, v, alpha, beta = _make_token_params(
        batch_size=batch_size,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        dtype=dtype,
    )
    block.append_token_update(k=k, v=v, alpha=alpha, beta=beta)

    assert block.has_affine
    expected = gdn_recurrent_update(state=state0, k=k, v=v, alpha=alpha, beta=beta)
    torch.testing.assert_close(block.recurrent_state, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(block.apply_affine(state0), expected, rtol=1e-10, atol=1e-10)


def test_invalid_shapes_raise_helpful_errors():
    block = GDNCacheBlock.empty(
        batch_size=2,
        num_heads=3,
        d_k=5,
        d_v=7,
        dtype=torch.float64,
        device="cpu",
    )

    good_k, good_v, alpha, beta = _make_token_params(
        batch_size=2,
        num_heads=3,
        d_k=5,
        d_v=7,
        dtype=torch.float64,
    )

    bad_k = torch.randn(2, 3, 6, dtype=torch.float64)
    with pytest.raises(ValueError, match="k must have shape"):
        block.append_token_update(k=bad_k, v=good_v, alpha=alpha, beta=beta)

    bad_v = torch.randn(2, 3, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="v must have shape"):
        block.append_token_update(k=good_k, v=bad_v, alpha=alpha, beta=beta)

    bad_state = torch.randn(2, 3, 8, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="state shape must match B_hat shape"):
        block.apply_affine(bad_state)
