"""Structural check: the AR patch must NOT replace the MoE sparse-MLP block.

`patch_qwen35_for_async_reasoning` walks anything whose class name ends in `GatedDeltaNet`.
This is a structural guarantee against future refactors: the MoE block lives in the
`layer.mlp` slot, has no `GatedDeltaNet` suffix, and must remain untouched. If a future
class rename accidentally caused the suffix matcher to grab a MoE module instead, expert
routing would silently break.

Also verifies the layer-type accounting: every `linear_attention` layer has its GDN patched,
every `full_attention` layer's self_attn is the stock class, and every layer's `mlp` is
the stock MoE sparse block.

Skipped if no Qwen3.5-MoE model is available locally.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MOE_CANDIDATES = [
    "Qwen3.5-35B-A3B",
    "Qwen3.5-MoE-A3B",
    "Qwen3-Next-80B-A3B-Instruct",
    "Qwen3-Next-80B-A3B",
]


def _find_local_moe() -> str | None:
    for name in _MOE_CANDIDATES:
        if os.path.isdir(f"/mnt/LLM/hub/models--Qwen--{name}"):
            return f"Qwen/{name}"
    return None


@pytest.mark.skipif(_find_local_moe() is None, reason="no Qwen3.5-MoE model in local HF cache")
def test_moe_sparse_block_is_not_patched():
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeAttention,
        Qwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeSparseMoeBlock,
    )
    from qwen35_gdn.qwen35_ar_patch import patch_qwen35_for_async_reasoning

    model_id = _find_local_moe()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()

    # Snapshot mlp class per layer BEFORE patch.
    mlp_classes_before = [type(layer.mlp).__name__ for layer in model.model.layers]
    # All MoE-decoder-layer mlps should be the sparse MoE block.
    assert all(c == "Qwen3_5MoeSparseMoeBlock" for c in mlp_classes_before), (
        f"unexpected mlp classes before patch: {set(mlp_classes_before)}"
    )

    patch = patch_qwen35_for_async_reasoning(model)
    try:
        # Per-layer post-patch invariants:
        layer_types = model.config.layer_types
        for i, layer in enumerate(model.model.layers):
            layer_type = layer_types[i]
            mlp = layer.mlp

            # Invariant 1: the MoE sparse block is untouched.
            assert isinstance(mlp, Qwen3_5MoeSparseMoeBlock), (
                f"layer {i}: mlp is now {type(mlp).__name__}, expected "
                f"Qwen3_5MoeSparseMoeBlock — the patch wrongly replaced the MoE block."
            )

            # Invariant 2: GDN layers' `linear_attn` is still the stock MoE GDN class
            # (the patch swaps the *forward method* on the instance, not the class).
            if layer_type == "linear_attention":
                assert hasattr(layer, "linear_attn"), f"layer {i}: missing linear_attn"
                assert isinstance(layer.linear_attn, Qwen3_5MoeGatedDeltaNet), (
                    f"layer {i}: linear_attn was replaced "
                    f"({type(layer.linear_attn).__name__})"
                )
                # The forward should now be our patched method, distinguishable by being
                # a bound method on a function (types.MethodType wraps `_patched_forward`).
                assert layer.linear_attn.forward.__func__.__name__ == "_patched_forward", (
                    f"layer {i}: GDN forward was NOT patched"
                )

            # Invariant 3: full-attention layers' self_attn remains the stock class.
            elif layer_type == "full_attention":
                assert hasattr(layer, "self_attn"), f"layer {i}: missing self_attn"
                assert isinstance(layer.self_attn, Qwen3_5MoeAttention), (
                    f"layer {i}: self_attn was replaced "
                    f"({type(layer.self_attn).__name__})"
                )
    finally:
        patch.unpatch(model)

    # After unpatch the patch dict must be empty.
    assert len(patch.originals) == 0, "unpatch left state behind"
