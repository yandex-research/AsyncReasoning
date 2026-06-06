"""End-to-end test that the AR affine cache composition is numerically equivalent to
single-shot prefill.

The math claim:
    For any split of a token sequence into prefix_a + prefix_b + ...,
    composing per-block GDN affines (A, B) and applying to S=0 must yield the same
    recurrent state as running the whole sequence through the model in one shot.

If this passes, AR's writer-fork limitation goes away: the writer worker reads the
chain-composed state of `[input_prompt, thinker_output, writer_output]` where
`thinker_output` carries the affine from all tokens the thinker has generated since
prefill — exactly what we want.

Skipped if Qwen3.5-4B isn't available locally.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _have_qwen35() -> bool:
    return os.path.isdir("/mnt/LLM/hub/models--Qwen--Qwen3.5-4B")


@pytest.mark.skipif(not _have_qwen35(), reason="Qwen3.5-4B not present in local HF cache")
def test_two_block_prefill_matches_single_shot():
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import shared_cache
    from qwen35_gdn.qwen35_ar_patch import patch_qwen35_for_async_reasoning

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    model.eval()
    patch = patch_qwen35_for_async_reasoning(model)

    text_a = "The capital of France is"
    text_b = " Paris, and it is famous for"
    ids_a = tokenizer(text_a, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    ids_b = tokenizer(text_b, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    ids_full = torch.cat([ids_a, ids_b], dim=-1)

    # --- single-shot path: prefill ids_full into one cache block ---
    block_full = shared_cache.CacheBlock(config=model.config)
    cm_full = shared_cache.SharedCacheManager(cache_structure=[[block_full]], write_to=[block_full])
    with torch.inference_mode():
        model(**cm_full.get_input_kwargs(ids_full))

    # --- AR path: prefill ids_a into block_a, then ids_b into block_b with [a,b] structure ---
    block_a = shared_cache.CacheBlock(config=model.config)
    block_b = shared_cache.CacheBlock(config=model.config)
    cm_a = shared_cache.SharedCacheManager(cache_structure=[[block_a]], write_to=[block_a])
    cm_ab = shared_cache.SharedCacheManager(
        cache_structure=[[block_a, block_b]], write_to=[block_b]
    )
    with torch.inference_mode():
        model(**cm_a.get_input_kwargs(ids_a))
        model(**cm_ab.get_input_kwargs(ids_b))

    # --- compare composed final state ---
    from shared_cache.gdn_cache_block import init_gdn_affine, compose_gdn_affines
    text_config = model.config
    layer_types = text_config.layer_types
    gdn_layer_indices = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    assert gdn_layer_indices, "model has no linear-attention layers"

    max_rel_err = 0.0
    bad_layers = []
    for layer_idx in gdn_layer_indices:
        # Single-shot stored an affine via block_full
        pair_full = block_full.linear_affine.get(layer_idx)
        assert pair_full is not None, f"single-shot didn't capture affine for layer {layer_idx}"
        A_full, B_full = pair_full
        S_full = B_full  # 0 @ A + B = B in block convention

        # AR path: compose block_a's affine with block_b's affine
        pair_a = block_a.linear_affine.get(layer_idx)
        pair_b = block_b.linear_affine.get(layer_idx)
        assert pair_a is not None, f"block_a missing affine for layer {layer_idx}"
        assert pair_b is not None, f"block_b missing affine for layer {layer_idx}"
        A_a, B_a = pair_a
        A_b, B_b = pair_b
        A_chain, B_chain = compose_gdn_affines(
            A_first=A_a, B_first=B_a,
            A_second=A_b, B_second=B_b,
        )
        S_ar = B_chain

        # Relative L2 error
        denom = S_full.float().norm().clamp_min(1e-8)
        err = (S_ar - S_full).float().norm() / denom
        max_rel_err = max(max_rel_err, err.item())
        if err.item() > 5e-2:
            bad_layers.append((layer_idx, err.item()))

    patch.unpatch(model)
    assert not bad_layers, (
        f"AR two-block prefill diverges from single-shot at layers {bad_layers}. "
        f"max_rel_err={max_rel_err:.4e}"
    )
    print(f"AR two-block prefill replays single-shot to max_rel_err={max_rel_err:.4e}")
