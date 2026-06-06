"""Validate AR affine captures vs the actual `chunk_gated_delta_rule` output for a single block.

If our captured (A_hat, B_hat) reproduce the model's own `last_recurrent_state`, the affine
math is correct and we can trust the chain composition.
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
def test_affine_capture_matches_kernel_state():
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

    text = "The quick brown fox jumps over the lazy dog. " * 3  # ~36 tokens
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    # AR path with one block — capture affine via our patch.
    block = shared_cache.CacheBlock(config=model.config)
    cm = shared_cache.SharedCacheManager(cache_structure=[[block]], write_to=[block])
    with torch.inference_mode():
        model(**cm.get_input_kwargs(ids))

    # Reference: install a separate hook that captures the kernel's last_recurrent_state directly.
    # We re-run the same prefill on a FRESH cache and intercept chunk_gated_delta_rule.
    captured_kernel_states = {}
    originals = {}
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    for mod in model.modules():
        if isinstance(mod, Qwen3_5GatedDeltaNet):
            layer_idx = mod.layer_idx
            orig = mod.chunk_gated_delta_rule
            originals[layer_idx] = orig

            def make_hook(orig_fn, lidx):
                def hook(*args, **kwargs):
                    out = orig_fn(*args, **kwargs)
                    if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
                        captured_kernel_states[lidx] = out[1].detach().clone()
                    return out
                return hook
            mod.chunk_gated_delta_rule = make_hook(orig, layer_idx)

    block2 = shared_cache.CacheBlock(config=model.config)
    cm2 = shared_cache.SharedCacheManager(cache_structure=[[block2]], write_to=[block2])
    with torch.inference_mode():
        model(**cm2.get_input_kwargs(ids))

    # Restore hooks
    for mod in model.modules():
        if isinstance(mod, Qwen3_5GatedDeltaNet):
            mod.chunk_gated_delta_rule = originals[mod.layer_idx]

    # For each linear layer, compare affine-composed state (block2's B_hat) to kernel's last_recurrent_state.
    # Both should represent S_final after the same token sequence.
    from shared_cache.gdn_cache_block import apply_gdn_affine, init_gdn_affine

    layer_types = model.config.layer_types
    gdn_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]

    max_rel_err = 0.0
    bad = []
    for layer_idx in gdn_layers:
        pair = block2.linear_affine.get(layer_idx)
        kernel_state_hf = captured_kernel_states.get(layer_idx)
        assert pair is not None
        assert kernel_state_hf is not None
        A_hat, B_hat = pair
        # Initial state = 0, so S_affine = B_hat (block convention [B, H, d_v, d_k])
        S_affine_block = B_hat
        # Kernel returns HF convention [B, H, d_k, d_v]
        S_kernel_block = kernel_state_hf.transpose(-1, -2).float()
        S_affine_block = S_affine_block.float()
        err = (S_affine_block - S_kernel_block).norm() / S_kernel_block.norm().clamp_min(1e-8)
        max_rel_err = max(max_rel_err, err.item())
        if err.item() > 5e-2:
            bad.append((layer_idx, err.item()))

    patch.unpatch(model)
    print(f"affine vs kernel max_rel_err = {max_rel_err:.4e}")
    assert not bad, f"affine diverges from kernel state at layers {bad}"
