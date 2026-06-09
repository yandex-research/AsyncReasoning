"""Math correctness on the MoE GDN: AR's captured affine summary (A_hat, B_hat) must
reproduce the actual `chunk_gated_delta_rule` last_recurrent_state for each GDN layer.

This is the MoE counterpart of `test_qwen35_ar_affine_vs_kernel.py`. The MoE GDN class
(`Qwen3_5MoeGatedDeltaNet`) is structurally identical to the dense one, so if the math is
right on dense it should be right on MoE — but the test exists to catch a future divergence
in the kernel call or attribute layout.

Skipped if no Qwen3.5-MoE model is available locally.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

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
def test_moe_affine_capture_matches_kernel_state():
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import shared_cache
    from qwen35_gdn.qwen35_ar_patch import patch_qwen35_for_async_reasoning

    model_id = _find_local_moe()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    patch = patch_qwen35_for_async_reasoning(model)

    text = "The quick brown fox jumps over the lazy dog. " * 3
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    # Run the AR-patched forward into a fresh cache block; this populates each GDN layer's
    # linear_affine pair via `capture_token_affines`.
    block = shared_cache.CacheBlock(config=model.config)
    cm = shared_cache.SharedCacheManager(cache_structure=[[block]], write_to=[block])
    with torch.inference_mode():
        model(**cm.get_input_kwargs(ids))

    # Reference path: re-run a fresh prefill while wrapping each GDN layer's
    # `chunk_gated_delta_rule` to capture its actual `last_recurrent_state`. The captured
    # state is the kernel's own bf16 output; AR's affine math should reproduce it.
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet

    captured_kernel_states: dict[int, torch.Tensor] = {}
    originals: dict[int, callable] = {}
    for mod in model.modules():
        if isinstance(mod, Qwen3_5MoeGatedDeltaNet):
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

    # Restore hooks before any assertion can fail.
    for mod in model.modules():
        if isinstance(mod, Qwen3_5MoeGatedDeltaNet):
            mod.chunk_gated_delta_rule = originals[mod.layer_idx]

    # Compare per-layer: AR's composed initial state from (A_hat, B_hat) starting from S=0
    # is just B_hat in block convention [B, H, d_v, d_k]; kernel returns HF convention
    # [B, H, d_k, d_v], so we transpose.
    layer_types = model.config.layer_types
    gdn_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    assert gdn_layers, "no GDN layers found in MoE model — wrong variant?"

    max_rel_err = 0.0
    bad: list[tuple[int, float]] = []
    for layer_idx in gdn_layers:
        pair = block2.linear_affine.get(layer_idx)
        kernel_state_hf = captured_kernel_states.get(layer_idx)
        assert pair is not None, f"layer {layer_idx}: AR captured no affine"
        assert kernel_state_hf is not None, f"layer {layer_idx}: kernel never produced a state"
        _, B_hat = pair
        S_affine_block = B_hat.float()
        S_kernel_block = kernel_state_hf.transpose(-1, -2).float()
        err = (S_affine_block - S_kernel_block).norm() / S_kernel_block.norm().clamp_min(1e-8)
        max_rel_err = max(max_rel_err, err.item())
        if err.item() > 5e-2:
            bad.append((layer_idx, err.item()))

    patch.unpatch(model)
    print(f"MoE affine vs kernel max_rel_err = {max_rel_err:.4e}")
    assert not bad, (
        f"AR-captured affine diverges from kernel state at MoE layers {bad}; this "
        f"breaks every downstream guarantee (multi-block compose, writer-fork view, etc)."
    )
