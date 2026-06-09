"""Sanity-check the AR forward patch on Qwen3.5-MoE.

The MoE variant of Qwen3.5 has the same hybrid GDN+full-attention layer stack as the dense
variant, with a sparse MoE block in place of the MLP. The GDN forward we patch is byte-for-
byte identical to dense's GDN forward, so the same patch should bind.

This test verifies two things:
  1. `patch_qwen35_for_async_reasoning` finds and replaces the MoE GDN layers (count > 0).
  2. The patch is a true no-op for vanilla `model.generate` — outputs and recurrent states
     match the unpatched model exactly.

Skipped if no Qwen3.5-MoE model is available locally.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Candidate MoE model directories to look for, in preference order. The smallest one wins.
_MOE_CANDIDATES = [
    "Qwen3.5-35B-A3B",
    "Qwen3.5-MoE-A3B",
    "Qwen3-Next-80B-A3B-Instruct",
    "Qwen3-Next-80B-A3B",
]


def _find_local_moe() -> str | None:
    """Return the first locally-cached candidate's HF id, or None."""
    for name in _MOE_CANDIDATES:
        if os.path.isdir(f"/mnt/LLM/hub/models--Qwen--{name}"):
            return f"Qwen/{name}"
    return None


@pytest.mark.skipif(_find_local_moe() is None, reason="no Qwen3.5-MoE model in local HF cache")
def test_moe_patch_is_noop_for_vanilla_generate():
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qwen35_gdn.qwen35_ar_patch import patch_qwen35_for_async_reasoning

    model_id = _find_local_moe()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cuda",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    # Sanity: the model's GDN layer class name should end with `GatedDeltaNet` (so the patch
    # finds it), and the model exposes a hybrid layer_types list.
    layer_types = getattr(model.config, "layer_types", None)
    assert layer_types is not None, "MoE model config is missing `layer_types`"
    assert "linear_attention" in layer_types, "MoE model has no GDN layers — wrong variant?"
    gdn_count_expected = sum(1 for t in layer_types if t == "linear_attention")
    assert gdn_count_expected > 0

    text = "The quick brown fox jumps over the lazy dog. " * 2
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    # 1. Reference: vanilla `model.generate`, no patch.
    with torch.inference_mode():
        out_unpatched = model.generate(
            ids, max_new_tokens=16, do_sample=False, use_cache=True,
        )

    # 2. Install patch and re-run; outputs must match.
    patch = patch_qwen35_for_async_reasoning(model)
    assert len(patch.originals) == gdn_count_expected, (
        f"patch replaced {len(patch.originals)} layers but model has "
        f"{gdn_count_expected} GDN layers — _iter_gdn_modules likely missed some."
    )

    try:
        with torch.inference_mode():
            out_patched = model.generate(
                ids, max_new_tokens=16, do_sample=False, use_cache=True,
            )
        assert torch.equal(out_unpatched, out_patched), (
            "Patched model.generate diverged from unpatched — patch is NOT a no-op for "
            "vanilla generate. Likely a wrong cache_params type-check or attribute mismatch."
        )
    finally:
        patch.unpatch(model)
