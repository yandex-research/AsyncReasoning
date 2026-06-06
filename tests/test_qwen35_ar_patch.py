"""Sanity-check the Qwen3.5 AR forward patch.

The patch must be a no-op for vanilla generate() (model.generate uses Qwen3_5DynamicCache,
which is the cache the model was designed for): outputs and per-step recurrent states must
match the unpatched model exactly.

Skipped if Qwen3.5-4B isn't available locally.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"


def _have_qwen35() -> bool:
    return os.path.isdir("/mnt/LLM/hub/models--Qwen--Qwen3.5-4B")


@pytest.mark.skipif(not _have_qwen35(), reason="Qwen3.5-4B not present in local HF cache")
def test_patch_is_noop_for_vanilla_generate():
    from qwen35_gdn.qwen35_ar_patch import patch_qwen35_for_async_reasoning

    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model.eval()

    prompt = "Hello, world."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        baseline = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    patch = patch_qwen35_for_async_reasoning(model)
    try:
        with torch.inference_mode():
            patched = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    finally:
        patch.unpatch(model)

    assert torch.equal(baseline, patched), (
        f"Patched generate diverged from baseline.\n"
        f"Baseline:\n{tokenizer.decode(baseline[0], skip_special_tokens=True)}\n"
        f"Patched:\n{tokenizer.decode(patched[0], skip_special_tokens=True)}"
    )
