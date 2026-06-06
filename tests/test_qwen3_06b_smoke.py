"""End-to-end smoke test for AsyncReasoning on Qwen3-0.6B (standard full-attention).

This is the regression test for the transformers Cache API rewrite (>=4.55).
It catches breakage in `shared_cache/cache_block.py` and `shared_cache/combined_cache.py`
even when the GDN-specific patch is uninvolved.

Skipped if Qwen3-0.6B isn't available locally.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _have_qwen3_06b() -> bool:
    return os.path.isdir("/mnt/LLM/hub/models--Qwen--Qwen3-0.6B")


@pytest.mark.skipif(not _have_qwen3_06b(), reason="Qwen3-0.6B not present in local HF cache")
def test_ar_runs_on_qwen3_06b():
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from async_reasoning.solver import AsyncReasoningSolver

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    model.eval()

    solver = AsyncReasoningSolver(model=model, tokenizer=tokenizer, use_fast_kernel=False)
    writer, thinker, times, eos = solver.solve(
        problem="What is attention? Answer briefly in one paragraph.",
        budget=64,
    )

    # We don't assert on the text content (model output varies), only that AR produced something
    # and didn't crash.
    assert len(times) > 0, "AR didn't emit any writer tokens"
    assert isinstance(writer, str) and len(writer) > 0
    assert isinstance(thinker, str) and len(thinker) > 0
