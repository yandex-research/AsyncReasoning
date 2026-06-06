"""End-to-end smoke test for AsyncReasoning on Qwen3.5-4B (hybrid GDN + full-attention).

Catches regressions in the GDN affine compose path, the conv-state composition, and the
patched `Qwen3_5GatedDeltaNet.forward`. We don't pin a specific token sequence (greedy output
shifts with tiny bf16 noise), only that AR produces a non-trivial multi-token answer.

Skipped if Qwen3.5-4B isn't available locally.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _have_qwen35_4b() -> bool:
    return os.path.isdir("/mnt/LLM/hub/models--Qwen--Qwen3.5-4B")


@pytest.mark.skipif(not _have_qwen35_4b(), reason="Qwen3.5-4B not present in local HF cache")
def test_ar_runs_on_qwen35_4b():
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from async_reasoning.solver import AsyncReasoningSolver
    from qwen35_gdn.qwen35_ar_patch import patch_qwen35_for_async_reasoning

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    model.eval()
    patch = patch_qwen35_for_async_reasoning(model)

    try:
        solver = AsyncReasoningSolver(model=model, tokenizer=tokenizer, use_fast_kernel=False)
        writer, thinker, times, eos = solver.solve(
            problem="Explain in 2-3 sentences why the sky is blue.",
            budget=80,
        )
    finally:
        patch.unpatch(model)

    # AR should emit a non-trivial writer output. The fix for the writer-fork limitation means
    # the writer should produce more than a one-sentence dump; this loosely checks that we
    # haven't regressed back to "The sky is blue.<|im_end|>"-style truncated output.
    assert len(times) > 5, f"AR produced too few writer tokens ({len(times)}) — likely a regression"
    # Strip the system prefix and verify there's real content past it.
    writer_content = writer.split("</think>", 1)[-1].strip()
    assert len(writer_content) > 20, f"writer content too short: {writer_content!r}"
