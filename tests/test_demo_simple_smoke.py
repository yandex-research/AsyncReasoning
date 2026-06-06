"""Smoke test for demo_simple.ipynb's flow on the fast-kernel path.

Runs the same code path as the notebook (use_fast_kernel=True) on Qwen3-8B (closest small
analogue to the demo's Qwen3-32B). Catches regressions in the inference_lib fast cache's
compatibility with transformers >=4.55 — the things this test would have caught:

- `AsyncReasoningCache.get_mask_sizes` missing (transformers >=4.55 requires it).
- `past_key_value` vs `past_key_values` kwarg rename in attention layers.
- `cache_kwargs['mask']` becoming 4D instead of 2D.
- `DynamicCache.key_cache` -> `self.layers[i].keys` rename.

Skipped if Qwen3-8B isn't available locally OR if compiled CUDA kernels aren't installed.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _have_qwen3_8b() -> bool:
    return os.path.isdir("/mnt/LLM/hub/models--Qwen--Qwen3-8B")


def _have_inference_lib() -> bool:
    try:
        import async_reasoning_inference  # noqa
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_qwen3_8b(), reason="Qwen3-8B not in local HF cache")
@pytest.mark.skipif(not _have_inference_lib(), reason="async_reasoning_inference not installed")
def test_demo_simple_flow_works_fast_kernel():
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    import transformers
    from async_reasoning.solver import AsyncReasoningSolver as Solver
    from utils.answer_processing import find_last_valid_expression

    MODEL = "Qwen/Qwen3-8B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype="auto", low_cpu_mem_usage=True, device_map="cuda",
    )
    sys_tok = [k for k in tokenizer.vocab if k.endswith("SYSTEM") or k.endswith("SYSTEM:")]
    wft = [tokenizer.vocab[x] for x in ["</think>", "<|im_start|>", "<|endoftext|>"] + sys_tok]
    tft = [tokenizer.vocab[x] for x in ["</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"] + sys_tok]

    solver = Solver(
        model, tokenizer,
        writer_forbidden_token_ix=wft,
        thinker_forbidden_token_ix=tft,
        use_fast_kernel=True,
    )
    writer_out, thinker_out, times, eos = solver.solve(
        "Calculate x - x^2 + x^3 for x = 5,6,7,8. Return all 4 answers in \\boxed{ }.",
        budget=1024,
    )

    # Two structural assertions, no answer-correctness check (greedy + bf16 drift is fine):
    #   1. writer actually emitted real tokens (token_times has entries that aren't the EMPTY sentinel)
    real_emissions = [t for t in times if t[0] != "EMPTY"]
    assert len(real_emissions) > 5, (
        f"writer never emitted real tokens — mode_switcher likely returning no every check. "
        f"Got times={times[:5]}"
    )
    #   2. answer extraction finds something rather than None
    extracted = find_last_valid_expression(writer_out, extract_result=lambda x: x[7:-1])
    assert extracted is not None, f"no \\boxed{{...}} in writer output: {writer_out[-300:]!r}"
