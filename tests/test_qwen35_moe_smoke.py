"""End-to-end smoke test for AsyncReasoning on Qwen3.5-MoE.

The MoE variant adds a sparse expert routing layer in place of the dense MLP. AR doesn't
touch the MLP at all, so functionally the only thing it needs from MoE is the GDN forward
to be patchable. This test exercises that full path: load model, patch, run the AR solver,
assert it produces non-trivial output.

We don't pin a specific token sequence — greedy output shifts with tiny bf16 noise and
varies further with expert routing. We only assert AR produces a multi-token writer answer
without crashing.

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
def test_ar_runs_on_qwen35_moe():
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from async_reasoning.solver import AsyncReasoningSolver
    from qwen35_gdn.qwen35_ar_patch import patch_qwen35_for_async_reasoning

    model_id = _find_local_moe()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()
    patch_qwen35_for_async_reasoning(model)

    system_tokens = [k for k in tokenizer.vocab.keys() if k.endswith("SYSTEM") or k.endswith("SYSTEM:")]
    writer_forbidden = [tokenizer.vocab[x] for x in ["</think>", "<|im_start|>", "<|endoftext|>"] + system_tokens]
    thinker_forbidden = [tokenizer.vocab[x] for x in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] + system_tokens]
    end_of_think = [tokenizer.vocab["</think>"]]

    solver = AsyncReasoningSolver(
        model,
        tokenizer,
        writer_forbidden_token_ix=writer_forbidden,
        thinker_forbidden_token_ix=thinker_forbidden,
        end_of_think_token_ix=end_of_think,
        use_fast_kernel=False,  # forced off for hybrid models (GDN doesn't support it)
    )

    problem = (
        "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        "How many positive integers less than 100 are divisible by 7?"
    )
    writer, thinker, token_times, eos = solver.solve(problem, budget=512)

    # Sanity: AR produced something non-trivial in both phases without crashing.
    assert len(thinker) > 0, "thinker produced no tokens"
    assert len(writer) > 0, "writer produced no tokens"
    real_writer_steps = [t for t in token_times if t[0] != "EMPTY"]
    assert len(real_writer_steps) >= 5, (
        f"writer emitted only {len(real_writer_steps)} real tokens; "
        f"either AR stalled on mode-switching or the MoE patch path crashed silently."
    )
    print(f"\nMoE smoke OK: thinker={len(thinker)} chars, writer={len(writer)} chars, "
          f"writer_tokens={len(real_writer_steps)}, eos={bool(eos)}")
