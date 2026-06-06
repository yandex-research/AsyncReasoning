"""Regression tests for `check_if_should_continue_writing`.

History: before fix, the mode-switcher always returned True because:
  (a) `tokenizer("yes")` returned the no-leading-space token id, but the model emits
      ` yes` (with leading space). Comparing zero-probability tokens, the boolean `>`
      randomly favoured yes.
  (b) `CacheBlock.clear()` didn't clear the GDN linear-attn state, so mode_switching_question
      accumulated stale state across calls.

After fix: solver._yes_no_token_ids collects all single-token yes/no variants and
logsumexp's over each side. CacheBlock.clear() now resets all state including linear-attn.

These tests assert behaviour that depends on the fix.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _have(name: str) -> bool:
    return os.path.isdir(f"/mnt/LLM/hub/models--Qwen--{name}")


@pytest.fixture(scope="module")
def qwen3_8b():
    """Qwen3-8B is large enough for the mode-switcher to give meaningful binary answers.
    On smaller models (0.6B, 4B) the prior is so yes-leaning we can't reliably elicit no."""
    if not _have("Qwen3-8B"):
        pytest.skip("Qwen3-8B not present in local HF cache")
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", device_map="cuda", torch_dtype="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    model.eval()
    yield model, tokenizer


def _make_solver_and_cache(model, tokenizer, problem):
    from async_reasoning.solver import AsyncReasoningSolver
    from async_reasoning.cache import AsyncReasoningCache, State
    from async_reasoning.prompting import AsyncReasoningPrompting

    solver = AsyncReasoningSolver(model=model, tokenizer=tokenizer, use_fast_kernel=False)
    prompting = AsyncReasoningPrompting(problem)
    cache = AsyncReasoningCache(
        model, tokenizer, prompting,
        tokenizer_kwargs=dict(return_tensors="pt", add_special_tokens=False),
        starting_state=State.thinker_only,
    )
    return solver, cache, prompting


def test_yes_no_ids_include_leading_space_variant(qwen3_8b):
    """Sanity: leading-space variants must be in the set the solver compares.
    On Qwen3-8B the no-leading-space tokens have essentially zero probability;
    if the comparison were against them alone we'd be back to the noise-floor bug."""
    model, tokenizer = qwen3_8b
    from async_reasoning.solver import AsyncReasoningSolver
    from async_reasoning.prompting import AsyncReasoningPrompting

    solver = AsyncReasoningSolver(model=model, tokenizer=tokenizer, use_fast_kernel=False)
    prompting = AsyncReasoningPrompting("placeholder")
    yes_ids, no_ids = solver._yes_no_token_ids(prompting)

    space_yes = tokenizer(" yes", add_special_tokens=False)["input_ids"]
    space_no = tokenizer(" no", add_special_tokens=False)["input_ids"]
    assert len(space_yes) == 1 and space_yes[0] in yes_ids, "missing ' yes' variant"
    assert len(space_no) == 1 and space_no[0] in no_ids, "missing ' no' variant"


def test_cacheblock_clear_resets_gdn_state():
    """CacheBlock.clear() must reset linear_affine/linear_conv_states/linear_recurrent_states
    so that re-prefill (used by check_if_should_continue_writing) starts from a clean slate.

    Pure-Python test — no model required."""
    import shared_cache
    import transformers

    # Use a tiny config from a small known model to instantiate CacheBlock.
    cfg = transformers.AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    b = shared_cache.CacheBlock(config=cfg)
    # Manually drop state into all three buckets.
    b.linear_affine[0] = (torch.eye(4).view(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    b.linear_conv_states[0] = torch.zeros(1, 4, 4)
    b.linear_recurrent_states[0] = torch.zeros(1, 1, 4, 4)
    assert b.linear_affine, b.linear_conv_states
    b.clear()
    assert not b.linear_affine
    assert not b.linear_conv_states
    assert not b.linear_recurrent_states
