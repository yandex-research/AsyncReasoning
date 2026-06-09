"""Multi-worker batched forward must produce, per-worker, the same logits as that
worker running alone.

This is the MoE-specific correctness invariant the dense tests don't isolate. When AR runs
`thinker_and_writer` mode, batch=2 enters every layer simultaneously. Standard top-k
softmax routers are per-token, so the same token's expert assignment must be identical
whether it shows up alone or batched with another worker. If a future MoE variant
introduces a batch-aware router (expert-choice, balanced top-k, etc.), this test catches it.

It also confirms block masking under batching: worker 0 sees chain=[input], worker 1 sees
chain=[input, fork]; worker 0 must not attend to `fork` even though it's in the same
batched K tensor (zero-padded for worker 0 and masked out by `cache_attention_mask`).

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


@pytest.fixture(scope="module")
def moe_model_and_tokenizer():
    model_id = _find_local_moe()
    if model_id is None:
        pytest.skip("no Qwen3.5-MoE model in local HF cache")
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qwen35_gdn.qwen35_ar_patch import patch_qwen35_for_async_reasoning

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    patch = patch_qwen35_for_async_reasoning(model)
    yield model, tokenizer
    patch.unpatch(model)


def test_moe_multi_worker_batched_matches_per_worker(moe_model_and_tokenizer):
    """Run the writer's chain alone, the thinker's chain alone, and both together batched.
    Each worker's batched logits must match its solo-forward logits to within bf16 noise.
    """
    import shared_cache

    model, tokenizer = moe_model_and_tokenizer
    device = model.device

    # Two prefix chunks: A is shared between workers, B is writer-only context.
    text_a = "Linear attention layers maintain a running recurrent state."
    text_b = " The state evolves by an affine update parameterised by per-token gates."
    ids_a = tokenizer(text_a, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ids_b = tokenizer(text_b, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # Block A: prefill with `text_a`. Block B: prefill with `text_b` chained after A.
    block_a = shared_cache.CacheBlock(config=model.config)
    block_b = shared_cache.CacheBlock(config=model.config)
    cm_prefill_a = shared_cache.SharedCacheManager(cache_structure=[[block_a]], write_to=[block_a])
    cm_prefill_b = shared_cache.SharedCacheManager(cache_structure=[[block_a, block_b]], write_to=[block_b])
    with torch.inference_mode():
        model(**cm_prefill_a.get_input_kwargs(ids_a))
        model(**cm_prefill_b.get_input_kwargs(ids_b))

    # Probe token (same for both workers; the per-worker context differs).
    probe = tokenizer(" Therefore", return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # --- arm 1: worker 0 alone, chain = [A] ---
    cm_w0_solo = shared_cache.SharedCacheManager(
        cache_structure=[[block_a]],
        write_to=[shared_cache.CacheBlock(config=model.config)],  # throwaway write target
    )
    with torch.inference_mode():
        logits_w0_solo = model(**cm_w0_solo.get_input_kwargs(probe)).logits[..., -1, :].clone()

    # --- arm 2: worker 1 alone, chain = [A, B] ---
    cm_w1_solo = shared_cache.SharedCacheManager(
        cache_structure=[[block_a, block_b]],
        write_to=[shared_cache.CacheBlock(config=model.config)],
    )
    with torch.inference_mode():
        logits_w1_solo = model(**cm_w1_solo.get_input_kwargs(probe)).logits[..., -1, :].clone()

    # --- arm 3: both workers batched (cm_thinker_and_writer-style view) ---
    cm_batched = shared_cache.SharedCacheManager(
        cache_structure=[[block_a], [block_a, block_b]],
        write_to=[shared_cache.CacheBlock(config=model.config),
                  shared_cache.CacheBlock(config=model.config)],
    )
    probe_batched = probe.repeat(2, 1)  # [2, T] — both workers feed the same new token
    with torch.inference_mode():
        out = model(**cm_batched.get_input_kwargs(probe_batched))
        logits_batched = out.logits[..., -1, :].clone()  # [2, vocab]

    # Each worker's batched logits should match its solo logits.
    # bf16 + batched matmul accumulation differs from per-row by tiny amounts;
    # we use a relative tolerance against the magnitude of the max-class logit.
    def _relmax(a: torch.Tensor, b: torch.Tensor) -> float:
        diff = (a - b).abs().max().float().item()
        scale = a.abs().max().float().clamp_min(1e-3).item()
        return diff / scale

    err_w0 = _relmax(logits_w0_solo.squeeze(0), logits_batched[0])
    err_w1 = _relmax(logits_w1_solo.squeeze(0), logits_batched[1])

    print(f"\nmulti-worker relmax err: worker_0={err_w0:.3e}  worker_1={err_w1:.3e}")

    # Argmax (top-1 next-token) must match — looser than full logit equality but the
    # most important behavioural property.
    top1_w0_solo = int(logits_w0_solo.argmax(-1).item())
    top1_w0_batched = int(logits_batched[0].argmax(-1).item())
    top1_w1_solo = int(logits_w1_solo.argmax(-1).item())
    top1_w1_batched = int(logits_batched[1].argmax(-1).item())

    assert top1_w0_solo == top1_w0_batched, (
        f"worker 0 top-1 token differs between solo and batched mode: "
        f"solo={tokenizer.decode([top1_w0_solo])!r}, "
        f"batched={tokenizer.decode([top1_w0_batched])!r}. "
        f"Likely a block-masking failure (worker 0 attended to fork block B) "
        f"or a batch-aware MoE router."
    )
    assert top1_w1_solo == top1_w1_batched, (
        f"worker 1 top-1 token differs between solo and batched mode: "
        f"solo={tokenizer.decode([top1_w1_solo])!r}, "
        f"batched={tokenizer.decode([top1_w1_batched])!r}."
    )

    # Logit magnitudes should also be close (catches subtle accumulator drift in MoE).
    assert err_w0 < 5e-2, f"worker 0 batched/solo relmax err too large: {err_w0:.3e}"
    assert err_w1 < 5e-2, f"worker 1 batched/solo relmax err too large: {err_w1:.3e}"
