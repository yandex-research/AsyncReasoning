"""Bit-exact equivalence of AR multi-block chains vs vanilla `model.generate` on Qwen3.5-MoE.

MoE counterpart of `test_qwen35_ar_chain_vs_vanilla.py`. Splitting a prefix into 1/2/3 AR
cache blocks, prefilling them sequentially through `SharedCacheManager`, and decoding from
the full chain must produce the same tokens as `model.generate` on the concatenated prefix.

This is the strongest functional check on the GDN affine compose + conv-state composition:
any drift in the recurrent or conv state would show up as a token divergence within a few
decode steps.

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


# Same split shape as the dense test, for an apples-to-apples comparison.
_SPLITS = [
    pytest.param(["The capital of France is"], id="1-block"),
    pytest.param(["The capital of", " France is"], id="2-block"),
    pytest.param(["The capital of", " France", " is"], id="3-block"),
]


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


@pytest.mark.parametrize("chunks", _SPLITS)
def test_moe_ar_chain_matches_vanilla_generate(moe_model_and_tokenizer, chunks):
    import shared_cache

    model, tokenizer = moe_model_and_tokenizer
    chunk_ids = [
        tokenizer(c, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        for c in chunks
    ]
    ids_full = torch.cat(chunk_ids, dim=-1)

    # Vanilla reference: greedy decode from the concatenated prefix in one shot.
    with torch.inference_mode():
        out_vanilla = model.generate(
            ids_full, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )

    # AR: prefill chunks sequentially into separate blocks; chain length grows by one block
    # per chunk. After the last prefill, use its final-position logits for the first
    # generated token; then decode the rest from the full chain.
    blocks = [shared_cache.CacheBlock(config=model.config) for _ in chunks]
    generated = ids_full.clone()
    with torch.inference_mode():
        for i, ids in enumerate(chunk_ids):
            chain = blocks[: i + 1]
            cm = shared_cache.SharedCacheManager(cache_structure=[chain], write_to=[blocks[i]])
            out = model(**cm.get_input_kwargs(ids))
        next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        cm_full = shared_cache.SharedCacheManager(
            cache_structure=[blocks], write_to=[blocks[-1]],
        )
        for _ in range(19):
            out = model(**cm_full.get_input_kwargs(next_token))
            next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

    assert torch.equal(out_vanilla, generated), (
        f"AR with {len(chunks)} MoE block(s) diverged from vanilla generate.\n"
        f"vanilla: {tokenizer.decode(out_vanilla[0], skip_special_tokens=True)!r}\n"
        f"AR    : {tokenizer.decode(generated[0], skip_special_tokens=True)!r}"
    )
