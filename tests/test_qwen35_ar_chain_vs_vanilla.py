"""Bit-exact equivalence of AR multi-block chains vs vanilla `model.generate`.

Splitting a prefix into 1/2/3 AR cache blocks, prefilling them sequentially through
`SharedCacheManager`, and decoding from the full chain must produce the same tokens
as running `model.generate` on the concatenated prefix.

This is the strongest functional check on the GDN affine compose + conv-state composition:
any drift in the recurrent or conv state would show up as a token divergence within a few
decode steps.

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


# Each parametrize entry: list of prefix-text chunks to split across cache blocks.
# n_blocks = len(chunks); chunks concatenated == the equivalent vanilla prompt.
_SPLITS = [
    pytest.param(["The capital of France is"], id="1-block"),
    pytest.param(["The capital of", " France is"], id="2-block"),
    pytest.param(["The capital of", " France", " is"], id="3-block"),
]


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if not _have_qwen35_4b():
        pytest.skip("Qwen3.5-4B not present in local HF cache")
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qwen35_gdn.qwen35_ar_patch import patch_qwen35_for_async_reasoning

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    model.eval()
    patch = patch_qwen35_for_async_reasoning(model)
    yield model, tokenizer
    patch.unpatch(model)


@pytest.mark.parametrize("chunks", _SPLITS)
def test_ar_chain_matches_vanilla_generate(model_and_tokenizer, chunks):
    import shared_cache

    model, tokenizer = model_and_tokenizer
    chunk_ids = [
        tokenizer(c, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        for c in chunks
    ]
    ids_full = torch.cat(chunk_ids, dim=-1)

    # --- vanilla path ---
    with torch.inference_mode():
        out_vanilla = model.generate(
            ids_full, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )

    # --- AR path: prefill each chunk into its own block, chain length grows with each chunk ---
    blocks = [shared_cache.CacheBlock(config=model.config) for _ in chunks]
    generated = ids_full.clone()
    with torch.inference_mode():
        for i, ids in enumerate(chunk_ids):
            chain = blocks[: i + 1]
            cm = shared_cache.SharedCacheManager(cache_structure=[chain], write_to=[blocks[i]])
            out = model(**cm.get_input_kwargs(ids))
        # the last `out` is from the final prefill chunk — use its last-position logits
        next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        # decode loop, full chain as the cache view
        cm_full = shared_cache.SharedCacheManager(
            cache_structure=[blocks], write_to=[blocks[-1]],
        )
        for _ in range(19):
            out = model(**cm_full.get_input_kwargs(next_token))
            next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

    assert torch.equal(out_vanilla, generated), (
        f"AR with {len(chunks)} block(s) diverged from vanilla generate.\n"
        f"vanilla: {tokenizer.decode(out_vanilla[0], skip_special_tokens=True)!r}\n"
        f"AR    : {tokenizer.decode(generated[0], skip_special_tokens=True)!r}"
    )
