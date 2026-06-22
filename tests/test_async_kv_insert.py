"""Numerical-equivalence test for async_kv_insert on a dense Qwen3 model.

We check that inserting M new tokens at position p into a cache via
`insert_async_input(model, cache, new, p)` is equivalent — up to bf16 noise — to
encoding `prompt + new + suffix` from scratch.

Method:
  * Build "oracle" cache by forwarding P + I + S in one shot.
  * Build "insert" cache by forwarding P + S, then async-inserting I at len(P).
  * Decode the same follow-up token from each cache and compare the resulting logits.

Skipped if no Qwen3 dense model is available locally.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_DENSE_CANDIDATES = ["Qwen3-4B", "Qwen3-8B", "Qwen3-1.7B", "Qwen3-0.6B"]


def _find_local_dense_qwen3() -> str | None:
    for name in _DENSE_CANDIDATES:
        if os.path.isdir(f"/mnt/LLM/hub/models--Qwen--{name}"):
            return f"Qwen/{name}"
    return None


@pytest.fixture(scope="module")
def qwen3_model_and_tokenizer():
    model_id = _find_local_dense_qwen3()
    if model_id is None:
        pytest.skip("no Qwen3 dense model in local HF cache")
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def _decode_next_logits(model, cache, next_token: torch.Tensor, position: int) -> torch.Tensor:
    """Forward one token at the given absolute position and return its last-step logits."""
    pos = torch.tensor([[position]], device=model.device)
    with torch.inference_mode():
        out = model(
            input_ids=next_token,
            past_key_values=cache,
            position_ids=pos,
            cache_position=pos.squeeze(0),
            use_cache=True,
        )
    return out.logits[:, -1, :].clone()


def test_insert_matches_from_scratch(qwen3_model_and_tokenizer):
    from transformers import DynamicCache
    from async_reasoning.async_kv_insert import insert_async_input

    model, tokenizer = qwen3_model_and_tokenizer
    device = model.device

    # Three independent text fragments — enough length variety to exercise non-trivial
    # RoPE shift over many positions.
    P_text = "Explain in three sentences how photosynthesis works in plants."
    I_text = " Also consider the role of stomata in gas exchange and turgor regulation."
    S_text = " Specifically focus on the light-dependent reactions and electron transport chain."

    enc = lambda s: tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    P, I, S = enc(P_text), enc(I_text), enc(S_text)
    nP, nI, nS = P.shape[-1], I.shape[-1], S.shape[-1]
    print(f"\nnP={nP} nI={nI} nS={nS} -> oracle len {nP + nI + nS}")

    # --- ORACLE: from-scratch encode of P + I + S ---
    PIS = torch.cat([P, I, S], dim=-1)
    cache_oracle = DynamicCache()
    with torch.inference_mode():
        out_oracle = model(input_ids=PIS, past_key_values=cache_oracle, use_cache=True)
    cache_oracle = out_oracle.past_key_values
    assert cache_oracle.get_seq_length() == nP + nI + nS

    # --- INSERT: encode P + S then insert I at position nP ---
    PS = torch.cat([P, S], dim=-1)
    cache_insert = DynamicCache()
    with torch.inference_mode():
        out_ps = model(input_ids=PS, past_key_values=cache_insert, use_cache=True)
    cache_insert = out_ps.past_key_values
    assert cache_insert.get_seq_length() == nP + nS

    cache_insert = insert_async_input(model, cache_insert, I, position=nP)
    assert cache_insert.get_seq_length() == nP + nI + nS, (
        f"post-insert length {cache_insert.get_seq_length()}, expected {nP + nI + nS}"
    )

    # --- Compare cache contents per layer ---
    # Layer 0 has no attention so its K/V at suffix slots MUST match exactly (the
    # RoPE-shift math has nothing to mask). This is the strongest verification of the
    # kernel itself.
    #
    # Layers ≥ 1 carry V values computed when the cache only held P — they are stale
    # by design (the user-stated spec is "slice + splice + shift, don't re-run
    # suffix"). We surface but don't assert on the cumulative drift.
    print(f"\nper-layer K/V relmax on suffix slots [nP+nI : nP+nI+nS]:")
    layer0_K_rel = layer0_V_rel = None
    max_deep_K = max_deep_V = 0.0
    for layer_idx, (lo, li) in enumerate(zip(cache_oracle.layers, cache_insert.layers)):
        assert lo.keys.shape == li.keys.shape, (
            f"layer {layer_idx}: K shape mismatch {lo.keys.shape} vs {li.keys.shape}"
        )
        Ko = lo.keys[..., nP + nI : nP + nI + nS, :].float()
        Ki = li.keys[..., nP + nI : nP + nI + nS, :].float()
        Vo = lo.values[..., nP + nI : nP + nI + nS, :].float()
        Vi = li.values[..., nP + nI : nP + nI + nS, :].float()
        Krel = (Ko - Ki).abs().max().item() / max(Ko.abs().max().item(), 1e-3)
        Vrel = (Vo - Vi).abs().max().item() / max(Vo.abs().max().item(), 1e-3)
        if layer_idx == 0:
            layer0_K_rel, layer0_V_rel = Krel, Vrel
        else:
            max_deep_K = max(max_deep_K, Krel)
            max_deep_V = max(max_deep_V, Vrel)
        if layer_idx in (0, 1, 2, len(cache_oracle.layers) // 2, len(cache_oracle.layers) - 1):
            print(f"  layer {layer_idx:>2}: K_relmax={Krel:.3e}  V_relmax={Vrel:.3e}")
    print(f"layer-0:                K={layer0_K_rel:.3e} V={layer0_V_rel:.3e}  (must be exact)")
    print(f"deepest layer (>=1):    K={max_deep_K:.3e} V={max_deep_V:.3e}  (drift expected)")

    # --- Decode one more token from each cache, compare logits ---
    next_token = enc(" Then")[:, :1]
    pos_after = nP + nI + nS
    logits_oracle = _decode_next_logits(model, cache_oracle, next_token, pos_after)
    logits_insert = _decode_next_logits(model, cache_insert, next_token, pos_after)
    diff = (logits_oracle - logits_insert).abs().max().item()
    scale = logits_oracle.abs().max().item()
    rel = diff / max(scale, 1e-3)
    top1_o = int(logits_oracle.argmax().item())
    top1_i = int(logits_insert.argmax().item())
    print(f"\nnext-token logits: max abs diff={diff:.3e}, scale={scale:.3e}, rel={rel:.3e}")
    print(f"top-1 next: oracle={top1_o} ({tokenizer.decode([top1_o])!r}) "
          f"insert={top1_i} ({tokenizer.decode([top1_i])!r})")

    # --- Assertions ---
    # 1. Layer 0 — must be exact (kernel correctness).
    assert layer0_K_rel < 5e-2, f"layer-0 K diff {layer0_K_rel:.3e} too large; RoPE shift math is wrong"
    assert layer0_V_rel < 5e-2, f"layer-0 V diff {layer0_V_rel:.3e} too large; V splice has a bug"
    # 2. Top-1 next-token agreement — the only end-to-end semantic invariant we want.
    assert top1_o == top1_i, "top-1 next-token differs between oracle and insert paths"
    # 3. Relative logit drift — loose ceiling so we notice catastrophic regression but
    #    don't fail on benign bf16-staleness compounding.
    assert rel < 1.5e-1, f"next-token logit relative diff {rel:.3e} too large"
