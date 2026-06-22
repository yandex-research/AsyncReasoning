"""Async-input KV-cache surgery: insert M new tokens into the middle of a DynamicCache.

This is the kernel for the async-input scenario where:
  - Every regular forward is single-batch decode on a single linear KV cache (no AR
    multi-block dance, no thinker/writer fork).
  - When new user input arrives mid-decode, it is *inserted* at a fixed position
    `p` (typically: end of original prompt) rather than appended. Tokens that were
    already decoded past `p` are shifted to positions `[p+M, N+M)` and their K cache
    entries are RoPE-rotated by +M to reflect the new positions. V cache has no RoPE
    so it is concatenated as-is.

The operation is:

    Before: cache holds KV for positions [0, N)
    After:  cache holds KV for positions [0, N + M)
              [0, p)        : unchanged (prompt prefix)
              [p, p + M)    : freshly encoded new input
              [p + M, N + M): old suffix, K rotated by R(+M)

The "encode the new input in one forward" step is bundled here for convenience —
caller passes `new_input_ids` and `position`; the kernel does the slice, the new
forward, the RoPE shift, and the splice.

Semantics — the suffix V cache is **stale**, by design.
  Layer 0 K/V at suffix positions matches a from-scratch encode exactly (just bf16
  noise), because layer 0's K, V depend only on input embeddings — no attention. But
  layers ≥ 1 carry V cache values that were computed during the original decode of the
  suffix, when the cache only contained `P`. After we splice `I` in, those cached V
  values do NOT reflect the influence `I` would have had on the suffix's hidden states.
  Cumulative drift grows with depth; for Qwen3-4B (36 layers) on small fragments the
  top-1 next-token logit matches and the relative logit drift is ~4 %. This is the
  user-stated design — accept staleness in exchange for not re-running the suffix.

Currently supports full-attention layers only (Qwen3 dense). For Qwen3.5 hybrid
(GDN+full) and Qwen3.5-MoE, GDN layers will need their recurrent state checkpointed
at `position` and replayed through the suffix; that's a follow-up.
"""
from __future__ import annotations

import torch
from transformers import DynamicCache


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """The split-half rotation used by HF's `apply_rotary_pos_emb`."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_shift(K: torch.Tensor, cos_shift: torch.Tensor, sin_shift: torch.Tensor) -> torch.Tensor:
    """Apply a single RoPE rotation by a fixed offset `M` to every position in `K`.

    Handles partial rotation: Qwen3.5 uses rotary_dim < head_dim (e.g. head_dim=256,
    rotary_dim=64). cos/sin returned by the model's `rotary_emb` have length =
    rotary_dim. We rotate the first rotary_dim dims of K via the standard split-half
    rotation; the remaining (head_dim - rotary_dim) dims are passed through unchanged.

    K            : [B, H, T, head_dim]   (post-RoPE cache for old positions [p, N))
    cos_shift    : [rotary_dim]          (cos at the absolute shift position M)
    sin_shift    : [rotary_dim]          (sin at the absolute shift position M)
    Returns      : [B, H, T, head_dim]   (post-RoPE for shifted positions [p+M, N+M))
    """
    rotary_dim = cos_shift.shape[-1]
    K_rot = K[..., :rotary_dim]
    K_pass = K[..., rotary_dim:]
    cos = cos_shift.view(1, 1, 1, -1)
    sin = sin_shift.view(1, 1, 1, -1)
    K_rot_shifted = (K_rot * cos) + (_rotate_half(K_rot) * sin)
    if K_pass.shape[-1] == 0:
        return K_rot_shifted
    return torch.cat([K_rot_shifted, K_pass], dim=-1)


@torch.inference_mode()
def insert_async_input(
    model,
    cache: DynamicCache,
    new_input_ids: torch.Tensor,
    position: int,
) -> DynamicCache:
    """
    Insert M new input tokens at `position` in the cache and return the updated cache.

    Parameters
    ----------
    model : transformers PreTrainedModel
        Used for the forward pass on the new tokens and for the rotary embedding module
        that supplies the RoPE shift cos/sin.
    cache : DynamicCache
        The existing cache. Must have length N >= position. Mutated in place.
    new_input_ids : [1, M] LongTensor
        The new tokens to insert. Single-batch only.
    position : int
        Insertion offset within the existing cache (0 <= position <= N).

    Returns
    -------
    The same `cache` object, now of length N + M.
    """
    device = next(model.parameters()).device
    new_input_ids = new_input_ids.to(device)
    B, M = new_input_ids.shape
    N = cache.get_seq_length()
    assert 0 <= position <= N, f"position {position} outside [0, {N}]"
    if M == 0:
        return cache

    # 1+2. Cache-class abstraction: both `DynamicCache` (Qwen3 dense) and
    # `Qwen3_5MoeDynamicCache` (Qwen3.5 hybrid / MoE) are supported.
    #
    # - DynamicCache: per-layer K/V live on `cache.layers[i].keys/.values`, full
    #   `crop(N)` truncates uniformly.
    # - Qwen3_5MoeDynamicCache: only full-attention layers populate `key_cache[i]` /
    #   `value_cache[i]`; GDN layers carry `conv_states[i]` and `recurrent_states[i]`
    #   that are NOT seq-length-dependent and are NOT updated here. After insertion
    #   the GDN recurrent state is stale (computed for the original prefix only) —
    #   same staleness story as the V cache for layers >= 1. A correct fix is to
    #   replay the suffix through GDN layers; that's the follow-up.
    K_suffix: list[torch.Tensor] = []
    V_suffix: list[torch.Tensor] = []
    if hasattr(cache, "layers") and hasattr(cache.layers[0], "keys"):
        # DynamicCache path.
        full_attn_layer_ids = list(range(len(cache.layers)))
        for layer in cache.layers:
            K_suffix.append(layer.keys[..., position:, :].clone())
            V_suffix.append(layer.values[..., position:, :].clone())
        cache.crop(position)
    else:
        # Qwen3_5MoeDynamicCache path: only full-attention layers have key/value.
        full_attn_layer_ids = list(getattr(cache, "transformer_layers"))
        for li in full_attn_layer_ids:
            K_suffix.append(cache.key_cache[li][..., position:, :].clone())
            V_suffix.append(cache.value_cache[li][..., position:, :].clone())
        for li in full_attn_layer_ids:
            cache.key_cache[li] = cache.key_cache[li][..., :position, :].contiguous()
            cache.value_cache[li] = cache.value_cache[li][..., :position, :].contiguous()

    # 3. Forward the new tokens with the truncated cache. They get positions
    #    [position, position + M). Same position layout for every batch row.
    new_position_ids = torch.arange(position, position + M, device=device).unsqueeze(0).expand(B, -1)
    cache_position = torch.arange(position, position + M, device=device)
    model(
        input_ids=new_input_ids,
        past_key_values=cache,
        position_ids=new_position_ids,
        cache_position=cache_position,
        use_cache=True,
    )

    # 4. Compute cos/sin for the RoPE shift (= cos/sin at absolute position M).
    rotary_emb = model.model.rotary_emb
    # Determine cache dtype from either layout.
    if hasattr(cache, "layers") and hasattr(cache.layers[0], "keys"):
        cache_dtype = cache.layers[0].keys.dtype
    else:
        cache_dtype = cache.key_cache[full_attn_layer_ids[0]].dtype
    # rotary_emb expects (hidden_states, position_ids); the hidden_states is only used
    # for dtype/device, not values.
    dummy_hs = torch.zeros(1, 1, model.config.hidden_size, device=device, dtype=cache_dtype)
    cos_full, sin_full = rotary_emb(dummy_hs, torch.tensor([[M]], device=device))
    cos_shift = cos_full.to(cache_dtype).reshape(-1)
    sin_shift = sin_full.to(cache_dtype).reshape(-1)

    # 5. RoPE-shift the snapshotted suffix K, and append (K, V) back onto the cache.
    if hasattr(cache, "layers") and hasattr(cache.layers[0], "keys"):
        for layer_idx, (Ks, Vs) in enumerate(zip(K_suffix, V_suffix)):
            K_shifted = _rope_shift(Ks, cos_shift, sin_shift)
            layer = cache.layers[layer_idx]
            layer.keys = torch.cat([layer.keys, K_shifted], dim=-2)
            layer.values = torch.cat([layer.values, Vs], dim=-2)
    else:
        for slot, (li, Ks, Vs) in enumerate(zip(full_attn_layer_ids, K_suffix, V_suffix)):
            K_shifted = _rope_shift(Ks, cos_shift, sin_shift)
            cache.key_cache[li] = torch.cat([cache.key_cache[li], K_shifted], dim=-2)
            cache.value_cache[li] = torch.cat([cache.value_cache[li], Vs], dim=-2)

    return cache
