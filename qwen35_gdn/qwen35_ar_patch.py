"""
Monkey-patch `Qwen3_5GatedDeltaNet.forward` to thread the cache's prior recurrent_state /
conv_state through the prefill (chunk) path.

Stock Qwen3.5 hardcodes `initial_state=None` in the chunk-mode delta rule and computes the
conv state from the current chunk alone — so AsyncReasoning's multi-block prefill silently
loses the state from previously-prefilled blocks. This patch fixes both:

- `chunk_gated_delta_rule(..., initial_state=cache_recurrent_state, ...)` when the cache
  reports `has_previous_state`.
- conv: when prior conv_state is known, prepend its last `conv_kernel_size - 1` columns to
  `mixed_qkv` as left context, conv the joined sequence, then slice the new seq_len outputs.

Both Qwen3.5 variants are supported by the same patch:

- Dense Qwen3.5 (`transformers.models.qwen3_5.Qwen3_5GatedDeltaNet`)
- MoE Qwen3.5 (`transformers.models.qwen3_5_moe.Qwen3_5MoeGatedDeltaNet`)

The patch is class-agnostic: `_iter_gdn_modules` walks any submodule whose class name ends
in `GatedDeltaNet` and which exposes `chunk_gated_delta_rule`. The MoE variant exposes the
exact same `(in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, conv1d, A_log, dt_bias, norm,
out_proj, chunk_gated_delta_rule, recurrent_gated_delta_rule, causal_conv1d_update,
causal_conv1d_fn, num_v_heads, head_k_dim, head_v_dim, conv_kernel_size, layer_idx,
activation)` attribute set as the dense variant, so the same `_patched_forward` binds
without changes.

The patch is reversible via `unpatch()`. Apply with:

    patch = patch_qwen35_for_async_reasoning(model)
    ...
    patch.unpatch(model)
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn.functional as F


def _iter_gdn_modules(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    for m in model.modules():
        if m.__class__.__name__.endswith("GatedDeltaNet") and hasattr(m, "chunk_gated_delta_rule"):
            yield m


def _apply_mask_to_padding_states(hidden_states: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    # Mirror transformers.models.qwen3_5.modeling_qwen3_5.apply_mask_to_padding_states.
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def _patched_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    # AR-style cache (CombinedCacheView) carries per-block affine summaries we compose here.
    # Vanilla `Qwen3_5DynamicCache` does not — for it we fall back to its flat recurrent_states.
    ar_cache = cache_params if (cache_params is not None and hasattr(cache_params, "capture_token_affines")) else None

    prior_conv_state = None
    prior_recurrent_state = None
    if ar_cache is not None:
        has_prev = ar_cache.has_previous_affine(self.layer_idx)
        prior_conv_state = ar_cache.conv_states[self.layer_idx]
        if has_prev:
            # For single-worker calls (prefill, thinker_only, writer_only) the kernel's
            # previously-stored output state is the correct initial state — using it directly
            # gives bit-equal generation to vanilla model.generate. For multi-worker batched
            # calls (thinker_and_writer) we keep affine composition because per-worker stored
            # states can be stale relative to chain mates: when the writer pauses while the
            # thinker advances, writer's stored state hasn't seen those new thinker tokens,
            # but the writer's chain [input, thinker, writer] still expects to include them.
            # Affine composition correctly threads the latest thinker affines into writer's
            # initial state on each batched call.
            num_workers = len(ar_cache.cache_structure)
            prior_recurrent_state = None
            if num_workers == 1:
                prior_recurrent_state = ar_cache.recurrent_states[self.layer_idx]
            if prior_recurrent_state is None:
                prior_recurrent_state = ar_cache.compose_initial_recurrent_state(
                    layer_idx=self.layer_idx,
                    num_heads=self.num_v_heads,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                    hf_convention=True,
                )
    elif cache_params is not None:
        has_prev = getattr(cache_params, "has_previous_state", False)
        prior_conv_state = cache_params.conv_states[self.layer_idx]
        prior_recurrent_state = cache_params.recurrent_states[self.layer_idx]
    else:
        has_prev = False

    use_decode_path = (
        cache_params is not None
        and has_prev
        and prior_conv_state is not None
        and seq_len == 1
        and cache_position is not None
    )

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)  # [batch, qkv_dim, seq_len]

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_decode_path:
        # `torch_causal_conv1d_update` calls `conv_state.copy_(...)` to advance the cache in place.
        # In the stock model that's fine — `conv_state` IS the write target. In our AR chain it
        # may be a reference to a prior block's conv_state (borrowed as left context when the
        # write target was empty), so we clone before mutating, then store the advanced clone as
        # the write target's new conv_state.
        decode_conv_state = prior_conv_state.detach().clone()
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            decode_conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
        if cache_params is not None:
            cache_params.conv_states[self.layer_idx] = decode_conv_state
    else:
        kernel_size = self.conv_kernel_size

        if has_prev and prior_conv_state is not None:
            # Prepend the last (kernel_size - 1) columns of the prior conv state as left context,
            # so the conv has the correct receptive field across the block boundary.
            ctx = prior_conv_state[..., -(kernel_size - 1):]
            full_input = torch.cat([ctx, mixed_qkv], dim=-1)
            # conv1d has padding=kernel_size-1 on both sides, so output length = L + k - 1.
            # The causal outputs for full_input positions [0, L-1] are at output indices [0, L-1].
            # Within those, the mixed_qkv portion sits at full_input positions [k-1, k-1+seq_len-1],
            # so the wanted output slice is [k-1, k-1+seq_len-1].
            conv_out = F.silu(self.conv1d(full_input))
            mixed_qkv_post_conv = conv_out[..., kernel_size - 1: kernel_size - 1 + seq_len]
            # Build the new conv_state = last kernel_size columns of [ctx; mixed_qkv].
            new_conv_state = full_input[..., -kernel_size:]
        else:
            # First prefill (no prior conv state).
            new_conv_state = F.pad(mixed_qkv, (kernel_size - mixed_qkv.shape[-1], 0))
            if self.causal_conv1d_fn is not None:
                mixed_qkv_post_conv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv_post_conv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv_post_conv
        if cache_params is not None:
            cache_params.conv_states[self.layer_idx] = new_conv_state

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    initial_state = prior_recurrent_state if has_prev else None

    if not use_decode_path:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=prior_recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if ar_cache is not None:
        # Build per-token alpha (= exp(g)) and beta in shapes matching capture_token_affines.
        alpha = g.exp()  # [batch, seq_len, num_v_heads]
        # When num_v_heads > num_k_heads, key was repeat_interleaved already, so it matches alpha/beta.
        ar_cache.capture_token_affines(
            layer_idx=self.layer_idx,
            key=key, value=value,
            alpha=alpha, beta=beta,
            l2norm_key=True,
        )
        # Also store the kernel's actual output state so the next single-worker call can
        # reuse it directly (matches baseline's path; affines are still used as fallback
        # and for multi-worker chain composition).
        if last_recurrent_state is not None:
            ar_cache.recurrent_states[self.layer_idx] = last_recurrent_state
    elif cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    return self.out_proj(core_attn_out)


@dataclass
class Qwen35ARPatch:
    originals: Dict[int, Callable] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.originals is None:
            self.originals = {}

    def patch(self, model: torch.nn.Module) -> "Qwen35ARPatch":
        for module in _iter_gdn_modules(model):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is None or layer_idx in self.originals:
                continue
            self.originals[layer_idx] = module.forward
            module.forward = types.MethodType(_patched_forward, module)
        return self

    def unpatch(self, model: torch.nn.Module) -> None:
        for module in _iter_gdn_modules(model):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is None:
                continue
            original = self.originals.pop(layer_idx, None)
            if original is not None:
                module.forward = original


def patch_qwen35_for_async_reasoning(model: torch.nn.Module) -> Qwen35ARPatch:
    """Install the AR-friendly Qwen3_5GatedDeltaNet.forward on every GDN layer of `model`.

    Returns the patch object; call `.unpatch(model)` to revert."""
    return Qwen35ARPatch().patch(model)
