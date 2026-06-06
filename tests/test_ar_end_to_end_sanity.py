"""END-TO-END SANITY CHECK for AsyncReasoning on Qwen3.5 (hybrid GDN + full-attention).

This is the most comprehensive correctness check we have for AR's cache and forward
semantics. It runs four stages on the same problem:

  1. baseline_think via `model.generate(enable_thinking=True)`. Saves the reasoning
     token stream (everything before `</think>`) and the response token stream
     (everything after).

  2. AR in pure thinker_only mode (mode-switching short-circuited to always-no).
     Compares AR's generated thinker tokens against baseline's. They should match
     for the first ~tens of tokens, then drift due to bf16 incremental-decode noise.

  3. AR with BASELINE's thinker INJECTED into the thinker_output cache block, then
     run in writer_only mode. Compares AR's generated writer tokens against baseline's
     response tokens. With baseline thinker as starting context, the writer path is
     bf16-deterministic the whole way through, so we expect near-bit-equal match.

  4. AR with AR's OWN thinker (from stage 2) INJECTED into the thinker_output cache
     block, then run in writer_only mode. This is the closest thing to a full AR
     pipeline check (AR thinker -> AR writer). Because AR's thinker may have stopped
     mid-derivation (budget exhaustion before it emitted `</think>`), the writer's
     output legitimately diverges from baseline — sometimes it rederives, sometimes
     it picks up where the thinker left off. The success criterion is that the writer
     produces a coherent response with a `\\boxed{...}` final answer, not bit-match
     against baseline.

A divergence in any stage points at a regression in one of:
  - multi-block prefill / chunk-kernel composition
  - GDN affine accumulation / composition (`shared_cache/gdn_cache_block.py`)
  - writer-fork `compose_initial_recurrent_state`
  - block masking in `CombinedCacheView`
  - RoPE rotation roundtrip on K cache reads (`shared_cache/cache_block.py`)
  - stored kernel state path in `qwen35_gdn/qwen35_ar_patch.py`

Uses Qwen3.5-27B because smaller models reason briefly and don't always reach the
writer phase within a sensible budget. Skipped if the model isn't available locally.
Total runtime is on the order of a few minutes on a single A100.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Moderate-complexity multi-step problem. Qwen3.5-27B's reasoning is verbose even on
# simple problems, so we pick a budget big enough for baseline_think to reach `</think>`
# and emit a meaningful writer phase that stage 3 can compare against.
PROBLEM = (
    "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
    "A train leaves station A at 60 km/h. Two hours later a second train leaves "
    "station A on the same track at 80 km/h. After how many hours will the second "
    "train catch up to the first?"
)
BUDGET = 2048
# Minimum number of leading tokens that must match before allowed bf16 drift sets in.
# Empirically AR matches baseline for ~20-50 tokens on hybrid models before incremental
# decode noise (the RoPE-rotation roundtrip on K reads) flips an argmax at a near-tied
# logit choice. Tightening these would catch regressions; loosening papers over them.
MIN_LEADING_MATCH_THINKER = 15           # stage 2: AR thinker_only vs baseline thinker
MIN_LEADING_MATCH_WRITER_BASELINE = 5    # stage 3: AR writer with BASELINE thinker injected
MIN_AR_TRACE_WRITER_TOKENS = 20          # stage 4: AR writer with AR's OWN thinker injected.
                                          # No bit-match requirement: the test only verifies
                                          # the writer produces a coherent response with a
                                          # `\boxed{...}` final answer, since AR thinker may
                                          # have stopped mid-derivation.


def _have_qwen35_27b() -> bool:
    return os.path.isdir("/mnt/LLM/hub/models--Qwen--Qwen3.5-27B")


def _max_leading_match(a_ids, b_ids, shift_range=range(-3, 5)) -> int:
    """Greatest number of leading matched tokens after sliding `a` by a few positions.

    `apply_chat_template` and AR's prefix can disagree by a small offset in how many
    tokens of bookkeeping precede the model's first real generation; this slides
    across that offset and returns the best alignment."""
    best = 0
    for shift in shift_range:
        a = a_ids[shift:] if shift >= 0 else a_ids
        b = b_ids[-shift:] if shift < 0 else b_ids
        common = min(len(a), len(b))
        match = 0
        for i in range(common):
            if a[i] != b[i]:
                break
            match += 1
        best = max(best, match)
    return best


@pytest.mark.skipif(not _have_qwen35_27b(), reason="Qwen3.5-27B not present in local HF cache")
def test_ar_end_to_end_sanity_qwen35_27b():
    os.environ.setdefault("HF_HOME", "/mnt/LLM")
    import transformers
    from async_reasoning.solver import AsyncReasoningSolver
    from async_reasoning.cache import State, AsyncReasoningCache
    import async_reasoning.prompting as Prompting
    from qwen35_gdn.qwen35_ar_patch import patch_qwen35_for_async_reasoning

    tk = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)
    m = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-27B", torch_dtype="auto", device_map="cuda",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    m.eval()
    patch_qwen35_for_async_reasoning(m)
    eot_id = tk.vocab["</think>"]
    eos_id = tk.eos_token_id

    # =========================================================================
    # Stage 1: baseline_think (the reference)
    # =========================================================================
    text = tk.apply_chat_template(
        [{"role": "user", "content": PROBLEM}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    input_ids = tk.encode(text, return_tensors="pt").to(m.device)
    with torch.inference_mode():
        outputs = m.generate(input_ids, max_new_tokens=BUDGET, do_sample=False)
    baseline_gen_ids = outputs[0][input_ids.shape[-1]:].tolist()
    # Split at the first </think> token (id = eot_id) into the reasoning and response halves.
    if eot_id in baseline_gen_ids:
        split_idx = baseline_gen_ids.index(eot_id)
        baseline_thinker_ids = baseline_gen_ids[:split_idx]
        baseline_writer_ids = baseline_gen_ids[split_idx + 1:]  # drop </think> itself
    else:
        baseline_thinker_ids, baseline_writer_ids = baseline_gen_ids, []
    assert len(baseline_thinker_ids) > MIN_LEADING_MATCH_THINKER, (
        f"baseline_think only produced {len(baseline_thinker_ids)} thinker tokens; "
        f"either the model degraded or the problem is too easy to compare against."
    )

    # =========================================================================
    # Stage 2: AR in pure thinker_only mode
    # =========================================================================
    system_tokens = [k for k in tk.vocab if k.endswith("SYSTEM") or k.endswith("SYSTEM:")]
    wft = [tk.vocab[x] for x in ["</think>", "<|im_start|>", "<|endoftext|>"] + system_tokens]
    tft = [tk.vocab[x] for x in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] + system_tokens]
    solver = AsyncReasoningSolver(
        m, tk,
        writer_forbidden_token_ix=wft, thinker_forbidden_token_ix=tft,
        end_of_think_token_ix=[eot_id],
        use_fast_kernel=False,
    )
    # Short-circuit the mode-switcher so the solver stays in thinker_only the entire run.
    solver.check_if_should_continue_writing = lambda *a, **kw: False
    _, ar_thinker_text, _, _ = solver.solve(PROBLEM, budget=BUDGET)
    # AR's reported thinker_text starts with the AR cache prefix `<|im_end|>...<think>\n`
    # plus the appended `\n` starter. Strip those so we compare the model's actual generation.
    prefix_idx = ar_thinker_text.find("<think>\n")
    ar_thinker_gen = ar_thinker_text[prefix_idx + len("<think>\n"):]
    if ar_thinker_gen.startswith("\n"):
        ar_thinker_gen = ar_thinker_gen[1:]
    ar_thinker_ids = tk.encode(ar_thinker_gen, add_special_tokens=False)

    thinker_match = _max_leading_match(ar_thinker_ids, baseline_thinker_ids)
    print(f"\n[stage 2] AR thinker_only matches baseline for {thinker_match}/"
          f"{min(len(ar_thinker_ids), len(baseline_thinker_ids))} leading tokens")
    assert thinker_match >= MIN_LEADING_MATCH_THINKER, (
        f"AR thinker_only diverged from baseline_think after only {thinker_match} leading "
        f"tokens (want >= {MIN_LEADING_MATCH_THINKER}). Likely a regression in AR's cache "
        f"or forward semantics."
    )

    # =========================================================================
    # Stage 3: AR writer-only with baseline thinker injected
    # =========================================================================
    if not baseline_writer_ids:
        pytest.skip("baseline_think did not produce a writer response within budget; "
                    "stage 3 cannot run. Increase BUDGET or pick a smaller problem.")
    baseline_thinker_text = tk.decode(baseline_thinker_ids)

    # Monkey-patch AsyncReasoningPrompting so AR's thinker_output_prefix carries baseline's
    # thinker content. The cache will prefill the combined string into the thinker_output
    # block in one chain pass, exactly as if the thinker had generated it itself.
    orig_init = Prompting.AsyncReasoningPrompting.__init__

    def patched_init(self, problem, _inject=baseline_thinker_text):
        orig_init(self, problem)
        self.thinker_output_prefix = self.thinker_output_prefix + "\n" + _inject

    Prompting.AsyncReasoningPrompting.__init__ = patched_init
    try:
        prompting = Prompting.AsyncReasoningPrompting(PROBLEM)
        tokenizer_kwargs = dict(add_special_tokens=False, return_tensors="pt",
                                padding=True, padding_side="left")
        cache = AsyncReasoningCache(m, tk, prompting,
                                    tokenizer_kwargs=tokenizer_kwargs,
                                    starting_state=State.writer_only)
        # Mirror solver.solve()'s writer-only loop directly; we can't reuse solver.solve
        # because it always starts in thinker_only and runs the mode-switcher.
        writer_output_tokens = tk.encode(
            prompting.writer_output_prefix, **tokenizer_kwargs).flatten().tolist()
        writer_output_tokens.append(tk.encode("\n\n", **tokenizer_kwargs).item())
        ar_writer_gen_ids: list[int] = []
        with torch.inference_mode():
            for _step in range(BUDGET):
                next_inputs = {"input_ids": torch.tensor(
                    [writer_output_tokens[-1:]], device=m.device)}
                logits = m(**cache.get_input_kwargs(**next_inputs)).logits[..., -1, :]
                logits[..., wft] -= 100
                next_tok = int(logits.argmax(-1))
                writer_output_tokens.append(next_tok)
                ar_writer_gen_ids.append(next_tok)
                if next_tok == eos_id:
                    break
    finally:
        Prompting.AsyncReasoningPrompting.__init__ = orig_init

    writer_match = _max_leading_match(ar_writer_gen_ids, baseline_writer_ids)
    print(f"[stage 3] AR writer_only (with baseline thinker injected) matches baseline "
          f"writer for {writer_match}/"
          f"{min(len(ar_writer_gen_ids), len(baseline_writer_ids))} leading tokens")
    assert writer_match >= MIN_LEADING_MATCH_WRITER_BASELINE, (
        f"AR writer_only (with baseline thinker injected) diverged from baseline_think's "
        f"response after only {writer_match} leading tokens (want >= "
        f"{MIN_LEADING_MATCH_WRITER_BASELINE}). Likely a regression in writer-side cache "
        f"or forward."
    )

    # =========================================================================
    # Stage 4: AR writer-only with AR's own thinker (from stage 2) injected
    # =========================================================================
    # Same machinery as stage 3 but injecting AR's drifted thinker trace instead of
    # baseline's. Verifies that the full AR pipeline (thinker -> writer) works end-to-end:
    # AR's writer can interpret AR's own thinker output and continue into a coherent
    # response. The leading writer tokens should still align with baseline's response
    # (the writer's first ~few tokens are usually structural — "To find ...", "The number
    # is ..." — and don't depend strongly on tail-of-thinker details).
    #
    # If AR's thinker_only run happened to emit `</think>` (which transitions state to
    # writer_only inside the solver), that `</think>` token is in thinker_output_tokens.
    # We must strip it before injection — the writer_output_prefix supplies its own
    # `</think>`, and a double-`</think>` context confuses the first writer token.
    eot_text = tk.decode([eot_id])
    if eot_text in ar_thinker_gen:
        ar_thinker_for_injection = ar_thinker_gen.split(eot_text, 1)[0].rstrip("\n ")
    else:
        ar_thinker_for_injection = ar_thinker_gen

    def patched_init_ar_trace(self, problem, _inject=ar_thinker_for_injection):
        orig_init(self, problem)
        self.thinker_output_prefix = self.thinker_output_prefix + "\n" + _inject

    Prompting.AsyncReasoningPrompting.__init__ = patched_init_ar_trace
    try:
        prompting = Prompting.AsyncReasoningPrompting(PROBLEM)
        tokenizer_kwargs = dict(add_special_tokens=False, return_tensors="pt",
                                padding=True, padding_side="left")
        cache = AsyncReasoningCache(m, tk, prompting,
                                    tokenizer_kwargs=tokenizer_kwargs,
                                    starting_state=State.writer_only)
        writer_output_tokens = tk.encode(
            prompting.writer_output_prefix, **tokenizer_kwargs).flatten().tolist()
        writer_output_tokens.append(tk.encode("\n\n", **tokenizer_kwargs).item())
        ar_writer_from_ar_thinker_ids: list[int] = []
        with torch.inference_mode():
            for _step in range(BUDGET):
                next_inputs = {"input_ids": torch.tensor(
                    [writer_output_tokens[-1:]], device=m.device)}
                logits = m(**cache.get_input_kwargs(**next_inputs)).logits[..., -1, :]
                logits[..., wft] -= 100
                next_tok = int(logits.argmax(-1))
                writer_output_tokens.append(next_tok)
                ar_writer_from_ar_thinker_ids.append(next_tok)
                if next_tok == eos_id:
                    break
    finally:
        Prompting.AsyncReasoningPrompting.__init__ = orig_init

    import re
    ar_writer_text = tk.decode(ar_writer_from_ar_thinker_ids)
    boxed_match = re.findall(r"\\boxed\{([^}]*)\}", ar_writer_text)
    print(f"[stage 4] AR writer_only (with AR thinker injected) produced "
          f"{len(ar_writer_from_ar_thinker_ids)} tokens; "
          f"boxed answers found: {boxed_match if boxed_match else '<none>'}")
    if len(ar_writer_from_ar_thinker_ids) < MIN_AR_TRACE_WRITER_TOKENS or not boxed_match:
        # Surface enough context to diagnose if this fails.
        print(f"  AR-trace writer text: {ar_writer_text!r}")
        print(f"  AR thinker injected, last 200 chars: "
              f"{ar_thinker_for_injection[-200:]!r}")
    assert len(ar_writer_from_ar_thinker_ids) >= MIN_AR_TRACE_WRITER_TOKENS, (
        f"Stage 4: AR writer (with AR thinker injected) produced only "
        f"{len(ar_writer_from_ar_thinker_ids)} tokens. Pipeline likely crashed or "
        f"stalled."
    )
    assert boxed_match, (
        f"Stage 4: AR writer (with AR thinker injected) produced "
        f"{len(ar_writer_from_ar_thinker_ids)} tokens but never reached a `\\boxed{{}}` "
        f"final answer. The full AR pipeline isn't producing a complete response."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
