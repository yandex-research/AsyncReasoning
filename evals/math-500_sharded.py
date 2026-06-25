import sys
sys.path.insert(0, __file__.rsplit("/", 2)[0])
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/utils")

import os
import json
import argparse

import torch
import transformers
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

# from tts_evaluator import TTSEvaluator  # tortoise-tts dep disabled
from utils.answer_processing import find_last_valid_expression, check_equality_judge, check_equality_local_model
from utils.gpu_parallel import get_worker_rank, init_worker_logger
from utils.task_queue import TaskQueue

if "NV_YT_OPERATION_ID" in os.environ:
    import nirvana_dl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue",
        type=str,
        default=None,
        help="Endpoint for a zmq-like task dispenser that dispenses task indices. Provide *either* this or start & end"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First task to be processed by script inclusive. E.g --start 0 --end 100 will process tasks [0-99]"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last task to be processed by script exclusive. E.g --start 0 --end 100 will process tasks [0-99]"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["async_reasoning", "async_inputs", "baseline_think", "baseline_no_think"],
        help="Select reasoning mode. async_inputs: single-cache decode, no thinker/writer "
             "fork; shard_2 is spliced into the cache via async_kv_insert at "
             "next_shard_every_steps decoded tokens (target must be 'input').",
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B", help="Model name from hf")
    parser.add_argument("--budget", type=int, default=16384, help="Budget to eval on")
    parser.add_argument("--use-slow-kernel", action="store_true", default=False, help="Disable fast kernel")
    parser.add_argument("--use-local-judge", action="store_true", default=False, help="Use the same model as a judge for result.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for async_inputs mode (other modes always use B=1). Larger B improves GPU utilization on MoE; OOM-ed batches are auto-retried at B=1.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="sharded math500 dataset path for load_from_disk")
    parser.add_argument("--path-to-results", type=str, help="path to store exp results", default="./eval_results/math-500")
    parser.add_argument("--dump_snapshot_freq", type=int, default=4, help="yandex-internal snapshotting frequency")
    parser.add_argument("--next_shard_every_steps", type=int, help="Setting to set up shards appearance frequency. Exceptions are: 0 -- concat, -1 -- never supply the rest of the shards.")
    parser.add_argument(
        "--shard_to_target",
        nargs="+",
        choices=["thinker", "writer", "input", "output"],
        default=None,
        help='Where to share live context. async_reasoning supports input | thinker | writer; '
             'async_inputs supports input | output (or both, e.g. --shard_to_target input output, '
             'inserting shard_2 in BOTH places simultaneously).',
    )
    parser.add_argument(
        "--target_reminders",
        nargs="+",
        choices=["thinker", "writer"],
        default=[],
        help='Which of shard_to_target are reminders. Use: --target_reminders thinker | writer',
    )

    parser.add_argument(
        "--shard_wait_step", action="store_true", default=False, help='Wait for \\n\\n before inserting shard?',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rank = get_worker_rank()
    logger = init_worker_logger()
    logger.info(f'The script was run in the following way:')
    logger.info(f"python {__file__} \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))
    use_fast_kernel = not args.use_slow_kernel
    assert (not args.use_local_judge) or (not use_fast_kernel), "You cannot use local model with kernel as a judge"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype='auto', device_map=device, low_cpu_mem_usage=True
    )

    # Qwen3.5 (hybrid GDN+full-attention dense, and MoE variant) requires the GDN
    # patch for AR mode — the AR cache machinery (per-layer affine capture, multi-block
    # views) only works on the patched `_patched_forward`. Vanilla HF GDN does not
    # understand `combined_cache_view`. The fast-kernel cache also doesn't support GDN,
    # so this combination implicitly requires `--use-slow-kernel`.
    _name_lower = args.model_name.lower()
    _model_type = getattr(model.config, "model_type", "")
    _has_gdn = (
        "qwen3.5" in _name_lower
        or _model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")
    )
    if _has_gdn and args.mode == "async_reasoning":
        from qwen35_gdn.qwen35_ar_patch import patch_qwen35_for_async_reasoning
        patch_qwen35_for_async_reasoning(model)
        n_gdn = sum(1 for t in model.config.layer_types if t == "linear_attention")
        logger.info(f"Patched {n_gdn} GDN layers for Qwen3.5 hybrid model")

    solver_kwargs = {}
    if args.mode == "async_reasoning":
        from async_reasoning.solver import AsyncReasoningSolver as Solver
        from async_reasoning.async_input_hook import async_input_hook_constructor

        system_tokens = [key for key in tokenizer.vocab.keys() if key.endswith("SYSTEM") or key.endswith("SYSTEM:")]
        writer_forbidden_token_ix = [tokenizer.vocab[x] for x in ["</think>", "<|im_start|>", "<|endoftext|>"] + system_tokens]
        thinker_forbidden_token_ix = [tokenizer.vocab[x] for x in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] + system_tokens]
        end_of_think_token_ix = [tokenizer.vocab[x] for x in ["</think>"]]
        solver_kwargs.update({
            "writer_forbidden_token_ix": writer_forbidden_token_ix,
            "thinker_forbidden_token_ix": thinker_forbidden_token_ix,
            "use_fast_kernel": use_fast_kernel,
            "end_of_think_token_ix": end_of_think_token_ix,
        })
        solver = Solver(model, tokenizer, **solver_kwargs)
    elif args.mode == "async_inputs":
        # Single linear cache, no thinker/writer fork, no AR machinery. Async user
        # input gets spliced into the cache via the async_kv_insert kernel at the
        # decode-step count given by --next_shard_every_steps. Two targets are
        # supported: "input" (splice at end of original prompt; suffix RoPE-shifted)
        # and "output" (append at current end of cache; degenerate insertion). Both
        # may be combined: shard_2 ends up duplicated in both places.
        _async_in_targets = set(args.shard_to_target or ["input"])
        assert _async_in_targets.issubset({"input", "output"}), \
            "async_inputs supports --shard_to_target chosen from {input, output} only"
        # async_inputs is a throughput-oriented mode: refuse to run if the GDN fast
        # path is missing on a Qwen3.5 hybrid model. Falling back to the torch
        # reference of chunk_gated_delta_rule would silently make the eval ~3-5×
        # slower without any warning past startup.
        if _has_gdn:
            try:
                import fla  # noqa: F401
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
                import causal_conv1d  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "async_inputs on a Qwen3.5 hybrid model requires the fast GDN kernels. "
                    "Install: `pip install flash-linear-attention>=0.5.0 causal-conv1d>=1.6.2`. "
                    f"Import failed: {e}"
                ) from e
        from async_reasoning.async_kv_insert import insert_async_input  # noqa: F401
        solver = None  # raw decode path below
    elif args.mode in ["baseline_think", "baseline_no_think"]:
        assert args.next_shard_every_steps is None, "shard timing is only for async modes."
        assert args.shard_to_target is None, "shard target is only for async modes."
        assert args.shard_wait_step is [], "shard wait is only for async modes."
        from evals.baseline_solver import BaselineSolver as Solver
        solver_kwargs.update({
            "thinker_enabled": (args.mode == "baseline_think"),
        })
        solver = Solver(model, tokenizer, **solver_kwargs)
    else:
        raise ValueError("unsupported mode")
    dataset_math = load_from_disk(args.dataset_path)
    accuracy_numerator = accuracy_denominator = 0
    exp_dir_path = f"{args.path_to_results}/math-500_sharded_{args.next_shard_every_steps}_steps/{args.mode}"
    os.makedirs(exp_dir_path, exist_ok=True)
    # evaluator = TTSEvaluator()  # tortoise-tts dep disabled

    def _solve_async_inputs_batch(batch_indices):
        """Run a batched single-cache decode over `batch_indices` and return one
        `(writer_str, "", token_times, eos)` tuple per sample, in the same shape
        the AR solver returns so the downstream judging code is identical.

        Returns `None` (instead of a list) on OOM — caller is expected to defer
        the whole batch and retry one-at-a-time at the end of the run.

        Insertion lands at `prompt_len` (end of input block) — the kernel's
        RoPE-shift on the suffix actually does its job: any decoded tokens
        produced before insertion get shifted forward by M, K cache is rotated
        to match. Padding inside shard_2 (right-pad to common M) does land in
        the cache; future attention to those positions is not masked out, same
        accuracy tradeoff as the pilot.

        Trigger semantics — `--next_shard_every_steps == 0`: both shards are
        concatenated in `instruction` by the caller, no insertion. `> 0`: splice
        at that many decoded tokens (optionally deferred to `\\n\\n` boundary
        when `--shard_wait_step`). `< 0` or unset: never insert.
        """
        from async_reasoning.async_kv_insert import insert_async_input
        import time as _time

        device = next(model.parameters()).device
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        B = len(batch_indices)
        # Targets for shard insertion. "input" splices at end of prompt and shifts the
        # decoded suffix by +M (the kernel does the RoPE shift). "output" appends at the
        # current end of cache (degenerate kernel call — empty suffix). Both can be on.
        async_in_targets = set(args.shard_to_target or ["input"])
        do_insert_input = "input" in async_in_targets
        do_insert_output = "output" in async_in_targets

        # Build per-sample inputs
        samples = [dataset_math[i] for i in batch_indices]
        prompts_text = []
        shard2_text_per_sample = []
        for s in samples:
            shards = s["problem_shards"]
            instruction = "".join(shards) if args.next_shard_every_steps == 0 else shards[0]
            prompts_text.append(f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{instruction}")
            shard2_text_per_sample.append(shards[1])

        # Tokenize + left-pad prompts to common length
        prompt_ids_list = [tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids[0] for p in prompts_text]
        L_prompt = max(t.shape[0] for t in prompt_ids_list)
        prompt_batch = torch.full((B, L_prompt), pad_id, dtype=torch.long, device=device)
        prompt_mask = torch.zeros((B, L_prompt), dtype=torch.long, device=device)
        for b, t in enumerate(prompt_ids_list):
            prompt_batch[b, -t.shape[0]:] = t.to(device)
            prompt_mask[b, -t.shape[0]:] = 1

        # Tokenize shard_2 + right-pad to common length (only if we will insert).
        do_insert = args.next_shard_every_steps is not None and args.next_shard_every_steps > 0
        shard2_batch = None
        M = 0
        if do_insert:
            shard2_wrapped = [f"\n\nADDITIONAL USER INPUT: {s}\n\n" for s in shard2_text_per_sample]
            shard2_ids_list = [tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids[0] for s in shard2_wrapped]
            M = max(t.shape[0] for t in shard2_ids_list)
            shard2_batch = torch.full((B, M), pad_id, dtype=torch.long, device=device)
            for b, t in enumerate(shard2_ids_list):
                shard2_batch[b, : t.shape[0]] = t.to(device)

        try:
            # Prefill (batched).
            with torch.inference_mode():
                prefill_pos_ids = (prompt_mask.cumsum(dim=-1) - 1).clamp(min=0)
                out = model(
                    input_ids=prompt_batch,
                    attention_mask=prompt_mask,
                    position_ids=prefill_pos_ids,
                    use_cache=True,
                )
            cache = out.past_key_values
            prompt_cache_len = cache.get_seq_length()  # = L_prompt for left-padded inputs
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            attn_mask = torch.cat([prompt_mask, torch.ones(B, 1, dtype=torch.long, device=device)], dim=1)

            # Per-sample state
            emitted = [[] for _ in range(B)]                # token ids
            writer_chars = [""] * B
            token_times = [[] for _ in range(B)]            # list[(str, t, char_offset)]
            eos_state = [False] * B
            # First token (= argmax of prefill last logits) is the model's response to the prompt.
            t0 = _time.time()
            inserted = False

            with torch.inference_mode():
                for step in range(args.budget):
                    # Record this step's emitted token per sample (skipping finished ones).
                    t_now = _time.time() - t0
                    for b in range(B):
                        if eos_state[b]:
                            continue
                        tok_id = int(next_token[b, 0].item())
                        tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
                        token_times[b].append((tok_str, t_now, len(writer_chars[b])))
                        emitted[b].append(tok_id)
                        writer_chars[b] += tok_str
                        if tok_id == eos_id:
                            eos_state[b] = True
                    if all(eos_state):
                        break

                    # Insert shard_2 BEFORE the next forward. Two possible splice points:
                    #   * "input"  → at end of the original prompt; the kernel RoPE-shifts
                    #                the decoded suffix by +M so it keeps its relative
                    #                position.
                    #   * "output" → at the current end of cache; degenerate kernel call
                    #                (empty suffix), effectively appends shard_2 after the
                    #                decoded tokens so far.
                    # Both can run on the same step; shard_2 then appears twice in the
                    # cache (once inside the input block, once at the output end).
                    if do_insert and not inserted and (step + 1) >= args.next_shard_every_steps:
                        # Boundary check. The insertion is one cache surgery applied to the whole
                        # batch — all samples splice at the same cache position — so requiring
                        # *every* alive sample to be at a "\n\n" boundary at the exact same step
                        # has probability ~0 at B>1 and means insertion never fires. Instead, fire
                        # when ANY alive sample has hit a boundary, OR after a small grace window
                        # past `next_shard_every_steps` (so a contrarian sample can't deadlock the
                        # whole batch). At B=1 this degenerates to "wait for the only sample's
                        # next \n\n".
                        if args.shard_wait_step:
                            any_boundary = any(
                                writer_chars[b].endswith("\n\n")
                                for b in range(B) if not eos_state[b]
                            )
                            grace_exceeded = (step + 1) >= args.next_shard_every_steps + 32
                            boundary_ok = any_boundary or grace_exceeded
                        else:
                            boundary_ok = True
                        if boundary_ok:
                            shard2_attn = torch.ones(B, M, dtype=torch.long, device=device)
                            if do_insert_input:
                                cache = insert_async_input(model, cache, shard2_batch, position=prompt_cache_len)
                                attn_mask = torch.cat([
                                    attn_mask[:, :prompt_cache_len],
                                    shard2_attn,
                                    attn_mask[:, prompt_cache_len:],
                                ], dim=1)
                            if do_insert_output:
                                cache = insert_async_input(model, cache, shard2_batch, position=cache.get_seq_length())
                                attn_mask = torch.cat([attn_mask, shard2_attn], dim=1)
                            inserted = True

                    cache_pos = torch.tensor([cache.get_seq_length()], device=device)
                    out = model(
                        input_ids=next_token,
                        past_key_values=cache,
                        attention_mask=attn_mask,
                        cache_position=cache_pos,
                        use_cache=True,
                    )
                    cache = out.past_key_values
                    new_attn_col = torch.ones(B, 1, dtype=torch.long, device=device)
                    attn_mask = torch.cat([attn_mask, new_attn_col], dim=1)
                    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Build per-sample results in the same shape as the AR solver returns.
            results = []
            for b in range(B):
                results.append((writer_chars[b], "", token_times[b], eos_state[b]))
            return results

        except torch.cuda.OutOfMemoryError:
            # Drop everything from this attempt and let the caller retry.
            try:
                del cache, out, next_token, attn_mask
            except Exception:
                pass
            torch.cuda.empty_cache()
            return None

    def _judge_and_save(idx: int, writer_output_str: str, thinker_output_str: str,
                        token_times, eos_generated: bool):
        """Save the per-sample result FIRST (with is_equal=None), then attempt to
        judge. A judge failure (OOM, missing API key, network error) leaves the
        saved JSON with is_equal=None — re-runnable offline — and DOES NOT lose the
        decode output. Decode-time OOMs are handled separately by the caller."""
        nonlocal accuracy_numerator, accuracy_denominator
        save_path = f"{exp_dir_path}/sample_{idx}.json"
        problem_shards = dataset_math[idx]['problem_shards']
        answer = str(dataset_math[idx]['answer'])
        response = find_last_valid_expression(writer_output_str, extract_result=lambda x: x[7:-1])
        assert len(token_times) > 0, f"empty token_times for idx={idx}"
        # 1. Save first with is_equal=None so the response is never lost.
        result = {
            "idx": idx,
            "is_equal": None,
            "token_times": token_times,
            "eos_generated": eos_generated,
            "response_answers": response,
            "correct_answer": answer,
            "writer_response": writer_output_str,
            "thinker_response": thinker_output_str,
        }
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        # 2. Try to judge. Any failure (CUDA OOM, missing OPENAI key, etc.) is logged
        #    but does NOT remove the saved result — judge offline later.
        is_equal = None
        try:
            if args.use_local_judge:
                is_equal = check_equality_local_model(model, tokenizer, response, answer)
            else:
                is_equal = check_equality_judge(response, answer)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            logger.warning(f"judge OOM on idx={idx}: {str(e)[:80]} — leaving is_equal=None")
        except Exception as e:
            logger.warning(f"judge failed on idx={idx}: {type(e).__name__}: {str(e)[:80]} — leaving is_equal=None")
        if is_equal is not None:
            result["is_equal"] = bool(is_equal)
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2)
            accuracy_numerator += int(bool(is_equal))
            accuracy_denominator += 1
            current_accuracy = (accuracy_numerator / accuracy_denominator)
            print(end=f'[{rank=}] {idx=}, {eos_generated=}, {is_equal=}\t| {current_accuracy=:.3f}',
                  file=sys.stderr)
        else:
            print(end=f'[{rank=}] {idx=}, {eos_generated=}, is_equal=None (judge unavailable)',
                  file=sys.stderr)
        if "NV_YT_OPERATION_ID" in os.environ and rank == 0 and (
                accuracy_denominator % args.dump_snapshot_freq == args.dump_snapshot_freq - 1):
            nirvana_dl.snapshot.dump_snapshot()
            logger.info("Dumped Nirvana snapshot")

    def _solve_task_and_save(idx: int):
        """Per-sample path used by async_reasoning and baseline modes."""
        save_path = f"{exp_dir_path}/sample_{idx}.json"
        if os.path.exists(save_path):
            return  # already solved by previous run and saved in snapshot

        problem_shards = dataset_math[idx]['problem_shards']
        assert len(problem_shards) == 2, f"Unexpected number of shards on id: {idx}, {len(problem_shards)}"
        instruction = "".join(problem_shards) if args.next_shard_every_steps == 0 else problem_shards[0]
        problem = f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{instruction}"

        if args.mode == "async_reasoning":
            writer_output_str, thinker_output_str, token_times, eos_generated = \
                solver.solve(
                    problem,
                    budget=args.budget,
                    on_new_tokens_generated=async_input_hook_constructor(
                        solver,
                        args.shard_to_target,
                        args.target_reminders,
                        args.next_shard_every_steps,
                        problem_shards[1],
                        args.shard_wait_step,
                    )
                )
        else:  # baseline_think / baseline_no_think
            writer_output_str, thinker_output_str, token_times, eos_generated = \
                solver.solve(problem, budget=args.budget)
        _judge_and_save(idx, writer_output_str, thinker_output_str, token_times, eos_generated)

    def _drain_batch_and_save(batch_indices, *, allow_retry: bool = True):
        """Try to solve `batch_indices` as one batch. Returns the list of indices
        that OOM'd (still pending). On OOM and allow_retry=True the caller is
        expected to retry each one at B=1 at the end of the run."""
        if not batch_indices:
            return []
        # Skip ones that have results already (resume support).
        live = [idx for idx in batch_indices
                if not os.path.exists(f"{exp_dir_path}/sample_{idx}.json")]
        if not live:
            return []
        results = _solve_async_inputs_batch(live)
        if results is None:
            # OOM
            if allow_retry:
                logger.warning(f"OOM on batch of {len(live)} — deferring {live}")
                return list(live)
            logger.error(f"OOM on retry at B=1 — giving up on {live}")
            return list(live)
        # _judge_and_save now writes the sample even when the judge fails, so we just
        # call it per sample. Any leftover OOM at the post-decode stage (rare) is
        # treated as a decode-side failure and deferred.
        try:
            for idx, (writer, thinker, ttimes, eos) in zip(live, results):
                _judge_and_save(idx, writer, thinker, ttimes, eos)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            unsaved = [j for j in live if not os.path.exists(f"{exp_dir_path}/sample_{j}.json")]
            if allow_retry and unsaved:
                logger.warning(f"OOM during save loop — deferring {unsaved}")
                return unsaved
            if unsaved:
                logger.error(f"OOM during save loop on retry — giving up on {unsaved}")
            return unsaved
        return []

    # --- iteration ---
    if args.start is not None and args.end is not None:
        idx_iter = range(args.start, args.end)
        logger.info(f'Generating tasks [{args.start}; {args.end})')
    elif args.queue is not None:
        idx_iter = TaskQueue.iterate_tasks_from_queue(endpoint=args.queue)
        logger.info(f'Generating tasks from {args.queue}')
    else:
        raise NotImplementedError("Please specify either --queue or both --start and --end")

    if args.mode == "async_inputs":
        # Batched path with deferred-OOM retry.
        deferred: list[int] = []
        batch_buf: list[int] = []
        progress = tqdm(desc=f'Process {rank}')
        for idx in idx_iter:
            if os.path.exists(f"{exp_dir_path}/sample_{idx}.json"):
                progress.update(1)
                continue
            batch_buf.append(idx)
            if len(batch_buf) >= args.batch_size:
                deferred.extend(_drain_batch_and_save(batch_buf))
                progress.update(len(batch_buf))
                batch_buf = []
        if batch_buf:
            deferred.extend(_drain_batch_and_save(batch_buf))
            progress.update(len(batch_buf))
            batch_buf = []
        # Retry OOM-deferred samples one at a time.
        if deferred:
            logger.info(f'Retrying {len(deferred)} OOM-deferred sample(s) at B=1')
            for idx in list(deferred):
                still_pending = _drain_batch_and_save([idx], allow_retry=False)
                if not still_pending:
                    deferred.remove(idx)
        if deferred:
            logger.error(f'{len(deferred)} sample(s) failed even at B=1: {deferred}')
        progress.close()
    else:
        for idx in tqdm(idx_iter, desc=f'Process {rank}'):
            _solve_task_and_save(idx)
    logger.info(f'Process {rank} has finished.')

if __name__ == "__main__":
    main()
