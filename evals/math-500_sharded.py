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
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="sharded math500 dataset path for load_from_disk")
    parser.add_argument("--path-to-results", type=str, help="path to store exp results", default="./eval_results/math-500")
    parser.add_argument("--dump_snapshot_freq", type=int, default=4, help="yandex-internal snapshotting frequency")
    parser.add_argument("--next_shard_every_steps", type=int, help="Setting to set up shards appearance frequency. Exceptions are: 0 -- concat, -1 -- never supply the rest of the shards.")
    parser.add_argument(
        "--shard_to_target",
        nargs="+",
        choices=["thinker", "writer", "input"],
        default=None,
        help='Where to share live context. Use: --shard_to_target input | thinker | writer',
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
        # decode-step count given by --next_shard_every_steps. Only shard_to_target
        # == ['input'] is meaningful here.
        assert args.shard_to_target is None or args.shard_to_target == ["input"], \
            "async_inputs supports only --shard_to_target input (or omit)"
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

    def _solve_async_inputs(problem, shard_2_text):
        """Single-cache decode with mid-stream KV insertion. Returns the same
        (writer_str, thinker_str, token_times, eos) shape as the AR solver does
        so the downstream judging/result-saving code is identical.

        Trigger semantics — when --next_shard_every_steps == 0, both shards are
        already concatenated in `instruction` (handled by caller), and no insertion
        happens. When > 0, shard_2 is spliced at that many decoded tokens (or just
        after a "\\n\\n" boundary, if --shard_wait_step is set). When < 0, no
        insertion at all.
        """
        from async_reasoning.async_kv_insert import insert_async_input
        import time as _time

        device = next(model.parameters()).device
        eos_id = tokenizer.eos_token_id

        # Prefill the prompt.
        ids = tokenizer(problem, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)

        # Pre-tokenize shard_2 once (with the same framing the async hook uses).
        do_insert = args.next_shard_every_steps is not None and args.next_shard_every_steps > 0
        if do_insert:
            shard2_text_wrapped = f"\n\nADDITIONAL USER INPUT: {shard_2_text}\n\n"
            shard2_ids = tokenizer(shard2_text_wrapped, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        inserted = False

        writer_chars = ""
        token_times = []
        t0 = _time.time()
        eos_generated = False

        with torch.inference_mode():
            for step in range(args.budget):
                tok_str = tokenizer.decode([int(next_token.item())], skip_special_tokens=False)
                t_now = _time.time() - t0
                token_times.append((tok_str, t_now, len(writer_chars)))
                writer_chars += tok_str

                if int(next_token.item()) == eos_id:
                    eos_generated = True
                    break

                # Decide whether to splice shard_2 in BEFORE the next forward.
                if do_insert and not inserted and (step + 1) >= args.next_shard_every_steps:
                    boundary_ok = (not args.shard_wait_step) or writer_chars.endswith("\n\n")
                    if boundary_ok:
                        cache = insert_async_input(model, cache, shard2_ids, position=cache.get_seq_length())
                        inserted = True

                cache_pos = torch.tensor([cache.get_seq_length()], device=device)
                out = model(
                    input_ids=next_token,
                    past_key_values=cache,
                    cache_position=cache_pos,
                    use_cache=True,
                )
                cache = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)

        return writer_chars, "", token_times, eos_generated

    def _solve_task_and_save(idx: int):
        save_path = f"{exp_dir_path}/sample_{idx}.json"
        if os.path.exists(save_path):
            return  # already solved by previous run and saved in snapshot

        nonlocal accuracy_numerator, accuracy_denominator
        problem_shards = dataset_math[idx]['problem_shards']
        answer = str(dataset_math[idx]['answer'])
        assert len(problem_shards) == 2, f"Unexpected number of shards on id: {idx}, {len(problem_shards)}"
        instruction = "".join(problem_shards) if args.next_shard_every_steps == 0 else problem_shards[0]
        problem = f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{instruction}"

        if args.mode == "async_inputs":
            writer_output_str, thinker_output_str, token_times, eos_generated = \
                _solve_async_inputs(problem, problem_shards[1])
        elif args.mode == "async_reasoning":
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
        response = find_last_valid_expression(writer_output_str, extract_result=lambda x: x[7:-1])
        assert len(token_times) > 0

        if args.use_local_judge:
            is_equal = check_equality_local_model(model, tokenizer, response, answer)
        else:
            is_equal = check_equality_judge(response, answer)

        # tortoise-tts dep disabled — no TTS-based delay metric
        # chunks = evaluator.get_chunks_with_tts(token_times[:-1] if eos_generated else token_times, k_chunks=5, return_audio=False)
        # metrics = evaluator(**chunks, add_tts_in_parrallel=True, return_delays=False)
        # total_delay = metrics["total_delay"]
        result = {
            "idx": idx,
            "is_equal": is_equal,
            # "metrics": metrics,  # tortoise-tts dep disabled
            "token_times": token_times,
            "eos_generated": eos_generated,
            "response_answers": response,
            "correct_answer": answer,
            "writer_response": writer_output_str,
            "thinker_response": thinker_output_str,
        }
        accuracy_numerator += int(is_equal)
        accuracy_denominator += 1
        current_accuracy = (accuracy_numerator / accuracy_denominator)
        print(end=f'[{rank=}] {idx=}, {eos_generated=}, {is_equal=}\t| {current_accuracy=:.3f}', file=sys.stderr)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        if "NV_YT_OPERATION_ID" in os.environ and rank == 0 and (
                accuracy_denominator % args.dump_snapshot_freq == args.dump_snapshot_freq - 1):
            nirvana_dl.snapshot.dump_snapshot()
            logger.info("Dumped Nirvana snapshot")

    if args.start is not None and args.end is not None:
        logger.info(f'Generating tasks [{args.start}; {args.end})')
        for idx in tqdm(range(args.start, args.end), desc=f'Process {rank}'):
            _solve_task_and_save(idx)
    elif args.queue is not None:
        logger.info(f'Generating tasks from {args.queue}')
        for idx in tqdm(TaskQueue.iterate_tasks_from_queue(endpoint=args.queue), desc=f"Process {rank}"):
            _solve_task_and_save(idx)
    else:
        raise NotImplementedError("Please specify either --queue or both --start and --end")
    logger.info(f'Process {rank} has finished.')

if __name__ == "__main__":
    main()
