import sys; 
sys.path.insert(0, __file__.rsplit("/", 2)[0])
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/utils")

import os
import json
import random
import argparse

import torch
import transformers
from tqdm import tqdm
from datasets import load_dataset

from tts_evaluator import TTSEvaluator
from utils.answer_processing import find_last_valid_expression
from utils.gpu_parallel import get_worker_rank, init_worker_logger
from utils.task_queue import TaskQueue

if "NV_YT_OPERATION_ID" in os.environ:
    import nirvana_dl

# GPQA has a Correct Answer and 3 incorrect options
CHOICES = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]


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
        choices=["async_reasoning", "baseline_think", "baseline_no_think"],
        help="Select reasoning mode",
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B", help="Model name from hf")
    parser.add_argument("--budget", type=int, default=16384, help="Budget to eval on")
    parser.add_argument("--use-slow-kernel", action="store_true", default=False, help="Disable fast kernel")
    parser.add_argument("--path-to-results", type=str, help="path to store exp results", default="./eval_results/gpqa-diamond")
    parser.add_argument("--dump_snapshot_freq", type=int, default=4, help="yandex-internal snapshotting frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for option shuffling")
    return parser.parse_args()


def main():
    args = parse_args()
    rank = get_worker_rank()
    logger = init_worker_logger()
    logger.info(f'The script was run in the following way:')
    logger.info(f"python {__file__} \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    mode = args.mode
    use_fast_kernel = not args.use_slow_kernel

    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("HF_HOME:", os.environ["HF_HOME"])
    print("OMP_NUM_THREADS:", os.environ["OMP_NUM_THREADS"])
    
    model_name = args.model_name
    assert model_name == "Qwen/Qwen3-32B", "We are yet to support forbidden token ids for other models"\
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device,
    )

    solver_kwargs = {}
    if mode in ["async_reasoning"]:
        from async_reasoning.solver import AsyncReasoningSolver as Solver
        system_tokens = [key for key in tokenizer.vocab.keys() if key.endswith("SYSTEM") or key.endswith("SYSTEM:")]
        writer_forbidden_token_ix = [tokenizer.vocab[x] for x in ["</think>", "<|im_start|>", "<|endoftext|>"] + system_tokens]
        thinker_forbidden_token_ix = [tokenizer.vocab[x] for x in ["</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"] + system_tokens]
        solver_kwargs.update({
            "writer_forbidden_token_ix": writer_forbidden_token_ix,
            "thinker_forbidden_token_ix": thinker_forbidden_token_ix,
            "use_fast_kernel": use_fast_kernel,
        })
    elif mode in ["baseline_think", "baseline_no_think"]:
        from evals.baseline_solver import BaselineSolver as Solver
        solver_kwargs.update({
            "thinker_enabled": (mode == "baseline_think"),
        })
    else:
        raise ValueError("unsupported mode")

    solver = Solver(model, tokenizer, **solver_kwargs)
    dataset_gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    accuracy_numerator = accuracy_denominator = 0
    exp_dir_path = f"{args.path_to_results}/gpqa-diamond/{args.mode}"
    os.makedirs(exp_dir_path, exist_ok=True)
    evaluator = TTSEvaluator()

    def _solve_task_and_save(idx: int):
        save_path = f"{exp_dir_path}/sample_{idx}.json"
        if os.path.exists(save_path):
            return  # already solved by previous run and saved in snapshot

        nonlocal accuracy_numerator, accuracy_denominator
        sample = dataset_gpqa[idx]

        system_prompt = (
            "Please reason step by step, and put your final answer within \\boxed{} "
            "using ONLY the letter (A, B, C, or D). Your final boxed answer must "
            "contain exactly one letter and nothing else.\n\n"
        )

        # We permute the choices to circumvent the potential position bias
        choices_order = [0, 1, 2, 3]
        # Without updating the seed, choices_order is fixed
        random.seed(args.seed + idx)
        random.shuffle(choices_order)
        choices = [CHOICES[i] for i in choices_order]
        gt_id = choices_order.index(0)
        answer = ["A", "B", "C", "D"][gt_id]

        problem = (
            system_prompt +
            f"Question: {sample['Question'].strip()}\n\n"
            f"Choices:\n"
            f"(A) {sample[choices[0]].strip()}\n"
            f"(B) {sample[choices[1]].strip()}\n"
            f"(C) {sample[choices[2]].strip()}\n"
            f"(D) {sample[choices[3]].strip()}"
        )

        writer_output_str, thinker_output_str, token_times, eos_generated = \
            solver.solve(problem, budget=args.budget)
        response = find_last_valid_expression(writer_output_str, extract_result=lambda x: x[7:-1])
        assert len(token_times) > 0

        is_equal = response.strip() == answer.strip() if response else False

        chunks = evaluator.get_chunks_with_tts(token_times[:-1] if eos_generated else token_times, k_chunks=5, return_audio=False)
        metrics = evaluator(**chunks, add_tts_in_parrallel=True, return_delays=False)
        total_delay = metrics["total_delay"]
        result = {
            "idx": idx,
            "is_equal": is_equal,
            "metrics": metrics,
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
        print(end=f'[{rank=}] {idx=}, {eos_generated=}, {is_equal=}, {total_delay=:.3f}\t| {current_accuracy=:.3f}', file=sys.stderr)
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
