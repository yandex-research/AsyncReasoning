import os
from pathlib import Path
import argparse
import torch
import transformers

from evals.tts_evaluator import TTSEvaluator

from utils.answer_processing import find_last_valid_expression, check_equality_judge, check_equality_local_model

from datasets import load_dataset
from tqdm import tqdm
import pickle
import json

import traceback

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo_split.log', encoding='utf-8', level=logging.DEBUG)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["async_reasoning", "baseline_think", "baseline_no_think"],
        help="Select reasoning mode",
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B", help="Model name from hf")
    parser.add_argument("--split-from", type=int, required=True, help="Split of math-500 from:")
    parser.add_argument("--split-to", type=int, required=True, help="Split of math-500 :to")
    parser.add_argument("--budget", type=int, default=16384, help="Budget to eval on")
    parser.add_argument("--use-slow-kernel", action="store_true", default=False, help="Disable fast kernel")
    parser.add_argument("--use-local-judge", action="store_true", default=False, help="Use the same model as a judge for result.")
    parser.add_argument("--path-to-results", type=str, help="path to store exp results", default="./eval_results/math-500")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Args:", *[f"{k}: {v}" for k, v in vars(args).items()], sep="\n")

    mode = args.mode
    split_from, split_to = args.split_from, args.split_to
    use_fast_kernel = not args.use_slow_kernel
    use_api_not_local = not args.use_local_judge
    assert use_api_not_local or not use_fast_kernel, "You cannot use local model with kernel as a judge"

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
        forbidden_token_ix = [tokenizer.vocab[x] for x in ("</think>", "<|im_start|>", "SYSTEM")]
        solver_kwargs.update({
            "forbidden_token_ix": forbidden_token_ix,
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
    dataset_math = load_dataset('HuggingFaceH4/MATH-500')

    exp_dir_path = f"{args.path_to_results}/{mode}_math-500_split_{split_from}-{split_to}"
    os.makedirs(exp_dir_path, exist_ok=True)

    measured_delays_over_dataset = []
    evaluator = TTSEvaluator()
    for idx, (instruction, answer) in tqdm(enumerate(
            zip(dataset_math["test"]["problem"], dataset_math["test"]["answer"])
        )):
        if idx < split_from or idx >= split_to:
            continue
        
        problem = f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{instruction}"

        writer_output_str, thinker_output_str, token_times, eos_generated = \
            solver.solve(problem, budget=args.budget)
        response = find_last_valid_expression(writer_output_str, extract_result=lambda x: x[7:-1])
    
        if use_api_not_local:
            is_equal = check_equality_judge(response, answer)
        else:
            is_equal = check_equality_local_model(response, answer)

        if not token_times:
            # If writer output is empty
            metrics = None
        else:
            chunks = evaluator.get_chunks_with_tts(token_times[:-1] if eos_generated else token_times, k_chunks=5, return_audio=False)
            metrics = evaluator(**chunks, add_tts_in_parrallel=True, return_delays=False)

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
        measured_delays_over_dataset.append(result)
        with open(f"{exp_dir_path}/sample_{idx}.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f'>>> {eos_generated=}, Total delay: {metrics["total_delay"] if metrics else None}')
    with open(f"{exp_dir_path}/all_results.pkl", "wb") as f:
        pickle.dump(measured_delays_over_dataset, f)


if __name__ == "__main__":
    main()
