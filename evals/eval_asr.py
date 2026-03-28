import argparse
import json
import os
import random
import sys
import time

import datasets
import librosa
import whisper
from tqdm import tqdm

sys.path.insert(0, __file__.rsplit("/", 2)[0])
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/utils")

from tts_evaluator import TTSEvaluator
from utils.gpu_parallel import get_worker_rank, init_worker_logger
from utils.task_queue import TaskQueue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue",
        type=str,
        default=None,
        help="Endpoint for a zmq-like task dispenser that dispenses task indices. Provide *either* this or start & end",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First task to be processed by script inclusive. E.g --start 0 --end 100 will process tasks [0-99]",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last task to be processed by script exclusive. E.g --start 0 --end 100 will process tasks [0-99]",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["math-500", "gpqa-diamond", "mmlu-pro"],
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--path-to-results",
        type=str,
        help="path to store exp results",
        default="./eval_results",
    )
    parser.add_argument(
        "--asr-model-name", type=str, default="base", help="Whisper ASR model name"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="The size of subset used for evaluation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed used for subset sampling"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    rank = get_worker_rank()
    logger = init_worker_logger()
    logger.info(f"The script was run in the following way:")
    logger.info(
        f"python {__file__} \\\n"
        + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items())
    )

    logger.info(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"HF_HOME: {os.getenv('HF_HOME')}")
    logger.info(f"OMP_NUM_THREADS: os.getenv('OMP_NUM_THREADS')")

    if args.dataset == "math-500":
        dataset = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    elif args.dataset == "gpqa-diamond":
        dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    elif args.dataset == "mmlu-pro":
        dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.num_samples is not None:
        logger.info(f"Subsampling {args.num_samples} samples (seed={args.seed})")
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

    asr_model = whisper.load_model(args.asr_model_name, device="cuda")
    evaluator = TTSEvaluator()

    exp_dir_path = f"{args.path_to_results}/{args.dataset}"
    os.makedirs(exp_dir_path, exist_ok=True)

    def _make_prompt(sample):
        if args.dataset == "math-500":
            instruction = str(sample["problem"])
            return f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{instruction}"
        elif args.dataset == "gpqa-diamond":
            CHOICES = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]

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

            return (
                system_prompt +
                f"Question: {sample['Question'].strip()}\n\n"
                f"Choices:\n"
                f"(A) {sample[choices[0]].strip()}\n"
                f"(B) {sample[choices[1]].strip()}\n"
                f"(C) {sample[choices[2]].strip()}\n"
                f"(D) {sample[choices[3]].strip()}"
            )
        elif args.dataset == "mmlu-pro":
            ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            num_options = len(sample["options"])
            question = sample["question"].strip('\n')

            system_prompt = (
                "Please reason step by step, and put your final answer within \\boxed{} "
                f"using ONLY the letter ({', '.join(ALPHABET[:num_options])}). Your final boxed answer must "
                "contain exactly one letter and nothing else.\n\n"
            )

            return (
                system_prompt +
                f"Question: {question}\n\n"
                f"Choices:\n"
                "\n".join([f"({ALPHABET[i]}) {option}" for i, option in enumerate(sample['options'])])
            )
        else:
            assert False, "Unreachable"

    def _solve_task_and_save(idx: int):
        save_path = f"{exp_dir_path}/sample_{idx}.json"
        if os.path.exists(save_path):  
            return  # already solved by previous run and saved in snapshot

        prompt = _make_prompt(sample=dataset[idx])

        audio, sample_rate, _, _ = evaluator.get_audio_track([prompt])
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        start = time.perf_counter()
        transcribed_prompt = asr_model.transcribe(audio, language="en")["text"]
        end = time.perf_counter()
        asr_duration_sec = end - start
        result = {
            "idx": idx,
            "asr_duration_sec": asr_duration_sec,
            "source_prompt": prompt,
            "transcribed_prompt": transcribed_prompt,
        }
        print(end=f"[{rank=}] {idx=}, {asr_duration_sec=}", file=sys.stderr)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)

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
