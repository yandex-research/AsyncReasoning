#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import eval_delay
import IPython.display as ipd
import argparse

import torch
import transformers
from hogwild.attention import model_surgery
from IPython.display import display, Markdown, clear_output
from typing import Callable, Optional, Sequence, TypeVar, Dict, Any, Protocol
T = TypeVar("T")

from datasets import load_dataset
from tqdm import tqdm
import pickle
import json

import traceback

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo_split.log', encoding='utf-8', level=logging.DEBUG)

from async_reasoning_prompting import AsyncReasoningPrompting
from async_reasoning_cache_fast_kernels import State, AsyncReasoningCacheFastKernels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_from", type=str, required=True, help="Split of math-500 from:")
    parser.add_argument("--split_to", type=str, required=True, help="Split of math-500 :to")
    # parser.add_argument("--gpu_idx", type=str, required=True, help="gpu idx")
    return parser.parse_args()

def find_last_valid_expression(
    response: str, prefix: str = "\\boxed{", extract_result: Callable[[str], str] = lambda x: x
) -> Optional[str]:
    """
    Find the last correct brace sequence that starts with prefix and passes extract_result; return it including prefix
    """
    while True:
        try:
            start = response.rindex(prefix)
            try:
                excerpt = parse_until_valid_brace_sequence(response[start:], keep_prefix=True)
                return extract_result(excerpt)
            except Exception:  # missing suffix or extract_result failed
                response = response[:start]
        except ValueError:
            return None


def parse_until_valid_brace_sequence(text: str, start: int = 0, end: Optional[int] = None, keep_prefix: bool = False) -> str:
    original_start = start
    start = text.index('{', start)
    balance = 1
    for i in range(start + 1, end if end is not None else len(text)):
        if text[i] == '{':
            balance += 1
        elif text[i] == '}':
            balance -= 1
        if balance == 0:
            return text[(original_start if keep_prefix else start): i + 1]
    raise ValueError("text does not have a correct bracket {/} ")


def main():
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
    # os.environ["HF_HOME"] = "/mnt/LLM"
    # os.environ["OMP_NUM_THREADS"] = "16"

    split_from, split_to = int(args.split_from), int(args.split_to)

    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("HF_HOME:", os.environ["HF_HOME"])
    print("OMP_NUM_THREADS:", os.environ["OMP_NUM_THREADS"])
    run_eval(split_from, split_to)

def run_eval(split_from, split_to):
    MODEL_NAME = "Qwen/Qwen3-32B"  # for 48GB gpus, use "Qwen/Qwen3-32B-AWQ" instead
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device,
        # Use eager PyTorch attention implementation for debugging
        # attn_implementation="eager"
    )

    forbidden_token_ix = [tokenizer.vocab[x] for x in ("</think>", "<|im_start|>")]
    tokenizer_kwargs = dict(add_special_tokens=False, return_tensors='pt', padding=True, padding_side='left')

    model_surgery(model)
    print(model)

    def compare_with_same_model(response, answer):
        return response == answer # TODO
        assert False, "It does not work with fast kernel"
        next_inputs = tokenizer(prompting.judge_prompt, **tokenizer_kwargs).to(device)
        logits = model(**next_inputs, use_cache=False).logits[..., -1, :]
        probs = logits.softmax(-1)

        yes_id = tokenizer(prompting.yes_token, **tokenizer_kwargs)["input_ids"].item()
        no_id  = tokenizer(prompting.no_token, **tokenizer_kwargs)["input_ids"].item()

        is_equal = (probs[..., yes_id] > probs[..., no_id]).item()
        return is_equal


    @torch.inference_mode()
    def check_if_should_continue_writing(cache: AsyncReasoningCacheFastKernels, use_trimming=False) -> bool:
        if use_trimming:
            # Trim cache instead of clearing
            cache.thinker_question.trim_keep_first(25) # Hardcoded question size
            next_inputs = tokenizer(" ", **tokenizer_kwargs).to(device)
        else:
            # Or clear and repopulate cache
            cache.thinker_question.crop(0)
            next_inputs = tokenizer(prompting.thinker_control_question, **tokenizer_kwargs).to(device)

        logits = model(**cache.cm_thinker_control.get_input_kwargs(**next_inputs)).logits[..., -1, :]
        logits[..., forbidden_token_ix] -= 100
        
        probs = logits.softmax(-1)  # TODO support more yes/no variants
        # Remove spaces
        yes_id = tokenizer(prompting.yes_token, **tokenizer_kwargs)["input_ids"].item()
        no_id  = tokenizer(prompting.no_token, **tokenizer_kwargs)["input_ids"].item()
        
        should_continue_writing = (probs[..., yes_id] > probs[..., no_id]).item()
        logger.debug(f'control: should continue writing? {should_continue_writing}')
        return should_continue_writing

    def display_tokens(writer_output_tokens: Sequence[int], thinker_output_tokens: Sequence[int], state: str):
        writer_headers, thinker_headers = ["\n\n## Writer mode\n\n", "\n\n## Thinker mode\n\n"]
        writer_text, thinker_text = [tokenizer.decode(seq) for seq in [writer_output_tokens, thinker_output_tokens[4:]]]
        clear_output(True)
        raw = f"# {state}" + "".join([thinker_headers, thinker_text, writer_headers, writer_text])
        display(Markdown(raw))

    def is_end_of_step(seq: Sequence[int]) -> bool:
        last_two_tokens = tokenizer.decode(seq[-2:])
        return last_two_tokens.endswith("\n\n")

    def async_reasoning_generate(prompting):
        token_times = []

        writer_output_tokens = tokenizer.encode(prompting.writer_output_prefix, **tokenizer_kwargs).flatten().tolist()
        thinker_output_tokens = tokenizer.encode(prompting.thinker_output_prefix, **tokenizer_kwargs).flatten().tolist()

        writer_output_tokens.append(tokenizer.encode("\n\n", **tokenizer_kwargs).item())
        thinker_output_tokens.append(tokenizer.encode("\n\n", **tokenizer_kwargs).item())
        eos_generated = False
        cache = AsyncReasoningCacheFastKernels(model, tokenizer, prompting, tokenizer_kwargs=tokenizer_kwargs, starting_state=State.thinker_only)
        with torch.inference_mode():
            for step in range(1024):
                t0 = time.perf_counter()
                if cache.state == State.thinker_only:
                    next_inputs = {"input_ids": torch.tensor([thinker_output_tokens[-1:]], device=device)}
                    logits = model(**cache.get_input_kwargs(**next_inputs)).logits[..., -1, :]
                    logits[..., forbidden_token_ix] -= 100
                    thinker_output_tokens.append(int(logits.argmax(-1)))

                elif cache.state == State.thinker_and_writer:
                    next_inputs = {"input_ids": torch.tensor([writer_output_tokens[-1:], thinker_output_tokens[-1:]], device=device)}
                    input_kwargs = cache.get_input_kwargs(**next_inputs)
                    logger.debug(f"input_kwargs: {input_kwargs}")
                    logits = model(**input_kwargs).logits[..., -1, :]
                    logits[..., forbidden_token_ix] -= 100
                    writer_next_token, thinker_next_token = logits.argmax(-1)
                    writer_output_tokens.append(int(writer_next_token))
                    thinker_output_tokens.append(int(thinker_next_token))
                    t1 = time.perf_counter()
                    token_times.append((tokenizer.decode(writer_next_token.item()), t1 - t0))
                    if is_end_of_step(writer_output_tokens):  # wait for the thinker's signal to continue
                        cache.state = State.thinker_only
                else:
                    raise ValueError(f"Unexpected state {cache.state}")

                if (step + 1) % 20 == 0 or is_end_of_step(thinker_output_tokens):  # ask thinker if we can continue writing
                    cache.state = State.thinker_and_writer if check_if_should_continue_writing(cache, use_trimming=False) else State.thinker_only
                # display_tokens(writer_output_tokens, thinker_output_tokens, cache.state)
                if writer_output_tokens[-1] == tokenizer.eos_token_id:
                    # print("EOS GENERATED, IMA TEMINATE NOW")
                    eos_generated = True
                    break
        return writer_output_tokens, thinker_output_tokens, token_times, eos_generated

    dataset_math = load_dataset('HuggingFaceH4/MATH-500', cache_dir="math-500")

    measured_delays_over_dataset = []
    os.makedirs(f"evals/math-500/math-500_split_{split_from}-{split_to}", exist_ok=True)
    evaluator = eval_delay.TTSEvaluator()
    for idx, (instruction, answer) in tqdm(enumerate(
            zip(dataset_math["test"]["problem"], dataset_math["test"]["answer"])
        )):
        if idx < split_from or idx >= split_to:
            continue
        problem = f"{instruction}. Please provide the final answer in \\boxed{{ }}"
        prompting = AsyncReasoningPrompting(problem)
        writer_output_tokens, thinker_output_tokens, token_times, eos_generated = async_reasoning_generate(prompting)
        result = {
            "idx": idx,
            "is_equal": None,
            "total_delay": None,
            "delays": None,
            "eos_generated": eos_generated,
            "response_answers": None,
            "correct_answer": answer,
            "error": None,
        }
        try:
            chunks, audio = evaluator.get_chunks_with_tts(token_times, k_chunks=5, return_audio=True)
            metrics = evaluator(**chunks, add_tts_in_parrallel=True)
            detokenized = tokenizer.decode(writer_output_tokens)
            response = find_last_valid_expression(detokenized)
            if response is None:
                response = detokenized
                is_equal = None
            else:
                is_equal = compare_with_same_model(response, answer)
            result.update({
                "is_equal": is_equal,
                "total_delay": float(metrics["total_delay"]),
                "delays": list(metrics["delays"]),
                "response_answers": response,
            })
        except Exception as e:
            traceback_str = traceback.format_exc()
            logger.error(f"Exception on sample {idx}: {traceback_str}")
            result.update({"error": traceback_str})

        measured_delays_over_dataset.append(result)
        with open(f"evals/math-500/math-500_split_{split_from}-{split_to}/{idx}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f'>>> Total delay: {result["total_delay"]}, {eos_generated=}, Answers: {result["response_answers"]} =?= {result["correct_answer"]}')

    with open(f"evals/math-500/math-500_split_{split_from}-{split_to}.pkl", "wb") as f:
        pickle.dump(measured_delays_over_dataset, f)


if __name__ == "__main__":
    main()