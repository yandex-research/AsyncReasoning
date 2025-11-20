import sys; sys.path.insert(0, "../.");

import time
import torch
import transformers
from IPython.display import display, Markdown, clear_output
from typing import Sequence, Union, Tuple, Dict, Any

from async_reasoning.async_reasoning_prompting import AsyncReasoningPrompting

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)

class BaselineSolver:
    def __init__(self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        thinker_enabled: bool = True,  
        forbidden_token_ix: Sequence[int] = [],
    ):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = dict(add_special_tokens=False, return_tensors='pt', padding=True, padding_side='left')
        self.forbidden_token_ix = forbidden_token_ix
        self.thinker_enabled = thinker_enabled
        assert str(model.name_or_path).startswith("Qwen/Qwen3"), f"Support only Qwen3 for now, but {model.name_or_path} provided"


    def display_tokens(self,
        writer_output_tokens: Sequence[int], 
        thinker_output_tokens: Sequence[int], 
        ):
        thinker_text = self.tokenizer.decode(thinker_output_tokens)
        writer_text = self.tokenizer.decode(writer_output_tokens)

        clear_output(True)
        raw = f"{thinker_text}\n\n{writer_text}"
        display(Markdown(raw))

    def solve(self,
            problem: str,
            display_generation_in_real_time: bool = False,
            budget: int = 1024,
        ):

        prompt = problem
        if not self.thinker_enabled:
            prompt += "<think>\n\n</think>"
            finished_thinking = True
        input_ids = self.tokenizer.encode(prompt, **self.tokenizer_kwargs).flatten().tolist()

        token_times = []
        writer_output_tokens = []
        thinker_output_tokens = []

        finished_thinking = False
        eos_generated = False

        with torch.inference_mode():
            for step in range(budget):
                t0 = time.perf_counter()

                next_inputs = {"input_ids": torch.tensor(input_ids, device=self.device)}
                logits = self.model(**next_inputs).logits[..., -1, :]
                if len(self.forbidden_token_ix) > 0:
                    logits[..., self.forbidden_token_ix] -= 100
                next_token = int(logits.argmax(-1))
                if not finished_thinking:
                    thinker_output_tokens.append(next_token)
                    if next_token == tokenizer.vocab["</think>"]:
                        finished_thinking = True
                else:
                    writer_output_tokens.append(next_token)
                    t1 = time.perf_counter()
                    token_times.append((self.tokenizer.decode(next_token), t1 - t0, step))

                    if writer_output_tokens[-1] == self.tokenizer.eos_token_id:
                        eos_generated = True
                        break

                if display_generation_in_real_time:
                    self.display_tokens(writer_output_tokens, thinker_output_tokens, cache.state)
        writer_putput_str = self.tokenizer.decode(writer_output_tokens)
        thinker_output_str = self.tokenizer.decode(thinker_output_tokens)
        return writer_putput_str, thinker_output_str, token_times, eos_generated
