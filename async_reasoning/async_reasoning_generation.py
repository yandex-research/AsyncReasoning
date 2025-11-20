import sys; sys.path.insert(0, "../.");

import time
import torch
import transformers
import shared_cache
from IPython.display import display, Markdown, clear_output
from typing import Sequence, Union, Tuple, Dict, Any

from async_reasoning.async_reasoning_prompting import AsyncReasoningPrompting
from async_reasoning.async_reasoning_cache import State, AsyncReasoningCache

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)

class AsyncReasoningSolver:
    def __init__(self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        forbidden_token_ix: Sequence[int] = [],
        use_fast_kernel: bool = True
    ):
        if use_fast_kernel:
            from async_reasoning.async_reasoning_cache_fast_kernels import AsyncReasoningCacheFastKernels
            from hogwild.attention import model_surgery
            model_surgery(model)
            self.Cache = AsyncReasoningCacheFastKernels
        else:
            self.Cache = AsyncReasoningCache

        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = dict(add_special_tokens=False, return_tensors='pt', padding=True, padding_side='left')
        self.forbidden_token_ix = forbidden_token_ix
        self.use_fast_kernel = use_fast_kernel

    @torch.inference_mode()
    def check_if_should_continue_writing(self,
        cache, # Union[AsyncReasoningCache, AsyncReasoningCacheFastKernels],
        prompting: AsyncReasoningPrompting,
        use_trimming=False) -> bool:
        if use_trimming:
            # Trim cache instead of clearing
            cache.thinker_question.trim_keep_first(25) # Hardcoded question size
            next_inputs = self.tokenizer(" ", **self.tokenizer_kwargs).to(self.device)
        else:
            # Or clear and repopulate cache
            if self.use_fast_kernel:
                cache.thinker_question.crop(0)
            else:
                cache.thinker_question.clear()
            next_inputs = self.tokenizer(prompting.thinker_control_question, **self.tokenizer_kwargs).to(self.device)

        logits = self.model(**cache.cm_thinker_control.get_input_kwargs(**next_inputs)).logits[..., -1, :]
        logits[..., self.forbidden_token_ix] -= 100
        
        probs = logits.softmax(-1)  # TODO support more yes/no variants
        # Remove spaces
        yes_id = self.tokenizer(prompting.yes_token, **self.tokenizer_kwargs)["input_ids"].item()
        no_id  = self.tokenizer(prompting.no_token, **self.tokenizer_kwargs)["input_ids"].item()
        
        should_continue_writing = (probs[..., yes_id] > probs[..., no_id]).item()
        logger.debug(f'control: should continue writing? {should_continue_writing}')
        return should_continue_writing

    def display_tokens(self,
        writer_output_tokens: Sequence[int], 
        thinker_output_tokens: Sequence[int], 
        state: str,
        ):
        writer_headers, thinker_headers = ["\n\n## Writer mode\n\n", "\n\n## Thinker mode\n\n"]
        writer_text, thinker_text = [self.tokenizer.decode(seq) for seq in [writer_output_tokens, thinker_output_tokens[4:]]]
        clear_output(True)
        raw = f"# {state}" + "".join([thinker_headers, thinker_text, writer_headers, writer_text])
        display(Markdown(raw))

    def is_end_of_step(self, seq: Sequence[int]) -> bool:
        last_two_tokens = self.tokenizer.decode(seq[-2:])
        return last_two_tokens.endswith("\n\n")

    def solve(self,
            problem: str,
            display_generation_in_real_time: bool = False,
            budget: int = 1024,
        ):
        
        prompting = AsyncReasoningPrompting(problem)

        token_times = []
        writer_output_tokens = self.tokenizer.encode(prompting.writer_output_prefix, **self.tokenizer_kwargs).flatten().tolist()
        thinker_output_tokens = self.tokenizer.encode(prompting.thinker_output_prefix, **self.tokenizer_kwargs).flatten().tolist()

        writer_output_tokens.append(self.tokenizer.encode("\n\n", **self.tokenizer_kwargs).item())
        thinker_output_tokens.append(self.tokenizer.encode("\n\n", **self.tokenizer_kwargs).item())
        eos_generated = False
        cache = self.Cache(self.model, self.tokenizer, prompting, tokenizer_kwargs=self.tokenizer_kwargs, starting_state=State.thinker_only)
        with torch.inference_mode():
            t0 = time.perf_counter()
            for step in range(budget):
                if cache.state == State.thinker_only:
                    next_inputs = {"input_ids": torch.tensor([thinker_output_tokens[-1:]], device=self.device)}
                    logits = self.model(**cache.get_input_kwargs(**next_inputs)).logits[..., -1, :]
                    logits[..., self.forbidden_token_ix] -= 100
                    thinker_output_tokens.append(int(logits.argmax(-1)))

                elif cache.state == State.thinker_and_writer:
                    next_inputs = {"input_ids": torch.tensor([writer_output_tokens[-1:], thinker_output_tokens[-1:]], device=self.device)}
                    input_kwargs = cache.get_input_kwargs(**next_inputs)
                    logger.debug(f"input_kwargs: {input_kwargs}")
                    logits = self.model(**input_kwargs).logits[..., -1, :]
                    logits[..., self.forbidden_token_ix] -= 100
                    writer_next_token, thinker_next_token = logits.argmax(-1)
                    writer_output_tokens.append(int(writer_next_token))
                    thinker_output_tokens.append(int(thinker_next_token))
                    t1 = time.perf_counter()
                    token_times.append((self.tokenizer.decode(writer_next_token.item()), t1 - t0, step))
                    if self.is_end_of_step(writer_output_tokens):  # wait for the thinker's signal to continue
                        cache.state = State.thinker_only
                else:
                    raise ValueError(f"Unexpected state {cache.state}")

                if (step + 1) % 20 == 0 or self.is_end_of_step(thinker_output_tokens):  # ask thinker if we can continue writing
                    cache.state = State.thinker_and_writer if self.check_if_should_continue_writing(cache, prompting, use_trimming=False) else State.thinker_only
                if display_generation_in_real_time:
                    self.display_tokens(writer_output_tokens, thinker_output_tokens, cache.state)
                if writer_output_tokens[-1] == self.tokenizer.eos_token_id:
                    eos_generated = True
                    break
        writer_output_str, thinker_output_str = self.tokenizer.decode(writer_output_tokens), self.tokenizer.decode(thinker_output_tokens)
        return writer_output_str, thinker_output_str, token_times, eos_generated
