import os
import time
import torch
import warnings
import transformers
from IPython.display import display, Markdown, clear_output
from typing import Sequence, Union, Callable, Optional

from async_reasoning.prompting import AsyncReasoningPrompting
from async_reasoning.cache import State, AsyncReasoningCache

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    logits: (..., vocab)
    returns: (...) token ids
    Greedy if temperature <= 0 (default).
    """
    if temperature is None or temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    # Top-k filtering
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        # threshold is the k-th largest logit
        kth_vals = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, -float("inf")), logits)

    # Top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)

        # mask tokens with cumulative prob above top_p (keep at least 1 token)
        sorted_mask = cumprobs > top_p
        sorted_mask[..., 0] = False
        # shift mask right to keep the first token that exceeds p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()

        sorted_logits = torch.where(sorted_mask, torch.full_like(sorted_logits, -float("inf")), sorted_logits)

        # scatter back to original order
        filtered_logits = torch.full_like(logits, -float("inf"))
        filtered_logits.scatter_(-1, sorted_idx, sorted_logits)
        logits = filtered_logits

    probs = torch.softmax(logits, dim=-1)
    # multinomial expects 2D; flatten batch dims safely
    orig_shape = probs.shape[:-1]
    probs_2d = probs.reshape(-1, probs.size(-1))
    next_ids = torch.multinomial(probs_2d, num_samples=1).squeeze(-1)
    return next_ids.reshape(orig_shape)


class AsyncReasoningSolver:
    def __init__(self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        forbidden_token_ix: Sequence[int] = [],
        thinker_forbidden_token_ix: Sequence[int] = [],
        writer_forbidden_token_ix: Sequence[int] = [],
        end_of_think_token_ix: Sequence[int] = [],
        use_fast_kernel: bool = True,
        use_torch_compile: bool = None,
    ):
        if use_torch_compile is None:
            use_torch_compile = bool(int(os.environ.get("USE_TORCH_COMPILE", use_fast_kernel)))
        if use_fast_kernel:
            from async_reasoning.cache_fast_kernels import AsyncReasoningCacheFastKernels
            from hogwild.attention import model_surgery
            model_surgery(model)
            self.Cache = AsyncReasoningCacheFastKernels
        else:
            self.Cache = AsyncReasoningCache
        if use_torch_compile:
            model = torch.compile(model)
        if forbidden_token_ix:
            assert not (thinker_forbidden_token_ix or writer_forbidden_token_ix)
            thinker_forbidden_token_ix = writer_forbidden_token_ix = forbidden_token_ix
            warnings.warn("forbidden_token_ix is deprecated, use separate thinker_/writer_forbidden_token_ix")

        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = dict(add_special_tokens=False, return_tensors='pt', padding=True, padding_side='left')
        self.thinker_forbidden_token_ix, self.writer_forbidden_token_ix = thinker_forbidden_token_ix, writer_forbidden_token_ix
        self.end_of_think_token_ix = end_of_think_token_ix
        self.use_fast_kernel = use_fast_kernel

    @torch.inference_mode()
    def check_if_should_continue_writing(self,
        cache: Union['AsyncReasoningCache', 'AsyncReasoningCacheFastKernels'], prompting: AsyncReasoningPrompting
     ) -> bool:
        # оставляем greedy, чтобы контроль был детерминированным
        if self.use_fast_kernel:
            cache.mode_switching_question.crop(0)
        else:
            cache.mode_switching_question.clear()
        next_inputs = self.tokenizer(prompting.mode_switching_question, **self.tokenizer_kwargs).to(self.device)

        logits = self.model(**cache.cm_mode_switching.get_input_kwargs(**next_inputs)).logits[..., -1, :]
        probs = logits.softmax(-1)
        yes_id = self.tokenizer(prompting.yes_token, **self.tokenizer_kwargs)["input_ids"].item()
        no_id  = self.tokenizer(prompting.no_token, **self.tokenizer_kwargs)["input_ids"].item()
        
        should_continue_writing = (probs[..., yes_id] > probs[..., no_id]).item()
        logger.debug(f'control: should continue writing? {should_continue_writing}')
        return should_continue_writing

    def display_tokens(self,
        writer_output_tokens: Sequence[int], 
        thinker_output_tokens: Sequence[int], 
        state: State,
        ):
        writer_headers, thinker_headers = ["\n\n## Writer mode\n\n", "\n\n## Thinker mode\n\n"]
        writer_text, thinker_text = [self.tokenizer.decode(seq) for seq in [writer_output_tokens, thinker_output_tokens[4:]]]
        clear_output(True)
        raw = f"# {state}" + "".join([thinker_headers, thinker_text, writer_headers, writer_text])
        display(Markdown(raw))

    def is_end_of_step(self, seq: Sequence[int]) -> bool:
        last_two_tokens = self.tokenizer.decode(seq[-2:])
        return last_two_tokens.endswith("\n\n")

    def solve(
        self,
        problem: str,
        display_generation_in_real_time: bool = False,
        budget: int = 1024,
        temperature: float = 0.0,  # 0.0 -> greedy (by default)
        top_p: float = 0.95,        # 1.0 -> отключено
        top_k: int = 20,            # 0 -> отключено
        on_new_tokens_generated: Optional[
            Callable[
                [Sequence[int], Sequence[int], tuple[str, float, int], bool, State],
                None,
            ]
        ] = None,
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
            starting_time = time.perf_counter()
            for step in range(budget):
                if cache.state == State.thinker_only:
                    next_inputs = {"input_ids": torch.tensor([thinker_output_tokens[-1:]], device=self.device)}
                    logits = self.model(**cache.get_input_kwargs(**next_inputs)).logits[..., -1, :]
                    logits[..., self.thinker_forbidden_token_ix] -= 100

                    next_id = sample_from_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    thinker_output_tokens.append(int(next_id.item()))

                elif cache.state == State.writer_only:
                    next_inputs = {"input_ids": torch.tensor([writer_output_tokens[-1:]], device=self.device)}
                    logits = self.model(**cache.get_input_kwargs(**next_inputs)).logits[..., -1, :]
                    logits[..., self.writer_forbidden_token_ix] -= 100

                    writer_next_token = sample_from_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    writer_output_tokens.append(int(writer_next_token.item()))
                    token_times.append((self.tokenizer.decode(writer_next_token.item()), time.perf_counter() - starting_time, step))

                elif cache.state == State.thinker_and_writer:
                    next_inputs = {"input_ids": torch.tensor([thinker_output_tokens[-1:], writer_output_tokens[-1:]], device=self.device)}
                    logits = self.model(**cache.get_input_kwargs(**next_inputs)).logits[..., -1, :]

                    logits[0, ..., self.thinker_forbidden_token_ix] -= 100
                    logits[1, ..., self.writer_forbidden_token_ix] -= 100

                    next_ids = sample_from_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    thinker_next_token, writer_next_token = next_ids[0], next_ids[1]

                    thinker_output_tokens.append(int(thinker_next_token.item()))
                    writer_output_tokens.append(int(writer_next_token.item()))
                    token_times.append((self.tokenizer.decode(writer_next_token.item()), time.perf_counter() - starting_time, step))

                    if self.is_end_of_step(writer_output_tokens):  # wait for the thinker's signal to continue
                        cache.state = State.thinker_only
                else:
                    raise ValueError(f"Unexpected state {cache.state}")
                
                if cache.state != State.writer_only and thinker_output_tokens[-1] in self.end_of_think_token_ix:
                    cache.state = State.writer_only
                if cache.state != State.writer_only and ((step + 1) % 20 == 0 or self.is_end_of_step(thinker_output_tokens)):
                    cache.state = State.thinker_and_writer if self.check_if_should_continue_writing(cache, prompting) else State.thinker_only

                if display_generation_in_real_time:
                    self.display_tokens(writer_output_tokens, thinker_output_tokens, cache.state)
                if writer_output_tokens[-1] == self.tokenizer.eos_token_id:
                    eos_generated = True

                if on_new_tokens_generated is not None:
                    on_new_tokens_generated(
                        writer_output_tokens,
                        thinker_output_tokens,
                        token_times,
                        eos_generated,
                        cache.state,
                    )

                if eos_generated:
                    break

            if len(token_times) == 0:
                token_times.append(("EMPTY", time.perf_counter() - starting_time, step))

        writer_output_str = self.tokenizer.decode(writer_output_tokens)
        thinker_output_str = self.tokenizer.decode(thinker_output_tokens)
        return writer_output_str, thinker_output_str, token_times, eos_generated
