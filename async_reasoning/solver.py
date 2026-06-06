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

from utils.modeling import prepare_model_for_inference

logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)

class AsyncReasoningSolver:
    def __init__(self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        forbidden_token_ix: Sequence[int] = [],
        thinker_forbidden_token_ix: Sequence[int] = [],
        writer_forbidden_token_ix: Sequence[int] = [],
        end_of_think_token_ix: Sequence[int] = [],
        use_fast_kernel: bool = True,
        **kwargs
    ):
        if use_fast_kernel:
            from async_reasoning.cache_fast_kernels import AsyncReasoningCacheFastKernels
            from async_reasoning_inference.attention import model_surgery
            model_surgery(model)
            self.Cache = AsyncReasoningCacheFastKernels
        else:
            self.Cache = AsyncReasoningCache
            kwargs.setdefault("use_torch_compile", False)  # do not compile unless explicitly asked to
        model = prepare_model_for_inference(model, **kwargs)
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
        if self.use_fast_kernel:
            cache.mode_switching_question.crop(0)
        else:
            cache.mode_switching_question.clear()
        next_inputs = self.tokenizer(prompting.mode_switching_question, **self.tokenizer_kwargs).to(self.device)

        logits = self.model(**cache.cm_mode_switching.get_input_kwargs(**next_inputs)).logits[..., -1, :]

        # We compare the model's "yes" preference against its "no" preference. Two subtleties:
        # (a) BPE tokenizers usually represent `yes`/`no` as distinct single-token variants
        #     with and without a leading space. Depending on the surrounding text the model
        #     may favour either. Sum over all single-token variants we can resolve.
        # (b) Many models (e.g. Qwen3.5) put most of their mass on neither variant — they
        #     want to keep writing prose. Comparing absolute softmax probabilities is
        #     therefore noisy; instead, restrict to {yes-variants, no-variants} and compare
        #     within that restricted set via logsumexp.
        yes_ids, no_ids = self._yes_no_token_ids(prompting)
        yes_score = torch.logsumexp(logits[..., yes_ids], dim=-1)
        no_score = torch.logsumexp(logits[..., no_ids], dim=-1)
        should_continue_writing = bool((yes_score > no_score).item())
        logger.debug(
            f"control: yes_score={float(yes_score):.3f}  no_score={float(no_score):.3f}  "
            f"continue_writing={should_continue_writing}"
        )
        return should_continue_writing

    def _yes_no_token_ids(self, prompting: AsyncReasoningPrompting) -> tuple[list[int], list[int]]:
        """Collect all single-token variants of yes/no the tokenizer can produce (with and
        without a leading space, both casings). Cached on the instance."""
        if getattr(self, "_yes_no_cache", None) is None:
            yes_ids: list[int] = []
            no_ids: list[int] = []
            for yes_str, no_str in [
                (prompting.yes_token,       prompting.no_token),
                (" " + prompting.yes_token, " " + prompting.no_token),
                (prompting.yes_token.capitalize(),       prompting.no_token.capitalize()),
                (" " + prompting.yes_token.capitalize(), " " + prompting.no_token.capitalize()),
            ]:
                yi = self.tokenizer(yes_str, **self.tokenizer_kwargs)["input_ids"].flatten().tolist()
                ni = self.tokenizer(no_str, **self.tokenizer_kwargs)["input_ids"].flatten().tolist()
                if len(yi) == 1 and yi[0] not in yes_ids:
                    yes_ids.append(yi[0])
                if len(ni) == 1 and ni[0] not in no_ids:
                    no_ids.append(ni[0])
            assert yes_ids and no_ids, "tokenizer didn't produce single-token yes/no variants"
            self._yes_no_cache = (yes_ids, no_ids)
        return self._yes_no_cache

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

        # Starter tokens — chosen so the model's first-step context exactly matches the
        # corresponding Qwen chat template ending:
        #   thinker context after starter = `<think>\n`     (= apply_chat_template(enable_thinking=True))
        #   writer  context after starter = `</think>\n\n`  (= apply_chat_template(enable_thinking=False))
        writer_output_tokens.append(self.tokenizer.encode("\n\n", **self.tokenizer_kwargs).item())
        thinker_output_tokens.append(self.tokenizer.encode("\n", **self.tokenizer_kwargs).item())
        eos_generated = False
        # Start the timer BEFORE cache creation. The AR cache constructor does the multi-block
        # prefill, which is analogous to a baseline solver's prompt prefill inside model.generate.
        # Both must be included in TTFT so the comparison against baselines is fair.
        starting_time = time.perf_counter()
        cache = self.Cache(self.model, self.tokenizer, prompting, tokenizer_kwargs=self.tokenizer_kwargs, starting_state=State.thinker_only)
        with torch.inference_mode():
            for step in range(budget):
                if cache.state == State.thinker_only:
                    next_inputs = {"input_ids": torch.tensor([thinker_output_tokens[-1:]], device=self.device)}
                    logits = self.model(**cache.get_input_kwargs(**next_inputs)).logits[..., -1, :]
                    logits[..., self.thinker_forbidden_token_ix] -= 100
                    thinker_output_tokens.append(int(logits.argmax(-1)))

                elif cache.state == State.writer_only:
                    next_inputs = {"input_ids": torch.tensor([writer_output_tokens[-1:]], device=self.device)}
                    logits = self.model(**cache.get_input_kwargs(**next_inputs)).logits[..., -1, :]
                    logits[..., self.writer_forbidden_token_ix] -= 100
                    writer_next_token = logits.argmax(-1)
                    writer_output_tokens.append(int(writer_next_token))
                    token_times.append((self.tokenizer.decode(writer_next_token.item()), time.perf_counter() - starting_time, step))

                elif cache.state == State.thinker_and_writer:
                    next_inputs = {"input_ids": torch.tensor([thinker_output_tokens[-1:], writer_output_tokens[-1:]], device=self.device)}
                    logits = self.model(**cache.get_input_kwargs(**next_inputs)).logits[..., -1, :]
                    logits[0, ..., self.thinker_forbidden_token_ix] -= 100
                    logits[1, ..., self.writer_forbidden_token_ix] -= 100
                    thinker_next_token, writer_next_token = logits.argmax(-1)
                    thinker_output_tokens.append(int(thinker_next_token))
                    writer_output_tokens.append(int(writer_next_token))
                    token_times.append((self.tokenizer.decode(writer_next_token.item()), time.perf_counter() - starting_time, step))
                    if self.is_end_of_step(writer_output_tokens):  # wait for the thinker's signal to continue
                        cache.state = State.thinker_only
                else:
                    raise ValueError(f"Unexpected state {cache.state}")
                
                if cache.state != State.writer_only and thinker_output_tokens[-1] in self.end_of_think_token_ix:
                    cache.state = State.writer_only
                if cache.state != State.writer_only and ((step + 1) % 20 == 0 or self.is_end_of_step(thinker_output_tokens)):  # ask thinker if we can continue writing
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
        writer_output_str, thinker_output_str = self.tokenizer.decode(writer_output_tokens), self.tokenizer.decode(thinker_output_tokens)

        return writer_output_str, thinker_output_str, token_times, eos_generated
