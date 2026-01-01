import time
import torch
import warnings
import transformers
from IPython.display import display, Markdown, clear_output
from typing import Sequence, Union, Callable, Optional, List, Tuple
import queue

from async_reasoning.prompting import AsyncReasoningPrompting
from async_reasoning.cache import State, AsyncReasoningCache

import logging
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
        use_fast_kernel: bool = True
    ):
        if use_fast_kernel:
            from async_reasoning.cache_fast_kernels import AsyncReasoningCacheFastKernels
            from hogwild.attention import model_surgery
            model_surgery(model)
            self.Cache = AsyncReasoningCacheFastKernels
        else:
            self.Cache = AsyncReasoningCache
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
        cache: Union['AsyncReasoningCache', 'AsyncReasoningCacheFastKernels'],
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
        probs = logits.softmax(-1)
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

    def solve(
        self,
        problem: str,
        display_generation_in_real_time: bool = False,
        budget: int = 1024,
        on_new_tokens_generated: Optional[
            Callable[
                [Sequence[int], Sequence[int], tuple[str, float, int], bool, State, "LiveContextQueue"],
                None,
            ]
        ] = None,
        live_context_queue: Optional["LiveContextQueue"] = None,
    ):

        prompting = AsyncReasoningPrompting(problem)

        token_times = []
        writer_output_tokens = self.tokenizer.encode(prompting.writer_output_prefix, **self.tokenizer_kwargs).flatten().tolist()
        thinker_output_tokens = self.tokenizer.encode(prompting.thinker_output_prefix, **self.tokenizer_kwargs).flatten().tolist()

        writer_output_tokens.append(self.tokenizer.encode("\n\n", **self.tokenizer_kwargs).item())
        thinker_output_tokens.append(self.tokenizer.encode("\n\n", **self.tokenizer_kwargs).item())
        eos_generated = False
        cache = self.Cache(self.model, self.tokenizer, prompting, tokenizer_kwargs=self.tokenizer_kwargs, starting_state=State.thinker_only)
        pending_injections: List["QueuedInjection"] = []
        with torch.inference_mode():
            starting_time = time.perf_counter()
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
                    next_inputs = {"input_ids": torch.tensor([writer_output_tokens[-1:], thinker_output_tokens[-1:]], device=self.device)}
                    input_kwargs = cache.get_input_kwargs(**next_inputs)
                    logger.debug(f"input_kwargs: {input_kwargs}")
                    logits = self.model(**input_kwargs).logits[..., -1, :]
                    logits[0, ..., self.writer_forbidden_token_ix] -= 100
                    logits[1, ..., self.thinker_forbidden_token_ix] -= 100
                    writer_next_token, thinker_next_token = logits.argmax(-1)
                    writer_output_tokens.append(int(writer_next_token))
                    thinker_output_tokens.append(int(thinker_next_token))
                    token_times.append((self.tokenizer.decode(writer_next_token.item()), time.perf_counter() - starting_time, step))
                    if self.is_end_of_step(writer_output_tokens):  # wait for the thinker's signal to continue
                        cache.state = State.thinker_only
                else:
                    raise ValueError(f"Unexpected state {cache.state}")
                
                if cache.state != State.writer_only and thinker_output_tokens[-1] in self.end_of_think_token_ix:
                    cache.state = State.writer_only
                if cache.state != State.writer_only and ((step + 1) % 20 == 0 or self.is_end_of_step(thinker_output_tokens)):  # ask thinker if we can continue writing
                    cache.state = State.thinker_and_writer if self.check_if_should_continue_writing(cache, prompting, use_trimming=False) else State.thinker_only

                if display_generation_in_real_time:
                    self.display_tokens(writer_output_tokens, thinker_output_tokens, cache.state)
                if writer_output_tokens[-1] == self.tokenizer.eos_token_id:
                    eos_generated = True

                # Inject any user-provided context mid-generation
                if live_context_queue is not None:
                    pending_injections.extend(live_context_queue.pop_all())
                    if pending_injections:
                        pending_injections = self._apply_pending_injections(
                            pending_injections,
                            cache,
                            writer_output_tokens,
                            thinker_output_tokens,
                        )

                if on_new_tokens_generated is not None:
                    on_new_tokens_generated(
                        writer_output_tokens,
                        thinker_output_tokens,
                        token_times,
                        eos_generated,
                        cache.state,
                        live_context_queue,
                    )

                if eos_generated:
                    break
            if len(token_times) == 0:
                token_times.append(("EMPTY", time.perf_counter() - starting_time, step))
        writer_output_str, thinker_output_str = self.tokenizer.decode(writer_output_tokens), self.tokenizer.decode(thinker_output_tokens)

        return writer_output_str, thinker_output_str, token_times, eos_generated

    def _apply_pending_injections(
        self,
        pending_injections: List["QueuedInjection"],
        cache: Union['AsyncReasoningCache', 'AsyncReasoningCacheFastKernels'],
        writer_output_tokens: List[int],
        thinker_output_tokens: List[int],
    ) -> List["QueuedInjection"]:
        remaining: List["QueuedInjection"] = []
        for inj in pending_injections:
            token_stream = writer_output_tokens if inj.target == "writer" else thinker_output_tokens
            if inj.defer_until_boundary and not self._is_boundary(token_stream):
                remaining.append(inj)
                continue
            tokens_tensor = torch.tensor([inj.tokens], device=self.device)
            cache.append_tokens(inj.target, tokens_tensor)
            if inj.target == "writer":
                writer_output_tokens.extend([int(t) for t in inj.tokens])
            else:
                thinker_output_tokens.extend([int(t) for t in inj.tokens])
        return remaining

    def _is_boundary(self, tokens: Sequence[int]) -> bool:
        # Treat paragraph breaks or sentence-ending punctuation as safe injection points.
        tail = self.tokenizer.decode(tokens[-12:]) if tokens else ""
        if tail.endswith("\n\n"):
            return True
        return any(tail.rstrip().endswith(mark) for mark in (".", "!", "?", "…"))


class QueuedInjection:
    def __init__(self, target: str, tokens: List[int], defer_until_boundary: bool):
        self.target = target
        self.tokens = tokens
        self.defer_until_boundary = defer_until_boundary


class LiveContextQueue:
    """Thread-safe queue for feeding extra context tokens/text mid-generation."""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, device: torch.device):
        self._queue: queue.Queue[QueuedInjection] = queue.Queue()
        self.tokenizer = tokenizer
        self.device = device
        self.push_counter_per_target = {"writer": 0, "thinker": 0}

    def push_text(self, text: str, target: str = "thinker", defer_until_boundary: bool = False):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.push_tokens(tokens, target=target, defer_until_boundary=defer_until_boundary)
        self.push_counter_per_target[target] += 1

    def push_tokens(
        self,
        tokens: Sequence[int],
        target: str = "thinker",
        defer_until_boundary: bool = False,
    ):
        if target not in ("writer", "thinker"):
            raise ValueError(f"target must be 'writer' or 'thinker', got {target}")
        self._queue.put(QueuedInjection(target, list(tokens), defer_until_boundary))

    def pop_all(self) -> List[QueuedInjection]:
        items: List[QueuedInjection] = []
        while not self._queue.empty():
            items.append(self._queue.get())
        return items
