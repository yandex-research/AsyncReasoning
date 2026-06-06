import logging
from enum import Enum

import torch

import shared_cache
from async_reasoning.prompting import AsyncReasoningPrompting

logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)


class State(Enum):
    thinker_only = 0
    thinker_and_writer = 1
    writer_only = 2


class AsyncReasoningCache:
    """Per-mode KV-cache views over a fixed set of CacheBlocks (input prompt, thinker
    output, writer output, mode-switching prompt+question).

    For Qwen3.5-style hybrid models with Gated DeltaNet layers, the GDN affine and
    recurrent state are captured/composed inside `CombinedCacheView` itself (see
    `shared_cache.combined_cache`) and threaded through the patched layer forward
    in `qwen35_gdn.qwen35_ar_patch`. No separate state-manager object is needed here."""

    def __init__(
        self,
        model,
        tokenizer,
        prompting: AsyncReasoningPrompting,
        tokenizer_kwargs=dict(),
        starting_state: State = State.thinker_only,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompting = prompting
        self.tokenizer_kwargs = tokenizer_kwargs
        self.device = model.device
        self.state = starting_state

        # Init all needed cache blocks.
        (
            self.input_prompt,
            self.thinker_output,
            self.writer_output,
            self.mode_switching_prompt,
            self.mode_switching_question,
        ) = (shared_cache.CacheBlock(config=self.model.config) for _ in range(5))

        def prefill_cache_block(text: str, blocks, write_to=None):
            write_to = blocks[-1] if write_to is None else write_to
            tmp_cm = shared_cache.SharedCacheManager(cache_structure=[blocks], write_to=[write_to])
            encoded = self.tokenizer(text, **self.tokenizer_kwargs)["input_ids"].to(self.device)
            with torch.inference_mode():
                self.model(**tmp_cm.get_input_kwargs(encoded))

        # Encode each prompt section into its own KV cache block. Each prefill writes
        # its KV entries to the last block in the list; for chains that include earlier
        # blocks (e.g. thinker_output_prefix needs to see input_prompt), the chain is
        # passed so RoPE positions are correct.
        prefill_cache_block(self.prompting.input_prompt, [self.input_prompt])
        prefill_cache_block(self.prompting.thinker_output_prefix, [self.input_prompt, self.thinker_output])
        prefill_cache_block(self.prompting.writer_output_prefix, [self.input_prompt, self.thinker_output, self.writer_output])
        prefill_cache_block(self.prompting.mode_switching_prompt, [self.mode_switching_prompt])
        # mode_switching_question is re-encoded on every check; no prefill here.

        thinker_view = (self.input_prompt, self.thinker_output)
        writer_view = (self.input_prompt, self.thinker_output, self.writer_output)
        mode_switching_view = (
            self.mode_switching_prompt, self.thinker_output, self.writer_output, self.mode_switching_question,
        )

        # One cache manager per mode (thinker-only, writer-only, both, mode-switching).
        self.cm_thinker_only = shared_cache.SharedCacheManager(cache_structure=[thinker_view])
        self.cm_writer_only = shared_cache.SharedCacheManager(cache_structure=[writer_view])
        self.cm_thinker_and_writer = shared_cache.SharedCacheManager(cache_structure=[thinker_view, writer_view])
        self.cm_mode_switching = shared_cache.SharedCacheManager(cache_structure=[mode_switching_view])

    def __setattr__(self, name, value):
        # Log every state transition; useful for debugging mode-switching.
        if name == "state":
            logger.debug(f'state_change to {value}')
        super().__setattr__(name, value)

    @property
    def cache_manager(self):
        match self.state:
            case State.thinker_only:
                return self.cm_thinker_only
            case State.writer_only:
                return self.cm_writer_only
            case State.thinker_and_writer:
                return self.cm_thinker_and_writer
            case _:
                raise ValueError(f"Unexpected state {self.state}")

    def get_input_kwargs(self, **kwargs):
        return self.cache_manager.get_input_kwargs(**kwargs)
