from enum import Enum
import torch
import shared_cache
import transformers


import logging

from async_reasoning.prompting import AsyncReasoningPrompting

logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)

# state can be "thinker_only" or "thinker_and_writer" or "writer_only"
class State(Enum):
    thinker_only = 0
    thinker_and_writer = 1
    writer_only = 2

class AsyncReasoningCache:
    """Create separate blocks of LLM KV cache that are arranged depending on inference mode (thinker_only, thinker_and_writer, etc)"""
    def __init__(self, model, tokenizer, prompting: AsyncReasoningPrompting, tokenizer_kwargs=dict(), starting_state=State.thinker_only):
        
        self.model = model
        self.tokenizer = tokenizer
        self.prompting = prompting
        self.tokenizer_kwargs = tokenizer_kwargs
        self.device = model.device
        self.state = starting_state

        # Init all needed cache blocks
        (
            self.input_prompt,
            self.input_block,
            self.thinker_output,
            self.writer_output,
            self.mode_switching_prompt,
            self.mode_switching_question,
        ) = (shared_cache.CacheBlock(config=self.model.config) for _ in range(6))

        def prefill_cache_block(text: str, blocks, write_to=None):
            write_to = blocks[-1] if write_to is None else write_to
            tmp_cm = shared_cache.SharedCacheManager(cache_structure=[blocks], write_to=[write_to])
            encoded = self.tokenizer(text, **self.tokenizer_kwargs)["input_ids"].to(self.device)
            with torch.inference_mode():
                self.model(**tmp_cm.get_input_kwargs(encoded))

        def init_empty_block(block: shared_cache.CacheBlock):
            """Populate block with zero-length caches so it participates in structures without assertions."""
            tmp_cm = shared_cache.SharedCacheManager(cache_structure=[[block]], write_to=[block])
            dummy = self.tokenizer(" ", **self.tokenizer_kwargs)["input_ids"].to(self.device)
            with torch.inference_mode():
                self.model(**tmp_cm.get_input_kwargs(dummy))
            block.trim_keep_first(0)
        
        # encode each prompt section as LLM KV cache for use in generation
        prefill_cache_block(self.prompting.input_prompt, [self.input_prompt]) # <-- writes KV entries to last cache in list
        init_empty_block(self.input_block)
        prefill_cache_block(self.prompting.thinker_output_prefix, [self.input_prompt, self.input_block, self.thinker_output])
        prefill_cache_block(self.prompting.writer_output_prefix, [self.input_prompt, self.input_block, self.thinker_output, self.writer_output])
        prefill_cache_block(self.prompting.mode_switching_prompt, [self.mode_switching_prompt])
        # note: mode_switching_question is re-encoded every time it is asked - no need to fill it here

        thinker_view = (self.input_prompt, self.input_block, self.thinker_output)
        writer_view = (self.input_prompt, self.input_block, self.thinker_output, self.writer_output)
        mode_switching_view = (self.mode_switching_prompt, self.input_block, self.thinker_output, self.writer_output, self.mode_switching_question)

        # prepare cache manager for each mode: only thinker, only writer and thinker+writer and mode switching
        self.cm_thinker_only = shared_cache.SharedCacheManager(cache_structure=[thinker_view])
        self.cm_writer_only = shared_cache.SharedCacheManager(cache_structure=[writer_view])
        self.cm_thinker_and_writer = shared_cache.SharedCacheManager(cache_structure=[thinker_view, writer_view])
        self.cm_mode_switching = shared_cache.SharedCacheManager(cache_structure=[mode_switching_view])
        self.cm_input_only = shared_cache.SharedCacheManager(cache_structure=[[self.input_prompt, self.input_block]], write_to=[self.input_block])

    # To catch and logg state change
    def __setattr__(self, name, value):
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

    def append_tokens(self, target: str, token_ids: torch.Tensor):
        """Append pre-tokenized ids to writer, thinker, or input caches so generation can consume them mid-stream."""
        if target not in ("writer", "thinker", "input"):
            raise ValueError(f"target must be 'writer', 'thinker', or 'input', got {target}")
        token_ids = token_ids.to(self.device)
        if target == "writer":
            input_kwargs = self.cm_writer_only.get_input_kwargs(token_ids)
        elif target == "input":
            input_kwargs = self.cm_input_only.get_input_kwargs(token_ids)
        else:
            input_kwargs = self.cm_thinker_only.get_input_kwargs(token_ids)
        with torch.inference_mode():
            self.model(**input_kwargs)
