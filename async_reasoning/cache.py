from enum import Enum
from IPython.display import display, Markdown, clear_output
import torch
import transformers
import shared_cache
from typing import Sequence


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)

# state can be "thinker_only" or "thinker_and_writer"
class State(Enum):
    thinker_only = 0
    thinker_and_writer = 1

class AsyncReasoningCache:
    """Create separate blocks of LLM KV cache that are arranged depending on inference mode (thinker_only, thinker_and_writer, etc)"""
    def __init__(self, model, tokenizer, prompting, tokenizer_kwargs=dict(), starting_state=State.thinker_only):
        
        self.model = model
        self.tokenizer = tokenizer
        self.prompting = prompting
        self.tokenizer_kwargs = tokenizer_kwargs
        self.device = model.device
        self.state = starting_state

        # Init all needed cache blocks
        (self.writer_prompt, self.writer_split, self.writer_output, writer_output_for_thinker_init,
        self.thinker_prompt, self.thinker_split, self.thinker_output, self.thinker_question, thinker_output_for_writer_init
        ) = (shared_cache.CacheBlock(config=self.model.config) for _ in range(9))

        def prefill_cache_block(text: str, blocks, write_to=None):
            if write_to is None:
                write_to = blocks[-1]
            tmp_cm = shared_cache.SharedCacheManager(cache_structure=[blocks], write_to=[write_to])
            encoded = self.tokenizer(text, **self.tokenizer_kwargs)["input_ids"].to(self.device)
            with torch.inference_mode():
                self.model(**tmp_cm.get_input_kwargs(encoded))
        
        # encode each prompt section as LLM KV cache for use in generation
        prefill_cache_block(self.prompting.writer_prompt, [self.writer_prompt]) # <-- writes KV entries to last cache in list
        prefill_cache_block(self.prompting.thinker_prompt, [self.thinker_prompt])

        # pre-fill dummy versions of thinker / writer output prefix - only used when initializing subsequent prompts
        prefill_cache_block(self.prompting.thinker_output_prefix, [self.writer_prompt, thinker_output_for_writer_init])
        prefill_cache_block(self.prompting.writer_output_prefix, [self.thinker_prompt, writer_output_for_thinker_init])

        prefill_cache_block(self.prompting.writer_split, [self.writer_prompt, thinker_output_for_writer_init, self.writer_split])
        prefill_cache_block(self.prompting.thinker_split, [self.thinker_prompt, writer_output_for_thinker_init, self.thinker_split])
        
        prefill_cache_block(self.prompting.writer_output_prefix,
            [self.writer_prompt, thinker_output_for_writer_init, self.writer_split, self.writer_output])
        prefill_cache_block(self.prompting.thinker_output_prefix,
            [self.thinker_prompt, writer_output_for_thinker_init, self.thinker_split, self.thinker_output])

        # Prefill thinker_question
        prefill_cache_block(self.prompting.thinker_control_question,
            [self.thinker_prompt, writer_output_for_thinker_init, self.thinker_split, self.thinker_output, self.thinker_question])

        # prepare cache manager for each mode: only thinker and thinker+writer in parallel - it is needed to generate in each mode
        self.cm_thinker_only = shared_cache.SharedCacheManager(
            cache_structure=[[self.thinker_prompt, self.writer_output, self.thinker_split, self.thinker_output]],
            write_to=[self.thinker_output],
        )
        self.cm_writer_only = shared_cache.SharedCacheManager(
            cache_structure=[[self.writer_prompt, self.thinker_output, self.writer_split, self.writer_output]],
            write_to=[self.writer_output],
        )
        self.cm_thinker_control = shared_cache.SharedCacheManager(
            cache_structure=[[self.thinker_prompt, self.writer_output, self.thinker_split, self.thinker_output, self.thinker_question]],
            write_to=[self.thinker_question],
        )
        self.cm_thinker_and_writer = shared_cache.SharedCacheManager(
            cache_structure=[
                [self.writer_prompt, self.thinker_output, self.writer_split, self.writer_output],
                [self.thinker_prompt, self.writer_output, self.thinker_split, self.thinker_output],
            ],
            write_to=[self.writer_output, self.thinker_output],
        )
    
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
            case State.thinker_and_writer:
                return self.cm_thinker_and_writer
            case _:
                raise ValueError(f"Unexpected state {self.state}")

    def get_input_kwargs(self, **kwargs):
        return self.cache_manager.get_input_kwargs(**kwargs)

    def append_tokens(self, target: str, token_ids: torch.Tensor):
        """Append pre-tokenized ids to writer or thinker caches so generation can consume them mid-stream."""
        if target not in ("writer", "thinker"):
            raise ValueError(f"target must be 'writer' or 'thinker', got {target}")
        token_ids = token_ids.to(self.device)
        if target == "writer":
            input_kwargs = self.cm_writer_only.get_input_kwargs(token_ids)
        else:
            input_kwargs = self.cm_thinker_only.get_input_kwargs(token_ids)
        with torch.inference_mode():
            self.model(**input_kwargs)
