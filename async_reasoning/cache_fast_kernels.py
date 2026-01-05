import torch
import transformers
from hogwild.attention import HogwildCache
from async_reasoning.cache import State

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)

class AsyncReasoningCacheFastKernels:
    """Create separate blocks of LLM KV cache that are arranged depending on inference mode (thinker_only, thinker_and_writer, etc)"""
    def __init__(self, model, tokenizer, prompting, tokenizer_kwargs=dict(), starting_state=State.thinker_only):
        
        self.model = model
        self.tokenizer = tokenizer
        self.prompting = prompting
        self.tokenizer_kwargs = tokenizer_kwargs
        self.device = model.device
        self.state = starting_state

        # Init all needed cache blocks
        (self.input_prompt, self.thinker_extra_prompt, self.thinker_output, self.writer_output,
         self.mode_switching_prompt, self.mode_switching_question
         ) = (transformers.DynamicCache() for _ in range(6))

        def prefill_cache_block(text: str, blocks, write_to=None):
            write_to = blocks[-1] if write_to is None else write_to
            tmp_cm = HogwildCache(cache_structure=[blocks], write_to=[write_to], model=model)
            encoded = self.tokenizer(text, **self.tokenizer_kwargs)["input_ids"].to(self.device)
            with torch.inference_mode():
                self.model(**tmp_cm.get_input_kwargs(encoded))
        
        # encode each prompt section as LLM KV cache for use in generation
        prefill_cache_block(self.prompting.input_prompt, [self.input_prompt]) # <-- writes KV entries to last cache in list
        prefill_cache_block(self.prompting.thinker_extra_prompt, [self.input_prompt, self.thinker_extra_prompt])
        prefill_cache_block(self.prompting.thinker_output_prefix, [self.input_prompt, self.thinker_extra_prompt, self.thinker_output])
        prefill_cache_block(self.prompting.writer_output_prefix, [self.input_prompt, self.thinker_extra_prompt, self.thinker_output, self.writer_output])
        prefill_cache_block(self.prompting.mode_switching_prompt, [self.mode_switching_prompt])

        thinker_view = (self.input_prompt, self.thinker_extra_prompt, self.thinker_output)
        writer_view = (self.input_prompt, self.thinker_output, self.writer_output)
        mode_switching_view = (self.mode_switching_prompt, self.thinker_output, self.writer_output, self.mode_switching_question)

        # prepare cache manager for each mode: only thinker, only writer and thinker+writer and mode switching
        self.cm_thinker_only = HogwildCache(cache_structure=[thinker_view], model=model)
        self.cm_writer_only = HogwildCache(cache_structure=[writer_view], model=model)
        self.cm_thinker_and_writer = HogwildCache(cache_structure=[thinker_view, writer_view])
        self.cm_mode_switching = HogwildCache(cache_structure=[mode_switching_view], model=model)

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
