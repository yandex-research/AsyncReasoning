import time
import torch
import transformers
from IPython.display import display, Markdown, clear_output
from typing import Sequence, Union, Tuple, Dict, Any

from async_reasoning.prompting import AsyncReasoningPrompting

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)

class BaselineSolver:
    def __init__(self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        thinker_enabled: bool = True,  
    ):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = dict(add_special_tokens=False, return_tensors='pt', padding=True, padding_side='left')
        self.thinker_enabled = thinker_enabled
        assert str(model.name_or_path).startswith("Qwen/Qwen3"), f"Support only Qwen3 for now, but {model.name_or_path} provided"
        self.eos_ids = model.generation_config.eos_token_id
        if isinstance(self.eos_ids, int):
            self.eos_ids = [self.eos_ids]
    
    def _init_token_times_counters(self):
        self.token_times = []
        self.current_step = 0
        self.starting_time = time.perf_counter()
        self.thinker_tokens = []
        self.writer_tokens = []
        self.in_thinking_mode = self.thinker_enabled
        self.eos_generated = False
    
    def forward_hook(self, model, _unused_args, output, **_unused_kwargs):
        assert not _unused_args and not _unused_kwargs
        if self.eos_generated: # do not do anything after eos was generated
            return
        next_token = int(output.logits.argmax(-1))
        if not self.in_thinking_mode:
            token_times_item = (self.tokenizer.decode(next_token), time.perf_counter() - self.starting_time, self.current_step)
            self.token_times.append(token_times_item)
            if next_token in self.eos_ids:
                self.eos_generated = True
            self.writer_tokens.append(next_token) 
        else:
            if next_token == self.tokenizer.vocab["</think>"]:
                self.in_thinking_mode = False
            self.thinker_tokens.append(next_token) 
        if self.display_generation_in_real_time:
            self.display_tokens(self.writer_tokens, self.thinker_tokens)
        self.current_step += 1


    def display_tokens(self,
        writer_output_tokens: Sequence[int], 
        thinker_output_tokens: Sequence[int], 
        ):
        writer_headers, thinker_headers = ["\n\n## Writer mode\n\n", "\n\n## Thinker mode\n\n"]
        thinker_text = self.tokenizer.decode(thinker_output_tokens)
        writer_text = self.tokenizer.decode(writer_output_tokens)

        clear_output(True)
        raw = "".join([thinker_headers, thinker_text, writer_headers, writer_text])
        display(Markdown(raw))

    def solve(self,
            problem: str,
            display_generation_in_real_time: bool = False,
            budget: int = 1024,
        ):
        self.display_generation_in_real_time = display_generation_in_real_time
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinker_enabled
        )
        input_ids = self.tokenizer.encode(text, **self.tokenizer_kwargs).to(self.device)

        handle = self.model.register_forward_hook(self.forward_hook)
        try:
            self._init_token_times_counters()
            outputs = self.model.generate(input_ids, 
                max_new_tokens=budget,
                return_dict_in_generate=True,
                output_scores=False,
            )
            if len(self.token_times) == 0:
                self.token_times.append(("EMPTY", time.perf_counter() - self.starting_time, self.current_step))
        finally:
            handle.remove()
        return (
            self.tokenizer.decode(self.writer_tokens), 
            self.tokenizer.decode(self.thinker_tokens[2:]), # here [2:] is "<think>\n""
            self.token_times, 
            self.eos_generated,
    )
