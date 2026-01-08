import os
import time
import torch
import transformers
from IPython.display import display, Markdown, clear_output
from typing import Sequence

import logging

from utils.modeling import prepare_model_for_inference

logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)


class BaselineSolver:
    def __init__(self,
                 model: transformers.PreTrainedModel,
                 tokenizer: transformers.PreTrainedTokenizer,
                 thinker_enabled: bool = True,
                 **kwargs
                 ):
        self.model = prepare_model_for_inference(model, **kwargs)
        self.device = model.device
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = dict(add_special_tokens=False, return_tensors='pt', padding=True, padding_side='left')
        self.thinker_enabled = thinker_enabled
        self.eos_token_ix = model.generation_config.eos_token_id
        if isinstance(self.eos_token_ix, int):
            self.eos_token_ix = [self.eos_token_ix]
        if model.config.model_type == "gpt_oss":
            self.end_of_thinking_token_ix = [self.tokenizer.vocab[i] for i in ['<|channel|>', 'final', '<|message|>']]
            assert thinker_enabled, "gpt-oss only works in thinking mode, set effort through $REASONING_EFFORT"
            self.reasoning_effort = os.environ["REASONING_EFFORT"]
        else:
            assert model.config.model_type.startswith(
                'qwen3'), "only tested with qwen3 and gpt-oss, remove this at your own risk"
            self.end_of_thinking_token_ix = [self.tokenizer.vocab["</think>"]]

    def _init_token_times_counters(self):
        self.token_times = []
        self.current_step = 0
        self.starting_time = time.perf_counter()
        self.thinker_tokens = []
        self.writer_tokens = []
        self.in_thinking_mode = self.thinker_enabled or self.model.config.model_type == 'gpt_oss'
        self.eos_generated = False

    def forward_hook(self, model, _unused_args, output, **_unused_kwargs):
        assert not _unused_args and not _unused_kwargs
        if self.eos_generated:  # do not do anything after eos was generated
            return
        next_token = int(output.logits.argmax(-1))
        if not self.in_thinking_mode:
            token_times_item = (self.tokenizer.decode(next_token), time.perf_counter() - self.starting_time,
                                self.current_step)
            self.token_times.append(token_times_item)
            if next_token in self.eos_token_ix:
                self.eos_generated = True
            self.writer_tokens.append(next_token)
        else:
            self.thinker_tokens.append(next_token)
            if self.thinker_tokens[-len(self.end_of_thinking_token_ix):] == self.end_of_thinking_token_ix:
                self.in_thinking_mode = False
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
        if self.model.config.model_type.startswith("qwen3"):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False, add_generation_prompt=True, enable_thinking=self.thinker_enabled)
        elif self.model.config.model_type == 'gpt_oss':
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False, reasoning_effort=self.reasoning_effort,
                add_generation_prompt=True)
        else:
            raise NotImplementedError(f"Unsupported chat template for model type {self.model.config.model_type}.")

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
            self.tokenizer.decode(
                self.thinker_tokens[2:] if self.model.config.model_type != "gpt_oss" else self.thinker_tokens[3:-6]),
            # by default, [2:] is "<think>\n"";  for openai, it's <|channel|>analysis<|message|>CONTENT<|end|><|start|>assistant<|channel|>final<|message|>
            self.token_times,
            self.eos_generated,
        )
