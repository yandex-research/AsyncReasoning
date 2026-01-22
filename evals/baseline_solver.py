import os
import time
import logging
from typing import Sequence

import torch
from IPython.display import display, Markdown, clear_output

import transformers
from transformers.generation.streamers import BaseStreamer

from utils.modeling import prepare_model_for_inference

logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)


class StreamRecorder(BaseStreamer):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 thinker_enabled: bool = True,
                 display_generation_in_real_time: bool = False,
                 eos_ids: Sequence[int] = (),
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.in_thinking_mode = thinker_enabled
        self.display_generation_in_real_time = display_generation_in_real_time
        self.token_times = []
        self.current_step = 0
        self.starting_time = time.perf_counter()
        self.thinker_tokens = []
        self.writer_tokens = []
        self.eos_generated = False
        self.eos_ids = eos_ids

    def put(self, input_ids: torch.Tensor):
        if self.eos_generated: # do not do anything after eos was generated
            return
        if self.current_step > 0:
            next_token, = input_ids.flatten().tolist()
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

    def end(self):
        if len(self.token_times) == 0:
            self.token_times.append(("EMPTY", time.perf_counter() - self.starting_time, self.current_step))

    def display_tokens(self, writer_output_tokens: Sequence[int], thinker_output_tokens: Sequence[int]):
        writer_headers, thinker_headers = ["\n\n## Writer mode\n\n", "\n\n## Thinker mode\n\n"]
        thinker_text = self.tokenizer.decode(thinker_output_tokens)
        writer_text = self.tokenizer.decode(writer_output_tokens)
        clear_output(True)
        raw = "".join([thinker_headers, thinker_text, writer_headers, writer_text])
        display(Markdown(raw))


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
        self.eos_ids = model.generation_config.eos_token_id
        if isinstance(self.eos_ids, int):
            self.eos_ids = [self.eos_ids]
        if model.config.model_type == "gpt_oss":
            self.end_of_thinking_token_ix = [self.tokenizer.vocab[i] for i in ['<|channel|>', 'final', '<|message|>']]
            assert thinker_enabled, "gpt-oss only works in thinking mode, set effort through $REASONING_EFFORT"
            self.reasoning_effort = os.environ["REASONING_EFFORT"]
            assert self.reasoning_effort in ["low", "medium", "high"]
        else:
            assert model.config.model_type.startswith(
                'qwen3'), "only tested with qwen3 and gpt-oss, remove this at your own risk"
            self.end_of_thinking_token_ix = [self.tokenizer.vocab["</think>"]]

    def solve(self,
            problem: str,
            display_generation_in_real_time: bool = False,
            budget: int = 1024,
            temperature: float = 0.0,
            top_p: float = 0.95,
            top_k: int = 20,
):
        if self.model.config.model_type.startswith("qwen3"):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False, add_generation_prompt=True, enable_thinking=self.thinker_enabled)
        elif self.model.config.model_type == 'gpt_oss':
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False, add_generation_prompt=True, reasoning_effort=self.reasoning_effort)
        else:
            raise NotImplementedError(f"Unsupported chat template for model type {self.model.config.model_type}.")
        input_ids = self.tokenizer.encode(text, **self.tokenizer_kwargs).to(self.device)
        streamer = StreamRecorder(
            tokenizer=self.tokenizer,
            thinker_enabled=self.thinker_enabled,
            display_generation_in_real_time=display_generation_in_real_time,
            eos_ids=self.eos_ids,
        )
        outputs = self.model.generate(input_ids,
            max_new_tokens=budget,
            return_dict_in_generate=True,
            output_scores=False,
            streamer=streamer,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
        )
        return (
            self.tokenizer.decode(streamer.writer_tokens),
            self.tokenizer.decode(
                streamer.thinker_tokens[2:] if self.model.config.model_type != "gpt_oss" else streamer.thinker_tokens[3:-6]),
                # by default, [2:] is "<think>\n"";  for openai, it's <|channel|>analysis<|message|>CONTENT<|end|><|start|>assistant<|channel|>final<|message|>
            streamer.token_times,
            streamer.eos_generated,
    )
