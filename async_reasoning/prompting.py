# thinker (t) is worker 0, writer (w) is worker 1, mode switching is queried separately
# thinker sees:    "{input_prompt}{thinker_output}"
# writer sees:     "{input_prompt}{thinker_output}{writer_output}
# mode switching:  "{mode_switching_prompt}{thinker_output}{writer_output}


class AsyncReasoningPrompting:
    def __init__(self, problem):
        # input_prompt ends right after `{problem}` (no trailing \n) so the concatenation
        # `input_prompt + thinker_output_prefix` produces `{problem}<|im_end|>\n...` — exactly
        # what tokenizer.apply_chat_template emits. Adding an extra `\n` here was inserting
        # token 198 between {problem} and <|im_end|>, drifting AR's prompt off-distribution.
        self.input_prompt = f"<|im_start|>user\n{problem}"
        # writer_output and thinker_output start with these prefixes.
        # IMPORTANT: prefixes intentionally end at a token boundary BEFORE the trailing newline.
        # The solver appends a starter token (one `\n` for the thinker, one `\n\n` for the writer)
        # so the model's first-step context lands at exactly `<think>\n` / `</think>\n\n` — the same
        # endings Qwen3.5's chat template produces for enable_thinking=True / False respectively.
        # If we leave an extra `\n` in the prefix, the thinker's first-forward context becomes
        # `<think>\n\n\n` which is the empty-block-close pattern → model emits `</think>` at p~0.999.
        # Writer prefix is bare on purpose: anything more (e.g. `Therefore,`) primes the writer to
        # jump to a conclusion before the thinker has derived it. The job of biasing pause-vs-write
        # belongs to the mode-switcher, not the writer's leading tokens.
        self.thinker_output_prefix = "<|im_end|>\n<|im_start|>assistant\n<think>"
        self.writer_output_prefix = "\n</think>"

        # mode_switching_prompt opens a user turn. The thinker_output block's prefix
        # closes that user turn with `<|im_end|>` and opens the assistant turn with `<think>`,
        # so this string MUST NOT close the user turn itself.
        self.mode_switching_prompt = f"""
<|im_start|>user
You are an AI assistant that can think and write responses concurrently, and you must decide whether or not you should pause writing and think more.
Read your current partial thoughts and partial response below, then decide whether you can continue writing the response without pausing.
 - Answer "yes" if your thoughts have enough information to write the next response paragraph, even if the full task is not solved yet.
 - Answer "no" if your thoughts aren't enough to write the next response paragraph, i.e. if your response ran out of thoughts.
""".strip() + "\n"

        # mode_switching_question closes the assistant turn (which holds the partial
        # thinker+writer state) and opens a NEW user turn asking the yes/no question.
        # We then pre-fill the assistant turn with an EMPTY `<think>\n\n</think>\n\n` block
        # — this is the `enable_thinking=False` template pattern. Without it, Qwen3.5 opens
        # a fresh `<think>` block to deliberate (top-1 = '<think>' at p=1.0), so the yes/no
        # answer never comes. With the empty block prefilled, the model is in "non-thinking"
        # mode and yes/no becomes its actual top prediction.
        self.mode_switching_question = (
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Looking at the partial thoughts and partial response above, do you already have enough information in your thoughts to continue writing the next paragraph or formula of your response without pausing to think more? Reply with a single word: yes or no.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        self.yes_token, self.no_token = "yes", "no"