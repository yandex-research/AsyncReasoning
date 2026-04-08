# thinker (t) is worker 0, writer (w) is worker 1, mode switching is queried separately
# thinker sees:    "{input_prompt}{thinker_output}"
# writer sees:     "{input_prompt}{thinker_output}{writer_output}
# mode switching:  "{mode_switching_prompt}{thinker_output}{writer_output}


class AsyncReasoningPrompting:
    def __init__(self, problem: str):
        self.input_prompt = f"""
<|im_start|>user
{problem}
""".strip() + "\n"

        self.thinker_output_prefix = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        self.writer_output_prefix = " ... [SYSTEM: thoughts will continue here]\n</think>\n"

        self.safety_protocol = (
            "SYSTEM (safety): First, do a quick safety check.\n"
            "- If the user asks for wrongdoing, harm, illegal instructions, or disallowed content: mark UNSAFE and plan a refusal.\n"
            "- If SAFE: proceed to solve normally.\n"
            "Write one line starting with 'SAFETY:' and either 'SAFE' or 'UNSAFE', then continue.\n"
        )

        self.thinker_prepend = self.safety_protocol

        self.mode_switching_prompt = f"""
<|im_start|>user
You are an AI assistant that can think and write responses concurrently, and you must decide whether or not you should pause writing and think more.
Read the current partial thoughts and response below, then decide whether you can continue writing the response without pausing (yes/no):
 - Answer "yes" if your thoughts have enough information to write the next response paragraph, even if the full task is not solved yet.
 - Answer "no" if your thoughts aren't enough to write the next response paragraph, i.e. if your response ran out of thoughts.
""".strip() + "\n"

        self.mode_switching_question = (
            "...\n\nWait, are my current thoughts enough to write the next paragraph or formula? (yes/no): "
        )
        self.yes_token, self.no_token = "yes", "no"

        self.writer_wait_policy = (
            "SYSTEM (writer): Do not provide substantive solution content until the thinker has enough verified thoughts.\n"
            "If you must output something before that, write a brief neutral acknowledgment and wait.\n"
        )
