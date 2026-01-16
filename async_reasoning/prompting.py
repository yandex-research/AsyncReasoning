# thinker (t) is worker 0, writer (w) is worker 1, mode switching is queried separately
# thinker sees:    "{input_prompt}{thinker_output}"
# writer sees:     "{input_prompt}{thinker_output}{writer_output}
# mode switching:  "{mode_switching_prompt}{thinker_output}{writer_output}


PLAN_FIRST_SYSTEM_PROMPT = (
    "Provide a brief, step-by-step plan as the first part of your response. "
    "Use 3-6 numbered steps. After the plan, continue with the solution and final answer. "
    "Keep the plan concise and user-facing; do not reveal internal reasoning."
)


def _merge_system_prompts(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part)


class AsyncReasoningPrompting:
    def __init__(
        self,
        problem: str,
        system_prompt: str | None = None,
        plan_first: bool = False,
        plan_text: str | None = None,
        plan_output_prefix: str = "Plan:\n",
        plan_output_suffix: str = "\n\nAnswer:\n",
        plan_context: str | None = None,
    ):
        merged_system_prompt = _merge_system_prompts(
            system_prompt,
            PLAN_FIRST_SYSTEM_PROMPT if plan_first else "",
            plan_context,
        )
        if merged_system_prompt:
            self.input_prompt = f"""
<|im_start|>system
{merged_system_prompt}
<|im_end|>
<|im_start|>user
{problem}
""".strip() + "\n"
        else:
            self.input_prompt = f"""
<|im_start|>user
{problem}
""".strip() + "\n"
        # writer_output and thinker_output starts with these prefixes
        self.thinker_output_prefix = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        plan_prefix = ""
        if plan_first:
            if plan_text:
                plan_prefix = f"{plan_output_prefix}{plan_text.strip()}{plan_output_suffix}"
            else:
                plan_prefix = plan_output_prefix
        self.writer_output_prefix = f""" ... [SYSTEM: thoughts will continue here]\n</think>\n{plan_prefix}"""

        self.mode_switching_prompt = f"""
<|im_start|>user
You are an AI assistant that can think and write responses concurrently, and you must decide whether or not you should pause writing and think more.
Read the current partial thoughts and response below, then decide whether you can continue writing the response without pausing (yes/no):
 - Answer "yes" if your thoughts have enough information to write the next response paragraph, even if the full task is not solved yet.
 - Answer "no" if your thoughts aren't enough to write the next response paragraph, i.e. if your response ran out of of thoughts.
""".strip() + "\n"

        # these questions are inserted to change mode depending on model answers
        self.mode_switching_question = "...\n\nWait, are my current thoughts enough to write the next paragraph or formula? (yes/no): "
        self.yes_token, self.no_token = "yes", "no"
