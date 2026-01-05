# thinker (t) is worker 0, writer (w) is worker 1, mode switching is queried separately
# thinker sees:    "{input_prompt}{thinker_extra_prompt}{thinker_output}"
# writer sees:     "{input_prompt}{thinker_output}{writer_output}
# mode switching:  "{mode_switching_prompt}{thinker_output}{writer_output}
# TODO mode switching does not see the original problem
# TODO maybe move thinker_extra_prompt into thinker output prefix in the first person?

class AsyncReasoningPrompting:
    def __init__(self, problem):
        self.input_prompt = f"""
<|im_start|>user
{problem}
""".strip() + "\n"
        self.thinker_extra_prompt = f"""
You are an AI assistant that can think and write outputs concurrently.
Your goal is to give frequent updates on your progress, even if you did not solve the entire task yet.
Reason in short paragraphs. Prioritize giving enough information for the system to begin responding to the user as soon as possible.
""".strip() + "\n"
        # writer_output and thinker_output starts with these prefixes
        self.thinker_output_prefix = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        self.writer_output_prefix = f""" ... [SYSTEM: thoughts will continue here]\n</think>\n"""

        self.mode_switching_prompt = f"""
<|im_start|>user
You are an AI assistant that can think and write outputs concurrently, but sometimes you need to wait for thoughts before you can write the next response paragraph.
Use the partial response to decide if you added enough new information to write one more passage in the user-facing response:
 - Reply "yes" if your thoughts have enough information to write the next paragraph or equation to your current response, even if the task is not fully solved yet.
 - Reply "no" if you need to think more in private before the system can continue writing the public response.
""".strip() + '\n'

        # these questions are inserted to change mode depending on model answers
        self.mode_switching_question = "...\n\nWait, are my current thoughts enough to write the next paragraph or formula? (yes/no): "
        self.yes_token, self.no_token = "yes", "no"