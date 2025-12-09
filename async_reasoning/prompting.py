# writer (w) is worker 0, thinker (t) is worker 1
# writer sees:  [writer_prompt, thinker_output, writer_split, writer_output]
# thinker sees: [thinker_prompt, writer_output, thinker_split, thinker_output]

class AsyncReasoningPrompting:
    def __init__(self, problem):
        self.writer_prompt = f"""
<|im_start|>user

You are an AI assistant that can think and write outputs concurrently.

You can write outputs for the user based on partial chain of thought that will be continued in the background by an automated system.
Your should first outline what you're going to do for the user, then gradually write your response as your thoughts progress, but not ahead of your thoughts.
When you are done, specify your final answer (e.g. \\boxed{{ }}).

You are given the following problem:
{problem}
""".strip()

        self.thinker_prompt = f"""
<|im_start|>user

You are an AI assistant that can think and write outputs concurrently.

You can reason in private and your thoughts will be used to form the public response in the background, by an automated system. Your task is to write thoughts and control when the automated system can continue writing the response.

Sometimes, an automated system will ask you to decide if your thoughts have enough information for it write an additional passage to the user. Use the partial response above yours thoughts to judge if you added enough new information to write one more passage in the user-facing response.

- Reply "yes" if you think there is enough information to write the next passage (paragraph, equation, etc).
- Reply "no" if you need to think more in private before the system can continue writing the public response.

Your goal is to give frequent updates on your progress, even if you did not solve the entire task yet. Reason in short paragraphs. Prioritize giving enough information for the system to begin responding to the user as soon as possible.

Please reason step by step. Validate final again and again until you see the same answer in the partial response above your thoughts.

Solve the following problem:
{problem}<|im_end|>
<|im_start|>assistant""".strip()

        self.writer_split = " SYSTEM: [additional thoughts will appear here]\n</think>\n"
        self.thinker_split = " SYSTEM: [the system will continute writing the response here]"

        # writer_output and thinker_output starts with these prefixes
        self.writer_output_prefix = f"""\n"""
        self.thinker_output_prefix =  f"""<|im_end|>\n<|im_start|>assistant\n<think>\n"""

        # these questions are inserted to change mode depending on model answers
        self.thinker_control_question = "...\n\nWait, are my private thoughts ahead of the written response by enough to write one more response paragraph? (yes/no): "
        self.yes_token, self.no_token = "yes", "no"