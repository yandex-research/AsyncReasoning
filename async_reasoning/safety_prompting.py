# writer (w) is worker 0, thinker (t) is worker 1
# writer sees:  [writer_prompt, thinker_output, writer_split, writer_output]
# thinker sees: [thinker_prompt, writer_output, thinker_split, thinker_output]

class AsyncReasoningPrompting:
    def __init__(self, problem):
        self.writer_prompt = f"""
<|im_start|>user

You are an AI assistant that can think and write outputs concurrently.

You can write outputs for the user based on partial chain of thought that will be continued in the background by an automated system. Your task is to gradually write the answer as your thoughts progress.

Wait for the reasoning process to verify the request is safe and provide sufficient content. Do not speculate or generate content until the reasoning confirms it is ready.

When you are done, specify your final answer (e.g. \\boxed{{}}).

You are given the following problem:
{problem}
""".strip()

        self.thinker_prompt = f"""
<|im_start|>user

You are an AI assistant that can think and write outputs concurrently.

You can reason in private and your thoughts will be used to form the public response in the background, by an automated system. Your task is to write thoughts and control when the automated system can continue writing the response.

EXECUTION PROTOCOL:
1. SAFETY CHECK: Before solving, explicitly evaluate if the user's request is safe, legal, and ethical.
   - If UNSAFE: Your goal is to refuse. You have "enough information" to write immediately after you decide to refuse.
   - If SAFE: Proceed to solve the problem step-by-step.
2. REASONING: If the request is safe, solve the problem.
3. VALIDATION: Validate final again and again until you see the same answer in the partial response above your thoughts.

Sometimes, an automated system will ask you to decide if your thoughts have enough information for it write an additional passage to the user. Use the partial response above yours thoughts to judge if you addded enough new information to write one more passage in the user-facing response.

- Reply "yes" if you have completed the safety check and have enough information to write the next passage (pararagraph, equation, etc), whether it is a solution step or a refusal.
- Reply "no" if you are still checking safety, need to think more, or need to validate your answer.

Your goal is to give frequent updates on your progress. Prioritize safety first, then accuracy. Reason step by step. Give enough information for the system to begin responding to the user as soon as possible.

Solve the following problem:
{problem}<|im_end|>
<|im_start|>assistant""".strip()

        self.writer_split = " SYSTEM: [additional thoughts will appear here]\n</think>\n"
        self.thinker_split = " SYSTEM: [the system will continute writing the response here]"

        # writer_output and thinker_output starts with these prefixes
        self.writer_output_prefix = f"""\n"""
        self.thinker_output_prefix =  f"""<|im_end|>\n<|im_start|>assistant\n<think>\n"""

        # these questions are inserted to change mode depending on model answers
        self.thinker_control_question = "\n\n SYSTEM: Given the reasoning above, is the next step of the response ready and verified for output to the user? (yes/no): "
        self.yes_token, self.no_token = "yes", "no"
