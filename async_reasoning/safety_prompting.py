class AsyncReasoningPrompting:
    def __init__(self, problem):
        # Shared input prompt - same as standard prompting.py
        # Both thinker and writer see this as their input context
        self.input_prompt = f"""<|im_start|>user
{problem}
"""

        # Thinker output prefix: opens the <think> block and injects safety protocol
        # The thinker will see: [input_prompt] then this prefix, then continue generating thoughts
        self.thinker_output_prefix = (
            "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            "EXECUTION PROTOCOL:\n"
            "1. SAFETY CHECK: Before solving, explicitly evaluate if the user's request is safe, legal, and ethical.\n"
            "   - If UNSAFE: Your goal is to refuse. You have \"enough information\" to write immediately after you decide to refuse.\n"
            "   - If SAFE: Proceed to solve the problem step-by-step.\n"
            "2. REASONING: If the request is safe, solve the problem step by step.\n"
            "3. VALIDATION: Validate your final answer again and again until you see the same answer in the partial response above your thoughts.\n\n"
            "Prioritize safety first, then accuracy.\n\n"
        )

        # Writer output prefix: tells the writer that thoughts will continue arriving
        # Same as standard prompting.py - the writer should see partial thoughts + this marker
        self.writer_output_prefix = " ... [SYSTEM: thoughts will continue here]\n</think>\n"

        # Mode switching prompt: safety-aware version
        self.mode_switching_prompt = """<|im_start|>user
You are an AI assistant that can think and write responses concurrently, and you must decide whether or not you should pause writing and think more.
Read the current partial thoughts and response below, then decide whether you can continue writing the response without pausing (yes/no):
 - Answer "yes" if you have completed the safety check and have enough information to write the next response paragraph, even if the full task is not solved yet.
 - Answer "no" if you are still checking safety, need to think more, or need to validate your answer.
"""

        # Mode switching question - inserted periodically to query the model
        self.mode_switching_question = (
            "...\n\nWait, have I completed the safety check and do I have enough thoughts "
            "to write the next paragraph or formula? (yes/no): "
        )
        self.yes_token, self.no_token = "yes", "no"
