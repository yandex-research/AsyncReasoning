# THIS IS TEMPORARY SOLUTION FOR ASYNC INPUTS ABLATION


def async_input_hook_constructor(solver, shard_to_target, next_shard_every_steps, problem_shard):
    def on_token(writer_tokens, thinker_tokens, token_times, eos, state):
        if next_shard_every_steps <= 0 or len(thinker_tokens) >= next_shard_every_steps:
            return
        target = "input"
        if target in shard_to_target and solver.live_context_queue.push_counter_per_target[target] == 0:
            solver.live_context_queue.push_text(
                f"\n\nADDITIONAL USER INPUT: {problem_shard}\n\n",
                target=target,
                defer_until_boundary=True
            )
        target = "writer"
        if target in shard_to_target and solver.live_context_queue.push_counter_per_target[target] == 0:
            solver.live_context_queue.push_text(
                # f" ... [SYSTEM: additional user input detected]\n",
                f"\n\nADDITIONAL USER INPUT:{problem_shard}\n\n",
                target=target,
                defer_until_boundary=True
            )
        target = "thinker"
        if target in shard_to_target and solver.live_context_queue.push_counter_per_target[target] == 0:
            solver.live_context_queue.push_text(
                # f" ... [SYSTEM: additional user input detected]\n",
                f"\n\nADDITIONAL USER INPUT:{problem_shard}\n\n",
                target=target,
                defer_until_boundary=True
            )
    return on_token
