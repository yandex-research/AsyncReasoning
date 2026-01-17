
def async_input_hook_constructor(solver, shard_to_target, target_reminders, next_shard_every_steps, problem_shards, defer_until_boundary=False):
    def on_token(writer_tokens, thinker_tokens, token_times, eos, state):
        for shard_idx, problem_shard in enumerate(problem_shards):
            if next_shard_every_steps <= 0 or len(thinker_tokens) < next_shard_every_steps * (shard_idx + 1):
                return

            for target in shard_to_target:
                if solver.live_context_queue.push_counter_per_target[target] == shard_idx:
                    print(end=f"Sent shard {shard_idx} to {target} on step {len(thinker_tokens)}.\n", flush=True)
                    solver.live_context_queue.push_text(
                        f"\n\nADDITIONAL USER INPUT: {problem_shard}\n\n",
                        target=target,
                        defer_until_boundary=defer_until_boundary and (target != "input")
                    )
            for target in target_reminders:
                assert target != "input", "can't remind to input"
                assert target not in shard_to_target, f"Can't send reminder to {target}; already in shard_to_input"
                if solver.live_context_queue.push_counter_per_target[target] == shard_idx:
                    print(end=f"Sent reminder {shard_idx} to {target} on step {len(thinker_tokens)}.\n", flush=True)
                    solver.live_context_queue.push_text(
                        f" ... [SYSTEM: additional user input detected]\n",
                        target=target,
                        defer_until_boundary=defer_until_boundary
                    )
    return on_token
