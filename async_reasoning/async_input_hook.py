# THIS IS TEMPORARY SOLUTION FOR ASYNC INPUTS ABLATION

from async_reasoning.cache import State

def async_input_hook_constructor(solver, shard_to_target, target_reminders, next_shard_every_steps, problem_shards, defer_until_boundary=False):
    def on_token(writer_tokens, thinker_tokens, token_times, eos, state):
        for shard_idx, problem_shard in enumerate(problem_shards):
            if next_shard_every_steps <= 0 or len(thinker_tokens) < next_shard_every_steps * (shard_idx + 1):
                return
            
            target = "input"
            if target in shard_to_target and solver.live_context_queue.push_counter_per_target[target] == shard_idx:
                solver.live_context_queue.push_text(
                    f"\n\nADDITIONAL USER INPUT: {problem_shard}\n\n",
                    target=target,
                    defer_until_boundary=False
                )

            target = "writer"
            if target in shard_to_target and solver.live_context_queue.push_counter_per_target[target] == shard_idx:
                solver.live_context_queue.push_text(
                    f" ... [SYSTEM: additional user input detected]\n" if target in target_reminders else f"\n\nADDITIONAL USER INPUT:{problem_shard}\n\n",
                    target=target,
                    defer_until_boundary=defer_until_boundary
                )
                
            target = "thinker"
            if target in shard_to_target and solver.live_context_queue.push_counter_per_target[target] == shard_idx:
                solver.live_context_queue.push_text(
                    f" ... [SYSTEM: additional user input detected]\n"  if target in target_reminders else f"\n\nADDITIONAL USER INPUT:{problem_shard}\n\n",
                    target=target,
                    defer_until_boundary=defer_until_boundary
                )
    return on_token
