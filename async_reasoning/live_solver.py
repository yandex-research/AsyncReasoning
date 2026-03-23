"""LiveSolver: AsyncReasoningSolver subclass that fires a callback on mode-switch checks."""

import torch
from typing import Callable, Optional

from async_reasoning.solver import AsyncReasoningSolver


class LiveSolver(AsyncReasoningSolver):
    """Adds an ``on_mode_switch(step, answer)`` callback to AsyncReasoningSolver."""

    def __init__(self, *args, on_mode_switch: Optional[Callable[[int, bool], None]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_mode_switch = on_mode_switch
        self._step = 0

    @torch.inference_mode()
    def check_if_should_continue_writing(self, cache, prompting) -> bool:
        result = super().check_if_should_continue_writing(cache, prompting)
        if self.on_mode_switch is not None:
            self.on_mode_switch(self._step, result)
        return result

    def solve(self, problem, on_new_tokens_generated=None, **kwargs):
        self._step = 0

        def _wrapped(writer_tokens, thinker_tokens, token_times, eos, state):
            self._step += 1
            if on_new_tokens_generated is not None:
                on_new_tokens_generated(writer_tokens, thinker_tokens, token_times, eos, state)

        return super().solve(problem, on_new_tokens_generated=_wrapped, **kwargs)
