"""
Terminal live demo: thinker (left) | writer (right) + mode-switch log (bottom).

Usage (from repo root):
    python live_demo.py                          # uses default problem
    python live_demo.py "Prove sqrt(2) is irrational."
    MODEL_NAME=Qwen/Qwen3-32B-AWQ BUDGET=512 python live_demo.py

Environment variables:
    MODEL_NAME      HuggingFace model id  (default: Qwen/Qwen3-32B)
    BUDGET          Max generation steps  (default: 1024)
    MAX_LOG_LINES   Lines in the log pane (default: 8)
    USE_FAST_KERNEL Use fast CUDA kernels (default: 0)
"""

import argparse
import os
import sys
import torch
import transformers
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

sys.path.insert(0, ".")

from async_reasoning.cache import State
from async_reasoning.live_solver import LiveSolver

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B")
BUDGET          = int(os.environ.get("BUDGET", "1024"))
MAX_LOG_LINES   = int(os.environ.get("MAX_LOG_LINES", "8"))
USE_FAST_KERNEL = os.environ.get("USE_FAST_KERNEL", "0") == "1"
DEFAULT_PROBLEM = r"Calculate x - x^2 + x^3 for x = 5,6,7,8. Return all 4 answers in \boxed{ }."

# ── State colours ─────────────────────────────────────────────────────────────
_STATE_COLOR = {
    State.thinker_only:       "yellow",
    State.thinker_and_writer: "green",
    State.writer_only:        "blue",
}


# ── Rich display ──────────────────────────────────────────────────────────────
class LiveDisplay:
    """Mutable state that knows how to render itself as a Rich Layout.

    Rich calls ``__rich__()`` on every Live refresh (4×/s by default).
    The solver callbacks just mutate the fields; the refresh loop picks them up.
    """

    def __init__(self, console: Console, problem: str = "", max_log: int = 8):
        self.console  = console
        self.problem  = problem
        self.max_log  = max_log
        self.log: deque[str] = deque(maxlen=max_log)
        self._thinker = ""
        self._writer  = ""
        self._state   = State.thinker_only
        self._step    = 0

    # ── solver callbacks ──────────────────────────────────────────────────────

    def on_tokens(self, writer_tokens, thinker_tokens, token_times, eos, state, tokenizer, step=0):
        self._state  = state
        self._step   = step
        # skip the 4-token prefix (<|im_end|>\n<|im_start|>assistant\n<think>\n)
        self._thinker = tokenizer.decode(thinker_tokens[4:])
        self._writer  = tokenizer.decode(writer_tokens)

    def on_mode_switch(self, step: int, answer: bool):
        tag = "[bold green]YES[/bold green]" if answer else "[bold red] NO[/bold red]"
        self.log.append(f"[dim]step {step:5d}[/dim] │ continue writing? → {tag}")

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tail_visual_rows(text: str, max_rows: int, panel_width: int) -> str:
        """Keep only the tail of *text* that fits within *max_rows* visual rows.

        A single logical line (no newline) wraps across ceil(len/panel_width)
        terminal rows, so this counts visual rows rather than newline count.
        Drops whole logical lines from the top — never mid-token.
        """
        lines = text.split("\n")
        kept, rows = [], 0
        for line in reversed(lines):
            line_rows = max(1, (len(line) + panel_width - 1) // panel_width)
            if rows + line_rows > max_rows:
                break
            kept.append(line)
            rows += line_rows
        return "\n".join(reversed(kept))

    # ── Rich renderable protocol ──────────────────────────────────────────────

    def __rich__(self) -> Layout:
        w, h       = self.console.width, self.console.height
        color      = _STATE_COLOR.get(self._state, "white")
        state_name = self._state.name if self._state else "unknown"

        # header: 2 rows (state line + prompt line), log panel, main panel borders
        header_size = 2
        panel_rows  = max(4, (h - header_size - self.max_log - 4) * 3 // 4)
        # inner width of each half-panel (border chars on each side)
        panel_width = max(20, w // 2 - 4)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=header_size),
            Layout(name="main"),
            Layout(name="log", size=self.max_log + 2),
        )
        layout["main"].split_row(
            Layout(name="thinker"),
            Layout(name="writer"),
        )

        prompt_display = self.problem if len(self.problem) <= w - 4 else self.problem[:w - 7] + "…"
        header_text = Text()
        header_text.append(f" ● {state_name}   step: {self._step}\n", style=f"bold {color}")
        header_text.append(f" {prompt_display}", style="dim")
        layout["header"].update(header_text)

        layout["thinker"].update(Panel(
            Text(self._tail_visual_rows(self._thinker, panel_rows, panel_width), overflow="fold"),
            title="[bold yellow]THINKER[/bold yellow]",
            border_style="yellow",
        ))
        layout["writer"].update(Panel(
            Text(self._tail_visual_rows(self._writer, panel_rows, panel_width), overflow="fold"),
            title="[bold blue]WRITER[/bold blue]",
            border_style="blue",
        ))
        layout["log"].update(Panel(
            Text.from_markup("\n".join(self.log) or "[dim]no checks yet…[/dim]"),
            title="[bold]Mode Switch Log[/bold]",
            border_style="dim",
        ))

        return layout


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AsyncReasoning live terminal demo")
    parser.add_argument("problem", nargs="?", default=DEFAULT_PROBLEM,
                        help="Problem to solve (default: built-in arithmetic example)")
    args = parser.parse_args()
    problem = args.problem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console = Console()
    console.print(f"Loading [bold]{MODEL_NAME}[/bold] on {device} …")

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", low_cpu_mem_usage=True, device_map=device
    )

    sys_toks = [k for k in tokenizer.vocab if k.endswith("SYSTEM") or k.endswith("SYSTEM:")]
    writer_forbidden  = [tokenizer.vocab[x] for x in ["</think>", "<|im_start|>", "<|endoftext|>"] + sys_toks]
    thinker_forbidden = [tokenizer.vocab[x] for x in ["</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"] + sys_toks]

    display = LiveDisplay(console, problem=problem, max_log=MAX_LOG_LINES)

    solver = LiveSolver(
        model=model,
        tokenizer=tokenizer,
        thinker_forbidden_token_ix=thinker_forbidden,
        writer_forbidden_token_ix=writer_forbidden,
        on_mode_switch=display.on_mode_switch,
        use_fast_kernel=USE_FAST_KERNEL,
    )

    with Live(display, console=console, refresh_per_second=4, screen=True) as live:
        writer_out, thinker_out, token_times, eos = solver.solve(
            problem,
            budget=BUDGET,
            on_new_tokens_generated=lambda *args: display.on_tokens(*args, tokenizer=tokenizer, step=solver._step),
        )
        if not eos:
            display.on_mode_switch(-1, False)  # surface budget-exhausted hint in log
        live.update(display)
        input("\n  Generation complete — press Enter to exit…")

    console.print(f"\n[bold]Writer output:[/bold]\n{writer_out}")


if __name__ == "__main__":
    main()
