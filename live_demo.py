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
    LOG_FILE        Path for visualization log (default: live_demo_output.log)
"""

import argparse
import logging
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

viz_logger = logging.getLogger("live_demo.viz")

sys.path.insert(0, ".")

from async_reasoning.cache import State
from async_reasoning.live_solver import LiveSolver
from async_reasoning.prompting import AsyncReasoningPrompting

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B")
BUDGET          = int(os.environ.get("BUDGET", "1024"))
MAX_LOG_LINES   = int(os.environ.get("MAX_LOG_LINES", "8"))
USE_FAST_KERNEL = os.environ.get("USE_FAST_KERNEL", "0") == "1"
DEFAULT_PROBLEM = r"Calculate x - x^2 + x^3 for x = 5,6,7,8. Return all 4 answers in \boxed{ }."
LOG_FILE        = os.environ.get("LOG_FILE", "live_demo_output.log")

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

    def __init__(self, console: Console, problem: str = "",
                 input_prompt: str = "", thinker_output_prefix: str = "",
                 writer_output_prefix: str = "",
                 thinker_prefix_ntokens: int = 0, writer_prefix_ntokens: int = 0,
                 mode_switch_question: str = "", max_log: int = 8):
        self.console  = console
        self.problem  = problem
        self.input_prompt = input_prompt
        self.thinker_output_prefix = thinker_output_prefix  # <|im_end|>\n<|im_start|>assistant\n<think>\n
        self.writer_output_prefix = writer_output_prefix      # ... [SYSTEM: ...]\n</think>\n
        self.thinker_prefix_ntokens = thinker_prefix_ntokens
        self.writer_prefix_ntokens = writer_prefix_ntokens
        # collapse to single line for compact log display
        self.mode_switch_question = " ".join(mode_switch_question.split()).strip()
        self.max_log  = max_log
        self.log: deque[str] = deque(maxlen=max_log)
        self._thinker_gen  = ""   # thinker generated text only (no prefix)
        self._writer_gen   = ""   # writer generated text only (no prefix)
        self._state   = State.thinker_only
        self._step    = 0

    # ── solver callbacks ──────────────────────────────────────────────────────

    def on_tokens(self, writer_tokens, thinker_tokens, token_times, eos, state, tokenizer, step=0):
        self._state  = state
        self._step   = step
        # thinker: skip prefix tokens to get generated text only
        self._thinker_gen = tokenizer.decode(thinker_tokens[self.thinker_prefix_ntokens:])
        # writer: skip prefix tokens to get generated text only
        self._writer_gen = tokenizer.decode(writer_tokens[self.writer_prefix_ntokens:])
        # log every 50 steps for post-run inspection
        if step % 50 == 0:
            viz_logger.info(
                "step=%d state=%s\n── thinker ──\n%s%s\n── writer ──\n%s%s",
                step, state.name,
                self.thinker_output_prefix, self._thinker_gen,
                self.writer_output_prefix, self._writer_gen,
            )

    def on_mode_switch(self, step: int, answer: bool):
        tag = "[bold green]YES[/bold green]" if answer else "[bold red] NO[/bold red]"
        self.log.append(f"[dim]step {step:5d}[/dim] │ {self.mode_switch_question} → {tag}")
        viz_logger.info("step=%d mode_switch=%s", step, "YES" if answer else "NO")

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

        # sizing
        prompt_size = 4            # fixed prompt panel height
        header_size = 1
        panel_width = max(20, w // 2 - 4)
        total_main  = max(8, h - header_size - prompt_size - self.max_log - 4)
        own_rows    = max(3, total_main * 2 // 3)
        other_rows  = max(2, total_main - own_rows)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=header_size),
            Layout(name="prompt", size=prompt_size),
            Layout(name="main"),
            Layout(name="log", size=self.max_log + 2),
        )
        layout["main"].split_row(
            Layout(name="thinker_col"),
            Layout(name="writer_col"),
        )
        layout["thinker_col"].split_column(
            Layout(name="thinker_other", ratio=1),
            Layout(name="thinker_own", ratio=2),
        )
        layout["writer_col"].split_column(
            Layout(name="writer_other", ratio=1),
            Layout(name="writer_own", ratio=2),
        )

        # ── header (state + step) ────────────────────────────────────────────
        layout["header"].update(
            Text(f" ● {state_name}   step: {self._step}", style=f"bold {color}")
        )

        # ── user prompt panel ─────────────────────────────────────────────────
        layout["prompt"].update(Panel(
            Text(self.input_prompt, style="dim", overflow="fold"),
            title="[dim]User Prompt[/dim]",
            border_style="dim",
        ))

        # ── helper: dim prefix (always visible) + trimmed body ────────────────
        def _prefixed(prefix: str, body: str, max_rows: int, pw: int) -> Text:
            """Render *prefix* (dim, always shown) followed by the tail of *body*."""
            prefix_lines = prefix.count("\n") + 1
            body_rows = max(1, max_rows - prefix_lines)
            trimmed = self._tail_visual_rows(body, body_rows, pw)
            text = Text(overflow="fold")
            text.append(prefix, style="dim")
            text.append(trimmed)
            return text

        # ── Thinker column ────────────────────────────────────────────────────
        # top: writer output (previous turn in thinker view) with </think> prefix
        layout["thinker_other"].update(Panel(
            _prefixed(self.writer_output_prefix, self._writer_gen,
                      other_rows, panel_width),
            title="[dim]Writer output (previous turn in thinker view)[/dim]",
            border_style="dim yellow",
        ))
        # bottom: <|im_end|><|im_start|>assistant<think> (dim, fixed) + generated thinker
        layout["thinker_own"].update(Panel(
            _prefixed(self.thinker_output_prefix, self._thinker_gen,
                      own_rows, panel_width),
            title="[bold yellow]THINKER[/bold yellow]",
            border_style="yellow",
        ))

        # ── Writer column ─────────────────────────────────────────────────────
        # top: <|im_end|><|im_start|>assistant<think> (dim, fixed) + thinker generated
        layout["writer_other"].update(Panel(
            _prefixed(self.thinker_output_prefix, self._thinker_gen,
                      other_rows, panel_width),
            title="[dim]Thinker context (<think> block in writer view)[/dim]",
            border_style="dim blue",
        ))
        # bottom: ...[SYSTEM: ...]</think> (dim, fixed) + writer generated
        layout["writer_own"].update(Panel(
            _prefixed(self.writer_output_prefix, self._writer_gen,
                      own_rows, panel_width),
            title="[bold blue]WRITER[/bold blue]",
            border_style="blue",
        ))

        # ── log ───────────────────────────────────────────────────────────────
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

    # set up visualization log file
    fh = logging.FileHandler(LOG_FILE, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    viz_logger.addHandler(fh)
    viz_logger.setLevel(logging.INFO)

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

    prompting = AsyncReasoningPrompting(problem)
    tokenizer_kwargs = dict(add_special_tokens=False, return_tensors='pt', padding=True, padding_side='left')
    # +1 for the \n\n token the solver appends after each prefix
    thinker_prefix_ntokens = len(tokenizer.encode(prompting.thinker_output_prefix, **tokenizer_kwargs).flatten()) + 1
    writer_prefix_ntokens = len(tokenizer.encode(prompting.writer_output_prefix, **tokenizer_kwargs).flatten()) + 1

    display = LiveDisplay(
        console, problem=problem,
        input_prompt=prompting.input_prompt,
        thinker_output_prefix=prompting.thinker_output_prefix,
        writer_output_prefix=prompting.writer_output_prefix,
        thinker_prefix_ntokens=thinker_prefix_ntokens,
        writer_prefix_ntokens=writer_prefix_ntokens,
        mode_switch_question=prompting.mode_switching_question,
        max_log=MAX_LOG_LINES,
    )

    solver = LiveSolver(
        model=model,
        tokenizer=tokenizer,
        thinker_forbidden_token_ix=thinker_forbidden,
        writer_forbidden_token_ix=writer_forbidden,
        on_mode_switch=display.on_mode_switch,
        use_fast_kernel=USE_FAST_KERNEL,
    )

    with Live(display, console=console, refresh_per_second=10, screen=True) as live:
        writer_out, thinker_out, token_times, eos = solver.solve(
            problem,
            budget=BUDGET,
            on_new_tokens_generated=lambda *args: display.on_tokens(*args, tokenizer=tokenizer, step=solver._step),
        )
        if not eos:
            display.on_mode_switch(-1, False)  # surface budget-exhausted hint in log
        viz_logger.info(
            "FINAL step=%d eos=%s\n── thinker ──\n%s%s\n── writer ──\n%s%s",
            solver._step, eos,
            display.thinker_output_prefix, display._thinker_gen,
            display.writer_output_prefix, display._writer_gen,
        )
        live.update(display)
        input("\n  Generation complete — press Enter to exit…")

    console.print(f"\n[bold]Writer output:[/bold]\n{writer_out}")
    console.print(f"\n[dim]Visualization log: {LOG_FILE}[/dim]")


if __name__ == "__main__":
    main()
