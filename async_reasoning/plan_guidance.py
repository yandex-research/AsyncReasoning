from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union
import random
import re

PlanRater = Callable[[str, str], Union[float, bool]]

DEFAULT_PLAN_SYSTEM_PROMPT = (
    "Provide a concise, step-by-step plan for the user's request. "
    "Only output the plan. Use 3-6 numbered steps. Do not provide the final answer."
)

PLAN_END_MARKERS = (
    "\n\n",
    "\nAnswer:",
    "\nSolution:",
    "\nFinal:",
)


@dataclass
class PlanGuidanceConfig:
    strategy: str = "single"  # single, best_of_n, rewind_and_repeat
    num_plans: int = 1
    max_retries: int = 3
    temperature: float = 1.2
    max_new_tokens: int = 128
    plan_system_prompt: str = DEFAULT_PLAN_SYSTEM_PROMPT
    plan_output_prefix: str = "Plan:\n"
    plan_output_suffix: str = "\n\nAnswer:\n"
    plan_rater: Optional[PlanRater] = None
    accept_threshold: float = 0.0


def select_plan(
    model,
    tokenizer,
    problem: str,
    config: PlanGuidanceConfig,
    tokenizer_kwargs: dict,
) -> Tuple[str, dict]:
    if config.strategy not in {"single", "best_of_n", "rewind_and_repeat"}:
        raise ValueError(f"Unknown plan guidance strategy: {config.strategy}")
    if config.strategy == "best_of_n":
        return _select_best_of_n(model, tokenizer, problem, config, tokenizer_kwargs)
    if config.strategy == "rewind_and_repeat":
        return _select_rewind_and_repeat(model, tokenizer, problem, config, tokenizer_kwargs)
    plan = _generate_plan(
        model,
        tokenizer,
        problem,
        config.plan_system_prompt,
        config.temperature,
        config.max_new_tokens,
        tokenizer_kwargs,
    )
    return plan, {"plans": [plan], "strategy": "single"}


def _select_best_of_n(
    model,
    tokenizer,
    problem: str,
    config: PlanGuidanceConfig,
    tokenizer_kwargs: dict,
) -> Tuple[str, dict]:
    plans = [
        _generate_plan(
            model,
            tokenizer,
            problem,
            config.plan_system_prompt,
            config.temperature,
            config.max_new_tokens,
            tokenizer_kwargs,
        )
        for _ in range(max(1, config.num_plans))
    ]
    if config.plan_rater is not None:
        scores = [_score_from_rater(config.plan_rater, problem, plan) for plan in plans]
        best_idx = max(range(len(plans)), key=lambda i: scores[i])
    else:
        scores = [_heuristic_plan_score(plan) for plan in plans]
        best_idx = max(range(len(plans)), key=lambda i: scores[i])
    return plans[best_idx], {"plans": plans, "scores": scores, "strategy": "best_of_n"}


def _select_rewind_and_repeat(
    model,
    tokenizer,
    problem: str,
    config: PlanGuidanceConfig,
    tokenizer_kwargs: dict,
) -> Tuple[str, dict]:
    rejected: list[str] = []
    decisions: list[bool] = []
    for _ in range(max(1, config.max_retries)):
        plan = _generate_plan(
            model,
            tokenizer,
            problem,
            config.plan_system_prompt,
            config.temperature,
            config.max_new_tokens,
            tokenizer_kwargs,
            rejected_plans=rejected,
        )
        accept = True
        if config.plan_rater is not None:
            accept = _accept_from_rater(
                config.plan_rater,
                problem,
                plan,
                config.accept_threshold,
            )
        decisions.append(accept)
        if accept:
            return plan, {
                "plans": rejected + [plan],
                "decisions": decisions,
                "strategy": "rewind_and_repeat",
            }
        rejected.append(plan)
    fallback = random.choice(rejected) if rejected else ""
    return fallback, {"plans": rejected, "decisions": decisions, "strategy": "rewind_and_repeat"}


def _build_plan_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt}".strip()
    return user_prompt


def _build_plan_user_prompt(problem: str, rejected_plans: Optional[Sequence[str]]) -> str:
    if not rejected_plans:
        return problem
    rejected_lines = "\n".join(f"- {plan}" for plan in rejected_plans if plan)
    return (
        f"{problem}\n\nPreviously rejected plans:\n{rejected_lines}\n\n"
        "Provide a new, different plan that addresses the issues. Only output the plan."
    )


def _generate_plan(
    model,
    tokenizer,
    problem: str,
    plan_system_prompt: str,
    temperature: float,
    max_new_tokens: int,
    tokenizer_kwargs: dict,
    rejected_plans: Optional[Sequence[str]] = None,
) -> str:
    user_prompt = _build_plan_user_prompt(problem, rejected_plans)
    prompt = _build_plan_prompt(tokenizer, plan_system_prompt, user_prompt)
    inputs = tokenizer([prompt], **tokenizer_kwargs).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    do_sample = temperature is not None and temperature > 0
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = temperature
    outputs = model.generate(**inputs, **gen_kwargs)
    raw_text = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
    return _normalize_plan_text(raw_text)


def _normalize_plan_text(text: str) -> str:
    cleaned = text.strip().replace("<think>", "").replace("</think>", "").strip()
    if not cleaned:
        return ""
    lines = cleaned.splitlines()
    if lines and lines[0].strip().lower().startswith("plan"):
        cleaned = "\n".join(lines[1:]).strip()
    for marker in PLAN_END_MARKERS:
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].strip()
            break
    return cleaned


def _score_from_rater(rater: PlanRater, problem: str, plan: str) -> float:
    score = rater(problem, plan)
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    return float(score)


def _accept_from_rater(
    rater: PlanRater,
    problem: str,
    plan: str,
    threshold: float,
) -> bool:
    result = rater(problem, plan)
    if isinstance(result, bool):
        return result
    return float(result) >= threshold


def _heuristic_plan_score(plan: str) -> Tuple[int, int]:
    steps = _count_step_lines(plan)
    length = len(plan.split())
    return (steps, length)


def _count_step_lines(plan: str) -> int:
    count = 0
    for line in plan.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(\d+[\).\s]|[-*]\s+)", line):
            count += 1
    return count
