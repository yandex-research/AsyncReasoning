import os
from pathlib import Path
import json
import transformers
import requests
import warnings

from typing import Callable, Optional, Sequence, TypeVar, Dict, Any, Protocol
T = TypeVar("T")

EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

def check_equality_judge(
        expr1: str, 
        expr2: str,
        max_tokens: int = 10,
    ):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}

    with open(f"{Path(__file__).resolve().parent}/eliza_config.json", "r") as f:
        config = json.loads("\n".join(f.readlines()))
    token, url, model = config["token"], config["url"], config["model"]
    headers = {"Authorization": f"OAuth {token}"}
    messages=[{"role": "user", "content": prompt}]
    payload = {
        "model": model, 
        "messages": messages,
        "max_tokens": max_tokens,
    }

    with warnings.catch_warnings(action="ignore"):
        response_json = requests.post(url, json=payload, headers=headers, verify=False).json()
    # response_text = response_json['response']['content'][0]["text"]
    response_text = response_json['response']['choices'][0]["message"]['content']
    return response_text.lower().strip() == "yes"


def check_equality_local_model(
        model: transformers.PreTrainedModel, 
        tokenizer: transformers.PreTrainedTokenizer, 
        expr1: str, 
        expr2: str,
    ):
    """
    IT DOES NOT YET WORK WITH FAST KERNEL!
    Example:
    from utils.answer_processing import check_equality_local_model
    >>> check_equality_local_model(model, tokenizer, "\\boxed{3, 2, 1}", "3, 2, 1") 
    <<< True
    """
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2} + "\n<think>\n\n</think>"
    tokenizer_kwargs = dict(add_special_tokens=False, return_tensors='pt', padding=True, padding_side='left')
    inputs = tokenizer([prompt], **tokenizer_kwargs).to(model.device)
    prompt_len = inputs["input_ids"][0].shape[0]
    outputs = model.generate(**inputs,
        use_cache=False, 
        past_key_values=None,
    )
    response_text = tokenizer.decode(outputs[0, prompt_len:-1])
    print(response_text)
    return response_text.lower().strip() == "yes"


def find_last_valid_expression(
    response: str, prefix: str = "\\boxed{", extract_result: Callable[[str], str] = lambda x: x[7:-1] # this is to extraced without \\boxed{...}
) -> Optional[str]:
    """
    Find the last correct brace sequence that starts with prefix and passes extract_result; return it including prefix
    """
    while True:
        try:
            start = response.rindex(prefix)
            try:
                excerpt = parse_until_valid_brace_sequence(response[start:], keep_prefix=True)
                return extract_result(excerpt)
            except Exception:  # missing suffix or extract_result failed
                response = response[:start]
        except ValueError:
            return None


def parse_until_valid_brace_sequence(text: str, start: int = 0, end: Optional[int] = None, keep_prefix: bool = False) -> str:
    original_start = start
    start = text.index('{', start)
    balance = 1
    for i in range(start + 1, end if end is not None else len(text)):
        if text[i] == '{':
            balance += 1
        elif text[i] == '}':
            balance -= 1
        if balance == 0:
            return text[(original_start if keep_prefix else start): i + 1]
    raise ValueError("text does not have a correct bracket {/} ")