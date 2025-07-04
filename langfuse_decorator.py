from __future__ import annotations

import json
import logging
import os
from functools import wraps
from typing import Callable, Iterable

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

os.environ["LANGFUSE_DEBUG"] = "True"
logging.getLogger("langfuse").setLevel(logging.DEBUG)

lf = get_client()
assert lf.auth_check(), "Langfuse creds / host invalid"

SYSTEM_PROMPT = "Summarize the following technical description in 3-4 sentences."
EVALUATOR_ID = "91e737ff-4706-4521-9fe1-660cdbca5e2e"


def evaluate_with_langfuse(
    *,
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    model: str = "gpt-4o",
    mode: str = "local",
    eval_fn: Callable[[str, str, str], float] | None = None,
) -> Callable[[Callable[[str], str]], Callable[[], None]]:
    """Decorate a function to run it over a dataset and log to Langfuse."""

    def decorator(func: Callable[[str], str]) -> Callable[[], None]:
        @wraps(func)
        def wrapper() -> None:
            if mode == "local":
                if not dataset_path:
                    raise ValueError("dataset_path must be provided for local mode")
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data: Iterable[dict[str, str]] = json.load(f)
                for idx, item in enumerate(data):
                    inp = item["input"]
                    ref = item.get("expected_output", "")
                    with lf.start_as_current_span(
                        name="decorator-eval",
                        input=inp,
                        metadata={
                            "dataset_id": EVALUATOR_ID,
                            "dataset_item_id": str(idx),
                        },
                    ) as root:
                        out = func(inp)
                        if eval_fn is not None:
                            score = eval_fn(inp, out, ref)
                            root.score_trace(
                                name="auto-eval",
                                value=score,
                                comment="decorator evaluation",
                            )
            elif mode == "remote":
                if not dataset_name:
                    raise ValueError("dataset_name must be provided for remote mode")
                dataset = lf.get_dataset(dataset_name)
                for item in dataset.items:
                    with item.run(
                        run_name="decorator-eval",
                        run_description="decorator evaluation",
                        run_metadata={"model": model},
                    ) as root:
                        out = func(item.input)
                        if eval_fn is not None:
                            score = eval_fn(item.input, out, item.expected_output)
                            root.score_trace(
                                name="auto-eval",
                                value=score,
                                comment="decorator evaluation",
                            )
            else:
                raise ValueError("mode must be 'local' or 'remote'")
            lf.flush()

        return wrapper

    return decorator


from langfuse.openai import openai


def my_eval_fn(inp: str, out: str, ref: str) -> float:
    return 1.0 if ref in out else 0.0


@evaluate_with_langfuse(
    dataset_path="./dataset.json",
    dataset_name="summarizer",
    model="gpt-4o",
    mode="local",
    eval_fn=my_eval_fn,
)
def summarize(text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    summarize()
