import os, json, logging
from functools import wraps
from typing import Callable, Literal

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import openai


load_dotenv()
os.environ["LANGFUSE_DEBUG"] = "True"
logging.getLogger("langfuse").setLevel(logging.DEBUG)

lf = get_client()
assert lf.auth_check(), "Langfuse creds / host invalid"

DATASET_PATH  = "./dataset.json"  # just a local path
DATASET_NAME  = "summarizer"  # same as in LangFuse
SYSTEM_PROMPT = "Summarize the following technical description in 3-4 sentences."


def evaluate_with_langfuse(
    *,
    dataset_path: str,
    dataset_name: str,
    model: str,
    mode: Literal["local", "remote"],
):
    def decorator(fn: Callable[[str], str]):
        @wraps(fn)
        def wrapper():
            if mode == "local":
                with open(dataset_path, encoding="utf-8") as f:
                    examples = json.load(f)

                for idx, ex in enumerate(examples):
                    inp = ex["input"]

                    with lf.start_as_current_span(
                        name="summary-eval-span",
                        input=inp,
                        metadata={
                            "dataset_id":   dataset_name,
                            "dataset_item_id": str(idx),
                        },
                    ) as span:
                        out = fn(inp)

                        span.score_trace(
                            name="Faithfulness",
                            value="pending",
                            data_type="CATEGORICAL",
                            comment="decorator auto-eval",
                        )

            elif mode == "remote":
                dataset = lf.get_dataset(dataset_name)
                for item in dataset.items:
                    with item.run(run_name="summary-eval-run",
                                  run_metadata={"model": model}) as span:
                        out = fn(item.input)

                        span.score_trace(
                            name="Faithfulness",
                            value="pending",
                            data_type="CATEGORICAL",
                            comment="remote auto-eval",
                        )
            else:
                raise ValueError("mode должен быть 'local' или 'remote'")

            lf.flush()
        return wrapper
    return decorator


@evaluate_with_langfuse(
    dataset_path=DATASET_PATH,
    dataset_name=DATASET_NAME,
    model="gpt-4o",
    mode="remote",  # or "local" to use local dataset
)
def summarize(text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    summarize()
