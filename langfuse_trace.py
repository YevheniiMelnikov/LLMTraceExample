import os
import json
import logging
from typing import Literal
from dotenv import load_dotenv
from langfuse import get_client, observe
from langfuse.openai import openai

load_dotenv()
os.environ["LANGFUSE_DEBUG"] = "True"
logging.getLogger("langfuse").setLevel(logging.DEBUG)

lf = get_client()
assert lf.auth_check(), "Langfuse creds / host invalid"

DATASET_PATH  = "./dataset.json"  # just a local path
DATASET_NAME  = "summarizer"  # same as in LangFuse
SYSTEM_PROMPT = "Summarize the following technical description in 3-4 sentences."
MODEL_NAME = "gpt-4o"
MODE: Literal["local", "remote"] = "remote"  # or "local"


@observe(name="summarize", as_type="generation", capture_input=True, capture_output=True)
def summarize(text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    resp = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def run_evaluation():
    if MODE == "local":
        with open(DATASET_PATH, encoding="utf-8") as f:
            examples = json.load(f)

        for idx, ex in enumerate(examples):
            with lf.start_span(
                name="local-summary-eval",
                input=ex["input"],
                metadata={"dataset": DATASET_NAME, "idx": str(idx), "model": MODEL_NAME},
            ) as span:
                summarize(ex["input"])
                span.score_trace(name="Faithfulness", value="pending", data_type="CATEGORICAL")

    elif MODE == "remote":
        dataset = lf.get_dataset(DATASET_NAME)
        for item in dataset.items:
            with item.run(run_name="summary-eval", run_metadata={"model": MODEL_NAME}) as span:
                summarize(item.input)
                span.score_trace(name="Faithfulness", value="pending", data_type="CATEGORICAL")

    else:
        raise ValueError("Invalid mode")

    lf.flush()


if __name__ == "__main__":
    run_evaluation()
