import json
import os
import logging
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import openai

load_dotenv()

os.environ["LANGFUSE_DEBUG"] = "True"
logging.getLogger("langfuse").setLevel(logging.DEBUG)

lf = get_client()
assert lf.auth_check(), "Langfuse creds / host invalid"

EVALUATOR_ID = "91e737ff-4706-4521-9fe1-660cdbca5e2e"
DATASET_PATH = "./dataset.json"
SYSTEM_PROMPT = "Summarize the following technical description in 3-4 sentences."


def run_evaluation_from_local_dataset(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, item in enumerate(data):
        input_text = item["input"]
        expected_output = item["expected_output"]

        with lf.start_as_current_span(
            name="summary-eval-span",
            input=input_text,
            metadata={
                "dataset_id": EVALUATOR_ID,
                "dataset_item_id": str(idx),
            },
        ) as root:

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ]

            with lf.start_as_current_generation(
                name="generate_summary",
                model="gpt-4o",
                input=messages,
                metadata={
                    "dataset_id": EVALUATOR_ID,
                    "dataset_item_id": str(idx),
                    "expected_output": expected_output,
                },
            ) as gen:

                resp = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.3,
                )
                summary = resp.choices[0].message.content.strip()
                gen.update(output=summary)

                gen.score(
                    name="Faithfulness",
                    value="pending",
                    data_type="CATEGORICAL",
                    comment="auto-eval pipeline",
                )


if __name__ == "__main__":
    run_evaluation_from_local_dataset(DATASET_PATH)
    lf.flush()
