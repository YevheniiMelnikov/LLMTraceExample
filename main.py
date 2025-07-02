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
DATASET_NAME = "summarizer"
SYSTEM_PROMPT = "Summarize the following technical description in 3-4 sentences."


def my_eval_fn(inp: str, out: str, ref: str) -> float:
    return 1.0 if ref in out else 0.0


def run_local(path: str) -> None:
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


def run_remote(dataset_name: str) -> None:
    dataset = lf.get_dataset(dataset_name)

    for item in dataset.items:
        with item.run(
            run_name="summary-eval-run",
            run_description="LLM summary + eval",
            run_metadata={"model": "gpt-4o"},
        ) as root:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.input},
            ]
            with lf.start_as_current_generation(
                name="generate_summary",
                model="gpt-4o",
                input=messages,
            ) as gen:
                resp = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.3,
                )
                summary = resp.choices[0].message.content.strip()
                gen.update(output=summary)

            root.score_trace(
                name="faithfulness-test",
                value=my_eval_fn(item.input, summary, item.expected_output),
                comment="string match for now",
            )


if __name__ == "__main__":
    mode = input("Выбери режим: [1] локальный JSON, [2] UI-датасет Langfuse: ").strip()
    if mode == "1":
        run_local(DATASET_PATH)
    elif mode == "2":
        run_remote(DATASET_NAME)
    else:
        print("Неверный режим.")
    lf.flush()
