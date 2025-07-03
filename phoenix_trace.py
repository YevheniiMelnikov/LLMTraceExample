import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from phoenix.evals import OpenAIModel, QAEvaluator, run_evals
from phoenix.trace.tracer import Tracer
from phoenix.trace.span import SpanKind

load_dotenv()
print("Phoenix is running on Docker at http://localhost:6006")

client = OpenAI()
tracer = Tracer(service_name="summarizer")

DATASET_PATH = "./dataset.json"
SYSTEM_PROMPT = "Summarize the following technical description in 3-4 sentences."

def run_summary_eval(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    print("Generating model predictions...")

    for item in data:
        input_text = item["input"]
        expected_output = item["expected_output"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ]

        with tracer.start_as_current_span(
            name="generate_summary",
            kind=SpanKind.LLM,
            attributes={"model": "gpt-4o"}
        ) as span:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
            )
            output_text = resp.choices[0].message.content.strip()
            span.set_attribute("input", input_text)
            span.set_attribute("output", output_text)

        records.append({
            "input": input_text,
            "output": output_text,
            "reference": expected_output
        })

    df = pd.DataFrame(records)
    print("Model predictions generated. Starting evaluations...")

    eval_model = OpenAIModel(model="gpt-4o")
    qa_evaluator = QAEvaluator(eval_model)

    evals_df = run_evals(
        dataframe=df,
        evaluators=[qa_evaluator],
        provide_explanation=True
    )[0]

    results_df = pd.concat([df, evals_df], axis=1)

    print("✅ Done: evaluations complete. Check the Phoenix UI.")
    print("Evaluation results:")
    print(results_df.head())

    print(f"\nPhoenix UI is running at: http://localhost:6006")
    input("Press Enter to close the script...\n")


if __name__ == "__main__":
    run_summary_eval(DATASET_PATH)
