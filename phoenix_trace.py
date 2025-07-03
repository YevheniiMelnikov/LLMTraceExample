import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from phoenix.evals import OpenAIModel, QAEvaluator, run_evals
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace

# --- ENV и OpenAI -----------------------------------------------------------
load_dotenv()
client = OpenAI()

# --- Phoenix OTEL + OpenInference -------------------------------------------
tracer_provider = register(project_name="default", batch=True)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = trace.get_tracer("summarizer")

print("Phoenix is running on Docker at http://localhost:6006")

# --- constants --------------------------------------------------------------
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

        with tracer.start_as_current_span("generate_summary") as span:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
            )
            output_text = resp.choices[0].message.content.strip()
            span.set_attribute("input", input_text)
            span.set_attribute("output", output_text)
            span_id = span.get_span_context().span_id.to_bytes(8, "big").hex()

        records.append({
            "context.span_id": span_id,
            "input": input_text,
            "output": output_text,
            "reference": expected_output
        })

    df = pd.DataFrame(records).set_index("context.span_id")
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

    print("\nPhoenix UI is running at: http://localhost:6006")
    input("Press Enter to close the script...\n")

if __name__ == "__main__":
    run_summary_eval(DATASET_PATH)

