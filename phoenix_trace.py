import os, json, pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from phoenix.evals import OpenAIModel, run_evals
from phoenix.evals.evaluators import SummarizationEvaluator
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace

load_dotenv()
client = OpenAI()

tracer_provider = register(project_name="default", batch=True)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = trace.get_tracer("summarizer")


DATASET_PATH   = "./dataset.json"
SYSTEM_PROMPT  = "Summarize the following technical description in 3-4 sentences."

def run_summary_eval(path: str) -> None:
    data = json.load(open(path, encoding="utf-8"))
    rows = []

    for item in data:
        with tracer.start_as_current_span("generate_summary") as span:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": item["input"]},
                ],
                temperature=0.3,
            )
            output = resp.choices[0].message.content.strip()
            span.set_attribute("input",  item["input"])
            span.set_attribute("output", output)
            span_id = span.get_span_context().span_id.to_bytes(8, "big").hex()

        rows.append({
            "context.span_id": span_id,
            "input":  item["input"],
            "output": output,
            "reference": item["expected_output"],
        })

    df = pd.DataFrame(rows).set_index("context.span_id")
    print("LLM generation complete, running evaluations...")

    summarization_eval = SummarizationEvaluator(OpenAIModel(model="gpt-4o"))
    evals_df = run_evals(df, evaluators=[summarization_eval])[0]

    for span_id, row in evals_df.iterrows():
        with tracer.start_as_current_span("summary_score") as span:
            span.set_attribute("parent_span_id", span_id)
            span.set_attribute("score", row.get("score", -1))

            explanation = row.get("explanation")
            if isinstance(explanation, str) and explanation.strip():
                span.set_attribute("explanation", explanation)

    print(pd.concat([df, evals_df], axis=1).head())

if __name__ == "__main__":
    run_summary_eval(DATASET_PATH)
