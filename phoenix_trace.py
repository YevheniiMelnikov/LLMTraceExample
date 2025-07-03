import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from opentelemetry import trace

from phoenix.otel import register
from phoenix.evals import OpenAIModel, QAEvaluator, run_evals
from phoenix.trace import SpanEvaluations, log_span_evaluations
from openinference.instrumentation.openai import OpenAIInstrumentor

# --- подключаемся к Phoenix через OpenTelemetry -----------------------------
register(project_name="default", batch=True)  # 👈 фикс: имя проекта = default
OpenAIInstrumentor().instrument()
tracer = trace.get_tracer("summarizer")

# --- загрузка ENV и OpenAI клиента -----------------------------------------
load_dotenv()
client = OpenAI()

# --- пути и системный prompt ------------------------------------------------
DATASET_PATH = Path("dataset.json")
SYSTEM_PROMPT = "Summarize the following technical description in 3-4 sentences."

# --- основной запуск --------------------------------------------------------
def run_summary_eval(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    for item in data:
        with tracer.start_as_current_span("generate_summary") as span:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["input"]},
                ],
                temperature=0.3,
            )
            output = resp.choices[0].message.content.strip()
            span.set_attribute("input", item["input"])
            span.set_attribute("output", output)
            span_id = span.get_span_context().span_id.to_bytes(8, "big").hex()

        rows.append({
            "context.span_id": span_id,  # 👈 нужен для привязки eval к спану
            "input": item["input"],
            "output": output,
            "reference": item["expected_output"],
        })

    df = pd.DataFrame(rows).set_index("context.span_id")

    eval_df = run_evals(
        df,
        evaluators=[QAEvaluator(OpenAIModel(model="gpt-4o"))],
        provide_explanation=True,
    )[0]

    log_span_evaluations(SpanEvaluations(
        dataframe=eval_df,
        eval_name="QA-score"
    ))

    trace.get_tracer_provider().force_flush()
    print("✅ Всё отправлено в Phoenix: http://localhost:6006")

if __name__ == "__main__":
    run_summary_eval(DATASET_PATH)
