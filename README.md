# LLM Trace

This script demonstrates how to generate a summary from technical text using OpenAI's GPT-4o model and log the process to Langfuse and Phoenix for traceability and quality evaluation.

## Features

* Traces the full generation flow using Langfuse spans
* Supports alternative trace and eval flow using Arize Phoenix
* Creates a fake reference summary span for future LLM-as-a-judge comparison (Langfuse)
* Sends prediction-reference pairs for LLM-based scoring (Phoenix)
* Logs prompt, output, and score for automatic evaluation
* Uses the official `langfuse` SDK and `langfuse.openai` proxy client
* Uses `arize-phoenix-client` for traces and LLM-as-a-judge evaluation

## Requirements

* Python 3.11+
* Langfuse instance running locally or remotely
* Phoenix instance running locally (`http://localhost:6006`)
* Valid `.env` file with:

```
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3000
PHOENIX_HOST=http://localhost:6006
OPENAI_API_KEY=...
```

## Installation

```bash
uv venv .venv
uv sync
```

## Usage

1. To run the trace+eval app with Langfuse:

```bash
uv run python langfuse_trace.py
```

2. To run the trace+eval app with Phoenix:

```bash
uv run python phoenix_trace.py
```

The script will:

1. Load the input and reference summary from `dataset.json`

2. For Langfuse:

   * Start a trace named `summary-flow`
   * Create two spans: a reference summary stub and a real OpenAI generation
   * Mark the generation span with `pending` eval status
   * Send metadata and model info

3. For Phoenix:

   * Collect prediction-reference pairs into a dataset
   * Send them to Phoenix’s LLM-as-a-judge scoring via OpenAI

## Output

If everything works, the script prints the trace ID and generation ID (Langfuse), or a `✅ Done` message for Phoenix evals.

## Notes

* The `langfuse.openai` module wraps OpenAI’s client and logs requests automatically.
* The Phoenix `OpenAIJudge` uses GPT-4o to evaluate your app’s predictions vs references.
* You can run both tools side-by-side and compare how they track quality and traceability.

## Decorator Example

The `langfuse_decorator.py` script shows how to hide the tracing boilerplate in
a decorator.  Simply annotate your generation function with
`evaluate_with_langfuse()` and call it once to run evaluation for every dataset
item:

```python
@evaluate_with_langfuse(
    dataset_path="./dataset.json",
    dataset_name="summarizer",
    model="gpt-4o",
    mode="local",
    eval_fn=my_eval_fn,
)
def summarize(text: str) -> str:
    ...

summarize()
```

Run it with:

```bash
uv run python langfuse_decorator.py
```
