# Langfuse LLM Summary Evaluator

This script demonstrates how to generate a summary from technical text using OpenAI's GPT-4o model and log the process to Langfuse for traceability and quality evaluation.

## Features

* Traces the full generation flow using Langfuse spans
* Creates a fake reference summary span for future LLM-as-a-judge comparison
* Logs prompt, output, and score for automatic evaluation
* Uses the official `langfuse` SDK and `langfuse.openai` proxy client

## Requirements

* Python 3.11+
* Langfuse instance running locally or remotely
* Valid `.env` file with:

```
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3000
OPENAI_API_KEY=...
```

## Installation

```bash
uv venv  .venv
uv sync
```

## Usage

```bash
python main.py
```

The script will:

1. Load the input and reference summary from `text.py`
2. Start a Langfuse trace (`summary-flow`)
3. Create two spans:

   * A fake reference generation span (`reference-summary-fake-gen`)
   * The real LLM call span (`generate_summary`)
4. Call OpenAI’s GPT-4o to generate a summary
5. Log the generation and mark it as `pending` for later quality scoring
6. Print the trace and generation IDs

## Output

If everything works, the script prints the trace ID, generation ID, and generated summary.

## Notes

* The `langfuse.openai` module wraps OpenAI’s client and logs requests automatically.
* All traces and scores are visible in the Langfuse UI under the configured project.
