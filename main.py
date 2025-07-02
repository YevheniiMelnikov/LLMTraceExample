import os
import logging
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import openai
from text import INPUT_TEXT, REFERENCE


load_dotenv()
os.environ["LANGFUSE_DEBUG"] = "True"
logging.getLogger("langfuse").setLevel(logging.DEBUG)

lf = get_client()
assert lf.auth_check(), "Langfuse creds / host invalid"

SYSTEM_PROMPT = "Summarize the following technical description in 3-4 sentences."

def generate_summary(prompt: str) -> str:
    # создаём корневой span (trace) для всего процесса генерации summary
    with lf.start_as_current_span(
        name="summary-flow",
        input=prompt
    ) as root:

        # создаём фейковую генерацию — якобы эталонное summary от LLM (для будущей оценки)
        with lf.start_as_current_generation(
            name="reference-summary-fake-gen",
            model="reference-placeholder",
            input=[{"role": "assistant", "content": REFERENCE}]
        ):
            pass  # здесь ничего не происходит, просто логируем как будто "сгенерировали" эталон

        # собираем промпт, который пойдёт в OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # создаём span для настоящего вызова модели
        with lf.start_as_current_generation(
            name="generate_summary",
            model="gpt-4o",
            input=messages,
        ) as gen:

            # делаем запрос к OpenAI и получаем summary
            resp = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
            )
            summary = resp.choices[0].message.content.strip()

            # логируем результат в Langfuse (output)
            gen.update(output=summary)

            # ставим метку для auto-eval: «оценить позже»
            gen.score(
                name="summary-quality",
                value="pending",
                data_type="CATEGORICAL",
                comment="auto-eval pipeline",
            )

            # выводим ID trace и generation в консоль
            print("🪪 trace:", root.id)
            print("🪪 generation:", gen.id)
            return summary



if __name__ == "__main__":
    summary = generate_summary(INPUT_TEXT.strip())
    lf.flush()
    print("\n✅ Summary:\n", summary)
