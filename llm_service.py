# core/llm_service.py

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLM:
    def __init__(
        self,
        _api_key=os.getenv("DASHSCOPE_API_KEY"),
        _base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        self.client = OpenAI(
            api_key=_api_key,
            base_url=_base_url,
        )
        self.prompt = ""

        self.model = "qwen3.5-plus"

    def chat(self, prompt: str) -> str:
        self.prompt = prompt
        message: list = [
            {"role": "user", "content": self.prompt},
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            temperature=0.01,
        )
        return str(completion.choices[0].message.content)


def main() -> None:
    from prompt import MedicalToMPromptGenerator

    file_path = Path("ehr_bench_decision_making.jsonl")
    mkp = MedicalToMPromptGenerator()
    mkp.read_datas(file_path)
    prompts: list[str] = mkp.run()

    prompt = prompts[2]
    llm = LLM()
    print(llm.chat(prompt))


if __name__ == "__main__":
    main()
