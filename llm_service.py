# core/llm_service.py

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        logger.info(f"已发送对话到 API \n Prompt: {self.prompt}")
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            temperature=0.01,
        )
        return str(completion.choices[0].message.content)


def main() -> None:
    llm = LLM()


if __name__ == "__main__":
    main()
