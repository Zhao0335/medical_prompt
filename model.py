import logging
from pathlib import Path

from ollama import ChatResponse, chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_path = Path("ehr_bench_decision_making.jsonl")


def qwq_chat(prompt: str) -> str:
    logger.info(f"已发送对话到 API \n Prompt: {prompt}")
    response: ChatResponse = chat(
        model="qwen2.5:14b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return str(response.message.content)
