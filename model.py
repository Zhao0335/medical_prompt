from pathlib import Path

from ollama import ChatResponse, chat

from prompt import MedicalToMPromptGenerator

file_path = Path("ehr_bench_decision_making.jsonl")
mkp = MedicalToMPromptGenerator()
mkp.read_datas(file_path)
prompts: list[str] = mkp.run()

prompt = prompts[0]


response: ChatResponse = chat(
    model="qwen2.5:14b",
    messages=[
        {
            "role": "user",
            "content": prompt,
        },
    ],
)
print(response.message.content)
