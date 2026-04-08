# main.py

import json
from pathlib import Path

from core.prompt import MedicalToMIncrementalGenerator
from core.reader import Reader
from core.vllm_service import LLM


def main() -> None:
    llm = LLM("models/base/Qwen2.5-7B-Coder-Instruct")
    llm.load_model()
    file = Path("ehr_bench_decision_making.jsonl")
    datas = Reader.read(file)
    assert datas is not None
    mp = MedicalToMIncrementalGenerator()
    count: int = 1
    for data in datas:
        result_file = Path(f"result{count}.json")
        with result_file.open("w", encoding="utf-8") as f:
            json.dump(mp.run(data, llm), f, ensure_ascii=False, indent=4)
        print(f"Done with line{count}!")
        count += 1


if __name__ == "__main__":
    main()
