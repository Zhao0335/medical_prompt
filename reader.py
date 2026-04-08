import json
from pathlib import Path


class Reader:
    @staticmethod
    def read(file_path: Path) -> list[dict] | None:
        with file_path.open("r", encoding="utf-8") as f:
            datas: list[dict] = [json.loads(line) for line in f if line.strip()]
        if not datas or datas == []:
            return None
        return datas


def main() -> None:
    file1 = Path("ehr_bench_decision_making.jsonl")
    datas = Reader.read(file1)
    assert datas is not None
    for line in datas:
        print(line)


if __name__ == "__main__":
    main()
