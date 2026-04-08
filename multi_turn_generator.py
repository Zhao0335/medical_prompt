import json
import re
import os

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from prompt import MedicalToMPromptGenerator

load_dotenv()


class MultiTurnGenerator:
    def __init__(
        self,
        turns: int = 4,
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-plus",
    ) -> None:
        self.generator = MedicalToMPromptGenerator(turns=turns)
        self.n_turns = turns
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return str(completion.choices[0].message.content)

    def _extract_json_from_response(self, response: str) -> dict[str, Any] | list[Any]:
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                return json.loads(json_match.group())
            array_match = re.search(r"\[[\s\S]*\]", response)
            if array_match:
                return json.loads(array_match.group())
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response[:500]}...")
            raise ValueError("Failed to parse JSON from LLM response")

    def generate_single_sample(self, sample_data: dict[str, Any]) -> dict[str, Any]:
        medical_input = MedicalToMPromptGenerator._safe_str(sample_data.get("input", ""))
        candidates = MedicalToMPromptGenerator._normalize_candidates(sample_data.get("candidates", []))
        ground_truth = MedicalToMPromptGenerator._extract_ground_truth(sample_data, candidates)

        if candidates:
            candidates_text = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(candidates)])
        else:
            candidates_text = "No candidates provided. Use best clinically justified recommendation."

        print("=" * 80)
        print(f"[Phase 1] Generating initial dialogue structure...")
        print("=" * 80)

        phase1_prompt = self.generator.generate_phase1_prompt(sample_data)
        phase1_response = self._call_llm(phase1_prompt)
        initial_structure = self._extract_json_from_response(phase1_response)

        if not isinstance(initial_structure, dict):
            raise ValueError("Phase 1 must return a dictionary (initial structure)")

        current_dialogue = initial_structure.get("prompt", [])
        if not current_dialogue:
            raise ValueError("Phase 1 did not generate any dialogue messages")

        print(f"✓ Generated {len(current_dialogue)} initial messages")
        print(f"  - System prompt + User complaint ready")
        print()

        for turn_num in range(1, self.n_turns + 1):
            is_final = turn_num == self.n_turns
            turn_label = f"[Final Turn {turn_num}]" if is_final else f"[Turn {turn_num}]"

            print("=" * 80)
            print(f"{turn_label} Generating assistant{' and patient' if not is_final else ''} message...")
            print("=" * 80)

            turn_prompt = self.generator.generate_intermediate_turn_prompt(
                current_dialogue=current_dialogue,
                turn_number=turn_num,
                total_turns=self.n_turns,
                ehr_evidence=medical_input,
                ground_truth=ground_truth,
                candidates_text=candidates_text,
            )

            turn_response = self._call_llm(turn_prompt)
            new_messages = self._extract_json_from_response(turn_response)

            if isinstance(new_messages, list):
                for msg in new_messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        current_dialogue.append(msg)
                        role = msg["role"]
                        content_preview = msg["content"][:100].replace("\n", " ")
                        print(f"  ✓ Added {role}: {content_preview}...")
            else:
                print(f"  ⚠ Unexpected format for turn {turn_num}")

            print()

        final_result = {
            **initial_structure,
            "prompt": current_dialogue,
        }

        return final_result

    def run(self, file_path: Path, output_path: Path | None = None, max_samples: int | None = None) -> list[dict[str, Any]]:
        ok = self.generator.read_datas(file_path)
        if not ok or self.generator.datas is None:
            raise RuntimeError("No valid data found in input file.")

        samples_to_process = self.generator.datas[:max_samples] if max_samples else self.generator.datas
        results: list[dict[str, Any]] = []

        for i, sample_data in enumerate(samples_to_process):
            idx = sample_data.get("idx", i)
            print(f"\n{'='*80}")
            print(f"[Sample {idx}] Processing...")
            print(f"{'='*80}\n")

            try:
                result = self.generate_single_sample(sample_data)
                results.append(result)

                print(f"\n✓ Sample {idx} completed successfully!")
                print(f"  Total dialogue turns: {len([m for m in result['prompt'] if m['role'] == 'assistant'])}")
                print()
            except Exception as e:
                print(f"\n✗ Error processing sample {idx}: {e}")
                continue

        if output_path and results:
            with output_path.open("w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False, indent=2))
                    f.write("\n")
            print(f"\n{'='*80}")
            print(f"Results saved to: {output_path}")
            print(f"Total successful samples: {len(results)}")
            print(f"{'='*80}")

        return results


def main() -> None:
    input_file = Path("ehr_bench_decision_making.jsonl")
    output_file = Path("multi_turn_results.jsonl")

    generator = MultiTurnGenerator(
        turns=4,
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    results = generator.run(
        file_path=input_file,
        output_path=output_file,
        max_samples=2,
    )

    print(f"\nGenerated {len(results)} complete multi-turn dialogues")


if __name__ == "__main__":
    main()
