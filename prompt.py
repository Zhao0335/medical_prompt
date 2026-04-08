import json
from pathlib import Path
from typing import Any


class MedicalToMPromptGenerator:
    """
    Generate high-fidelity multi-turn medical dialogue prompts with explicit
    Theory-of-Mind (ToM) constraints and safe recommendation selection.
    """

    def __init__(self, turns: int = 4) -> None:
        if turns < 2:
            raise ValueError("turns must be >= 2")
        self.file_path: Path | None = None
        self.datas: list[dict[str, Any]] | None = None
        self.n_turns: int = turns

    @staticmethod
    def _safe_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _normalize_candidates(candidates: Any) -> list[str]:
        if not isinstance(candidates, list):
            return []
        normalized: list[str] = []
        for c in candidates:
            s = MedicalToMPromptGenerator._safe_str(c).strip()
            if s:
                normalized.append(s)
        return normalized

    @staticmethod
    def _extract_ground_truth(
        sample_data: dict[str, Any], candidates: list[str]
    ) -> str:
        """
        Try to extract ground truth from common fields.
        Fallback: first candidate if present, else empty string.
        """
        # Direct common keys
        for key in ("ground_truth", "label", "answer", "target", "recommendation"):
            v = sample_data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # Nested reward_model keys
        reward_model = sample_data.get("reward_model")
        if isinstance(reward_model, dict):
            for key in ("ground_truth", "label", "answer", "target"):
                v = reward_model.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        # Numeric index styles
        for key in ("ground_truth_idx", "label_idx", "answer_idx", "target_idx"):
            v = sample_data.get(key)
            if isinstance(v, int) and 0 <= v < len(candidates):
                return candidates[v]

        if candidates:
            return candidates[0]
        return ""

    @staticmethod
    def _escape_for_prompt(value: str) -> str:
        # Keep prompt stable and JSON-friendly when embedded in f-string blocks
        return value.replace("{", "{{").replace("}", "}}")

    def generate(self, sample_data: dict[str, Any]) -> str:
        medical_input = self._safe_str(sample_data.get("input", "")).strip()
        instruction = self._safe_str(sample_data.get("instruction", "")).strip()
        idx = sample_data.get("idx", 0)

        candidates = self._normalize_candidates(sample_data.get("candidates", []))
        ground_truth = self._extract_ground_truth(sample_data, candidates)

        # Ensure ground truth is one of candidates when possible
        if ground_truth and candidates and ground_truth not in candidates:
            candidates = [ground_truth] + [c for c in candidates if c != ground_truth]
        elif ground_truth and not candidates:
            candidates = [ground_truth]

        # Render candidates for instruction block
        if candidates:
            candidates_text = "\n".join(
                [f"{i + 1}. {c}" for i, c in enumerate(candidates)]
            )
        else:
            candidates_text = (
                "No candidates provided. Use best clinically justified recommendation."
            )

        # A robust, enforceable ToM protocol
        master_prompt = f"""
            ### [Role: Expert Clinical Dataset Simulator]
            You generate a HIGH-FIDELITY multi-turn medical consultation in JSON format.
            You must model the psychological state transition of the patient over time.

            ### [Case Metadata]
            - sample_idx: {idx}
            - total_assistant_turns: {self.n_turns}
            - optional_instruction: {self._escape_for_prompt(instruction)}

            ### [Knowledge Partition]
            - Assistant (Doctor): has full EHR access, labs, and differential risk awareness.
            - User (Patient): only knows symptoms, discomfort, fear, practical concerns; does NOT know raw labs unless explained by doctor.

            ### [Source EHR Evidence]
            {self._escape_for_prompt(medical_input)}

            ### [Recommendation Candidates]
            {self._escape_for_prompt(candidates_text)}

            ### [Core Objective: ToM-Driven Communication]
            For EVERY assistant turn, enforce:
            1) Infer current patient emotion and belief.
            2) Choose one communication strategy based on that mental state.
            3) Reveal at most ONE new key medical fact per turn.
            4) Pair medical information with empathy and clear next-step guidance.

            ### [Dynamic Multi-turn Protocol]

            - Turn 1 (Doctor): acknowledge acute symptom distress; ask ONE focused clarifying question.
            - Middle turns (Doctor): gradually disclose abnormalities and implications, while managing anxiety escalation.
            - Final turn (Doctor): provide final recommendation selected ONLY from candidates when candidates are provided; if no candidates, provide best justified recommendation.

            ### [Strict Output Format]
            Return ONLY valid JSON with this structure:
            {{
            "data_source": "mimic_iv_tom_multi",
            "topic": "Clinical Admission Reasoning",
            "department": "Emergency Medicine",
            "subdepartment": "Internal Medicine",
            "disease": "Chest pain evaluation",
            "prompt": [
                {{
                "role": "system",
                "content": "You are a professional physician. Always output <think>...</think><answer>...</answer>. No extra text."
                }},
                {{
                "role": "user",
                "content": "Initial patient complaint in plain language."
                }},
                {{
                "role": "assistant",
                "content": "<think>\\n[ToM Order: 1]\\n[Emotion: ...]\\n[Belief: ...]\\n[Strategy: ...]\\n[One New Medical Fact: ...]\\n</think>\\n<answer>\\nEmpathy: ...\\nQuestion: ...\\n</answer>"
                }}
            ],
            "ability": "medical_consultation",
            "reward_model": {{
                "ground_truth": "{self._escape_for_prompt(ground_truth)}",
                "style": "rule"
            }}
            }}

            ### [Hard Constraints]
            - Exactly {self.n_turns} assistant turns in total.
            - Every assistant turn must include BOTH <think> and <answer>.
            - <think> must contain exactly these fields:
            [ToM Order], [Emotion], [Belief], [Strategy], [One New Medical Fact]
            - <answer> must include:
            - one empathy statement,
            - one plain-language explanation,
            - and exactly one question (except final turn, which must include Recommendation).
            - Do not expose chain-of-thought outside <think>.
            - Do not output anything outside the final JSON object.
            - Final recommendation must be: "{self._escape_for_prompt(ground_truth)}" if available in candidates; otherwise choose the most clinically justified candidate.
            """.strip()

        return master_prompt

    def read_datas(self, file_path: Path) -> bool:
        self.file_path = file_path
        with file_path.open("r", encoding="utf-8") as f:
            self.datas = [json.loads(line) for line in f if line.strip()]
        return bool(self.datas)

    def run(self) -> list[str]:
        if self.datas is None:
            raise ValueError("No data loaded. Call read_datas() first.")
        return [self.generate(data) for data in self.datas]


def main() -> None:
    file_path = Path("ehr_bench_decision_making.jsonl")
    generator = MedicalToMPromptGenerator(turns=4)
    ok = generator.read_datas(file_path)
    if not ok:
        raise RuntimeError("No valid data found in input file.")

    prompts = generator.run()
    for prompt in prompts[:5]:
        print(prompt)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
