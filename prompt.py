import json
from pathlib import Path


class MedicalToMPromptGenerator:
    def __init__(self) -> None:
        self.file_path: Path | None = None
        self.datas: list[dict] | None = None

    def generate(self, sample_data: dict) -> str:
        # 1. 提取原始信息
        idx = sample_data.get("idx", 0)
        medical_input = sample_data.get("input", "")
        candidates = sample_data.get("candidates", [])

        # 2. 构建 Master Prompt 模板
        # 这里综合了两篇论文的核心：ToM Order (Paper 1) 和 BID 心理轨迹 (Paper 2)
        master_prompt = f"""
            ### [System Role: Advanced Medical ToM Reasoning AI]
            You are an advanced AI specialized in Medical Theory of Mind (ToM). Your objective is to analyze the psychological evolution of a patient through a temporal sequence of clinical events, provide a formal clinical decision, and simulate a realistic doctor-patient dialogue.

            ---

            ### [Step 1: Meta-Cognition (Ref: Mind Your Theory)]
            Determine the required **ToM Order (Depth of Mentalizing)** to answer the clinical decision in this case:
            - **Order 0**: Factual extraction (e.g., specific lab values, vitals).
            - **Order 1**: Internal state inference (e.g., the patient's overall clinical status and psychological outlook).
            - **Order 2**: Nested/Recursive inference (e.g., how the physician anticipates the patient's reaction to the diagnosis).
            **Task: Explicitly state the identified ToM Order in your output.**

            ---

            ### [Step 2: Mental State Trajectory (Ref: DynToM)]
            Based on the [Medical Event Sequence], analyze the temporal evolution of the patient's psychological state across these dimensions:
            - **Beliefs**: What does the patient currently "believe" about the severity of their condition? What is their understanding of the lab results?
            - **Intentions**: At this specific clinical juncture, what is the patient's primary goal or the most urgent question they intend to ask?
            - **Desires/Emotions**: Following the sequence of events (e.g., chest pain, ER triage, multiple blood draws, waiting), what is the patient's core emotional need or concern?

            ---

            ### [Step 3: Clinical Decision & Dialogue Simulation]
            1. **Clinical Suggestion**: Select the most appropriate admission type from the following candidates: {candidates}
            2. **Dialogue Generation**: Roleplay as the 57-year-old female patient described in the data. Based on the "Beliefs" and "Intentions" analyzed above, generate a dialogue with the attending physician (Requirements: Empathetic tone, age-appropriate language, expressing doubt or concern regarding lab findings, 3-5 conversational turns).

            ---

            ### [Input Data: Medical Event Sequence]
            {medical_input}

            ---

            ### [Output Format: Strict JSON Schema]
            {{
            "idx": {idx},
            "metacognition": {{
                "tom_order": "0/1/2",
                "rationale": "Briefly explain the reasoning behind the ToM Order selection."
            }},
            "tom_profile": {{
                "beliefs": "Analysis of the patient's beliefs",
                "intentions": "Analysis of the patient's intentions",
                "desires": "Analysis of the patient's emotional desires"
            }},
            "decision": {{
                "selected_candidate": "Your chosen option",
                "clinical_reason": "Medical justification for the chosen admission type."
            }},
            "simulated_dialogue": [
                {{"role": "Patient", "content": "..."}},
                {{"role": "Doctor", "content": "..."}}
            ]
            }}
            """
        return master_prompt

    def read_datas(self, file_path: Path) -> bool:
        with file_path.open("r", encoding="utf-8") as f:
            self.datas = [json.loads(line) for line in f]
            if not self.datas or self.datas == []:
                return False
            return True

    def run(self) -> list[str]:
        assert self.datas is not None
        prompts: list[str] = [self.generate(data) for data in self.datas]
        return prompts


def main() -> None:
    file_path = Path("ehr_bench_decision_making.jsonl")
    mkp = MedicalToMPromptGenerator()
    mkp.read_datas(file_path)
    prompts: list[str] = mkp.run()
    for prompt in prompts[:5]:
        print(prompt)


if __name__ == "__main__":
    main()


# ==========================================
# 使用示例 (直接运行此文件即可生成 Prompt)
# ==========================================
# if __name__ == "__main__":
#     # 模拟你给出的原始输入数据
#     my_sample = {
#         "idx": 0,
#         "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Admissions suggestion for the patiens.",
#         "input": "## Patient Demographics [None]\n- Anchor_Age: 57\n- Gender: F\n\n## Triage [2115-11-02 13:59:01]\n- Chiefcomplaint: Pleuritic chest pain\n- Heartrate: 101.0\n\n## Labotary Test Events\n| Item_Name | Valuenum |\n| Bilirubin | nan |\n| White Blood Cells | 16.1 (abnormal) |\n| Calcium, Total | 13.2 (abnormal) |",
#         "candidates": [
#             "SURGICAL SAME DAY ADMISSION",
#             "AMBULATORY OBSERVATION",
#             "DIRECT OBSERVATION",
#             "ELECTIVE",
#             "OBSERVATION ADMIT",
#             "DIRECT EMER.",
#             "EW EMER.",
#             "EU OBSERVATION",
#         ],
#     }

#     # 执行生成
#     final_prompt = MedicalToMPromptGenerator.generate(my_sample)

#     # 打印结果
#     print(final_prompt)

#     # 如果你想保存到文件直接发给老板
#     with open("final_prompt_to_llm.txt", "w", encoding="utf-8") as f:
#         f.write(final_prompt)
#     print("\n--- Prompt 已保存至 final_prompt_to_llm.txt ---")
