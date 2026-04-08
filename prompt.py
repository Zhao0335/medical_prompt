import json
from pathlib import Path
from typing import Any

from llm_service import LLM
from model import qwq_chat
from reader import Reader


class MedicalToMIncrementalGenerator:
    def __init__(self):
        # 预定义的系统提示词模板（固定格式）
        self.system_content = (
            "You are an experienced doctor tasked with providing a professional diagnosis and treatment plan "
            "for a patient through a consultation dialogue. Please carefully listen to the patient's responses, "
            "ask targeted questions.\n\nQuick Guide\nObjective:\n"
            "1. Gather key information through effective questioning. Each question should be based on the previous roundʼs information. Avoid repeating questions.\n\n"
            "Rules:\n1. Complete both action per turn: provide thinking and ask a question.\n"
            "2. Repetitive or similar questions are strictly prohibited.\n\n"
            "Response Format:\n<think> [Your reasoning] </think>\n"
            "<answer> If information is insufficient, ask one question only, in the following format:\n"
            "Question: (Your question).\n</answer>\n"
            "<answer> If information is sufficient, provide diagnosis and recommendation, in the following format:\n"
            "Recommendation: (Your diagnosis and recommendation)\n</answer>.\n\n"
            "Decide your next action:\nAlways output: <think> [Your reasoning] </think> <answer> [Your reply] </answer> "
            "Do not include any additional text. Follow this format strictly."
        )

    def phase_1_initialize_structure(
        self, sample_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        第一阶段：生成数据头和系统指令
        """
        # 从原始数据提取疾病特征
        disease_name = sample_data.get("task_info", {}).get(
            "event", "General Consultation"
        )

        return {
            "data_source": "mimic_iv_tom_dynamic",
            "topic": "Clinical Admission Decision",
            "department": "Emergency Medicine",
            "subdepartment": "Internal Medicine",
            "disease": disease_name,
            "prompt": [{"role": "system", "content": self.system_content}],
            "ability": "medical_consultation",
            "reward_model": {
                "ground_truth": sample_data.get("output", "OBSERVATION ADMIT"),
                "style": "rule",
            },
        }

    def phase_2_first_user_turn_prompt(self, sample_data: dict[str, Any]) -> str:
        """
        第二阶段：基于 EHR 生成患者的第一句主诉
        """
        ehr_input = sample_data.get("input", "")
        prompt = f"""
        ### [Task]
        You are simulating a patient. Based on the following EHR record, write your INITIAL complaint to the doctor in PLAIN language.

        ### [Source EHR]
        {ehr_input}

        ### [Constraints]
        - Speak like a person in distress (Age: {sample_data.get("input", "").split("Age: ")[-1][:2] if "Age" in str(sample_data) else "57"}).
        - Only mention symptoms (e.g., chest pain, fever).
        - DO NOT mention specific lab values like WBC or Calcium yet.
        - Be brief and natural.

        Output only the patient's spoken words.
        """.strip()
        return prompt

    def phase_3_assistant_step_prompt(
        self,
        current_json: dict[str, Any],
        ehr_data: str,
        target_gt: str,
    ) -> str:
        history = json.dumps(current_json["prompt"], indent=2, ensure_ascii=False)

        # 这里的关键：让 LLM 知道它最终必须证明这个 ground_truth 是对的
        prompt = f"""
            ### [Role]
            You are the Doctor. Use Theory of Mind to analyze the patient's state.

            ### [Target Diagnosis]
            The final admission suggestion for this patient MUST be: {target_gt}

            ### [Clinical Evidence (EHR)]
            {ehr_data}

            ### [Current Dialogue Context]
            {history}

            ### [Your Goal]
            1. In <think>: Analyze the patient's BID (Beliefs, Intentions, Desires).
               - Do they understand why {target_gt} is necessary?
               - If not, what information (WBC, Calcium, etc.) do you need to reveal next?
            2. In <answer>:
               - If you haven't explained enough evidence, ask ONE targeted question.
               - If you have provided enough context, provide the final recommendation in the format: "Recommendation: {target_gt} (and your explanation)".

            ### [Strict Format]
            Always output: <think> [Reasoning] </think> <answer> [Reply] </answer>
            """.strip()
        return prompt

    def phase_4_user_step_prompt(
        self, current_json: dict[str, Any], ehr_data: str
    ) -> str:
        """
        第三阶段（患者反馈端）：让 LLM 模拟患者对医生上一轮提问的回答
        """
        last_doctor_reply = current_json["prompt"][-1]["content"]
        prompt = f"""
        ### [Role]
        You are the Patient. You just heard the doctor say: "{last_doctor_reply}"

        ### [Your Internal Truth (EHR)]
        {ehr_data}

        ### [Task]
        Reply to the doctor.
        - If the doctor asked about pain, describe it based on the EHR (e.g., pleuritic).
        - If the doctor mentioned labs, express confusion or concern.
        - Keep it in plain language.

        Output only the patient's response text.
        """.strip()
        return prompt

    def run(self, data: dict) -> dict:
        # llm = LLM()
        result: dict = self.phase_1_initialize_structure(data)
        target_gt = result["reward_model"]["ground_truth"]  # 获取终点

        # 1. 生成患者开场
        # patient_first_msg = llm.chat(self.phase_2_first_user_turn_prompt(data))
        patient_first_msg = qwq_chat(self.phase_2_first_user_turn_prompt(data))
        result["prompt"].append({"role": "user", "content": patient_first_msg})

        # 2. 动态对话循环（设置最大轮数，如 4 轮）
        max_turns = 3
        for i in range(max_turns):
            # A. 医生端：传入 target_gt 确保逻辑收敛
            doc_prompt = self.phase_3_assistant_step_prompt(
                result, data["input"], target_gt
            )
            # doc_reply = llm.chat(doc_prompt)
            doc_reply = qwq_chat(doc_prompt)
            result["prompt"].append({"role": "assistant", "content": doc_reply})

            # 检查是否已经结案
            if "Recommendation:" in doc_reply or i == max_turns - 1:
                # 如果最后一轮还没结案，我们需要强制 LLM 在下一轮输出结案（或者在这里处理）
                if "Recommendation:" not in doc_reply:
                    # 补充逻辑：强制结案的 Prompt
                    pass
                break

            # B. 患者端：模拟反馈
            pat_prompt = self.phase_4_user_step_prompt(result, data["input"])
            # pat_reply = llm.chat(pat_prompt)
            pat_reply = qwq_chat(pat_prompt)
            result["prompt"].append({"role": "user", "content": pat_reply})

        return result


def main() -> None:
    file1 = Path("ehr_bench_decision_making.jsonl")
    datas = Reader.read(file1)
    assert datas is not None
    mp = MedicalToMIncrementalGenerator()
    result_file = Path("result.json")
    with result_file.open("w", encoding="utf-8") as f:
        json.dump(mp.run(datas[6]), f, ensure_ascii=False, indent=4)
    print("Done!")


if __name__ == "__main__":
    main()
