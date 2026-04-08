# core/prompt.py

import json
from typing import Any

from core.vllm_service import LLM


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

    def _get_task_metadata(self, task_type: str) -> tuple[str, str]:
        """根据任务类型动态获取 topic 和 ability 标签"""
        task_type = task_type.lower()
        if "diagnosis" in task_type:
            return "Clinical Diagnosis Reasoning", "medical_diagnosis"
        elif "medrecon" in task_type:
            return "Medication Reconciliation", "medical_medrecon"
        elif "prescription" in task_type:
            return "Medication Prescription Reasoning", "medical_prescribing"
        else:
            return "Clinical Admission Decision", "medical_consultation"

    def phase_1_initialize_structure(
        self, sample_data: dict[str, Any], task_category: str
    ) -> dict[str, Any]:
        """
        第一阶段：生成数据头和系统指令
        """
        disease_name = sample_data.get("task_info", {}).get(
            "event", "General Consultation"
        )
        ground_truth = sample_data.get("output", "UNKNOWN")
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0]

        topic, ability = self._get_task_metadata(task_category)

        return {
            "data_source": f"mimic_iv_{task_category}_tom",
            "topic": topic,
            "department": "Emergency Medicine",
            "subdepartment": "Internal Medicine",
            "disease": disease_name,
            "prompt": [{"role": "system", "content": self.system_content}],
            "ability": ability,
            "reward_model": {
                "ground_truth": ground_truth,
                "style": "rule",
            },
        }

    def phase_2_first_user_turn_prompt(self, sample_data: dict[str, Any]) -> str:
        """
        第二阶段：基于 EHR 生成患者的第一句主诉
        """
        ehr_input = sample_data.get("input", "")
        age = (
            sample_data.get("input", "").split("Age: ")[-1][:2]
            if "Age" in str(sample_data)
            else "57"
        )

        prompt = f"""
        ### [Task]
        You are simulating a patient. Based on the following EHR record, write your INITIAL complaint to the doctor in PLAIN language.

        ### [Source EHR]
        {ehr_input}

        ### [Constraints]
        - Speak like a person in distress (Age: {age}).
        - Only mention symptoms (e.g., chest pain, fever, nausea).
        - DO NOT mention specific lab values like WBC, Calcium, or specific diagnoses yet.
        - Be brief and natural.

        Output only the patient's spoken words.
        """.strip()
        return prompt

    # =====================================================================
    # 下方三个函数分别实现了 Diagnosis, Medrecon, Prescriptions 任务的核心逻辑
    # =====================================================================

    def prompt_for_diagnosis(
        self, history: str, ehr_data: str, target_gt: str, is_final_turn: bool
    ) -> str:
        """任务 1: 诊断任务 (Diagnosis)"""
        clinical_focus = "Focus on analyzing the patient's symptoms, vitals, and laboratory results to deduce the underlying disease/diagnosis."
        tom_guide = "- Do they understand what their symptoms mean?\n           - If not, what lab results or medical facts do you need to reveal to explain the diagnosis?"
        return self._build_assistant_prompt(
            history, ehr_data, target_gt, is_final_turn, clinical_focus, tom_guide
        )

    def prompt_for_medrecon(
        self, history: str, ehr_data: str, target_gt: str, is_final_turn: bool
    ) -> str:
        """任务 2: 药物重整任务 (Medrecon)"""
        clinical_focus = "Focus on Medication Reconciliation (Medrecon). Review the patient's current home medications in the EHR. Identify any discrepancies, omissions, or necessary continuations."
        tom_guide = "- Does the patient accurately remember their home medications and dosages?\n           - How can you clarify their medication history without causing confusion?"
        return self._build_assistant_prompt(
            history, ehr_data, target_gt, is_final_turn, clinical_focus, tom_guide
        )

    def prompt_for_prescription(
        self, history: str, ehr_data: str, target_gt: str, is_final_turn: bool
    ) -> str:
        """任务 3: 处方任务 (Prescriptions)"""
        clinical_focus = "Focus on prescribing new medications for the acute symptoms. You MUST critically check existing home medications (Medrecon) to avoid drug-drug interactions before prescribing."
        tom_guide = "- Does the patient desire immediate symptom relief?\n           - Are they anxious about taking new medications or potential side effects?"
        return self._build_assistant_prompt(
            history, ehr_data, target_gt, is_final_turn, clinical_focus, tom_guide
        )

    def _build_assistant_prompt(
        self,
        history: str,
        ehr_data: str,
        target_gt: str,
        is_final_turn: bool,
        clinical_focus: str,
        tom_guide: str,
    ) -> str:
        """底层共用的 Prompt 构建引擎"""
        if is_final_turn:
            action_constraint = f"THIS IS THE FINAL TURN. You MUST wrap up the consultation and provide the final recommendation strictly in the format: 'Recommendation: {target_gt}\\nDoctor: (Your explanation)'. DO NOT ask any more questions."
        else:
            action_constraint = f"If you haven't explained enough evidence, ask ONE targeted question. If you have provided enough context, you MAY provide the final recommendation in the format: 'Recommendation: {target_gt}\\nDoctor: (Your explanation)'."

        prompt = f"""
        ### [Role]
        You are the Doctor. Use Theory of Mind to analyze the patient's state.

        ### [Target Clinical Goal]
        The final decision/recommendation for this patient MUST be exactly: {target_gt}

        ### [Clinical Evidence (EHR)]
        {ehr_data}

        ### [Clinical Focus]
        {clinical_focus}

        ###[Current Dialogue Context]
        {history}

        ### [Your Goal]
        1. In <think>: Analyze the patient's BID (Beliefs, Intentions, Desires).
           {tom_guide}
        2. In <answer>:
           - {action_constraint}

        ### [Strict Format]
        Always output: <think>[Reasoning] </think> <answer> [Reply] </answer>
        """.strip()
        return prompt

    def phase_3_assistant_step_prompt(
        self,
        current_json: dict[str, Any],
        ehr_data: str,
        target_gt: str,
        is_final_turn: bool,
        task_category: str,
    ) -> str:
        """
        第三阶段（医生端）：路由函数，动态分配给三个核心任务处理
        """
        history = json.dumps(current_json["prompt"], indent=2, ensure_ascii=False)
        task_cat_lower = task_category.lower()

        if "diagnosis" in task_cat_lower:
            return self.prompt_for_diagnosis(
                history, ehr_data, target_gt, is_final_turn
            )
        elif "medrecon" in task_cat_lower:
            return self.prompt_for_medrecon(history, ehr_data, target_gt, is_final_turn)
        elif "prescription" in task_cat_lower:
            return self.prompt_for_prescription(
                history, ehr_data, target_gt, is_final_turn
            )
        else:
            # 默认兜底任务 (Admission 等)
            return self.prompt_for_diagnosis(
                history, ehr_data, target_gt, is_final_turn
            )

    def phase_4_user_step_prompt(
        self, current_json: dict[str, Any], ehr_data: str
    ) -> str:
        """
        第四阶段（患者反馈端）：让 LLM 模拟患者对医生上一轮提问的回答
        """
        last_doctor_reply = current_json["prompt"][-1]["content"]
        prompt = f"""
        ### [Role]
        You are the Patient. You just heard the doctor say: "{last_doctor_reply}"

        ### [Your Internal Truth (EHR)]
        {ehr_data}

        ### [Task]
        Reply to the doctor.
        - If the doctor asked about pain/symptoms, describe it based on the EHR.
        - If the doctor mentioned labs or new medications, express confusion, concern, or ask about side effects.
        - Keep it in plain language.

        Output only the patient's response text.
        """.strip()
        return prompt

    def run(self, data: dict, llm: LLM) -> dict:
        # 获取任务类型，支持 diagnosis, medrecon, prescriptions
        task_category = data.get("task_info", {}).get("task", "diagnosis")

        result: dict = self.phase_1_initialize_structure(data, task_category)
        target_gt = result["reward_model"]["ground_truth"]

        # 1. 生成患者开场
        patient_first_msg = llm.chat(self.phase_2_first_user_turn_prompt(data))
        result["prompt"].append({"role": "user", "content": patient_first_msg})

        # 2. 动态对话循环
        max_turns = 4
        for i in range(max_turns):
            is_final = i == max_turns - 1

            # A. 医生端：自动路由到对应的三个任务逻辑中
            doc_prompt = self.phase_3_assistant_step_prompt(
                result, data["input"], target_gt, is_final, task_category
            )
            doc_reply = llm.chat(doc_prompt)
            result["prompt"].append({"role": "assistant", "content": doc_reply})

            # 检查是否已经结案
            if "Recommendation:" in doc_reply:
                break

            # B. 患者端：模拟反馈
            if not is_final:
                pat_prompt = self.phase_4_user_step_prompt(result, data["input"])
                pat_reply = llm.chat(pat_prompt)
                result["prompt"].append({"role": "user", "content": pat_reply})

        return result
