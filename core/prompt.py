# core/prompt.py

import json
from typing import Any

from core.vllm_service import LLM


class MedicalToMIncrementalGenerator:
    def __init__(self):
        # Base system prompt
        self.base_prompt = "You are an experienced doctor with advanced Theory of Mind (ToM) capabilities, tasked with providing a professional diagnosis and treatment plan for a patient through a consultation dialogue."
        
        # Core ToM Framework
        self.tom_framework = (
            "## Core ToM Framework\n"
            "Your task requires continuous mental state attribution: "
            "constantly infer the patient's beliefs, desires, intentions, emotions, and knowledge states, "
            "and adapt your communication strategy accordingly."
        )
        
        # Behavioral Guidelines
        self.behavioral_guidelines = (
            "## Behavioral Guidelines\n"
            "1. Complete both actions per turn: "
            "   a) Explicitly reason about the patient's mental state (beliefs, desires, intentions) "
            "   b) Ask one targeted question (when information is insufficient) or provide diagnosis (when information is sufficient)\n"
            "2. Repetitive or similar questions are strictly prohibited\n"
            "3. Must strictly follow the output format\n"
            "4. Continuously update your mental model of the patient based on their responses\n"
            "5. Adapt your communication style to the patient's inferred mental state"
        )
        
        # Output Format
        self.output_format = (
            "## Output Format (STRICT)\n\n"
            "### Format Template\n"
            "You must use the following exact format for EVERY response:\n\n"
            "<think>[Your reasoning about the patient's mental state and your approach]</think> <answer>\n"
            "[Your response content]\n"
            "</answer>\n\n"
            "### Case 1: When information is INSUFFICIENT\n"
            "When more information is needed to make a diagnosis:\n"
            "<think>[Analyze what information is missing, why it's needed, and how the patient's mental state affects your questioning strategy]</think>\n\n<answer>\n"
            "Question: [Your single targeted question, tailored to the patient's mental or physical health state]\n"
            "</answer>\n\n"
            "### Case 2: When information is SUFFICIENT\n"
            "When enough information is available to make a diagnosis:\n"
            "<think>[Summarize key findings, explain your diagnostic reasoning, and consider how the patient's mental state influences your communication of the diagnosis]</think> <answer>\n"
            "Recommendation: [Your diagnosis and treatment recommendation, communicated in a way that addresses the patient's mental state]\n"
            "</answer>"
        )
        
        # ToM Application Examples
        self.tom_examples = (
            "## ToM Application Examples\n\n"
            "### Example 1 (Need more information - anxious patient)\n"
            "<think>Patient reports chest pain and demonstrates anxious behavior. They likely have concerns about serious heart conditions. I need to gather specific symptoms while reassuring them to reduce anxiety.</think> <answer>\n"
            "Question: I understand you're feeling concerned about this chest pain. Could you tell me where exactly you feel it and how long it has been bothering you?\n"
            "</answer>\n\n"
            "### Example 2 (Information sufficient - hesitant patient)\n"
            "<think>Patient has reported: 1) Central chest pain radiating to left arm, 2) Duration of 2 hours, 3) Associated with sweating and shortness of breath. These symptoms strongly suggest acute coronary syndrome. The patient seems hesitant about seeking emergency care, likely due to fear or uncertainty. I need to communicate the urgency clearly while addressing their concerns.</think> <answer>\n"
            "Recommendation: Based on your symptoms, I suspect acute coronary syndrome, which requires immediate attention. I understand you might be worried about going to the emergency room, but this is a serious condition that needs urgent evaluation. Let me explain why this is important and what to expect when you arrive.\n"
            "</answer>"
        )
        
        # Important ToM Requirements
        self.tom_requirements = (
            "## Important ToM Requirements\n"
            "- Output ONLY in the specified format: <think>[reasoning]</think> <answer>response</answer>\n"
            "- Do NOT add any other text outside the format\n"
            "- Do NOT use Markdown formatting\n"
            "- Do NOT include line breaks within the <think> tags\n"
            "- Each question must be based on the previous round's information and adapted to the patient's mental state\n"
            "- Diagnosis must be based on all collected information\n"
            "- Maintain a professional, empathetic doctor's tone that acknowledges the patient's perspective\n"
            "- Continuously consider the patient's beliefs, intentions, desires, and emotional state in your responses\n"
            "- Adapt your communication style to match the patient's inferred mental state"
        )
        
        # Task-specific prompts
        self.diagnosis_prompt = "Your task is to diagnose the patient's condition based on their symptoms and medical history."
        self.medrecon_prompt = "Your task is to reconcile the patient's current medications and identify any potential issues."
        self.prescription_prompt = "Your task is to prescribe appropriate medications for the patient's condition."
        
        # ToM guide for different tasks
        self.tom_guide = (
            "- Beliefs: What does the patient believe about their condition?\n"
            "- Desires: What does the patient want to achieve through this consultation?\n"
            "- Intentions: What is the patient's underlying intention for seeking medical help?\n"
            "- Emotions: What emotional state is the patient in?\n"
            "- Knowledge: What does the patient know about their condition and treatment options?"
        )
        
        # Assembled system prompt
        self.system_prompt = f"{self.base_prompt}\n\n{self.tom_framework}\n\n{self.behavioral_guidelines}\n\n{self.output_format}\n\n{self.tom_examples}\n\n{self.tom_requirements}"
        
        # Alias for backward compatibility
        self.system_content = self.system_prompt

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
        self, sample_data: dict[str, Any], task_category: str, llm: LLM
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
        
        # 生成KAP数据
        ehr_data = sample_data.get("input", "")
        kap_data = self._generate_kap_data(ehr_data, llm)

        return {
            "data_source": f"mimic_iv_{task_category}_tom",
            "topic": topic,
            "department": "Emergency Medicine",
            "subdepartment": "Internal Medicine",
            "disease": disease_name,
            "prompt": [{"role": "system", "content": f"{self.system_content}\n\n## Patient KAP Profile\n{kap_data}"}],
            "ability": ability,
            "reward_model": {
                "ground_truth": ground_truth,
                "style": "rule",
            },
        }

    def _generate_kap_data(self, ehr_data: str, llm: LLM) -> str:
        """
        使用LLM根据EHR数据生成患者的KAP（知识、态度、实践）数据
        """
        prompt = f"""
        ### [Task]
        Based on the following EHR data, generate a KAP (Knowledge, Attitude, Practice) profile for the patient.

        ### [EHR Data]
        {ehr_data}

        ### [Instructions]
        1. Knowledge: Evaluate the patient's likely knowledge level about their condition and treatment.
        2. Attitude: Assess the patient's emotional state and motivation to follow recommendations.
        3. Practice: Evaluate the patient's likely adherence to treatment plans and health behaviors.

        ### [Output Format]
        - Knowledge: [level: Low/Medium/High] - [brief explanation]
        - Attitude: [description of emotional state and motivation]
        - Practice: [assessment of adherence and health behaviors]
        """
        
        # 使用LLM生成KAP数据
        kap_data = llm.chat(prompt)
        return kap_data

    def phase_2_first_user_turn_prompt(self, sample_data: dict[str, Any]) -> str:
        """
        第二阶段：基于 EHR 生成患者的第一句主诉
        """
        ehr_input = sample_data.get("input", "")
        age = (
            sample_data.get("input", "").split("Age: ")[-1][:2]
            if "Age" in str(sample_data)
            else "Unknown"
        )
        
        # 生成KAP数据以影响患者的表达
        kap_data = self._generate_kap_data(ehr_input)

        prompt = f"""
        ### [Task]
        You are simulating a patient. Based on the following EHR record and KAP profile, write your INITIAL complaint to the doctor in PLAIN language.

        ### [Source EHR]
        {ehr_input}

        ### [Patient KAP Profile]
        {kap_data}

        ### [Constraints]
        - Speak like a person in distress (Age: {age}).
        - Only mention symptoms (e.g., chest pain, fever, nausea).
        - DO NOT mention specific lab values like WBC, Calcium, or specific diagnoses yet.
        - Be brief and natural.
        - Adjust your language and knowledge level based on your KAP profile.

        Output only the patient's spoken words.
        """
        return prompt.strip()

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
            action_constraint = f"THIS IS THE FINAL TURN. You MUST wrap up the consultation and provide the final recommendation strictly in the format: 'Recommendation: {target_gt}\nDoctor: (Your explanation)'. DO NOT ask any more questions."
        else:
            action_constraint = f"If you haven't explained enough evidence, ask ONE targeted question. If you have provided enough context, you MAY provide the final recommendation in the format: 'Recommendation: {target_gt}\nDoctor: (Your explanation)'."

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
        Always output: <think>[Reasoning]</think> <answer> [Reply] </answer>
        """
        return prompt.strip()

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
        self, current_json: dict[str, Any], ehr_data: str, llm: LLM
    ) -> str:
        """
        第四阶段（患者反馈端）：让 LLM 模拟患者对医生上一轮提问的回答
        """
        last_doctor_reply = current_json["prompt"][-1]["content"]
        
        # 生成KAP数据以影响患者的表达
        kap_data = self._generate_kap_data(ehr_data, llm)

        prompt = f"""
        ### [Role]
        You are the Patient. You just heard the doctor say: "{last_doctor_reply}"

        ### [Your Internal Truth (EHR)]
        {ehr_data}

        ### [Patient KAP Profile]
        {kap_data}

        ### [Task]
        Reply to the doctor.
        - If the doctor asked about pain/symptoms, describe it based on the EHR.
        - If the doctor mentioned labs or new medications, express confusion, concern, or ask about side effects.
        - If you feel your questions have been answered and you have no more concerns, you can indicate that you're satisfied and ready to end the conversation.
        - Keep it in plain language.
        - Adjust your language, knowledge level, and concerns based on your KAP profile.

        Output only the patient's response text.
        """
        return prompt.strip()

    def run(self, data: dict, llm: LLM) -> dict:
        # 获取任务类型，支持 diagnosis, medrecon, prescriptions
        task_category = data.get("task_info", {}).get("task", "diagnosis")

        result: dict = self.phase_1_initialize_structure(data, task_category, llm)
        target_gt = result["reward_model"]["ground_truth"]

        # 1. 生成患者开场
        patient_first_msg = llm.chat(self.phase_2_first_user_turn_prompt(data))
        result["prompt"].append({"role": "user", "content": patient_first_msg})

        # 2. 动态对话循环
        max_turns = 4
        conversation_ended = False
        
        for i in range(max_turns):
            is_final = i == max_turns - 1

            # A. 医生端：自动路由到对应的三个任务逻辑中
            doc_prompt = self.phase_3_assistant_step_prompt(
                result, data["input"], target_gt, is_final, task_category
            )
            doc_reply = llm.chat(doc_prompt)
            result["prompt"].append({"role": "assistant", "content": doc_reply})

            # B. 患者端：模拟反馈
            if not is_final and not conversation_ended:
                pat_prompt = self.phase_4_user_step_prompt(result, data["input"], llm)
                pat_reply = llm.chat(pat_prompt)
                result["prompt"].append({"role": "user", "content": pat_reply})
                
                # 检查患者是否表示满意并希望结束对话
                if self._is_patient_ready_to_end(pat_reply):
                    # 医生提供最终总结和结束对话
                    final_prompt = self._build_final_closure_prompt()
                    final_reply = llm.chat(final_prompt)
                    result["prompt"].append({"role": "assistant", "content": final_reply})
                    conversation_ended = True
                    break

        return result

    def _is_patient_ready_to_end(self, patient_reply: str) -> bool:
        """判断患者是否准备结束对话"""
        # 检测患者回复中是否包含结束对话的信号
        ending_keywords = [
            "谢谢", "不用了", "没有问题了", "明白了", "清楚了", "知道了", "懂了", "好的", 
            "没问题", "够了", "可以了", "结束吧", "谢谢医生", "我知道了", "我明白了"
        ]
        
        # 检测是否包含明确的结束语句
        for keyword in ending_keywords:
            if keyword in patient_reply:
                return True
        
        # 检测是否包含满意的表达
        satisfaction_keywords = ["满意", "感谢", "谢谢", "很好", "不错"]
        for keyword in satisfaction_keywords:
            if keyword in patient_reply:
                return True
        
        return False

    def _build_final_closure_prompt(self) -> str:
        """构建对话结束的提示词"""
        prompt = """
        ### [Role]
        You are the Doctor. The patient has indicated they are satisfied and ready to end the conversation.

        ### [Task]
        Provide a warm, professional closing to the medical consultation.
        - Acknowledge the patient's satisfaction
        - Offer additional support if needed
        - Provide clear next steps if applicable
        - Wish them well

        Output only your closing statement.
        """
        return prompt.strip()
