import json
import os
from typing import Any
from core.vllm_service import LLM

class MedicalToMIncrementalGenerator:
    def __init__(self):
        self.base_prompt = "You are an experienced doctor tasked with providing a professional diagnosis and treatment plan for a patient through a consultation dialogue."

        self.behavioral_guidelines = (
            "## Behavioral Guidelines\n"
            "1. Complete both actions per turn:\n"
            "   a) Provide clinical reasoning about the patient's symptoms and medical evidence\n"
            "   b) Ask one targeted question (when information is insufficient) or provide diagnosis (when information is sufficient)\n"
            "2. Repetitive or similar questions are strictly prohibited\n"
            "3. Must strictly follow the output format\n"
            "4. Base your decisions only on medical evidence, not on speculation about patient's mental state"
        )

        self.output_format = (
            "## Output Format (STRICT)\n\n"
            "You must use the following exact format for EVERY response:\n\n"
            "<think>[Your clinical reasoning based on symptoms and medical evidence]</think> <answer>\n"
            "[Your response content]\n"
            "</answer>\n\n"
            "### Case 1: When information is INSUFFICIENT\n"
            "<think>[Analyze what medical information is missing and why it is needed]</think>\n\n<answer>\n"
            "Question: [Your single targeted question]\n"
            "</answer>\n\n"
            "### Case 2: When information is SUFFICIENT\n"
            "<think>[Summarize key findings and explain your diagnostic reasoning]</think> <answer>\n"
            "Recommendation: [Your diagnosis and treatment recommendation]\n"
            "</answer>"
        )

        self.clinical_examples = (
            "## Clinical Examples\n\n"
            "### Example 1 (Need more information)\n"
            "<think>Patient reports chest pain. Need to differentiate cardiac vs. non-cardiac causes. Missing: location, radiation, duration, associated symptoms.</think> <answer>\n"
            "Question: Could you describe where exactly you feel the pain and how long it has been present?\n"
            "</answer>\n\n"
            "### Example 2 (Information sufficient)\n"
            "<think>Patient has central chest pain radiating to left arm for 2 hours with sweating and shortness of breath. Symptoms strongly suggest acute coronary syndrome.</think> <answer>\n"
            "Recommendation: I suspect acute coronary syndrome. You need immediate emergency evaluation.\n"
            "</answer>"
        )

        self.general_requirements = (
            "## Important Requirements\n"
            "- Output ONLY in the specified format: <think>[reasoning]</think> <answer>response</answer>\n"
            "- Do NOT add any other text outside the format\n"
            "- Do NOT use Markdown formatting\n"
            "- Do NOT include line breaks within the <think> tags\n"
            "- Each question must be based on the previous round's information\n"
            "- Diagnosis must be based on all collected information"
        )

        self.diagnosis_prompt = "Your task is to diagnose the patient's condition based on their symptoms and medical history."
        self.medrecon_prompt = "Your task is to reconcile the patient's current medications and identify any potential issues."
        self.prescription_prompt = "Your task is to prescribe appropriate medications for the patient's condition."

        self.system_prompt = f"{self.base_prompt}\n\n{self.behavioral_guidelines}\n\n{self.output_format}\n\n{self.clinical_examples}\n\n{self.general_requirements}"
        self.system_content = self.system_prompt

    @staticmethod
    def _get_task_metadata(task_type: str) -> tuple[str, str]:
        task_type = task_type.lower()
        if "diagnosis" in task_type:
            return "Clinical Diagnosis Reasoning", "medical_diagnosis"
        elif "medrecon" in task_type:
            return "Medication Reconciliation", "medical_medrecon"
        elif "prescription" in task_type:
            return "Medication Prescription Reasoning", "medical_prescribing"
        else:
            return "Clinical Admission Decision", "medical_consultation"

    @staticmethod
    def _generate_kap_data(ehr_data: str, llm: LLM) -> str:
        prompt = f"""
Based on the following EHR data, generate a KAP (Knowledge, Attitude, Practice) profile for the patient.

EHR Data:
{ehr_data}

Output format:
- Knowledge: [level: Low/Medium/High] - [brief explanation]
- Attitude: [description of emotional state and motivation]
- Practice: [assessment of adherence and health behaviors]
"""
        return llm.chat(prompt)

    @staticmethod
    def phase_2_first_user_turn_prompt(sample_data: dict[str, Any]) -> str:
        ehr_input = sample_data.get("input", "")
        age = (
            sample_data.get("input", "").split("Age: ")[-1][:2]
            if "Age" in str(sample_data)
            else "Unknown"
        )
        prompt = f"""
You are simulating a patient. Based on the following EHR record, write your INITIAL complaint to the doctor in PLAIN language.

EHR:
{ehr_input}

Constraints:
- Speak like a person in distress (Age: {age}).
- Only mention symptoms (e.g., chest pain, fever, nausea).
- DO NOT mention specific lab values or diagnoses.
- Be brief and natural.

Output only the patient's spoken words.
"""
        return prompt.strip()

    def prompt_for_diagnosis(
        self, history: str, target_gt: str, is_final_turn: bool
    ) -> str:
        clinical_focus = "Focus on analyzing the patient's symptoms, vitals, and laboratory results to deduce the underlying disease/diagnosis."
        return self._build_assistant_prompt(
            history, target_gt, is_final_turn, clinical_focus
        )

    def prompt_for_medrecon(
        self, history: str, target_gt: str, is_final_turn: bool
    ) -> str:
        clinical_focus = "Focus on Medication Reconciliation (Medrecon). Review the patient's current home medications. Identify discrepancies, omissions, or necessary continuations."
        return self._build_assistant_prompt(
            history, target_gt, is_final_turn, clinical_focus
        )

    def prompt_for_prescription(
        self, history: str, target_gt: str, is_final_turn: bool
    ) -> str:
        clinical_focus = "Focus on prescribing new medications for the acute symptoms. Critically check existing home medications to avoid drug-drug interactions."
        return self._build_assistant_prompt(
            history, target_gt, is_final_turn, clinical_focus
        )

    @staticmethod
    def _build_assistant_prompt(
        history: str,
        target_gt: str,
        is_final_turn: bool,
        clinical_focus: str,
    ) -> str:
        if is_final_turn:
            action_constraint = f"THIS IS THE FINAL TURN. You MUST wrap up the consultation and provide the final recommendation strictly in the format: 'Recommendation: {target_gt}\nDoctor: (Your explanation)'. DO NOT ask any more questions."
        else:
            action_constraint = f"If you haven't explained enough evidence, ask ONE targeted question. If you have provided enough context, you MAY provide the final recommendation in the format: 'Recommendation: {target_gt}\nDoctor: (Your explanation)'."

        prompt = f"""
You are the Doctor.

Target Clinical Goal: {target_gt}

Clinical Focus:
{clinical_focus}

Current Dialogue Context:
{history}

Your Goal:
{action_constraint}

Strict Format:
Always output: <think>[Your clinical reasoning]</think> <answer> [Reply] </answer>
"""
        return prompt.strip()

    def phase_3_assistant_step_prompt(
        self,
        current_json: dict[str, Any],
        target_gt: str,
        is_final_turn: bool,
        task_category: str,
    ) -> str:
        # 简化对话历史格式，只保留角色和内容，不包含完整的JSON结构
        history_lines = []
        for msg in current_json["prompt"]:
            if msg["role"] == "system":
                continue  # 跳过system提示词，因为它已经包含在上下文里了
            history_lines.append(f"{msg['role']}: {msg['content']}")
        history = "\n".join(history_lines)
        
        task_cat_lower = task_category.lower()

        if "diagnosis" in task_cat_lower:
            return self.prompt_for_diagnosis(history, target_gt, is_final_turn)
        elif "medrecon" in task_cat_lower:
            return self.prompt_for_medrecon(history, target_gt, is_final_turn)
        elif "prescription" in task_cat_lower:
            return self.prompt_for_prescription(history, target_gt, is_final_turn)
        else:
            return self.prompt_for_diagnosis(history, target_gt, is_final_turn)

    def phase_4_user_step_prompt(
        self, current_json: dict[str, Any], ehr_data: str, kap_data: str
    ) -> str:
        last_doctor_reply = current_json["prompt"][-1]["content"]

        # user复用system中关于assistant回复格式的部分，以及KAP和EHR数据的部分
        prompt = f"""
You are the Patient. You just heard the doctor say: "{last_doctor_reply}"

## Patient KAP Profile
{kap_data}

## EHR Data
{ehr_data}

Now, before replying, you MUST perform Theory of Mind (ToM) reasoning about the doctor. Think about:
- What does the doctor believe about your condition?
- What information does the doctor want to obtain from you?
- What is the doctor's intention (e.g., to diagnose, to prescribe, to reassure)?
- How can you adjust your reply to be most helpful while staying true to your symptoms?

After this reasoning, reply to the doctor as the patient.

Guidelines:
- If the doctor asked about pain/symptoms, describe them based on the EHR.
- If the doctor mentioned labs or new medications, express confusion, concern, or ask about side effects.
- If you feel your questions have been answered and you have no more concerns, indicate that you're satisfied.
- Keep it in plain language.
- Adjust your language, knowledge level, and concerns based on your KAP profile.

Output format:
ahre[Your ToM reasoning about the doctor]ahr/ <patient>[Your reply to the doctor]</patient>

Do not output anything else.
"""
        return prompt.strip()

    @staticmethod
    def _is_patient_ready_to_end(patient_reply: str) -> bool:
        ending_keywords = [
            "谢谢", "不用了", "没有问题了", "明白了", "清楚了", "知道了", "懂了", "好的",
            "没问题", "够了", "可以了", "结束吧", "谢谢医生", "我知道了", "我明白了"
        ]
        for keyword in ending_keywords:
            if keyword in patient_reply:
                return True
        satisfaction_keywords = ["满意", "感谢", "谢谢", "很好", "不错"]
        for keyword in satisfaction_keywords:
            if keyword in patient_reply:
                return True
        return False

    @staticmethod
    def _build_final_closure_prompt() -> str:
        prompt = """
You are the Doctor. The patient has indicated they are satisfied and ready to end the conversation.

Provide a warm, professional closing to the medical consultation.
- Acknowledge the patient's satisfaction
- Offer additional support if needed
- Provide clear next steps if applicable
- Wish them well

Output only your closing statement.
"""
        return prompt.strip()

    @staticmethod
    def _extract_answer_content(text: str) -> str:
        """提取<answer>标签内的内容"""
        if "<answer>" in text and "</answer>" in text:
            return text.split("<answer>")[1].split("</answer>")[0].strip()
        return text

    @staticmethod
    def _extract_patient_content(text: str) -> str:
        """提取<patient>标签内的内容"""
        if "<patient>" in text and "</patient>" in text:
            return text.split("<patient>")[1].split("</patient>")[0].strip()
        return text

    def run(self, data: dict, llm: LLM) -> dict:
        task_category = data.get("task_info", {}).get("task", "diagnosis")
        result: dict = self.phase_1_initialize_structure(data, task_category, llm)
        target_gt = result["reward_model"]["ground_truth"]
        ehr_data = result["ehr_data"]
        kap_data = result["kap_data"]

        # 创建raw目录
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # 生成患者的第一条消息
        patient_first_msg = llm.chat(self.phase_2_first_user_turn_prompt(data))
        result["prompt"].append({"role": "user", "content": patient_first_msg})
        result["raw_prompt"].append({"role": "user", "content": patient_first_msg})

        max_turns = 4

        for i in range(max_turns):
            is_final = i == max_turns - 1

            # 医生回复
            doc_prompt = self.phase_3_assistant_step_prompt(
                result, target_gt, is_final, task_category
            )
            doc_reply = llm.chat(doc_prompt)
            
            # 保存完整的原始回复到raw_prompt
            result["raw_prompt"].append({"role": "assistant", "content": doc_reply})
            
            # 截断医生的思考部分，只保留<answer>内的内容
            doc_answer = self._extract_answer_content(doc_reply)
            result["prompt"].append({"role": "assistant", "content": doc_answer})

            if not is_final:
                # 患者回复
                pat_prompt = self.phase_4_user_step_prompt(result, ehr_data, kap_data)
                pat_reply = llm.chat(pat_prompt)
                
                # 保存完整的原始回复到raw_prompt（包括思考部分和<answer>部分）
                result["raw_prompt"].append({"role": "user", "content": pat_reply})
                
                # 提取<patient>标签内的内容
                pat_answer = self._extract_patient_content(pat_reply)
                result["prompt"].append({"role": "user", "content": pat_answer})

                if self._is_patient_ready_to_end(pat_answer):
                    final_prompt = self._build_final_closure_prompt()
                    final_reply = llm.chat(final_prompt)
                    result["prompt"].append({"role": "assistant", "content": final_reply})
                    result["raw_prompt"].append({"role": "assistant", "content": final_reply})
                    break

        # 保存完整的原始对话到raw文件夹
        raw_file = os.path.join(raw_dir, f"raw_{data.get('task_info', {}).get('event', 'unknown')}_{os.getpid()}.json")
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump({
                "data_source": result["data_source"],
                "topic": result["topic"],
                "department": result["department"],
                "subdepartment": result["subdepartment"],
                "disease": result["disease"],
                "raw_prompt": result["raw_prompt"],
                "ability": result["ability"],
                "reward_model": result["reward_model"],
            }, f, ensure_ascii=False, indent=2)

        # 返回截断后的结果
        return {
            "data_source": result["data_source"],
            "topic": result["topic"],
            "department": result["department"],
            "subdepartment": result["subdepartment"],
            "disease": result["disease"],
            "prompt": result["prompt"],
            "ability": result["ability"],
            "reward_model": result["reward_model"],
        }

    def phase_1_initialize_structure(
        self, sample_data: dict[str, Any], task_category: str, llm: LLM
    ) -> dict[str, Any]:
        disease_name = sample_data.get("task_info", {}).get(
            "event", "General Consultation"
        )
        ground_truth = sample_data.get("output", "UNKNOWN")
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0]

        topic, ability = self._get_task_metadata(task_category)
        ehr_data = sample_data.get("input", "")
        kap_data = self._generate_kap_data(ehr_data, llm)

        # system部分只包含给assistant的系统提示词（包括KAP和EHR数据）
        system_content = f"{self.system_content}\n\n## Patient KAP Profile\n{kap_data}\n\n## EHR Data\n{ehr_data}"

        return {
            "data_source": f"mimic_iv_{task_category}_tom",
            "topic": topic,
            "department": "Emergency Medicine",
            "subdepartment": "Internal Medicine",
            "disease": disease_name,
            "prompt": [{"role": "system", "content": system_content}],
            "ability": ability,
            "reward_model": {
                "ground_truth": ground_truth,
                "style": "rule",
            },
            "raw_prompt": [],  # 用于存储完整的原始对话
            "ehr_data": ehr_data,  # 保存EHR数据供后续使用
            "kap_data": kap_data,  # 保存KAP数据供后续使用
        }
