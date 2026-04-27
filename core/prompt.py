from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.vllm_service import LLM


class MedicalToMIncrementalGenerator:
    def __init__(self) -> None:
        self.base_prompt = (
            "You are an experienced doctor tasked with providing a professional diagnosis "
            "and treatment plan for a patient through a consultation dialogue."
        )

        self.behavioral_guidelines = (
            "## Behavioral Guidelines\n"
            "1. Complete both actions per turn:\n"
            "   a) Provide clinical reasoning about the patient's symptoms and medical evidence\n"
            "   b) Ask one targeted question (when information is insufficient) or provide suggestion "
            "(when information is sufficient)\n"
            "2. Repetitive or similar questions are strictly prohibited\n"
            "3. Must strictly follow the output format\n"
            "4. Base your decisions only on medical evidence, not on speculation about patient's mental state"
        )

        self.output_format = (
            "## Output Format (STRICT)\n\n"
            "Your clinical reasoning happens automatically in your thinking process.\n"
            "After your thinking, you MUST output your response in this EXACT format:\n\n"
            "<answer>[Your response content]\n"
            "</answer>\n\n"
            "### Case 1: When information is INSUFFICIENT\n"
            "<answer>[Your single targeted question]\n"
            "</answer>\n\n"
            "### Case 2: When information is SUFFICIENT\n"
            "<answer>[Your suggestion or treatment recommendation]\n"
            "</answer>"
        )

        self.clinical_examples = (
            "## Clinical Examples\n\n"
            "### Example 1 (Need more information)\n"
            "<answer>Question: Could you describe where exactly you feel the pain and how long it has been present?\n"
            "</answer>\n"
            "### Example 2 (Information sufficient)\n"
            "<answer>Suggestion: I suspect acute coronary syndrome. You need immediate emergency evaluation.\n"
            "</answer>"
        )

        self.general_requirements = (
            "## Important Requirements\n"
            "- Your clinical reasoning is automatic (it happens in your thinking)\n"
            "- After your thinking, output ONLY the <answer> tag with your response\n"
            "- Do NOT add any text outside the <answer> tags\n"
            "- Do NOT use Markdown formatting inside <answer>\n"
            "- Each question must be based on the previous round's information\n"
            "- Suggestion must be based on all collected information\n"
            "- You MUST respond in English at all times. Do NOT use any other language."
        )

        self.suggestion_prompt = "Your task is to suggest the patient's condition based on their symptoms and medical history."
        self.medrecon_prompt = "Your task is to reconcile the patient's current medications and identify any potential issues."
        self.prescription_prompt = "Your task is to prescribe appropriate medications for the patient's condition."

        self.system_prompt = (
            f"{self.base_prompt}\n\n"
            f"{self.behavioral_guidelines}\n\n"
            f"{self.output_format}\n\n"
            f"{self.clinical_examples}\n\n"
            f"{self.general_requirements}"
        )
        self.system_content = self.system_prompt

    # ------------------------------------------------------------------
    # 元数据 / 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _get_task_metadata(task_type: str) -> tuple[str, str]:
        task_type = task_type.lower()
        if "suggestion" in task_type:
            return "Clinical Suggestion Reasoning", "medical_suggestion"
        elif "medrecon" in task_type:
            return "Medication Reconciliation", "medical_medrecon"
        elif "prescription" in task_type:
            return "Medication Prescription Reasoning", "medical_prescribing"
        else:
            return "Clinical Admission Decision", "medical_consultation"

    @staticmethod
    def _extract_kap_content(text: str) -> str:
        """解析 KAP 生成结果，去掉模型可能输出的 <think> 块"""
        content = text.strip()
        if "<think>" in content:
            parts = content.split("</think>", 1)
            if len(parts) > 1:
                content = parts[1].strip()
        return content.strip()

    @staticmethod
    def _build_kap_prompt(ehr_data: str) -> str:
        """构建 KAP 画像生成 prompt（纯字符串，不调用 LLM）"""
        return (
            "Based on the following EHR data, generate a KAP (Knowledge, Attitude, Practice) "
            "profile for the patient.\n\n"
            f"EHR Data:\n{ehr_data}\n\n"
            "Output format:\n"
            "- Knowledge: [level: Low/Medium/High] - [brief explanation]\n"
            "- Attitude: [description of emotional state and motivation]\n"
            "- Practice: [assessment of adherence and health behaviors]\n\n"
            "You MUST respond in English."
        )

    @staticmethod
    def _generate_kap_data(ehr_data: str, llm: LLM) -> str:
        """单条 KAP 生成（单样本模式使用）"""
        prompt = MedicalToMIncrementalGenerator._build_kap_prompt(ehr_data)
        raw_reply = llm.chat(prompt)
        return MedicalToMIncrementalGenerator._extract_kap_content(raw_reply)

    # ------------------------------------------------------------------
    # Phase 2：患者首次主诉
    # ------------------------------------------------------------------

    @staticmethod
    def phase_2_first_user_turn_prompt(sample_data: dict[str, Any]) -> str:
        ehr_input = sample_data.get("input", "")

        # 稳健地提取年龄
        age = "Unknown"
        if "Age: " in ehr_input:
            age_fragment = ehr_input.split("Age: ", 1)[1][:5]
            age_digits = "".join(c for c in age_fragment if c.isdigit())
            if age_digits:
                age = age_digits[:3]  # 最多 3 位，避免截到下一个字段

        return (
            "You are simulating a patient. Based on the following EHR record, write your INITIAL "
            "complaint to the doctor in PLAIN language.\n\n"
            f"EHR:\n{ehr_input}\n\n"
            "Constraints:\n"
            f"- Speak like a person in distress (Age: {age}).\n"
            "- Only mention symptoms (e.g., chest pain, fever, nausea).\n"
            "- DO NOT mention specific lab values or diagnoses.\n"
            "- Be brief and natural.\n"
            "- You MUST speak in English.\n\n"
            "Output only the patient's spoken words.\n\n"
            "Example output:\n"
            "My chest pain started yesterday and feels like pressure in the middle of my chest.\n\n"
            "Do not include any additional text. Follow this format strictly."
        )

    # ------------------------------------------------------------------
    # Phase 3：医生回复 prompt 构建
    # ------------------------------------------------------------------

    def prompt_for_diagnosis(self, history: str, is_final_turn: bool) -> str:
        clinical_focus = (
            "Focus on analyzing the patient's symptoms, vitals, and laboratory results "
            "to deduce the underlying disease/diagnosis."
        )
        return self._build_assistant_prompt(history, is_final_turn, clinical_focus)

    def prompt_for_medrecon(self, history: str, is_final_turn: bool) -> str:
        clinical_focus = (
            "Focus on Medication Reconciliation (Medrecon). Review the patient's current home medications. "
            "Identify discrepancies, omissions, or necessary continuations."
        )
        return self._build_assistant_prompt(history, is_final_turn, clinical_focus)

    def prompt_for_prescription(self, history: str, is_final_turn: bool) -> str:
        clinical_focus = (
            "Focus on prescribing new medications for the acute symptoms. "
            "Critically check existing home medications to avoid drug-drug interactions."
        )
        return self._build_assistant_prompt(history, is_final_turn, clinical_focus)

    @staticmethod  # 修复：原代码缺少此装饰器，导致 self 被误传为 history 参数造成崩溃
    def _build_assistant_prompt(
        history: str,
        is_final_turn: bool,
        clinical_focus: str,
    ) -> str:
        if is_final_turn:
            action_constraint = (
                "THIS IS THE FINAL TURN. You MUST provide the final recommendation. "
                "Your <answer> must contain your diagnosis and explanation. "
                "DO NOT ask any more questions."
            )
        else:
            action_constraint = (
                "If information is insufficient, ask ONE targeted question. "
                "If information is sufficient, provide suggestion and recommendation. "
                "You may also provide general guidance before the final answer."
            )

        format_instruction = (
            "Your clinical reasoning happens automatically in your thinking.\n"
            "After your thinking, output ONLY:\n"
            "<answer>[Your response content]</answer>\n\n"
            "Example:\n"
            "<answer>Question: Could you describe when the pain started and what makes it better or worse?</answer>\n\n"
            "Do not include any additional text outside the <answer> tags."
        )

        return (
            f"You are the Doctor.\n\n"
            f"Clinical Focus:\n{clinical_focus}\n\n"
            f"Current Dialogue Context:\n{history}\n\n"
            f"Your Goal:\n{action_constraint}\n\n"
            f"{format_instruction}"
        ).strip()

    def phase_3_assistant_step_prompt(
        self,
        current_json: dict[str, Any],
        is_final_turn: bool,
        task_category: str,
    ) -> str:
        """构建医生回复 prompt（医生可见自己的 <think> 内容和患者的纯文本回复）"""
        history_lines: list[str] = []
        for msg in current_json["prompt"]:
            if msg["role"] == "system":
                continue
            history_lines.append(f"{msg['role']}: {msg['content']}")
        history = "\n".join(history_lines)

        task_cat_lower = task_category.lower()
        if "diagnosis" in task_cat_lower:
            return self.prompt_for_diagnosis(history, is_final_turn)
        elif "medrecon" in task_cat_lower:
            return self.prompt_for_medrecon(history, is_final_turn)
        elif "prescription" in task_cat_lower:
            return self.prompt_for_prescription(history, is_final_turn)
        else:
            return self.prompt_for_diagnosis(history, is_final_turn)

    # ------------------------------------------------------------------
    # Phase 4：患者 ToM 回复 prompt 构建
    # ------------------------------------------------------------------

    def phase_4_user_step_prompt(
        self, current_json: dict[str, Any], ehr_data: str, kap_data: str
    ) -> str:
        """
        构建患者 ToM 回复 prompt。
        历史呈现规则：
          - 医生的回复：仅展示 <answer> 部分（不含 <think>）
          - 患者的历史回复：展示 raw_prompt 里的原始内容（含 <think>）
        """
        history_lines: list[str] = []
        for i, msg in enumerate(current_json["prompt"]):
            if msg["role"] == "system":
                continue
            if msg["role"] == "assistant":
                doc_answer = self._extract_answer_content(msg["content"])
                history_lines.append(f"assistant: {doc_answer}")
            elif msg["role"] == "user":
                # 患者侧用 raw_prompt（含自身思考）
                history_lines.append(
                    f"user: {current_json['raw_prompt'][i]['content']}"
                )
        history = "\n".join(history_lines)

        last_doctor_reply = self._extract_answer_content(
            current_json["prompt"][-1]["content"]
        )

        format_instruction = (
            "Your ToM reasoning about yourself happens automatically in your thinking.\n"
            "After your thinking, output ONLY your reply in this format:\n"
            "<patient>[Your reply to the doctor]</patient>\n\n"
            "If fully satisfied with NO moreExample (patient still has concerns, NO [END]):\n"
            "<think>I'm still worried about the side effects of this medication. "
            "I should ask the doctor about it.</think>\n"
            "<patient>But what about the side effects? "
            "Will this medication interact with anything I'm already taking?</patient>\n\n"
            "Example (patient is fully satisfied, WITH [END]):\n"
            "<think>The doctor answered all my questions clearly. I understand the treatment plan now.</think>\n"
            "<patient>Thank you, doctor. I understand everything now and will follow the plan. [END]</patient>"
        )

        return (
            f"You are the Patient.\n\n"
            f"## Previous Conversation\n{history}\n\n"
            f"## Patient KAP Profile\n{kap_data}\n\n"
            f"## EHR Data\n{ehr_data}\n\n"
            f'You just heard the doctor say: "{last_doctor_reply}"\n\n'
            f"Now, before replying, you MUST perform Theory of Mind (ToM) reasoning about YOURSELF. Think about:\n"
            f"- What are my current symptoms and how do they affect me?\n"
            f"- What am I most worried about right now?\n"
            f"- What do I need from the doctor (information, reassurance, treatment)?\n"
            f"- Based on my KAP profile, how should I express my concerns?\n"
            f"- What question did the doctor ask, and how should I answer it truthfully?\n\n"
            f"After this reasoning, reply to the doctor as the patient.\n\n"
            f"Guidelines:\n"
            f"- If the doctor asked about pain/symptoms, describe them based on the EHR.\n"
            f"- If the doctor mentioned labs or new medications, express confusion, concern, or ask about side effects.\n"
            f"- If you feel your questions have been answered and you have no more concerns, indicate satisfaction.\n"
            f"- Keep it in plain language.\n"
            f"- Adjust your language, knowledge level, and concerns based on your KAP profile.\n\n"
            f"### CRITICAL RULE about [END]:\n"
            f"- Add [END] ONLY if you have NO more questions AND you are completely satisfied.\n"
            f"- If your reply contains ANY question mark (?), do NOT add [END].\n"
            f"- If you are still confused, worried, or need more information, do NOT add [END].\n"
            f"- Most replies should NOT contain [END]. Only add it when you are truly done.\n\n"
            f"You MUST reply in English at all times. Do NOT use any other language.\n\n"
            f"{format_instruction}"
        ).strip()

    # ------------------------------------------------------------------
    # 终止 / 结束语
    # ------------------------------------------------------------------

    @staticmethod
    def _is_patient_ready_to_end(patient_reply: str) -> bool:
        if "[END]" not in patient_reply:
            return False
        # 提取 <patient> 标签内容来检测是否还有问号
        patient_content = patient_reply
        if "<patient>" in patient_content:
            parts = patient_content.split("<patient>", 1)
            if len(parts) > 1:
                patient_content = parts[1]
        return "?" not in patient_content

    @staticmethod
    def _build_final_closure_prompt() -> str:
        return (
            "You are the Doctor. The patient has indicated they are satisfied and ready to end the conversation.\n\n"
            "Provide a warm, professional closing to the medical consultation.\n"
            "- Acknowledge the patient's satisfaction\n"
            "- Offer additional support if needed\n"
            "- Provide clear next steps if applicable\n"
            "- Wish them well\n\n"
            "Always output your response in this exact format:\n"
            "<think>[Your closing reasoning]</think>\n"
            "<answer>[Your closing statement]</answer>\n\n"
            "Example:\n"
            "<think>Patient is satisfied with the consultation and has no further questions.</think>\n"
            "<answer>I'm glad we could address your concerns today. Please don't hesitate to reach out "
            "if you have any follow-up questions. Take care and follow the treatment plan we discussed.</answer>\n\n"
            "Do not include any additional text outside this format."
        ).strip()

    # ------------------------------------------------------------------
    # 内容提取工具
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_patient_content(text: str) -> str:
        """提取患者回复中的纯文本（去掉 <think> 块和 [END] 标记）"""
        content = text
        if "<think>" in content:
            parts = content.split("</think>", 1)
            if len(parts) > 1:
                content = parts[1]
        content = content.replace("[END]", "").strip()
        if "<patient>" in content:
            parts = content.split("<patient>", 1)
            if len(parts) > 1:
                content = parts[1]
                if "</patient>" in content:
                    content = content.split("</patient>", 1)[0]
        content = content.strip()
        if not content or content in {
            "Your reply to the doctor",
            "Your reply to the doctor\n",
        }:
            return ""
        return content

    @staticmethod
    def _extract_answer_content(text: str) -> str:
        """提取 <answer> 标签内的内容（去掉 <think> 块）"""
        content = text
        if "<think>" in content:
            parts = content.split("</think>", 1)
            if len(parts) > 1:
                content = parts[1]
        if "<answer>" in content:
            parts = content.split("<answer>", 1)
            if len(parts) > 1:
                content = parts[1]
                if "</answer>" in content:
                    content = content.split("</answer>", 1)[0]
        return content.strip()

    # ------------------------------------------------------------------
    # Phase 1：初始化对话结构（KAP 已在外部生成好，作为参数传入）
    # ------------------------------------------------------------------

    def phase_1_initialize_structure(
        self,
        sample_data: dict[str, Any],
        task_category: str,
        kap_data: str,  # 修复：不再在内部调用 LLM，改为接收已生成好的 KAP
    ) -> dict[str, Any]:
        disease_name = sample_data.get("task_info", {}).get(
            "event", "General Consultation"
        )
        ground_truth = sample_data.get("output", "UNKNOWN")
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0]

        topic, ability = self._get_task_metadata(task_category)
        ehr_data = sample_data.get("input", "")

        system_content = (
            f"{self.system_content}\n\n"
            f"## Patient KAP Profile\n{kap_data}\n\n"
            f"## EHR Data\n{ehr_data}"
        )

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
            "raw_prompt": [{"role": "system", "content": system_content}],
            "ehr_data": ehr_data,
            "kap_data": kap_data,
        }

    # ------------------------------------------------------------------
    # 存储：将对话结果写入 raw/ 和 dataset/ 目录
    # ------------------------------------------------------------------

    @staticmethod
    def _save_result(result: dict[str, Any], data: dict, index: int) -> dict[str, Any]:
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "raw")
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        event = data.get("task_info", {}).get("event", "unknown")

        # raw/：保留所有思考链（患者 <think> 可见）
        raw_output = {
            "data_source": result["data_source"],
            "topic": result["topic"],
            "department": result["department"],
            "subdepartment": result["subdepartment"],
            "disease": result["disease"],
            "raw_prompt": result["raw_prompt"],
            "ability": result["ability"],
            "reward_model": result["reward_model"],
        }
        raw_file = os.path.join(raw_dir, f"raw_{event}_{index}.json")
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(raw_output, f, ensure_ascii=False, indent=2)

        # dataset/：医生 <think> 保留，患者 <think> 去除
        # prompt 和 raw_prompt 对医生消息内容相同，对患者消息 prompt 已去掉思考
        final_prompt: list[dict] = []
        for i, msg in enumerate(result["prompt"]):
            if msg["role"] == "assistant":
                # 使用 raw_prompt 里的医生消息（内容相同，语义上更明确）
                final_prompt.append(result["raw_prompt"][i])
            else:
                # system / user：使用 prompt 里已截断的版本
                final_prompt.append(msg)

        dataset_output = {
            "data_source": result["data_source"],
            "topic": result["topic"],
            "department": result["department"],
            "subdepartment": result["subdepartment"],
            "disease": result["disease"],
            "prompt": final_prompt,
            "ability": result["ability"],
            "reward_model": result["reward_model"],
        }
        dataset_file = os.path.join(dataset_dir, f"{event}_{index}.json")
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(dataset_output, f, ensure_ascii=False, indent=2)

        return dataset_output

    # ------------------------------------------------------------------
    # 单样本运行（兼容旧接口）
    # ------------------------------------------------------------------

    def run(self, data: dict, llm: LLM, index: int, max_turns: int = 10) -> dict:
        task_category = data.get("task_info", {}).get("task", "diagnosis")

        # KAP 生成
        ehr_data = data.get("input", "")
        kap_data = self._generate_kap_data(ehr_data, llm)

        result = self.phase_1_initialize_structure(data, task_category, kap_data)

        # Phase 2：患者首次主诉
        patient_first_msg = llm.chat(self.phase_2_first_user_turn_prompt(data))
        result["raw_prompt"].append({"role": "user", "content": patient_first_msg})
        patient_first_content = self._extract_patient_content(patient_first_msg)
        result["prompt"].append({"role": "user", "content": patient_first_content})

        # 主对话循环
        turn = 0
        while True:
            is_final = turn >= max_turns - 1

            # 医生回复
            doc_prompt = self.phase_3_assistant_step_prompt(
                result, is_final, task_category
            )
            doc_reply = llm.chat(doc_prompt, self.system_content)
            result["raw_prompt"].append({"role": "assistant", "content": doc_reply})
            result["prompt"].append({"role": "assistant", "content": doc_reply})

            # 患者回复
            pat_prompt = self.phase_4_user_step_prompt(result, ehr_data, kap_data)
            pat_reply = llm.chat(pat_prompt)
            result["raw_prompt"].append({"role": "user", "content": pat_reply})
            pat_content = self._extract_patient_content(pat_reply)
            result["prompt"].append({"role": "user", "content": pat_content})

            turn += 1

            if self._is_patient_ready_to_end(pat_reply) or turn >= max_turns:
                closure_reply = llm.chat(
                    self._build_final_closure_prompt(), self.system_content
                )
                result["raw_prompt"].append(
                    {"role": "assistant", "content": closure_reply}
                )
                result["prompt"].append({"role": "assistant", "content": closure_reply})
                break

        return self._save_result(result, data, index)

    # ------------------------------------------------------------------
    # 批量运行：充分利用多卡算力，每步批量推理所有活跃对话
    # ------------------------------------------------------------------

    def batch_run(
        self,
        datas: list[dict],
        llm: LLM,
        start_index: int = 1,
        batch_size: int = 8,
        max_turns: int = 10,
    ) -> list[dict]:
        """
        批量生成医患对话数据集。

        核心思路：将多个对话的同一阶段 prompt 打包成一个 batch，
        通过 LLM.batch_chat() 单次提交，最大化 GPU 吞吐量。

        Args:
            datas:       待处理的 EHR 数据列表
            llm:         已加载的 LLM 实例
            start_index: 输出文件编号起点
            batch_size:  每批并行处理的样本数（建议 8~16）
            max_turns:   单次对话最大轮数，防止无限循环

        Returns:
            所有生成结果的列表（dataset/ 格式）
        """
        all_results: list[dict] = []
        total_chunks = (len(datas) + batch_size - 1) // batch_size

        for chunk_idx, chunk_start in enumerate(range(0, len(datas), batch_size)):
            chunk = datas[chunk_start : chunk_start + batch_size]
            chunk_indices = list(
                range(start_index + chunk_start, start_index + chunk_start + len(chunk))
            )
            print(
                f"\n=== Chunk {chunk_idx + 1}/{total_chunks} | "
                f"Samples {chunk_indices[0]}~{chunk_indices[-1]} ==="
            )

            # ── Step 1: 批量生成 KAP 画像 ───────────────────────────────
            kap_prompts = [self._build_kap_prompt(d.get("input", "")) for d in chunk]
            kap_replies = llm.batch_chat(kap_prompts)
            kap_datas = [self._extract_kap_content(r) for r in kap_replies]
            print(f"  [1/4] KAP profiles generated ({len(chunk)} samples)")

            # ── Step 2: 初始化所有对话状态 ──────────────────────────────
            task_categories = [
                d.get("task_info", {}).get("task", "diagnosis") for d in chunk
            ]
            states: list[dict[str, Any]] = []
            for data, kap_data, task_cat in zip(chunk, kap_datas, task_categories):
                result = self.phase_1_initialize_structure(data, task_cat, kap_data)
                states.append(
                    {
                        "result": result,
                        "data": data,
                        "task_category": task_cat,
                        "done": False,
                        "closed": False,
                        "turns": 0,
                    }
                )

            # ── Step 3: 批量生成患者首次主诉 ────────────────────────────
            phase2_prompts = [
                self.phase_2_first_user_turn_prompt(s["data"]) for s in states
            ]
            phase2_replies = llm.batch_chat(phase2_prompts)
            for s, reply in zip(states, phase2_replies):
                s["result"]["raw_prompt"].append({"role": "user", "content": reply})
                patient_content = self._extract_patient_content(reply)
                s["result"]["prompt"].append(
                    {"role": "user", "content": patient_content}
                )
            print("  [2/4] Patient opening complaints generated")

            # ── Step 4: 主对话循环（每步批量推理）──────────────────────
            while any(not s["done"] for s in states):
                active = [s for s in states if not s["done"]]

                # 医生回复（批量）
                doc_prompts = [
                    self.phase_3_assistant_step_prompt(
                        s["result"],
                        s["turns"] >= max_turns - 1,
                        s["task_category"],
                    )
                    for s in active
                ]
                doc_replies = llm.batch_chat(doc_prompts, self.system_content)
                for s, reply in zip(active, doc_replies):
                    s["result"]["raw_prompt"].append(
                        {"role": "assistant", "content": reply}
                    )
                    s["result"]["prompt"].append(
                        {"role": "assistant", "content": reply}
                    )

                # 患者回复（批量）
                pat_prompts = [
                    self.phase_4_user_step_prompt(
                        s["result"],
                        s["result"]["ehr_data"],
                        s["result"]["kap_data"],
                    )
                    for s in active
                ]
                pat_replies = llm.batch_chat(pat_prompts)
                for s, reply in zip(active, pat_replies):
                    s["result"]["raw_prompt"].append({"role": "user", "content": reply})
                    pat_content = self._extract_patient_content(reply)
                    s["result"]["prompt"].append(
                        {"role": "user", "content": pat_content}
                    )
                    s["turns"] += 1

                    if self._is_patient_ready_to_end(reply) or s["turns"] >= max_turns:
                        s["done"] = True

                # 对本轮刚结束的对话批量生成结束语（避免跨轮混入下一批）
                newly_done = [s for s in active if s["done"] and not s["closed"]]
                if newly_done:
                    closure_prompt = self._build_final_closure_prompt()
                    closure_replies = llm.batch_chat(
                        [closure_prompt] * len(newly_done), self.system_content
                    )
                    for s, reply in zip(newly_done, closure_replies):
                        s["result"]["raw_prompt"].append(
                            {"role": "assistant", "content": reply}
                        )
                        s["result"]["prompt"].append(
                            {"role": "assistant", "content": reply}
                        )
                        s["closed"] = True

                remaining = sum(1 for s in states if not s["done"])
                print(
                    f"  [3/4] Turn done | Active: {remaining}/{len(states)} | "
                    f"Max turns reached: {sum(1 for s in states if s['turns'] >= max_turns)}"
                )

            # ── Step 5: 保存本 chunk 所有结果 ──────────────────────────
            for s, idx in zip(states, chunk_indices):
                final = self._save_result(s["result"], s["data"], idx)
                all_results.append(final)
            print(
                f"  [4/4] Saved {len(states)} results (indices {chunk_indices[0]}~{chunk_indices[-1]})"
            )

        print(f"\n✓ All done! Total samples processed: {len(all_results)}")
        return all_results
