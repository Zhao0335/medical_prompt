import torch
import vllm
from transformers import AutoTokenizer


class LLM:
    def __init__(
        self,
        _modelpath: str,
        _temperature: float = 0.8,
        _top_p: float = 0.8,
        _max_tokens: int = 8192,
        _gpu_memory_utilization: float = 0.9,
    ) -> None:
        """
        初始化 LLM 配置
        """
        self.model_path: str = _modelpath
        self.temperature: float = _temperature
        self.top_p: float = _top_p
        self.max_tokens: int = _max_tokens  # 最大回复长度
        self.gpu_memory_utilization: float = _gpu_memory_utilization  # 显存占用率

        self.llm = None
        self.tokenizer = None
        self.sampling_params = None

    def load_model(self) -> None:
        """
        新增接口：启动配置的 LLM 模型
        """
        # 找到模型的存储路径
        import os

        home_dir = os.path.expanduser("~")
        models_dir = os.path.join(home_dir, "models")
        full_model_path = os.path.join(models_dir, self.model_path)

        if self.llm is None:
            print(f"正在加载模型：{full_model_path}")

            # 使用 vLLM 加载模型
            self.llm = vllm.LLM(
                model=full_model_path,
                gpu_memory_utilization=0.9,  # 增加内存使用率以容纳模型和KV缓存
                download_dir=models_dir,
                trust_remote_code=True,
                tensor_parallel_size=4,  # 使用8个GPU以获得更多内存
                max_model_len=40960,  # 恢复最大序列长度
            )

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                full_model_path, local_files_only=True
            )

            # 模型参数设置：选词方式
            self.sampling_params = vllm.SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            print("模型加载完成")
        else:
            print("模型已经在运行中")

    def unload_model(self) -> None:
        """
        新增接口：关闭模型
        """
        if self.llm is not None:
            print("正在关闭模型")
            del self.llm
            self.llm = None

            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None

            torch.cuda.empty_cache()
            print("模型已关闭")
        else:
            print("模型并未启动")

    @property
    def get_model_status(self) -> bool:
        return self.llm is not None

    def chat(self, text: str, system_prompt: str = "") -> str:
        """
        用来产出对话

        Args:
            text:          输入文本
            system_prompt: 可选的系统提示词，传入后以 system 角色发送给模型

        Returns:
            answer_text: llm回复
        """
        if self.tokenizer is None or self.llm is None:
            print("模型未启动")
            return ""

        assert self.tokenizer is not None
        assert self.llm is not None

        # AI 对话的标准格式：系统指令（可选）+ 用户提问
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        # 关闭模型内置思考模式（避免产生"我现在是一个医生..."之类的冗长垃圾），
        # 同时在 prompt 末尾注入 <think> 开头，让模型直接填充我们要求的简洁推理内容，
        # 再在输出前拼回 <think> 前缀，得到完整的 <think>...</think><answer>...</answer>。
        thinking_disabled = False
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_text += "<think>"  # 注入开头，模型续写推理内容后自行关闭标签
            thinking_disabled = True
        except TypeError:
            # 旧版 transformers 不支持 enable_thinking，退回内置思考模式
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # 记录prompt到文件
        self._log_prompt(prompt_text)

        # 生成回答
        response = self.llm.generate([prompt_text], self.sampling_params)
        answer_text = str(response[0].outputs[0].text)

        if thinking_disabled:
            # 把注入的 <think> 开头拼回，得到完整格式
            answer_text = "<think>" + answer_text
        else:
            # fallback：内置思考模式产生冗长推理，直接裁掉第一个 <think>…</think> 块
            answer_text = self._strip_first_think_block(answer_text)

        return answer_text

    def batch_chat(self, texts: list[str], system_prompt: str = "") -> list[str]:
        """
        批量推理：一次性提交多个 prompt，最大化 GPU 利用率

        Args:
            texts:         输入文本列表
            system_prompt: 可选的系统提示词，对批次中每条消息都生效

        Returns:
            answers: LLM 回复列表，顺序与输入一致
        """
        if not texts:
            return []
        if self.tokenizer is None or self.llm is None:
            print("模型未启动")
            return [""] * len(texts)

        # 关闭内置思考模式并注入 <think> 前缀（与 chat() 逻辑一致）
        thinking_disabled = False
        prompt_texts: list[str] = []
        for text in texts:
            messages: list[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": text})
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_text += "<think>"
                thinking_disabled = True
            except TypeError:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            prompt_texts.append(prompt_text)

        # 记录批量 prompt 到一个文件
        self._log_prompt(
            f"[BATCH SIZE={len(texts)}]\n\n"
            + "\n\n---\n\n".join(
                f"[Item {i + 1}]\n{p}" for i, p in enumerate(prompt_texts)
            )
        )

        # 单次 generate 调用处理所有 prompt
        responses = self.llm.generate(prompt_texts, self.sampling_params)

        if thinking_disabled:
            return ["<think>" + str(r.outputs[0].text) for r in responses]
        else:
            return [
                self._strip_first_think_block(str(r.outputs[0].text)) for r in responses
            ]

    @staticmethod
    def _strip_first_think_block(text: str) -> str:
        """fallback 模式（旧版 transformers）：裁掉内置思考产生的第一个 <think>…</think> 块。"""
        text = text.strip()
        if text.startswith("<think>"):
            end = text.find("</think>")
            if end != -1:
                return text[end + len("</think>") :].strip()
        return text

    def _log_prompt(self, prompt: str) -> None:
        """
        记录每次发送给模型的prompt到文件
        """
        import datetime
        import os

        # 创建logs目录（如果不存在）
        logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # 生成日志文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(logs_dir, f"prompt_{timestamp}.txt")

        # 写入prompt
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        print(f"Prompt logged to: {log_file}")

    def set_sampling_params(
        self, _temperature: float, _top_p: float, _max_tokens: int
    ) -> None:
        """
        这个函数用来调整模型采样参数
        """
        self.temperature = _temperature
        self.top_p = _top_p
        self.max_tokens = _max_tokens
        if self.sampling_params:
            self.sampling_params = vllm.SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

    def set_gpu_memory_utilization(self, gpu_memory_utilization: float) -> None:
        if self.llm is None:
            self.gpu_memory_utilization = gpu_memory_utilization
        else:
            self.unload_model()
            self.gpu_memory_utilization = gpu_memory_utilization
            self.load_model()

    def __del__(self) -> None:
        """
        析构函数: 删除实例对象后清理显存
        """
        if self.llm:
            del self.llm
            torch.cuda.empty_cache()
