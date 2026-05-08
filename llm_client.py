"""
LLM 客户端 - 支持多种 LLM Provider

支持：perplexity / openai / ollama / grok
参考: https://github.com/zwy/zimage-prompt/blob/main/zimage_prompt/llm_client.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    统一 LLM 调用接口

    用法:
        llm = LLMClient.from_env()
        text = llm.chat(system_prompt="你是...", user_prompt="输入内容")
    """

    PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
    MODEL = os.getenv("LLM_MODEL", "")
    API_KEY = os.getenv("LLM_API_KEY", "")
    BASE_URL = os.getenv("LLM_BASE_URL", "")

    def __init__(self, provider: str, model: str, api_key: str = "", base_url: str = ""):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = self._build_client()

    @classmethod
    def from_env(cls) -> "LLMClient":
        """从环境变量创建客户端"""
        return cls(
            provider=cls.PROVIDER,
            model=cls.MODEL or cls._default_model(cls.PROVIDER),
            api_key=cls.API_KEY or cls._get_api_key(cls.PROVIDER),
            base_url=cls.BASE_URL,
        )

    @classmethod
    def _default_model(cls, provider: str) -> str:
        defaults = {
            "openai": "gpt-4o-mini",
            "perplexity": "google/gemini-3-flash-preview",
            "ollama": "qwen2.5:7b",
            "grok": "grok-4-1-fast-non-reasoning",
        }
        return defaults.get(provider, "gpt-4o-mini")

    @classmethod
    def _get_api_key(cls, provider: str) -> str:
        env_map = {
            "openai": "OPENAI_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY",
            "grok": "XAI_API_KEY",
        }
        key = env_map.get(provider, "")
        return os.environ.get(key, "")

    def _build_client(self):
        if self.provider in ("openai", "perplexity", "grok"):
            from openai import OpenAI
            kwargs = {"api_key": self.api_key} if self.api_key else {}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            return OpenAI(**kwargs)
        elif self.provider == "ollama":
            from openai import OpenAI
            base = self.base_url or "http://localhost:11434/v1"
            return OpenAI(base_url=base, api_key="ollama")
        else:
            raise ValueError(f"不支持的 provider: {self.provider}")

    def chat(self, messages: list[dict] | None = None,
             system_prompt: str = "", user_prompt: str = "") -> str:
        """
        发送消息，返回模型回复文本。
        支持两种调用方式：
          1. chat(messages=[...])               # 直接传 messages 列表
          2. chat(system_prompt=..., user_prompt=...)  # 自动组装 messages
        """
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})

        if self.provider == "perplexity":
            return self._chat_perplexity(messages)
        else:
            return self._chat_openai_compat(messages)

    def _chat_perplexity(self, messages: list[dict]) -> str:
        """使用 Perplexity Responses API"""
        from perplexity import Perplexity

        client = Perplexity(api_key=self.api_key) if self.api_key else Perplexity()

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_parts = [m["content"] for m in messages if m["role"] != "system"]

        parts = []
        if system_parts:
            parts.append("[Instructions]\n" + "\n\n".join(system_parts))
        parts.append("[Task]\n" + "\n\n".join(user_parts))
        input_text = "\n\n".join(parts)

        response = client.responses.create(
            model=self.model or "google/gemini-3-flash-preview",
            input=input_text,
        )
        return response.output_text

    def _chat_openai_compat(self, messages: list[dict]) -> str:
        """OpenAI / Ollama / Grok 标准接口"""
        response = self._client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            messages=messages,
            temperature=0.2,   # 结构化输出任务用低温度
            max_tokens=2000,
        )
        return response.choices[0].message.content


# 全局单例，模块级直接 import 使用
_llm_instance: LLMClient | None = None


def get_llm() -> LLMClient:
    """获取 LLM 客户端单例（懒加载）"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient.from_env()
    return _llm_instance
