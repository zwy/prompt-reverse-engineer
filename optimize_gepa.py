"""
GEPA 自动优化主流程
基于反思进化自动迭代 System Prompt
使用 LLMClient 统一接口，支持 perplexity/openai/ollama/grok
"""
import json
import os
from typing import Any
import dspy
from pathlib import Path
from llm_client import get_llm, LLMClient
from evaluate import metric_with_feedback


# ───────────────────────────────────────────────────────────────────────────────
class PerplexityAgentLM(dspy.LM):
    """
    自定义 DSPy LM，直接用 OpenAI SDK 调用 Perplexity Agent API
    (client.responses.create → POST /v1/responses)。

    为什么要自定义：
      dspy.LM 底层用 LiteLLM，LiteLLM 调 perplexity 走 Chat Completions 接口，
      并且自动附加 response_format: {type: json_object}。
      但 Perplexity Agent API 不支持该字段，所以绕过 LiteLLM 直接调用。
    """

    def __init__(self, model: str, api_key: str, **kwargs):
        # 传入一个虚拟的占位符 model 名让父类初始化，
        # 但实际调用完全重写
        super().__init__(model=f"openai/{model}", api_key=api_key, **kwargs)
        from openai import OpenAI
        self._pplx_client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai/v1",
        )
        self._pplx_model = model  # 原始模型名，如 xai/grok-4-1-fast-non-reasoning

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        DSPy 会以 messages 列表调用此方法。
        我们拿到 messages 后，拆分 system/user 传给 Agent API。
        """
        # 从 kwargs 中过滤掉 LiteLLM 专用字段，避免传给 OpenAI SDK 报错
        _drop = {"response_format", "num_retries", "cache", "metadata",
                 "acompletion", "mock_response", "api_base", "api_version"}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in _drop}

        # 拆分 system instructions 和 user input
        instructions = None
        input_messages = []
        for msg in (messages or []):
            if msg.get("role") == "system":
                instructions = msg["content"]
            else:
                input_messages.append(msg)

        create_kwargs: dict[str, Any] = {
            "model": self._pplx_model,
            "input": input_messages if len(input_messages) != 1
                     else input_messages[0]["content"],
        }
        if instructions:
            create_kwargs["instructions"] = instructions
        create_kwargs.update(clean_kwargs)

        response = self._pplx_client.responses.create(**create_kwargs)
        text = response.output_text

        # DSPy 期望返回与 LiteLLM 相同的 choices 结构
        # 简单包装成 ModelResponse 兼容结构
        from types import SimpleNamespace
        choice = SimpleNamespace(
            message=SimpleNamespace(content=text, role="assistant"),
            finish_reason="stop",
        )
        result = SimpleNamespace(
            choices=[choice],
            model=self._pplx_model,
            usage=getattr(response, "usage", None),
        )
        return [text]  # dspy.LM.__call__ 期望返回 completions 列表


# ───────────────────────────────────────────────────────────────────────────────
def build_dspy_lm(llm: LLMClient) -> dspy.LM:
    """
    根据 LLMClient 配置构建 DSPy LM。

    perplexity → PerplexityAgentLM（绕过 LiteLLM，直接用 Agent API）
    ollama     → dspy.LM("ollama/...")
    grok       → dspy.LM("openai/...", api_base=xai)
    openai     → dspy.LM("openai/...")
    """
    bare = llm.model.split("/")[-1] if "/" in llm.model else llm.model

    if llm.provider == "perplexity":
        return PerplexityAgentLM(model=llm.model, api_key=llm.api_key)

    elif llm.provider == "ollama":
        return dspy.LM(
            f"ollama/{bare}",
            api_base=llm.base_url or "http://localhost:11434",
        )
    elif llm.provider == "grok":
        return dspy.LM(
            f"openai/{bare}",
            api_key=llm.api_key,
            api_base="https://api.x.ai/v1",
        )
    else:  # openai
        kwargs: dict = {"api_key": llm.api_key} if llm.api_key else {}
        if llm.base_url:
            kwargs["api_base"] = llm.base_url
        return dspy.LM(f"openai/{bare}", **kwargs)


# ───────────────────────────────────────────────────────────────────────────────
class PromptToImageJSON(dspy.Signature):
    """你是专业的 SD 提示词结构化专家。
    将文生图 prompt 解析为结构化 image-json。
    严格只输出 JSON 对象，不含任何额外文字、解释或 markdown。

    输出字段：
    - subject: 主体描述（人物/物体/场景）
    - style: 画风列表
    - lighting: 光线描述
    - environment: 背景/场景
    - camera: 镜头/构图
    - quality_tags: 质量 tag 数组（masterpiece 等）
    - lora_tags: <lora:xxx:weight> 格式的 tag 数组
    - artist_reference: 艺术家参考数组
    - mood: 氛围/情绪
    - negative_prompt: 负面提示词
    """
    raw_prompt: str = dspy.InputField(
        desc="Stable Diffusion / NovelAI 风格的文生图提示词，可能包含逗号分隔的 tag 或自然语言描述"
    )
    image_json: str = dspy.OutputField(
        desc="合法的 JSON 字符串，包含上述所有字段，无多余文字"
    )


class PromptParser(dspy.Module):
    def __init__(self):
        self.parser = dspy.ChainOfThought(PromptToImageJSON)

    def forward(self, raw_prompt: str):
        return self.parser(raw_prompt=raw_prompt)


def main():
    # ── 配置 LLM ──
    llm = get_llm()
    print(f"使用 Provider: {llm.provider} | 模型: {llm.model}")

    task_lm = build_dspy_lm(llm)

    # GEPA reflection 用强模型（可选，通过 GEPA_PROMPT_MODEL 指定）
    strong_model = os.getenv("GEPA_PROMPT_MODEL", "")
    if strong_model:
        reflection_lm = build_dspy_lm(LLMClient(
            provider=llm.provider,
            model=strong_model,
            api_key=llm.api_key,
            base_url=llm.base_url,
        ))
    else:
        reflection_lm = task_lm

    dspy.configure(lm=task_lm)

    # ── 加载数据集 ──
    data_path = Path("data/prompts.json")
    if not data_path.exists():
        raise FileNotFoundError("请先运行 data/fetch_civitai.py")

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    examples = [
        dspy.Example(raw_prompt=d["prompt"]).with_inputs("raw_prompt")
        for d in data
    ]
    trainset, devset = examples[:70], examples[70:]
    print(f"训练集: {len(trainset)}，验证集: {len(devset)}")

    # ── GEPA 优化 ──
    auto_budget = os.getenv("GEPA_AUTO_BUDGET", "medium")  # light / medium / heavy
    log_dir = os.getenv("GEPA_LOG_DIR", "outputs/gepa_logs")

    optimizer = dspy.GEPA(
        metric=metric_with_feedback,
        reflection_lm=reflection_lm,
        auto=auto_budget,
        log_dir=log_dir,
    )

    module = PromptParser()
    print("\n开始 GEPA 优化...")
    optimized = optimizer.compile(module, trainset=trainset, valset=devset)

    # ── 保存结果 ──
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    optimized.save(str(out_dir / "optimized_module.json"))
    print("已保存优化模块至 outputs/optimized_module.json")

    sig = optimized.parser.signature
    best_prompt = sig.instructions
    sp_path = out_dir / "system_prompt_best.txt"
    sp_path.write_text(best_prompt, encoding="utf-8")
    print(f"\n=== 最优 System Prompt ===")
    print(best_prompt)
    print(f"\n已保存至 {sp_path}")


if __name__ == "__main__":
    main()
