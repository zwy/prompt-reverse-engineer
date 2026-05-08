"""
GEPA 自动优化主流程
基于反思进化自动迭代 System Prompt
使用 LLMClient 统一接口，支持 perplexity/openai/ollama/grok
"""
import json
import os
from typing import Any
from datetime import datetime
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
        super().__init__(model=f"openai/{model}", api_key=api_key, **kwargs)
        from openai import OpenAI
        self._pplx_client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai/v1",
        )
        self._pplx_model = model

    def __call__(self, prompt=None, messages=None, **kwargs):
        _drop = {"response_format", "num_retries", "cache", "metadata",
                 "acompletion", "mock_response", "api_base", "api_version"}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in _drop}

        instructions = None
        input_messages = []

        if messages:
            for msg in messages:
                if msg.get("role") == "system":
                    instructions = msg["content"]
                else:
                    input_messages.append(msg)
        elif prompt:
            input_messages = [{"role": "user", "content": prompt}]

        input_val = (
            input_messages[0]["content"] if len(input_messages) == 1
            else input_messages
        )

        create_kwargs: dict[str, Any] = {
            "model": self._pplx_model,
            "input": input_val,
        }
        if instructions:
            create_kwargs["instructions"] = instructions
        create_kwargs.update(clean_kwargs)

        response = self._pplx_client.responses.create(**create_kwargs)
        return [response.output_text]


# ───────────────────────────────────────────────────────────────────────────────
def build_dspy_lm(llm: LLMClient) -> dspy.LM:
    """
    根据 LLMClient 配置构建 DSPy LM。
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
    else:
        kwargs: dict = {"api_key": llm.api_key} if llm.api_key else {}
        if llm.base_url:
            kwargs["api_base"] = llm.base_url
        return dspy.LM(f"openai/{bare}", **kwargs)


# ───────────────────────────────────────────────────────────────────────────────
class PromptToImageJSON(dspy.Signature):
    """你是专业的文生图提示词结构化专家。
    将文生图 prompt 解析为结构化 image-json，核心目标是：
    让每个视觉维度独立可控——修改任意一个字段，其余字段保持不变。

    严格只输出 JSON 对象，不含任何额外文字、解释或 markdown。

    字段说明：
    - subject: 主体。简单场景可用字符串；人物场景强烈建议使用嵌套对象，
      包含 demographics / face / hair / pose / attire / expression 等子字段
    - background: 背景元素（与 environment 区分：background 是「画面中有什么」）
    - environment: 环境氛围（「是什么感觉」，如 humid, misty, warm）
    - time_of_day: 时间段，如 "golden hour", "midnight"
    - style: 画风数组，如 ["anime illustration", "painterly"]
    - artist_reference: 艺术家参考数组
    - mood: 整体情绪/氛围
    - color_palette: 主色调描述
    - lighting: 光线对象，包含 type / source / details 子字段
    - camera: 镜头对象，包含 shot_scale / lens / aperture / angle / focus 子字段
    - must_keep: 【关键】必须保留的视觉元素列表，确保维度稳定性
    - avoid: 【关键】应当避免的内容列表
    - negative_prompt: 负面提示词列表
    - sd_extras: 仅 SD/ComfyUI 流程时填写，包含 quality_tags / lora_tags

    重要规则：
    1. must_keep 和 avoid 必须填写，这是精准控制的核心
    2. 人物图像中 subject 请尽量使用嵌套对象以获得逐字段可控性
    3. lighting 和 camera 请使用嵌套对象（非字符串）
    4. 不要把摄影参数写入 mood 或 style
    5. 若原始 prompt 无 SD 专属内容（lora/quality tags），sd_extras 留 null
    """
    raw_prompt: str = dspy.InputField(
        desc="文生图提示词，可能是逗号分隔的 tag 列表或自然语言描述"
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
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"本次运行 ID: {run_id}")

    llm = get_llm()
    print(f"使用 Provider: {llm.provider} | 模型: {llm.model}")

    task_lm = build_dspy_lm(llm)

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

    auto_budget = os.getenv("GEPA_AUTO_BUDGET", "medium")
    log_dir = os.getenv("GEPA_LOG_DIR", f"outputs/gepa_logs/{run_id}")

    optimizer = dspy.GEPA(
        metric=metric_with_feedback,
        reflection_lm=reflection_lm,
        auto=auto_budget,
        log_dir=log_dir,
    )

    module = PromptParser()
    print("\n开始 GEPA 优化...")
    optimized = optimizer.compile(module, trainset=trainset, valset=devset)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    module_archive = out_dir / f"optimized_module_{run_id}.json"
    sp_archive     = out_dir / f"system_prompt_best_{run_id}.txt"
    module_latest  = out_dir / "optimized_module_latest.json"
    sp_latest      = out_dir / "system_prompt_best.txt"

    optimized.save(str(module_archive))
    optimized.save(str(module_latest))
    print(f"已保存优化模块至 {module_archive}")
    print(f"已更新 latest 至 {module_latest}")

    parser = optimized.parser
    if hasattr(parser, "predict"):
        sig = parser.predict.signature
    elif hasattr(parser, "prog"):
        sig = parser.prog.signature
    else:
        sig = optimized.predictors()[0].signature

    best_prompt = sig.instructions

    sp_archive.write_text(best_prompt, encoding="utf-8")
    sp_latest.write_text(best_prompt, encoding="utf-8")
    print(f"\n=== 最优 System Prompt ===")
    print(best_prompt)
    print(f"\n已保存至 {sp_archive}")
    print(f"已更新 latest 至 {sp_latest}")


if __name__ == "__main__":
    main()
