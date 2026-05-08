"""
GEPA 自动优化主流程
基于反思进化自动迭代 System Prompt
使用 LLMClient 统一接口，支持 perplexity/openai/ollama/grok
"""
import json
import os
import dspy
from pathlib import Path
from llm_client import get_llm, LLMClient
from evaluate import metric_with_feedback


def build_dspy_lm(llm: LLMClient) -> dspy.LM:
    """
    根据 LLMClient 配置构建 DSPy LM。
    与 llm_client.py 中 _build_client 逻辑保持一致：

      perplexity → 直接用 perplexity/<model> + api_base
                    LiteLLM 的 perplexity/ 前缀会路由到 api.perplexity.ai
      ollama     → ollama/<model> + api_base
      grok       → openai/<model> + api_base (xAI 兼容 OpenAI 协议)
      openai     → openai/<model>
    """
    if llm.provider == "perplexity":
        # llm.model 可能是 “xai/grok-4-1-fast-non-reasoning” 这样带前缀的格式，
        # 也可能是 “openai/gpt-4o-mini” 。
        # LiteLLM perplexity provider 要求格式： perplexity/<bare-model-name>
        # 所以去掉已有 provider 前缀再拼接。
        bare_model = llm.model.split("/")[-1] if "/" in llm.model else llm.model
        return dspy.LM(
            f"perplexity/{bare_model}",
            api_key=llm.api_key,
        )
    elif llm.provider == "ollama":
        base = llm.base_url or "http://localhost:11434"
        return dspy.LM(
            f"ollama/{llm.model}",
            api_base=base,
        )
    elif llm.provider == "grok":
        # xAI 兴趣商兼容 OpenAI 协议，模型名原样传
        bare_model = llm.model.split("/")[-1] if "/" in llm.model else llm.model
        return dspy.LM(
            f"openai/{bare_model}",
            api_key=llm.api_key,
            api_base="https://api.x.ai/v1",
        )
    else:  # openai 及其他 OpenAI 兴趣商
        kwargs: dict = {"api_key": llm.api_key} if llm.api_key else {}
        if llm.base_url:
            kwargs["api_base"] = llm.base_url
        bare_model = llm.model.split("/")[-1] if "/" in llm.model else llm.model
        return dspy.LM(f"openai/{bare_model}", **kwargs)


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
