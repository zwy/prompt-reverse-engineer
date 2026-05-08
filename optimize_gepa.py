"""
GEPA 自动优化主流程
基于反思进化自动迭代 System Prompt
使用 LLMClient 统一接口，支持 perplexity/openai/ollama/grok
"""
import json
import dspy
from pathlib import Path
from llm_client import get_llm
from evaluate import metric_with_feedback


def build_dspy_lm(llm) -> dspy.LM:
    """根据 LLMClient 配置构建对应的 DSPy LM"""
    if llm.provider == "perplexity":
        return dspy.LM(
            f"openai/{llm.model}",
            api_key=llm.api_key,
            api_base="https://api.perplexity.ai",
        )
    elif llm.provider == "ollama":
        return dspy.LM(
            f"ollama/{llm.model}",
            api_base=llm.base_url or "http://localhost:11434",
        )
    elif llm.provider == "grok":
        return dspy.LM(
            f"openai/{llm.model}",
            api_key=llm.api_key,
            api_base="https://api.x.ai/v1",
        )
    else:  # openai
        return dspy.LM(f"openai/{llm.model}", api_key=llm.api_key)


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

    # GEPA 反思用强模型（如果是 perplexity 则同 provider 切换强模型）
    import os
    from llm_client import LLMClient
    strong_model = os.getenv("GEPA_PROMPT_MODEL", "")
    if strong_model:
        prompt_llm = LLMClient(
            provider=llm.provider,
            model=strong_model,
            api_key=llm.api_key,
            base_url=llm.base_url,
        )
        prompt_lm = build_dspy_lm(prompt_llm)
    else:
        prompt_lm = task_lm  # 没配置则复用同一模型

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
    optimizer = dspy.GEPA(
        metric=metric_with_feedback,
        prompt_model=prompt_lm,
        task_model=task_lm,
        num_iterations=10,
        verbose=True,
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
