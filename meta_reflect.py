"""
手动 Meta-Prompting 反思循环
让 LLM 自己分析失败案例并改写 System Prompt
使用 LLMClient 统一接口，支持 perplexity/openai/ollama/grok
"""
import json
from pathlib import Path
from llm_client import get_llm


def reflect_and_improve(system_prompt: str, failed_cases: list[dict], version: int) -> str:
    """
    让 LLM 分析失败案例，自动改写 System Prompt。

    Args:
        system_prompt: 当前版本的 System Prompt
        failed_cases: [{"raw_prompt": ..., "output": ..., "error": ...}, ...]
        version: 当前版本号

    Returns:
        改进后的新 System Prompt
    """
    cases = failed_cases[:5]
    cases_text = "\n\n".join([
        f"【案例 {i+1}】\n输入：{c['raw_prompt'][:300]}\n输出：{c['output'][:500]}\n问题：{c['error']}"
        for i, c in enumerate(cases)
    ])

    user_prompt = f"""你是一个 System Prompt 优化专家，专注于文生图提示词结构化任务。

当前 System Prompt（v{version}）：
```
{system_prompt}
```

以下是使用该 System Prompt 失败或低分的案例：
{cases_text}

请分析失败原因，并输出改进后的 System Prompt（v{version+1}）。

优化要求：
1. 保持核心指令：严格输出 JSON，包含所有必要字段
2. 针对失败案例的具体问题，添加约束规则或 few-shot 示例
3. 如有必要，可加入边界条件说明（如 lora_tags 的提取规则）
4. 只输出新的 System Prompt 文本，不要有任何解释或前言"""

    llm = get_llm()
    return llm.chat(user_prompt=user_prompt)


def run_reflection_loop(max_versions: int = 5):
    """完整的反思迭代循环"""
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    report_path = out_dir / "eval_report.json"
    if not report_path.exists():
        print("请先运行 evaluate.py 生成评估报告")
        return

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    failures = report.get("failures", [])
    avg_score = report.get("avg_score", 0)
    print(f"当前平均分: {avg_score:.3f}，失败案例: {len(failures)} 条")

    if avg_score >= 0.85:
        print("平均分已达 0.85，无需继续优化")
        return

    # 找到当前最新版 System Prompt
    version = 0
    while (out_dir / f"system_prompt_v{version + 1}.txt").exists():
        version += 1

    current_sp_path = out_dir / f"system_prompt_v{version}.txt"
    if not current_sp_path.exists():
        print(f"未找到 {current_sp_path}，请先创建 outputs/system_prompt_v0.txt")
        return

    current_prompt = current_sp_path.read_text(encoding="utf-8")
    print(f"当前版本: v{version}")

    for i in range(max_versions):
        print(f"\n--- 第 {i + 1} 轮反思 ---")
        new_prompt = reflect_and_improve(current_prompt, failures, version)

        new_version = version + 1
        new_path = out_dir / f"system_prompt_v{new_version}.txt"
        new_path.write_text(new_prompt, encoding="utf-8")
        print(f"新版 System Prompt 已保存至 {new_path}")
        print("--- 新 Prompt 预览（前500字）---")
        print(new_prompt[:500])

        current_prompt = new_prompt
        version = new_version

        cont = input("\n继续下一轮反思？(y/n): ").strip().lower()
        if cont != "y":
            break

    print(f"\n反思循环结束，最终版本: v{version}")


if __name__ == "__main__":
    run_reflection_loop()
