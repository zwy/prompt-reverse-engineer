"""
双通道评估模块：规则评分 + LLM Judge（带 Feedback）
支持通过 LLMClient 切换 provider
"""
import json
import dspy
from schema import parse_and_validate, ImageJSON
from llm_client import get_llm


# ── 规则评分 ──────────────────────────────────────────────────────────────────────────────────────

def rule_score(raw_prompt: str, parsed: ImageJSON) -> float:
    """
    基于规则的评分，满分 1.0。
    - 字段覆盖率 (40%)
    - 关键词保留率 (40%)
    - lora_tags 单独提取奖励 (20%)
    """
    score = 0.0
    key_fields = ["subject", "style", "lighting", "environment", "quality_tags"]
    filled = sum(1 for f in key_fields if getattr(parsed, f))
    score += (filled / len(key_fields)) * 0.4

    tokens = {t for t in raw_prompt.lower().replace(",", " ").split() if len(t) > 3}
    all_text = " ".join([
        parsed.subject,
        " ".join(parsed.style),
        parsed.mood or "",
        " ".join(parsed.quality_tags),
        parsed.environment or "",
        parsed.lighting or "",
    ]).lower()
    overlap = sum(1 for t in tokens if t in all_text)
    score += min((overlap / max(len(tokens), 1)) * 0.4, 0.4)

    if parsed.lora_tags:
        score += 0.2

    return round(score, 4)


# ── LLM Judge ────────────────────────────────────────────────────────────────────────────────────────────

class JSONQualityJudge(dspy.Signature):
    """你是专业的 AI 图像提示词质量评估专家。
    给定原始 SD prompt 和转换后的 image-json，评估转换质量并给出具体改进建议，
    用于优化 System Prompt。"""
    raw_prompt: str = dspy.InputField(desc="原始 Stable Diffusion prompt")
    image_json_str: str = dspy.InputField(desc="LLM 输出的 image-json 字符串")
    score: float = dspy.OutputField(desc="0.0~1.0 的质量分，1.0=完美转换")
    feedback: str = dspy.OutputField(desc="具体失败原因及改进建议，将用于优化 System Prompt")


judge_module = dspy.ChainOfThought(JSONQualityJudge)


# ── GEPA metric（带 Feedback）───────────────────────────────────────────────────────────────────────

def metric_with_feedback(
    gold,
    pred,
    trace=None,
    pred_name: str = "",
    pred_trace=None,
):
    """
    GEPA 优化器要求的 metric 函数格式。
    必须接受 5 个参数: (gold, pred, trace, pred_name, pred_trace)
    返回 dspy.Prediction(score=..., feedback=...)
    """
    raw_prompt = gold.raw_prompt
    json_str = getattr(pred, "image_json", "")

    parsed, err = parse_and_validate(json_str)

    if parsed is None:
        feedback = f"JSON 解析/校验失败：{err}。请确保严格输出合法 JSON，不含任何额外文字或 markdown。"
        return dspy.Prediction(score=0.0, feedback=feedback)

    r = rule_score(raw_prompt, parsed)

    # LLM Judge 仅在规则分 < 0.7 时调用，节省 token
    if r < 0.7:
        try:
            judge_result = judge_module(
                raw_prompt=raw_prompt,
                image_json_str=json_str,
            )
            llm_score = float(judge_result.score)
            final = (r + llm_score) / 2
            feedback = judge_result.feedback
        except Exception as e:
            final = r
            feedback = f"LLM Judge 调用失败: {e}"
    else:
        final = r
        feedback = "输出质量良好，无需改进。"

    return dspy.Prediction(score=round(final, 4), feedback=feedback)


# ── 批量评估入口 ───────────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from optimize_gepa import build_dspy_lm

    llm = get_llm()
    # 统一使用 build_dspy_lm，避免 LiteLLM 对 perplexity 自动附加 response_format
    dspy_lm = build_dspy_lm(llm)
    dspy.configure(lm=dspy_lm)

    data_path = Path("data/prompts.json")
    if not data_path.exists():
        print("请先运行 data/fetch_civitai.py 采集数据")
        exit(1)

    with open(data_path, encoding="utf-8") as f:
        dataset = json.load(f)

    sp_path = Path("outputs/system_prompt_v0.txt")
    if not sp_path.exists():
        print(f"请先创建 {sp_path}")
        exit(1)
    system_prompt = sp_path.read_text(encoding="utf-8")

    scores, failures = [], []
    for i, item in enumerate(dataset):
        raw = item["prompt"]
        output = llm.chat(system_prompt=system_prompt, user_prompt=raw)
        parsed, err = parse_and_validate(output)
        if parsed:
            s = rule_score(raw, parsed)
            scores.append(s)
            if s < 0.5:
                failures.append({"raw_prompt": raw, "output": output, "error": f"低分: {s}"})
        else:
            scores.append(0.0)
            failures.append({"raw_prompt": raw, "output": output, "error": err})

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(dataset)}] 当前均分: {sum(scores)/len(scores):.3f}")

    avg = sum(scores) / len(scores)
    print(f"\n最终平均分: {avg:.3f}")
    print(f"失败案例数: {len(failures)}")

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    with open(out / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump({"avg_score": avg, "scores": scores, "failures": failures}, f, ensure_ascii=False, indent=2)
    print("评估报告已保存至 outputs/eval_report.json")
