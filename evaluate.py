"""
双通道评估模块：规则评分 + LLM Judge（带 Feedback）
支持通过 LLMClient 切换 provider

评分说明（rule_score）：
  - 字段覆盖率      (40%)：6 个关键字段非空 + 内容充分性
  - 字段填充质量    (40%)：各字段内容长度是否充分（替代原来的跨语言字面匹配）
  - lora 提取准确性 (20%)：原始 prompt 含 lora 标签时检测是否正确提取
                           无 lora 标签的 prompt 此项满分，不作惩罚
"""
import json
import dspy
from schema import parse_and_validate, ImageJSON
from llm_client import get_llm


# ── 规则评分 ──────────────────────────────────────────────────────────────────

def rule_score(raw_prompt: str, parsed: ImageJSON) -> float:
    """
    基于规则的评分，满分 1.0。

    修复说明（相比 v1）：
    1. 字段覆盖率新增 camera 字段（原来漏掉了）
    2. 关键词保留率 → 字段填充质量：用内容长度代替跨语言字面匹配，
       解决中文 prompt → 英文 JSON 时 overlap 恒为 0 导致 40 分白扣的问题
    3. lora 逻辑修正：
       - 原始 prompt 含 <lora:> 但未提取 → 扣 0.2
       - 原始 prompt 含 <lora:> 且正确提取 → 满分
       - 原始 prompt 不含 <lora:> → 此项直接满分（0.2），不受影响
    """
    score = 0.0

    # ── Part 1: 字段覆盖率 (40%) ──
    # 检测 6 个关键字段是否非空
    key_fields = ["subject", "style", "lighting", "environment", "camera", "quality_tags"]
    filled = sum(1 for f in key_fields if getattr(parsed, f))
    score += (filled / len(key_fields)) * 0.4

    # ── Part 2: 字段填充质量 (40%) ──
    # 用各字段内容长度评估填充质量，不依赖字面 token 匹配
    # 每个字段内容长度达到阈值即视为充分填充
    THRESHOLDS = {
        "subject":      30,   # subject 应有足够描述
        "style":        10,   # style 列表拼接后至少 10 字符
        "lighting":     10,
        "environment":  10,
        "camera":        8,
        "mood":          5,
    }
    quality_scores = []
    for field, threshold in THRESHOLDS.items():
        val = getattr(parsed, field, None)
        if val is None:
            content_len = 0
        elif isinstance(val, list):
            content_len = len(" ".join(val))
        else:
            content_len = len(str(val))
        # 达到阈值得满分，未达到按比例给分
        quality_scores.append(min(content_len / threshold, 1.0))
    score += (sum(quality_scores) / len(quality_scores)) * 0.4

    # ── Part 3: lora 提取准确性 (20%) ──
    has_lora_in_raw = "<lora:" in raw_prompt.lower()
    if not has_lora_in_raw:
        # 原始 prompt 无 lora 标签，此项满分
        score += 0.2
    elif parsed.lora_tags:
        # 有 lora 且正确提取
        score += 0.2
    # else: 有 lora 但未提取，得 0，相当于扣 0.2

    return round(score, 4)


# ── LLM Judge ────────────────────────────────────────────────────────────────

class JSONQualityJudge(dspy.Signature):
    """你是专业的 AI 图像提示词质量评估专家。
    给定原始 SD prompt 和转换后的 image-json，评估转换质量并给出具体改进建议，
    用于优化 System Prompt。"""
    raw_prompt: str = dspy.InputField(desc="原始 Stable Diffusion prompt")
    image_json_str: str = dspy.InputField(desc="LLM 输出的 image-json 字符串")
    score: float = dspy.OutputField(desc="0.0~1.0 的质量分，1.0=完美转换")
    feedback: str = dspy.OutputField(desc="具体失败原因及改进建议，将用于优化 System Prompt")


judge_module = dspy.ChainOfThought(JSONQualityJudge)


# ── GEPA metric（带 Feedback）────────────────────────────────────────────────

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


# ── 批量评估入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from optimize_gepa import build_dspy_lm

    llm = get_llm()
    dspy_lm = build_dspy_lm(llm)
    dspy.configure(lm=dspy_lm)

    # 优先读取 filtered 数据，回退到原始数据
    data_path = Path("data/prompts_filtered.json")
    if not data_path.exists():
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

    scores, failures, converted_results = [], [], []

    for i, item in enumerate(dataset):
        raw = item["prompt"]
        output = llm.chat(system_prompt=system_prompt, user_prompt=raw)
        parsed, err = parse_and_validate(output)

        if parsed:
            s = rule_score(raw, parsed)
            scores.append(s)

            # 收集转换结果，供生图平台测试用
            converted_results.append({
                "index": i,
                "score": s,
                "raw_prompt": raw,
                "image_json": parsed.model_dump(),
            })

            if s < 0.5:
                failures.append({"raw_prompt": raw, "output": output, "error": f"低分: {s}"})
        else:
            scores.append(0.0)
            failures.append({"raw_prompt": raw, "output": output, "error": err})
            converted_results.append({
                "index": i,
                "score": 0.0,
                "raw_prompt": raw,
                "image_json": None,
                "parse_error": err,
            })

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(dataset)}] 当前均分: {sum(scores)/len(scores):.3f}")

    avg = sum(scores) / len(scores)
    print(f"\n最终平均分: {avg:.3f}")
    print(f"失败案例数: {len(failures)}")

    out = Path("outputs")
    out.mkdir(exist_ok=True)

    # 评估报告
    with open(out / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump({"avg_score": avg, "scores": scores, "failures": failures}, f, ensure_ascii=False, indent=2)
    print("评估报告已保存至 outputs/eval_report.json")

    # 转换结果（全量，含 image_json，供生图平台测试）
    with open(out / "converted_results.json", "w", encoding="utf-8") as f:
        json.dump(converted_results, f, ensure_ascii=False, indent=2)
    print(f"转换结果已保存至 outputs/converted_results.json（共 {len(converted_results)} 条）")

    # 高分转换结果（score >= 0.7，最干净的一批）
    top_results = [r for r in converted_results if r.get("score", 0) >= 0.7]
    with open(out / "converted_top.json", "w", encoding="utf-8") as f:
        json.dump(top_results, f, ensure_ascii=False, indent=2)
    print(f"高分转换结果已保存至 outputs/converted_top.json（共 {len(top_results)} 条，score >= 0.7）")
