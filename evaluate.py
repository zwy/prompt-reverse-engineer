"""
双通道评估模块：规则评分 + LLM Judge（带 Feedback）
支持通过 LLMClient 切换 provider

评分说明（rule_score）v2：
  - 核心字段覆盖率  (35%)：subject / style / lighting / camera / background / mood
  - 字段填充质量    (30%)：各字段内容长度充分性（不依赖字面 token 匹配）
  - Constraints 覆盖 (20%)：must_keep + avoid 是精准维度控制的核心
  - 负面提示词质量  (15%)：negative_prompt 列表是否有实质内容

  移除：quality_tags / lora_tags 评分（SD 专属字段，不计入通用评分）
"""
import json
import dspy
from schema import parse_and_validate, ImageJSON
from llm_client import get_llm


# ── 辅助：获取字段内容字符长度 ───────────────────────────────────────────────

def _content_len(val) -> int:
    """统一计算字段内容的字符长度。"""
    if val is None:
        return 0
    if isinstance(val, list):
        return len(" ".join(str(x) for x in val))
    if isinstance(val, dict):
        return len(" ".join(str(v) for v in val.values() if v))
    # Pydantic model（如 PersonSubject, LightingDetail 等）
    if hasattr(val, "model_dump"):
        return len(" ".join(str(v) for v in val.model_dump().values() if v))
    return len(str(val))


# ── 规则评分 ──────────────────────────────────────────────────────────────────

def rule_score(raw_prompt: str, parsed: ImageJSON) -> float:
    """
    基于规则的评分，满分 1.0。

    v2 变更：
    1. 核心字段移除 quality_tags（SD 专属），加入 background
    2. 字段填充质量：lighting/camera 改为读取嵌套对象的总字符长度
    3. 新增 Constraints 覆盖评分（must_keep + avoid），权重 20%
    4. 新增 negative_prompt 质量评分，权重 15%
    5. 总权重：35% + 30% + 20% + 15% = 100%
    """
    score = 0.0

    # ── Part 1: 核心字段覆盖率 (35%) ──
    # 检测 6 个核心字段是否有实质内容
    core_fields = ["subject", "style", "lighting", "camera", "background", "mood"]
    filled = sum(1 for f in core_fields if _content_len(getattr(parsed, f, None)) > 0)
    score += (filled / len(core_fields)) * 0.35

    # ── Part 2: 字段填充质量 (30%) ──
    # 各字段内容达到阈值即视为充分，lighting/camera 以嵌套对象总长度计
    THRESHOLDS = {
        "subject":     30,   # 主体描述足够详细
        "style":       10,   # 至少一两个风格词
        "lighting":    15,   # 嵌套对象拼接后至少 15 字符
        "camera":      15,   # 同上
        "background":  10,
        "mood":         5,
    }
    quality_scores = []
    for field, threshold in THRESHOLDS.items():
        clen = _content_len(getattr(parsed, field, None))
        quality_scores.append(min(clen / threshold, 1.0))
    score += (sum(quality_scores) / len(quality_scores)) * 0.30

    # ── Part 3: Constraints 覆盖率 (20%) ──
    # must_keep 和 avoid 是「精准维度控制」的核心字段
    # 各占 10%，有实质内容则得满分
    has_must_keep = _content_len(parsed.must_keep) > 3
    has_avoid     = _content_len(parsed.avoid) > 3
    score += 0.10 * (1.0 if has_must_keep else 0.0)
    score += 0.10 * (1.0 if has_avoid else 0.0)

    # ── Part 4: negative_prompt 质量 (15%) ──
    # 有实质内容（至少 3 个 token）则得满分
    neg_len = _content_len(parsed.negative_prompt)
    score += 0.15 * min(neg_len / 20, 1.0)   # 20 字符作为充分阈值

    return round(score, 4)


# ── LLM Judge ────────────────────────────────────────────────────────────────

class JSONQualityJudge(dspy.Signature):
    """你是专业的 AI 图像提示词质量评估专家。
    给定原始图像提示词和转换后的 image-json，评估转换质量并给出具体改进建议，
    用于优化 System Prompt。

    评估重点：
    1. subject 字段粒度是否足够（人物场景是否使用了嵌套的 face/hair/pose/attire）
    2. lighting / camera 是否精确描述了独立维度
    3. must_keep / avoid（constraints）是否有效捕捉了原 prompt 的关键要素
    4. negative_prompt 是否完整
    5. 是否存在信息丢失或维度混淆（把摄影参数写进 mood 等）
    """
    raw_prompt: str = dspy.InputField(desc="原始文生图 prompt")
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
        feedback = (
            f"JSON 解析/校验失败：{err}。"
            "请确保严格输出合法 JSON，不含任何额外文字或 markdown。"
            "subject 支持字符串或包含 face/hair/pose/attire 等字段的对象；"
            "lighting/camera 支持嵌套对象或字符串。"
        )
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

    with open(out / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump({"avg_score": avg, "scores": scores, "failures": failures}, f, ensure_ascii=False, indent=2)
    print("评估报告已保存至 outputs/eval_report.json")

    with open(out / "converted_results.json", "w", encoding="utf-8") as f:
        json.dump(converted_results, f, ensure_ascii=False, indent=2)
    print(f"转换结果已保存至 outputs/converted_results.json（共 {len(converted_results)} 条）")

    top_results = [r for r in converted_results if r.get("score", 0) >= 0.7]
    with open(out / "converted_top.json", "w", encoding="utf-8") as f:
        json.dump(top_results, f, ensure_ascii=False, indent=2)
    print(f"高分转换结果已保存至 outputs/converted_top.json（共 {len(top_results)} 条，score >= 0.7）")
