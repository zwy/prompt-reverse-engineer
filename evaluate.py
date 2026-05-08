"""
双通道评估模块：规则评分 + LLM Judge（带 Feedback）
支持通过 LLMClient 切换 provider

评分说明（rule_score）v3：
  - 核心字段覆盖率  (35%)：subject / style / lighting / camera / background / mood
  - 字段填充质量    (25%)：各字段内容长度充分性
    + 结构质量加成  ( 5%)：lighting/camera 使用嵌套对象时额外给分
  - Constraints 覆盖 (20%)：must_keep + avoid
  - 负面提示词质量  (15%)：negative_prompt 列表是否有实质内容

用法示例：
  python evaluate.py                      # 全量评估
  python evaluate.py -n 10                # 只测前 10 条
  python evaluate.py -n 10 --offset 20   # 跳过前 20 条，测第 21~30 条
  python evaluate.py -n 10 --shuffle     # 随机抽 10 条测试
"""
import json
import argparse
import random
import dspy
from schema import parse_and_validate, ImageJSON, LightingDetail, CameraDetail
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

    v3 变更：
    1. lighting/camera 兼容字符串：有内容就给基础分，嵌套对象额外给结构质量加成
    2. 总权重：35% + 30%(=25%质量+5%结构加成) + 20% + 15% = 100%
    """
    score = 0.0

    # ── Part 1: 核心字段覆盖率 (35%) ──
    core_fields = ["subject", "style", "lighting", "camera", "background", "mood"]
    filled = sum(1 for f in core_fields if _content_len(getattr(parsed, f, None)) > 0)
    score += (filled / len(core_fields)) * 0.35

    # ── Part 2: 字段填充质量 (25%) + 结构质量加成 (5%) ──
    THRESHOLDS = {
        "subject":     30,
        "style":       10,
        "lighting":    10,   # 字符串模式阈值放宽到 10（嵌套模式天然更长）
        "camera":      10,
        "background":  10,
        "mood":         5,
    }
    quality_scores = []
    for field, threshold in THRESHOLDS.items():
        clen = _content_len(getattr(parsed, field, None))
        quality_scores.append(min(clen / threshold, 1.0))
    score += (sum(quality_scores) / len(quality_scores)) * 0.25

    # 结构质量加成：lighting/camera 使用嵌套对象时各给 2.5%
    if parsed.lighting_is_nested():
        score += 0.025
    if parsed.camera_is_nested():
        score += 0.025

    # ── Part 3: Constraints 覆盖率 (20%) ──
    has_must_keep = _content_len(parsed.must_keep) > 3
    has_avoid     = _content_len(parsed.avoid) > 3
    score += 0.10 * (1.0 if has_must_keep else 0.0)
    score += 0.10 * (1.0 if has_avoid else 0.0)

    # ── Part 4: negative_prompt 质量 (15%) ──
    neg_len = _content_len(parsed.negative_prompt)
    score += 0.15 * min(neg_len / 20, 1.0)

    return round(score, 4)


# ── LLM Judge ────────────────────────────────────────────────────────────────

class JSONQualityJudge(dspy.Signature):
    """你是专业的 AI 图像提示词质量评估专家。
    给定原始图像提示词和转换后的 image-json，评估转换质量并给出具体改进建议，
    用于优化 System Prompt。

    评估重点：
    1. subject 字段粒度是否足够（人物场景是否使用了嵌套的 face/hair/pose/attire）
    2. lighting / camera 是否精确描述了独立维度（嵌套对象优于字符串）
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
            "lighting/camera 可以是字符串（兼容模式）或嵌套对象（精细模式，推荐）。"
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


# ── CLI 参数解析 ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="批量评估 image-json 转换质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python evaluate.py                      全量评估
  python evaluate.py -n 10                只测前 10 条
  python evaluate.py -n 10 --offset 20   跳过前 20 条，测第 21~30 条
  python evaluate.py -n 10 --shuffle     随机抽 10 条测试
  python evaluate.py --sp outputs/system_prompt_v1.txt  指定 System Prompt 文件
        """,
    )
    p.add_argument("-n", "--limit",
                   type=int, default=None, metavar="N",
                   help="只评估 N 条数据（默认全部）")
    p.add_argument("--offset",
                   type=int, default=0, metavar="K",
                   help="跳过前 K 条（先 offset 再 limit，不与 --shuffle 同用）")
    p.add_argument("--shuffle",
                   action="store_true",
                   help="随机打乱后再取前 N 条（需配合 -n 使用）")
    p.add_argument("--seed",
                   type=int, default=42,
                   help="shuffle 时的随机种子（默认 42，保证可复现）")
    p.add_argument("--sp", "--system-prompt",
                   dest="sp_path",
                   default="outputs/system_prompt_v0.txt",
                   metavar="FILE",
                   help="指定 System Prompt 文件路径")
    p.add_argument("--data",
                   dest="data_path",
                   default=None, metavar="FILE",
                   help="指定数据文件路径（默认自动查找 prompts_filtered.json / prompts.json）")
    return p.parse_args()


# ── 批量评估入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from optimize_gepa import build_dspy_lm

    args = _parse_args()

    llm = get_llm()
    dspy_lm = build_dspy_lm(llm)
    dspy.configure(lm=dspy_lm)

    # ── 加载数据 ──
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = Path("data/prompts_filtered.json")
        if not data_path.exists():
            data_path = Path("data/prompts.json")
    if not data_path.exists():
        print(f"数据文件不存在：{data_path}\n请先运行 data/fetch_civitai.py 采集数据")
        exit(1)

    with open(data_path, encoding="utf-8") as f:
        dataset = json.load(f)

    # ── 按参数切片 ──
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(dataset)
    if args.offset:
        dataset = dataset[args.offset:]
    if args.limit is not None:
        if args.limit <= 0:
            print("--limit 必须是正整数")
            exit(1)
        dataset = dataset[:args.limit]

    total = len(dataset)
    tag = ""
    if args.shuffle:  tag += f" shuffle(seed={args.seed})"
    if args.offset:   tag += f" offset={args.offset}"
    if args.limit:    tag += f" limit={args.limit}"
    print(f"评估数据集：{data_path}  共 {total} 条{tag}")

    # ── 加载 System Prompt ──
    sp_path = Path(args.sp_path)
    if not sp_path.exists():
        print(f"System Prompt 文件不存在：{sp_path}")
        exit(1)
    system_prompt = sp_path.read_text(encoding="utf-8")
    print(f"System Prompt：{sp_path}")

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
                "lighting_nested": parsed.lighting_is_nested(),
                "camera_nested":   parsed.camera_is_nested(),
                "raw_prompt":  raw,
                "image_json":  parsed.model_dump(),
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
            print(f"[{i+1}/{total}] 当前均分: {sum(scores)/len(scores):.3f}")

    avg = sum(scores) / len(scores)
    nested_light = sum(1 for r in converted_results if r.get("lighting_nested"))
    nested_cam   = sum(1 for r in converted_results if r.get("camera_nested"))
    print(f"\n最终平均分: {avg:.3f}")
    print(f"失败案例数: {len(failures)}")
    print(f"lighting 嵌套率: {nested_light}/{total}  camera 嵌套率: {nested_cam}/{total}")

    out = Path("outputs")
    out.mkdir(exist_ok=True)

    with open(out / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "avg_score": avg,
            "scores": scores,
            "lighting_nested_count": nested_light,
            "camera_nested_count":   nested_cam,
            "failures": failures,
        }, f, ensure_ascii=False, indent=2)
    print("评估报告已保存至 outputs/eval_report.json")

    with open(out / "converted_results.json", "w", encoding="utf-8") as f:
        json.dump(converted_results, f, ensure_ascii=False, indent=2)
    print(f"转换结果已保存至 outputs/converted_results.json（共 {len(converted_results)} 条）")

    top_results = [r for r in converted_results if r.get("score", 0) >= 0.7]
    with open(out / "converted_top.json", "w", encoding="utf-8") as f:
        json.dump(top_results, f, ensure_ascii=False, indent=2)
    print(f"高分转换结果已保存至 outputs/converted_top.json（共 {len(top_results)} 条，score >= 0.7）")
