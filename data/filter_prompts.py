"""
规则筛选模块 - 过滤不适合训练的 prompt 数据

筛选规则：
  1. 去掉长度 < 50 的过短 prompt
  2. 去掉 prompt 里已有 JSON 结构的（如已包含 "subject":, "style": 等字段）
  3. 去掉过长（> 5000 字符）的 prompt

用法：
    python data/filter_prompts.py
    python data/filter_prompts.py --input data/prompts.json --output data/prompts_filtered.json
    python data/filter_prompts.py --dry-run     # 只显示统计，不写文件
"""
import json
import argparse
from pathlib import Path


# 已结构化字段特征：如果 prompt 里出现这些 key，说明它本身已经是结构化的，不适合作为训练样本
JSON_STRUCTURE_SIGNALS = [
    '"subject"',
    '"style"',
    '"lighting"',
    '"environment"',
    '"camera"',
    '"quality_tags"',
    '"lora_tags"',
    '"artist_reference"',
    '"mood"',
]


def is_too_short(prompt: str, min_len: int = 50) -> bool:
    """prompt 过短，无法提供足够训练信息"""
    return len(prompt.strip()) < min_len


def is_already_structured(prompt: str) -> bool:
    """prompt 里已包含结构化字段（即目标任务的输出格式），不适合作为训练输入"""
    # 至少命中 2 个字段才判定为已结构化，避免误判
    hits = sum(1 for signal in JSON_STRUCTURE_SIGNALS if signal in prompt)
    return hits >= 2


def is_too_long(prompt: str, max_len: int = 5000) -> bool:
    """prompt 过长，可能是的异常数据或嵌入了额外内容"""
    return len(prompt) > max_len


def filter_record(record: dict) -> tuple[bool, str]:
    """
    对单条记录进行筛选判断。

    Returns:
        (keep: bool, reason: str)  reason 在 keep=False 时说明被过滤的原因
    """
    prompt = record.get("prompt", "")

    if is_too_short(prompt):
        return False, f"too_short (len={len(prompt)})"

    if is_already_structured(prompt):
        return False, "already_structured"

    if is_too_long(prompt):
        return False, f"too_long (len={len(prompt)})"

    return True, ""


def run_filter(
    input_path: Path,
    output_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """
    执行筛选并输出结果。

    Returns:
        统计信息字典
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    kept = []
    removed = []
    reason_counts: dict[str, int] = {}

    for record in data:
        keep, reason = filter_record(record)
        if keep:
            kept.append(record)
        else:
            removed.append({"id": record.get("id"), "reason": reason,
                            "prompt_preview": record.get("prompt", "")[:80]})
            # 按原因分类计数
            category = reason.split(" ")[0]  # e.g. "too_short", "already_structured", "too_long"
            reason_counts[category] = reason_counts.get(category, 0) + 1

    stats = {
        "total": len(data),
        "kept": len(kept),
        "removed": len(removed),
        "removal_rate": f"{len(removed) / len(data) * 100:.1f}%",
        "by_reason": reason_counts,
    }

    print(f"输入: {len(data)} 条")
    print(f"保留: {len(kept)} 条")
    print(f"过滤: {len(removed)} 条 ({stats['removal_rate']})")
    print(f"按原因分类: {reason_counts}")

    if verbose and removed:
        print("\n被过滤的记录：")
        for r in removed:
            print(f"  [{r['reason']}] id={r['id']} | {r['prompt_preview']!r}")

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        print(f"\n已保存筛选后数据至 {output_path}")
    else:
        print("\n[dry-run 模式]，未写入文件")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Civitai Prompt 数据筛选工具")
    parser.add_argument(
        "--input", "-i",
        default="data/prompts.json",
        help="输入文件路径（默认: data/prompts.json）"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/prompts_filtered.json",
        help="输出文件路径（默认: data/prompts_filtered.json）"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="直接覆盖 prompts.json（谨慎使用）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出统计，不写入文件"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出被过滤记录的详情"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：找不到输入文件 {input_path}")
        exit(1)

    output_path = Path(args.input) if args.inplace else Path(args.output)

    if args.inplace and not args.dry_run:
        confirm = input(f"确认覆盖 {input_path}？ (y/n): ").strip().lower()
        if confirm != "y":
            print("已取消")
            exit(0)

    run_filter(
        input_path=input_path,
        output_path=output_path,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
