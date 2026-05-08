"""
从 GEPA 保存的 optimized_module.json 中直接提取最优 System Prompt。

用法：
    python extract_best_prompt.py
    python extract_best_prompt.py --input outputs/optimized_module.json --output outputs/system_prompt_best.txt
"""
import argparse
import json
from pathlib import Path


def extract_prompt(data: dict) -> str | None:
    """
    尝试从 DSPy 保存的不同格式中找到 instructions。

    DSPy 保存格式可能的大致结构：

    方案 A（新版 dspy.Module.save）:
    {
      "predict": {
        "signature": { "instructions": "...", ... },
        ...
      }
    }

    方案 B（ChainOfThought 包装了一层 predict）:
    {
      "parser": {
        "predict": {
          "signature": { "instructions": "..." }
        }
      }
    }

    方案 C（常见的 flat 格式）:
    {
      "parser.predict.signature.instructions": "..."
    }

    方案 D（predictors 列表）:
    {
      "predictors": [
        { "signature": { "instructions": "..." } }
      ]
    }
    """

    # 方案 C：flat key 形式
    for key, val in data.items():
        if "instructions" in key and isinstance(val, str) and val.strip():
            return val.strip()

    # 递归搜索所有嵌套结构中的 instructions
    def search(obj, depth=0):
        if depth > 10:
            return None
        if isinstance(obj, dict):
            # 直接命中
            if "instructions" in obj and isinstance(obj["instructions"], str) and obj["instructions"].strip():
                return obj["instructions"].strip()
            # 先找 signature 子对象
            if "signature" in obj:
                result = search(obj["signature"], depth + 1)
                if result:
                    return result
            # 再遍历其就其它 key
            for v in obj.values():
                result = search(v, depth + 1)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = search(item, depth + 1)
                if result:
                    return result
        return None

    return search(data)


def main():
    parser = argparse.ArgumentParser(description="从 optimized_module.json 提取最优 System Prompt")
    parser.add_argument(
        "--input", "-i",
        default="outputs/optimized_module.json",
        help="输入 JSON 文件路径（默认: outputs/optimized_module.json）"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/system_prompt_best.txt",
        help="输出文本文件路径（默认: outputs/system_prompt_best.txt）"
    )
    parser.add_argument(
        "--print-json", action="store_true",
        help="打印 JSON 顶层 key 结构，帮助调试"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：找不到文件 {input_path}")
        return

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if args.print_json:
        print("=== JSON 顶层结构 ===")
        def show_keys(obj, prefix="", depth=0):
            if depth > 4:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    full = f"{prefix}.{k}" if prefix else k
                    vtype = type(v).__name__
                    vpreview = str(v)[:60].replace("\n", " ") if not isinstance(v, (dict, list)) else ""
                    print(f"  {full} ({vtype}) {vpreview}")
                    show_keys(v, full, depth + 1)
            elif isinstance(obj, list) and obj:
                show_keys(obj[0], f"{prefix}[0]", depth + 1)
        show_keys(data)
        print()

    prompt = extract_prompt(data)

    if not prompt:
        print("未能自动提取到 instructions，请使用 --print-json 查看 JSON 结构手动定位。")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")

    print(f"=== 最优 System Prompt ===")
    print(prompt)
    print(f"\n已保存至 {output_path}")


if __name__ == "__main__":
    main()
