"""Civitai 数据采集模块 - 使用 Cursor 分页抓取图片 prompt"""
import requests
import json
import time
from pathlib import Path


def fetch_civitai_prompts(target: int = 100, nsfw: bool = False, base_model: str = "") -> list[dict]:
    """
    使用 Cursor 分页从 Civitai API 采集图片 prompt.

    Args:
        target: 目标采集数量
        nsfw: 是否包含 NSFW 内容
        base_model: 模型筛选 (如 ZImageTurbo / ZImageBase / Flux.1 D 等)

    Returns:
        包含 prompt 及元数据的列表
    """
    url = "https://civitai.red/api/v1/images"

    session = requests.Session()
    session.trust_env = False  # 禁用代理，避免 SSL 问题
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    })
    params = {
        "limit": 20,
        "sort": "Most Reactions",
        "nsfw": str(nsfw).lower(),
    }
    if base_model:
        params["baseModels"] = base_model
    results = []
    cursor = None

    print(f"开始采集，目标：{target} 条...")

    while len(results) < target:
        if cursor:
            params["cursor"] = cursor

        try:
            resp = session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"请求失败: {e}，3秒后重试...")
            time.sleep(3)
            continue

        for item in data.get("items", []):
            meta = item.get("meta") or {}
            prompt = meta.get("prompt", "").strip()
            # 过滤过短或无意义的 prompt
            if not prompt or len(prompt) < 20:
                continue
            results.append({
                "id": item.get("id"),
                "prompt": prompt,
                "negative_prompt": meta.get("negativePrompt", ""),
                "model": meta.get("Model", ""),
                "steps": meta.get("steps"),
                "cfg_scale": meta.get("cfgScale"),
                "sampler": meta.get("sampler"),
                "image_url": item.get("url", ""),
            })

        print(f"已采集: {len(results)} 条")

        # Cursor 翻页
        cursor = data.get("metadata", {}).get("nextCursor")
        if not cursor:
            print("已到达最后一页")
            break

        time.sleep(0.5)  # 礼貌爬取，避免被限速

    final = results[:target]
    print(f"采集完成，共 {len(final)} 条")
    return final


def merge_prompts(existing: list[dict], new: list[dict]) -> list[dict]:
    """合并新旧数据，按 id 去重"""
    existing_ids = {p["id"] for p in existing}
    merged = existing + [p for p in new if p["id"] not in existing_ids]
    return merged


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Civitai Prompt 采集器")
    parser.add_argument("--target", "-n", type=int, default=100, help="目标数量")
    parser.add_argument("--nsfw", action="store_true", help="包含 NSFW 内容")
    parser.add_argument("--base-model", "-m", type=str, default="", help="模型筛选 (如 ZImageTurbo / Flux.1 D 等)")
    args = parser.parse_args()

    out_path = Path(__file__).parent / "prompts.json"

    # 加载已有数据，避免覆盖
    existing = []
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"已加载历史数据: {len(existing)} 条")

    prompts = fetch_civitai_prompts(
        target=args.target,
        nsfw=args.nsfw,
        base_model=args.base_model,
    )

    # 合并去重
    merged = merge_prompts(existing, prompts)
    added = len(merged) - len(existing)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"新增 {added} 条，合并后共 {len(merged)} 条，已保存至 {out_path}")
