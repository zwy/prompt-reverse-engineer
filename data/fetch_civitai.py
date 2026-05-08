"""Civitai 数据采集模块 - 使用 Cursor 分页抓取图片 prompt"""
import requests
import json
import time
from pathlib import Path


def fetch_civitai_prompts(target: int = 100, nsfw: bool = False) -> list[dict]:
    """
    使用 Cursor 分页从 Civitai API 采集图片 prompt。
    
    Args:
        target: 目标采集数量
        nsfw: 是否包含 NSFW 内容
    
    Returns:
        包含 prompt 及元数据的列表
    """
    url = "https://civitai.com/api/v1/images"
    params = {
        "limit": 20,
        "sort": "Most Reactions",
        "nsfw": str(nsfw).lower(),
    }
    results = []
    cursor = None

    print(f"开始采集，目标：{target} 条...")

    while len(results) < target:
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(url, params=params, timeout=15)
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


if __name__ == "__main__":
    out_path = Path(__file__).parent / "prompts.json"
    prompts = fetch_civitai_prompts(target=100)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"数据已保存至 {out_path}")
