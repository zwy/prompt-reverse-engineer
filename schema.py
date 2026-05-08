"""image-json Schema 定义与 Pydantic 强校验"""
import json
from typing import List, Optional
from pydantic import BaseModel, field_validator


class ImageJSON(BaseModel):
    """结构化 image-json，用于描述文生图提示词的各个维度。"""
    subject: str                          # 主体：人物/物体/场景
    style: List[str] = []                 # 画风列表，如 ["anime", "watercolor"]
    lighting: Optional[str] = None       # 光线描述
    environment: Optional[str] = None    # 背景/场景
    camera: Optional[str] = None         # 镜头/构图参数
    quality_tags: List[str] = []          # masterpiece / best quality 等
    lora_tags: List[str] = []             # <lora:xxx:0.8> 类标签（单独提取）
    artist_reference: List[str] = []      # 艺术家风格参考
    mood: Optional[str] = None           # 氛围/情绪
    negative_prompt: Optional[str] = None  # 负面提示词

    @field_validator("subject")
    @classmethod
    def subject_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("subject 不能为空")
        return v.strip()

    @field_validator("style", "quality_tags", "lora_tags", "artist_reference", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        """兼容 LLM 输出字符串的情况，自动转为列表"""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


EXAMPLE_OUTPUT = ImageJSON(
    subject="1girl, white dress, long hair",
    style=["anime", "watercolor", "illustration"],
    lighting="soft natural light, rim light",
    environment="flower field, golden hour, outdoor",
    camera="medium shot, shallow depth of field",
    quality_tags=["masterpiece", "best quality", "ultra-detailed"],
    lora_tags=["<lora:detail_tweaker:0.8>"],
    artist_reference=["artgerm", "wlop"],
    mood="peaceful, dreamy",
    negative_prompt="blurry, bad anatomy, lowres, watermark"
)


def parse_and_validate(json_str: str) -> tuple:
    """
    解析并校验 LLM 输出的 JSON 字符串。
    
    Returns:
        (ImageJSON 对象 | None, 错误信息字符串)
    """
    # 尝试提取 JSON 块（LLM 可能包裹在 markdown code block 里）
    if "```" in json_str:
        import re
        match = re.search(r"```(?:json)?\s*({.*?})\s*```", json_str, re.DOTALL)
        if match:
            json_str = match.group(1)

    try:
        raw = json.loads(json_str)
        obj = ImageJSON(**raw)
        return obj, ""
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败: {e}"
    except Exception as e:
        return None, f"Schema 校验失败: {e}"


if __name__ == "__main__":
    print("示例 image-json:")
    print(EXAMPLE_OUTPUT.model_dump_json(indent=2))
