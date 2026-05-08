"""image-json Schema 定义与 Pydantic 强校验

设计原则：JSON prompt 的核心价值不是「把 tag 结构化」，
而是「让 LLM 能精准修改单个维度、同时保持其他维度稳定」。
因此字段粒度越细、越独立，可控性越高。

主要变更（相比 v1）：
  - 移除 SD 专属字段 quality_tags / lora_tags（进入可选的 SDExtras 子块）
  - subject 支持 str（通用）或 PersonSubject（人物细粒度控制）
  - lighting / camera 支持嵌套对象（细粒度）或字符串（兼容模式）
    - 嵌套对象可逐子字段修改，评分时给质量加成
    - 字符串兼容老格式/LLM 简短输出，不报错，评分时给基础分
  - 新增 must_keep / avoid（constraints），是「保持维度稳定」的核心字段
  - negative_prompt 改为 List[str]，方便逐条追加/删除
"""
import json
import re
from typing import List, Optional, Union
from pydantic import BaseModel, field_validator, model_validator


# ── 子结构：人物主体（仅含人物时使用） ────────────────────────────────────────

class PersonSubject(BaseModel):
    """人物主体的细粒度描述，每个维度独立、互不干扰。"""
    demographics: Optional[str] = None   # 年龄/性别/种族，如 "early 20s female, East Asian"
    face:         Optional[str] = None   # 面部特征，如 "sharp jawline, large almond eyes"
    hair:         Optional[str] = None   # 发型/发色，如 "long silver hair, twin tails"
    pose:         Optional[str] = None   # 姿势/动作，如 "sitting cross-legged, looking away"
    attire:       Optional[str] = None   # 服装/配饰，如 "white qipao, gold earrings"
    body:         Optional[str] = None   # 身材/体型（可选）
    expression:   Optional[str] = None   # 表情/神态，如 "soft smile, contemplative"


# ── 子结构：光线（摄影级精度） ────────────────────────────────────────────────

class LightingDetail(BaseModel):
    """光线的三个独立维度：类型 / 光源 / 补充细节。"""
    type:    Optional[str] = None   # "soft diffuse", "volumetric", "hard rim"
    source:  Optional[str] = None   # "golden hour sun", "large softbox", "candle"
    details: Optional[str] = None   # 额外修饰，如 "warm orange tones, long shadows"


# ── 子结构：镜头/构图 ─────────────────────────────────────────────────────────

class CameraDetail(BaseModel):
    """摄影参数的独立维度，修改其中一项不影响其他项。"""
    shot_scale: Optional[str] = None   # "medium close-up", "full body", "extreme wide"
    lens:       Optional[str] = None   # "85mm portrait", "24mm wide angle", "macro"
    aperture:   Optional[str] = None   # "f/1.8 bokeh", "f/8 deep focus"
    angle:      Optional[str] = None   # "low angle", "bird's eye", "dutch tilt"
    focus:      Optional[str] = None   # "tack sharp on eyes, soft shoulder fall-off"


# ── 可选：SD/ComfyUI 专属扩展块（通用流程不使用）─────────────────────────────

class SDExtras(BaseModel):
    """仅在目标平台为 Stable Diffusion / ComfyUI 时填写。"""
    quality_tags: List[str] = []   # "masterpiece", "best quality", "ultra-detailed"
    lora_tags:    List[str] = []   # "<lora:detail_tweaker:0.8>" 格式


# ── 主结构 ────────────────────────────────────────────────────────────────────

class ImageJSON(BaseModel):
    """
    结构化 image-json：每个字段对应一个独立的视觉控制维度。
    修改任意一个字段，其余字段保持不变，即可实现精准的单维度编辑。

    lighting / camera 兼容策略：
      - 优先：LightingDetail / CameraDetail 嵌套对象（评分时给质量加成）
      - 兼容：普通字符串（不报错，evaluate 时给基础分，system prompt 会引导逐步升级）
    """

    # ── 主体 ──
    # 通用场景用字符串；人物场景用 PersonSubject 获得逐字段控制
    subject: Union[str, PersonSubject]

    # ── 场景 / 背景 ──
    background:  Optional[str] = None
    environment: Optional[str] = None
    time_of_day: Optional[str] = None

    # ── 视觉风格 ──
    style:            List[str] = []
    artist_reference: List[str] = []
    mood:             Optional[str] = None
    color_palette:    Optional[str] = None

    # ── 摄影参数 ──
    # 支持嵌套对象（精细控制）或字符串（兼容模式）
    # Union 顺序：先尝试 dict→嵌套对象，若 LLM 输出字符串则保留为 str
    lighting: Optional[Union[LightingDetail, str]] = None
    camera:   Optional[Union[CameraDetail,   str]] = None

    # ── Constraints（核心）──
    must_keep: List[str] = []
    avoid:     List[str] = []

    # ── 负面提示词 ──
    negative_prompt: List[str] = []

    # ── SD 专属扩展（非 SD 流程留空）──
    sd_extras: Optional[SDExtras] = None

    # ── 兼容旧格式：quality_tags / lora_tags 直接出现在顶层时自动迁移 ──
    quality_tags: Optional[List[str]] = None
    lora_tags:    Optional[List[str]] = None

    @model_validator(mode="after")
    def _migrate_sd_extras(self):
        """把顶层 quality_tags / lora_tags 自动迁移到 sd_extras，保持向后兼容。"""
        qt = self.quality_tags or []
        lt = self.lora_tags or []
        if qt or lt:
            if self.sd_extras is None:
                self.sd_extras = SDExtras(quality_tags=qt, lora_tags=lt)
            else:
                self.sd_extras.quality_tags = list(set(self.sd_extras.quality_tags + qt))
                self.sd_extras.lora_tags    = list(set(self.sd_extras.lora_tags    + lt))
            self.quality_tags = None
            self.lora_tags    = None
        return self

    @field_validator("subject", mode="before")
    @classmethod
    def subject_not_empty(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("subject 不能为空")
        return v

    @field_validator("style", "artist_reference", "must_keep", "avoid",
                     "negative_prompt", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        """兼容 LLM 输出字符串的情况，自动转为列表。"""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    def lighting_is_nested(self) -> bool:
        """True 表示 lighting 使用了嵌套对象（精细结构），False 表示字符串兼容模式。"""
        return isinstance(self.lighting, LightingDetail)

    def camera_is_nested(self) -> bool:
        """True 表示 camera 使用了嵌套对象（精细结构），False 表示字符串兼容模式。"""
        return isinstance(self.camera, CameraDetail)


# ── 示例输出（人物场景）────────────────────────────────────────────────────────

EXAMPLE_OUTPUT = ImageJSON(
    subject=PersonSubject(
        demographics="early 20s female",
        face="large expressive eyes, soft features",
        hair="long silver hair, loose",
        pose="sitting on windowsill, knees drawn up",
        attire="oversized white linen shirt, bare feet",
        expression="wistful, gazing at the rain",
    ),
    background="rain-streaked glass window, blurred city lights",
    environment="cozy indoor attic room, warm lamplight",
    time_of_day="late evening",
    style=["anime illustration", "painterly", "soft linework"],
    artist_reference=["ilya kuvshinov", "krenz cushart"],
    mood="melancholic, intimate",
    color_palette="muted blues and warm amber accents",
    lighting=LightingDetail(
        type="soft diffuse",
        source="single table lamp behind subject",
        details="warm orange rim light, cool blue fill from window",
    ),
    camera=CameraDetail(
        shot_scale="medium close-up",
        lens="85mm portrait",
        aperture="f/2.0 shallow depth of field",
        angle="eye-level, slight side angle",
        focus="sharp on face, soft fall-off on background",
    ),
    must_keep=["silver hair", "white shirt", "rain on window"],
    avoid=["extra limbs", "text", "watermark", "bad anatomy"],
    negative_prompt=["blurry", "lowres", "bad anatomy", "watermark", "jpeg artifacts"],
)


# ── 示例输出（非人物/场景）────────────────────────────────────────────────────

EXAMPLE_LANDSCAPE = ImageJSON(
    subject="ancient stone temple ruins overtaken by jungle vines",
    background="dense tropical rainforest",
    environment="humid, misty, shafts of sunlight through canopy",
    time_of_day="early morning golden hour",
    style=["concept art", "matte painting", "hyper-detailed"],
    artist_reference=["craig mullins", "james gurney"],
    mood="mysterious, awe-inspiring",
    color_palette="deep greens, warm gold highlights, soft mist blue",
    lighting=LightingDetail(
        type="volumetric",
        source="sunbeams breaking through jungle canopy",
        details="god rays, dappled light on stone",
    ),
    camera=CameraDetail(
        shot_scale="wide establishing shot",
        lens="24mm wide angle",
        aperture="f/8 deep focus",
        angle="low angle looking up at ruins",
    ),
    must_keep=["temple ruins", "jungle overgrowth", "god rays"],
    avoid=["people", "modern objects", "text"],
    negative_prompt=["blurry", "oversaturated", "cartoon", "flat lighting"],
)


def parse_and_validate(json_str: str) -> tuple:
    """
    解析并校验 LLM 输出的 JSON 字符串。

    Returns:
        (ImageJSON 对象 | None, 错误信息字符串)
    """
    # 尝试提取 JSON 块（LLM 可能包裹在 markdown code block 里）
    if "```" in json_str:
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
    print("=== 示例 image-json（人物）===")
    print(EXAMPLE_OUTPUT.model_dump_json(indent=2))
    print("\n=== 示例 image-json（场景）===")
    print(EXAMPLE_LANDSCAPE.model_dump_json(indent=2))

    # 验证字符串兼容模式不报错
    test_str = ImageJSON(
        subject="a girl",
        lighting="soft natural light",
        camera="85mm close-up shot",
    )
    print("\n=== 字符串兼容模式 ===")
    print(test_str.model_dump_json(indent=2))
    print(f"lighting_is_nested: {test_str.lighting_is_nested()}")
    print(f"camera_is_nested:   {test_str.camera_is_nested()}")
