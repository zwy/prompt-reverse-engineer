# prompt-reverse-engineer

> 逆向推理文生图 System Prompt：将 Civitai 的真实 prompt 数据集转化为结构化 image-json，基于 DSPy GEPA 自动优化 System Prompt。

## 项目架构

```
Civitai Prompt 数据集 (100条)
        ↓
   数据采集模块 (data/fetch_civitai.py)
        ↓
  LLM 转换模块 (optimize_gepa.py)
        ↓
  双通道评估 (evaluate.py: 规则分 + LLM Judge)
        ↓
  GEPA 自动进化 / Meta-Prompting 手动反思
        ↓
  最优 System Prompt 导出 (outputs/)
```

## 文件结构

```
prompt-reverse-engineer/
├── data/
│   ├── fetch_civitai.py        # Cursor 分页采集 Civitai 数据
│   └── prompts.json            # 采集后的原始数据集（运行后生成）
├── schema.py                   # Pydantic 强校验 image-json Schema
├── evaluate.py                 # 双通道评分（规则分 + LLM Judge with Feedback）
├── optimize_gepa.py            # GEPA 自动优化主流程
├── meta_reflect.py             # 手动 meta-prompting 反思循环
├── outputs/                    # 各版本 System Prompt 及评估报告
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 采集数据
python data/fetch_civitai.py

# 2. 基线评估（需先设置 OPENAI_API_KEY）
export OPENAI_API_KEY=your_key
python evaluate.py

# 3. GEPA 自动优化
python optimize_gepa.py

# 4. 或使用手动反思循环
python meta_reflect.py
```

## 推荐执行顺序

1. 跑 `fetch_civitai.py` 采集 100 条 prompt，人工抽查 10 条确认质量
2. 手写初始 `outputs/system_prompt_v0.txt`
3. 用 `evaluate.py` 跑全量，找出评分 < 0.5 的失败案例
4. 把失败案例喂给 `meta_reflect.py`，得到 v1
5. 以 v1 为起点，用 GEPA 自动进化 10 轮
6. 提取最终 `sig.instructions` 作为生产用 System Prompt

## image-json 输出格式

```json
{
  "subject": "1girl, white dress",
  "style": ["anime", "watercolor"],
  "lighting": "soft natural light, rim light",
  "environment": "flower field, golden hour",
  "camera": "medium shot, shallow depth of field",
  "quality_tags": ["masterpiece", "best quality"],
  "lora_tags": ["<lora:detail_tweaker:0.8>"],
  "artist_reference": ["artgerm", "wlop"],
  "mood": "peaceful, dreamy",
  "negative_prompt": "blurry, bad anatomy"
}
```

## 参考资源

- [DSPy GEPA Optimizer](https://dspy.ai/api/optimizers/GEPA/overview/)
- [CivitAI Prompt Scraper](https://github.com/jeffjbowie/CivitAI_Prompt_Scraper)
- [DSPy GEPA HuggingFace Cookbook](https://huggingface.co/learn/cookbook/dspy_gepa)
