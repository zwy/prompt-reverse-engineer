# prompt-reverse-engineer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-experimental-orange)

> 逆向推理文生图 System Prompt：将 Civitai 的真实 prompt 数据集转化为结构化 image-json，基于 DSPy GEPA 自动优化 System Prompt。

## 背景 & 动机

手写文生图 System Prompt 依赖经验，难以量化优劣。本项目通过将 Civitai 社区真实高质量 prompt 作为黄金标准数据集，利用 DSPy GEPA 优化框架自动迭代 System Prompt，实现从"靠感觉调 prompt"到"数据驱动自动进化"的转变。

## 项目架构

```
Civitai Prompt 数据集
        ↓
   数据采集模块 (data/fetch_civitai.py)
        ↓
   数据筛选模块 (data/filter_prompts.py)
        ↓
  LLM 转换模块 (optimize_gepa.py)
        ↓
  双通道评估 (evaluate.py: 规则分 + LLM Judge)
        ↓
  GEPA 自动进化 / Meta-Prompting 手动反思
        ↓
  最优 System Prompt 导出 (extract_best_prompt.py → outputs/)
```

## 文件结构

```
prompt-reverse-engineer/
├── data/
│   ├── fetch_civitai.py        # Cursor 分页采集 Civitai 数据
│   ├── filter_prompts.py       # 规则筛选：去掉低质量数据
│   └── prompts.json            # 采集后的原始数据集（运行后生成，不入 git）
├── outputs/                    # 各版本 System Prompt 及评估报告（不入 git）
├── schema.py                   # Pydantic 强校验 image-json Schema
├── llm_client.py               # 统一 LLM 调用封装（OpenAI / OpenRouter）
├── evaluate.py                 # 双通道评分（规则分 + LLM Judge with Feedback）
├── optimize_gepa.py            # GEPA 自动优化主流程
├── meta_reflect.py             # 手动 meta-prompting 反思循环
├── extract_best_prompt.py      # 从 GEPA 历史中提取最优版本
├── requirements.txt            # 依赖声明（最低版本约束）
├── .env.example                # 环境变量模板
└── LICENSE
```

## 快速开始

### 1. 环境准备

```bash
# 推荐使用虚拟环境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 等
```

### 3. 采集数据

```bash
python data/fetch_civitai.py
```

> **注意**：需要 [Civitai API Key](https://civitai.com/user/account)，在 `.env` 中配置 `CIVITAI_API_KEY`。

### 4. 数据筛选

```bash
# 预览筛选结果，不写文件
python data/filter_prompts.py --dry-run

# 筛选并保存到 data/prompts_filtered.json
python data/filter_prompts.py

# 或直接覆盖原文件
python data/filter_prompts.py --inplace
```

筛选规则：
- 去掉 prompt 长度 < 50 的过短数据
- 去掉 prompt 里已包含结构化字段的（如 `"subject":`, `"style":` 等）
- 去掉 prompt 长度 > 5000 的过长数据

### 5. 基线评估

```bash
# 快速测前 5 条（最简单）
python evaluate.py -n 5

# 跳过前 20 条，测第 21~30 条
python evaluate.py -n 10 --offset 20

# 随机抽 10 条测试（seed=42 保证每次结果一样）
python evaluate.py -n 10 --shuffle

# 测试新版 System Prompt 效果
python evaluate.py -n 20 --sp outputs/system_prompt_v1.txt

# 查看帮助
python evaluate.py --help
```

### 6. GEPA 自动优化

```bash
python optimize_gepa.py
```

### 7. 或使用手动反思循环

```bash
python meta_reflect.py
```

### 8. 提取最优 Prompt

```bash
# 从 outputs/ 中的 GEPA 历史文件提取评分最高的版本
python extract_best_prompt.py
```

## 推荐执行顺序

1. 跑 `fetch_civitai.py` 采集 100 条 prompt
2. 跑 `filter_prompts.py --dry-run` 预览筛选结果，确认无误过滤
3. 跑 `filter_prompts.py` 生成干净数据集
4. 手写初始 `outputs/system_prompt_v0.txt`
5. 用 `evaluate.py` 跑全量，找出评分 < 0.5 的失败案例
6. 把失败案例喂给 `meta_reflect.py`，得到 v1
7. 以 v1 为起点，用 GEPA 自动进化 10 轮
8. 用 `extract_best_prompt.py` 提取最终最优版本作为生产用 System Prompt

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

## 费用说明

GEPA 优化每轮需调用 LLM 多次（评估 + 进化），100 条数据跑 10 轮约消耗 **$1~3 USD**（视所用模型而定）。建议先用少量数据（10~20 条）测试流程。

## 参考资源

- [DSPy GEPA Optimizer](https://dspy.ai/api/optimizers/GEPA/overview/)
- [CivitAI Prompt Scraper](https://github.com/jeffjbowie/CivitAI_Prompt_Scraper)
- [DSPy GEPA HuggingFace Cookbook](https://huggingface.co/learn/cookbook/dspy_gepa)

## License

[MIT](./LICENSE)
