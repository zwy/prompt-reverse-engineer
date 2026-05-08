# 扩展性重构方案：任务配置化架构

> **文档状态**：待实施  
> **创建时间**：2026-05  
> **背景**：当前项目强耦合于 `image_json` 单一任务，扩展新任务需改动 4 个核心文件，几乎等同重写。本文档记录重构思路，供下次实施参考。

---

## 一、当前问题：任务定义散落在 4 个文件

如果要训练第二个任务（如"将用户描述转为分镜脚本"、"将商品描述转为 SEO 结构"），需要同时修改：

| 文件 | 强耦合内容 | 改动规模 |
|------|-----------|----------|
| `optimize_gepa.py` | `PromptToImageJSON` Signature、`PromptParser` Module、数据加载格式 | ~40 行 |
| `evaluate.py` | `rule_score()` 完全针对 `ImageJSON`、LLM Judge prompt、`metric_with_feedback` 调用 `parse_and_validate` | ~50 行 |
| `schema.py` | `ImageJSON` Pydantic 模型、`parse_and_validate` 函数 | ~30 行（新建模型）|
| `data/fetch_civitai.py` | 数据来源和字段结构完全绑定 Civitai | 全新文件 |

**结论**：添加新任务 ≈ 重写整个项目，只复用了 `llm_client.py` 和 GEPA 调用的壳。

---

## 二、目标架构：任务插件化

### 目录结构

```
prompt-reverse-engineer/
├── tasks/
│   ├── __init__.py             # 任务注册表：task_name → TaskClass
│   ├── base.py                 # 抽象基类，定义 Task 接口
│   └── image_json/             # 当前任务迁入此处
│       ├── __init__.py
│       ├── schema.py           # ImageJSON Pydantic 模型（从根目录移入）
│       ├── signature.py        # PromptToImageJSON DSPy Signature
│       ├── metric.py           # rule_score + metric_with_feedback
│       └── dataset.py          # 从 prompts.json 加载数据的逻辑
├── optimize_gepa.py            # 主流程，通过 --task 参数加载任务
├── evaluate.py                 # 评估框架，metric 由 task 提供
├── meta_reflect.py             # 反思循环，prompt 模板由 task 提供
├── data/
│   ├── fetch_civitai.py        # 保持不变（image_json 专属数据采集）
│   └── filter_prompts.py       # 保持不变
└── schema.py                   # 可保留为向后兼容的别名导入
```

### 使用方式（重构后）

```bash
# 当前任务（默认）
python optimize_gepa.py --task image_json

# 未来新任务，主流程代码零改动
python optimize_gepa.py --task scene_script
python optimize_gepa.py --task seo_keywords
python evaluate.py --task image_json
```

---

## 三、核心接口设计：`tasks/base.py`

```python
"""任务抽象基类 - 每个任务必须实现此接口"""
from abc import ABC, abstractmethod
from pathlib import Path
import dspy


class BaseTask(ABC):
    """
    任务插件接口。
    新增任务只需继承此类并实现以下 4 个方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """任务唯一标识，如 'image_json'、'scene_script'"""
        ...

    @abstractmethod
    def get_dspy_module(self) -> dspy.Module:
        """
        返回该任务的 DSPy Module 实例。
        内部定义 Signature 和 Module 结构。
        """
        ...

    @abstractmethod
    def load_dataset(self, data_path: Path) -> list[dspy.Example]:
        """
        从数据文件加载并转换为 DSPy Example 列表。
        每个 Example 必须调用 .with_inputs() 标记输入字段。
        """
        ...

    @abstractmethod
    def metric(self, gold, pred, trace=None, pred_name="", pred_trace=None) -> dspy.Prediction:
        """
        GEPA metric 函数。
        返回 dspy.Prediction(score=float, feedback=str)
        score 范围 0.0~1.0
        """
        ...

    def get_reflection_prompt_template(self) -> str:
        """
        （可选覆盖）meta_reflect.py 使用的反思 prompt 模板。
        默认模板适用于大多数任务，特殊任务可覆盖此方法。
        """
        return """你是一个 System Prompt 优化专家。

当前 System Prompt（v{version}）：
```
{system_prompt}
```

以下是使用该 System Prompt 失败或低分的案例：
{cases_text}

请分析失败原因，输出改进后的 System Prompt（v{new_version}）。
只输出新的 System Prompt 文本，不要有任何解释或前言。"""

    def get_default_data_path(self) -> Path:
        """（可选覆盖）该任务默认的数据文件路径"""
        return Path("data/prompts.json")
```

---

## 四、image_json 任务迁移示例：`tasks/image_json/__init__.py`

```python
"""image_json 任务：将 SD prompt 转为结构化 image-json"""
from pathlib import Path
import dspy
from tasks.base import BaseTask
from tasks.image_json.schema import ImageJSON, parse_and_validate
from tasks.image_json.metric import rule_score, make_metric


class ImageJsonTask(BaseTask):

    @property
    def name(self) -> str:
        return "image_json"

    def get_dspy_module(self) -> dspy.Module:
        from tasks.image_json.signature import PromptParser
        return PromptParser()

    def load_dataset(self, data_path: Path) -> list[dspy.Example]:
        import json
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        return [
            dspy.Example(raw_prompt=d["prompt"]).with_inputs("raw_prompt")
            for d in data
        ]

    def metric(self, gold, pred, trace=None, pred_name="", pred_trace=None):
        return make_metric(gold, pred, trace, pred_name, pred_trace)

    def get_default_data_path(self) -> Path:
        return Path("data/prompts_filtered.json")
```

---

## 五、主流程改动：`optimize_gepa.py` 改动点

重构后主流程只需改动约 10 行：

```python
# 重构前（硬编码）
module = PromptParser()
optimizer = dspy.GEPA(metric=metric_with_feedback, ...)
examples = [dspy.Example(raw_prompt=d["prompt"])... for d in data]

# 重构后（任务插件化）
import argparse
from tasks import load_task

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="image_json")
args = parser.parse_args()

task = load_task(args.task)                                  # 加载任务插件
module = task.get_dspy_module()                              # 获取 DSPy Module
examples = task.load_dataset(task.get_default_data_path())   # 加载数据
optimizer = dspy.GEPA(metric=task.metric, ...)               # 注入 metric
```

---

## 六、任务注册表：`tasks/__init__.py`

```python
"""任务注册表 - 新增任务在此注册"""
from tasks.base import BaseTask
from tasks.image_json import ImageJsonTask

_REGISTRY: dict[str, type[BaseTask]] = {
    "image_json": ImageJsonTask,
    # 新增任务：在此添加一行即可
    # "scene_script": SceneScriptTask,
    # "seo_keywords": SEOKeywordsTask,
}


def load_task(name: str) -> BaseTask:
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"未知任务: '{name}'。可用任务: {available}")
    return _REGISTRY[name]()


def list_tasks() -> list[str]:
    return list(_REGISTRY.keys())
```

---

## 七、新增任务示例：`tasks/scene_script/`

假设未来要训练"将用户描述转为分镜脚本"任务，只需：

1. 新建 `tasks/scene_script/` 目录，实现 `BaseTask` 的 4 个方法
2. 在 `tasks/__init__.py` 注册表添加一行
3. 准备对应数据集

**无需改动** `optimize_gepa.py`、`evaluate.py`、`meta_reflect.py` 的任何主流程代码。

---

## 八、实施步骤（待执行）

1. **新建 `tasks/` 目录结构**，写好 `base.py` 接口
2. **迁移现有代码**：将 `schema.py`、evaluate 的 metric 逻辑、optimize 的 Signature 移入 `tasks/image_json/`
3. **根目录保留向后兼容导入**：`schema.py` 改为 `from tasks.image_json.schema import *`，避免破坏现有脚本
4. **改造 `optimize_gepa.py`**：加 `--task` 参数，通过 `load_task()` 加载
5. **改造 `evaluate.py`** 和 **`meta_reflect.py`**：同上
6. **验证现有 `image_json` 任务**：确保行为与重构前完全一致
7. **添加第二个任务**：验证扩展机制是否工作

**预估工作量**：半天~一天（主要是迁移 + 测试，无新功能开发）

---

## 九、重构前后对比

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| 新增任务改动范围 | 4 个核心文件，~120 行 | 1 个新目录 + 注册表加 1 行 |
| 主流程代码复用 | 不复用，需复制改写 | 零改动 |
| 任务间隔离 | 无（同一文件混写） | 完全隔离（独立目录） |
| 新人理解成本 | 需通读所有文件 | 看 `base.py` 接口即可 |
| 向后兼容 | — | 通过别名导入保持兼容 |
