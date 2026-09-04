"""全局配置：模型端点、模型名、路径、各指标阈值。

红线（CLAUDE.md §1.7）：API key 绝不写在这里或任何代码文件里，
只从环境变量 DEEPSEEK_API_KEY 读，由 python-dotenv 从 .env 载入，
而 .env 已被 .gitignore 排除。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# override=False：已经存在的环境变量优先，方便 CI / 临时覆盖
load_dotenv(PROJECT_ROOT / ".env", override=False)

# CLAUDE.md §1.7：除对 DeepSeek 的模型调用外，不向任何外部服务发数据。
# langchain-core 会顺带装上 langsmith，这里显式关掉它的追踪上报（可观测性属 v2）。
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

# ---------------------------------------------------------------- 模型 / 端点
# DeepSeek 提供 OpenAI 兼容端点，所以直接用 openai SDK 打过去。
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

# 裁判模型（M5）。v1 先与被测模型同款；想避免"自己判自己"的自评偏好，
# 可改成 deepseek-reasoner —— 这是一个有意识的评测设计选择。
JUDGE_MODEL_NAME = "deepseek-chat"

API_KEY_ENV = "DEEPSEEK_API_KEY"

# 评测要可复现，所以默认温度必须是 0；需要造多样性的地方（多轮一致性）显式传参覆盖。
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
REQUEST_TIMEOUT_S = 60.0
MAX_RETRIES = 2

# ---------------------------------------------------------------------- 路径
CORPUS_DIR = PROJECT_ROOT / "corpus"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
GOLDEN_SET_PATH = DATA_DIR / "golden_capability.jsonl"
HUMAN_LABELS_PATH = DATA_DIR / "human_labels.jsonl"

# ------------------------------------------------------------- 检索（M2 生效）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# 允许用环境变量临时换小模型跑通逻辑（bge-small-zh-v1.5 约 95MB，large 约 1.3GB）。
# 出正式指标时必须用默认的 large，报告里也会记下实际用的是哪个。
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

# BGE 官方建议：查询侧加指令前缀，文档侧不加。
# 但 M2 用本仓库语料做的 3 条探针显示前缀反而拉低了命中排名，
# 样本太少不足以据此定口径——留成开关，等 M3 用 recall@k 正式裁决。
USE_QUERY_INSTRUCTION = True
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

RETRIEVE_TOP_K = 4

# corpus/README.md 是写给人看的放置说明，不是语料，不进索引。
CORPUS_EXCLUDE = ("README.md",)
CORPUS_SUFFIXES = (".md", ".txt")
INDEX_DIR = PROJECT_ROOT / "index"  # FAISS 索引落盘位置（已 gitignore，可由代码重建）

# 中文优先的切分分隔符：先按段落/换行，再按中文句读，最后才退到字符级
SPLIT_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")

# ------------------------------------------------------------ Agent（M2 生效）
# ReAct 循环的硬上限：模型打转时必须能停下来，否则一道题能烧掉一把 token。
MAX_AGENT_STEPS = 6

# ------------------------------------------------------------ 评测（M3+ 生效）
CONSISTENCY_RUNS = 5  # 多轮一致性：同题跑几次
# 一致性要跑 K 倍的量，成本随题数线性涨；默认只在前若干题上测，报告里写明覆盖数。
CONSISTENCY_SUBSET_SIZE = 10

# -------------------------------------------------------- 回归门禁（M6 生效）
# 与信贷风控项目里 PSI 触发告警同一思路：指标相对基线跌超容差就 fail。
METRIC_DROP_TOLERANCE = 0.05
GATED_METRICS = ("task_success_rate", "tool_accuracy", "faithfulness")


def get_api_key() -> str:
    """取 DeepSeek API key；缺失时明确报错，绝不静默降级。"""
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"环境变量 {API_KEY_ENV} 未设置。请复制 .env.example 为 .env 并填入 key。"
        )
    return key


def has_api_key() -> bool:
    """是否具备真实调用条件（给 live 测试做 skip 判断用）。"""
    return bool(os.environ.get(API_KEY_ENV, "").strip())
