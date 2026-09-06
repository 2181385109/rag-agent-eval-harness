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

# 裁判模型。**刻意与被测模型不同源**：被测是 deepseek-chat，裁判用 deepseek-reasoner，
# 避免"自己判自己"的自评偏好（同一模型倾向于给自己的输出打高分）。
# 注意这个值同时决定 RAGAS 的裁判模型，改它会让 faithfulness 等指标整体变动，
# 报告的 meta 里记了实际用的是哪个，换模型必须重跑才能与配置对上。
JUDGE_MODEL_NAME = "deepseek-reasoner"

API_KEY_ENV = "DEEPSEEK_API_KEY"

# 评测要可复现，所以默认温度必须是 0；需要造多样性的地方（多轮一致性）显式传参覆盖。
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
REQUEST_TIMEOUT_S = 60.0

# RAGAS 裁判的调用比 Agent 重得多（faithfulness 要拆句逐条核验，
# context_precision 还要对每个 context 单独判），60s 会大面积超时——
# 实测：60s/16 并发 -> context_precision 只打出 16/35；
# 180s/4 并发 -> 8/9；300s/2 并发 -> chat 裁判 35/35。
# 换 deepseek-reasoner 后 300s 又不够（推理 token 让单次调用更慢，
# faithfulness 只打了 31/35），故放宽到 600s。
# 评测是离线批处理，稳比快重要。
JUDGE_TIMEOUT_S = 600.0
# 裁判打分的输出上限。reasoner 的推理 token 也计入 completion，
# 给少了会在推理阶段就被截断、正文吐不出来，解析必然失败。
JUDGE_MAX_TOKENS = 4096

RAGAS_MAX_WORKERS = 2  # RAGAS 默认 16 并发，对 DeepSeek 太激进；降到 2 换取长尾任务不超时
MAX_RETRIES = 2

# ---------------------------------------------------------------------- 路径
CORPUS_DIR = PROJECT_ROOT / "corpus"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
GOLDEN_SET_PATH = DATA_DIR / "golden_capability.jsonl"
HUMAN_LABELS_PATH = DATA_DIR / "human_labels.jsonl"
# 一致率一节的人写解读（数字自动算，叙述人写，两者在报告里分区显示）
AGREEMENT_INTERPRETATION_PATH = DATA_DIR / "agreement_interpretation.md"

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
# 主评测锁 temp=0 保可复现，但那样一致性几乎恒为 1.0、没有信息量。
# 另跑一档 temp>0 专门测鲁棒性，两者并存报数，互不替代。
CONSISTENCY_TEMPERATURE = 0.7

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
