"""RAGAS 指标接入（CLAUDE.md §9 M4）。

四个指标：faithfulness / answer_relevancy / context_recall / context_precision。

两条硬约束：
  1. **裁判 LLM 走 DeepSeek**，但 **embedding 必须留在本地 BGE**。
     §2 把 embedding 锁成"本地免费不走 API"，§1.7 又规定除对 DeepSeek 的模型调用外
     不把语料发往任何外部服务——所以这里用 LocalBGEEmbeddings 适配器包住已有的
     BGEEmbedder，而不是引入某个云端 embedding。
  2. **没有答案的样本不进均值。** cap_036 那类跑满步数没收口、最终答案为空的样本，
     faithfulness 和 answer_relevancy 对它无从谈起；混进均值只会让指标失去意义。
     一律排除并如实记下原因和条数（同 recall@k 的分母纪律）。

ragas 本体在函数内部延迟导入：CI 上不装 ragas、不调真实 API，也能跑通本模块
关于数据组织与排除逻辑的全部测试。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from langchain_core.embeddings import Embeddings

from src import config
from src.agent import rag
from src.agent.trace import AgentTrace
from src.eval.datasets import GoldenSample

def _ensure_ragas_importable() -> None:
    """填平 RAGAS 0.4.3 与 LangGraph 1.x 技术栈之间的一个死导入。

    ragas/llms/base.py 在**模块加载时**就 `from langchain_community.chat_models.vertexai
    import ChatVertexAI`，而这个模块只存在于 langchain-community 0.3.x 的布局里。
    我们这边装的是 0.4.2（langchain 1.x 世代），vertexai 早已拆成独立集成包。

    为什么选择填平而不是降级：把 langchain-community 降到 0.3.x 会连带把
    langchain-core 拽到 0.3.86（实测 pip dry-run 结论），而 langgraph 1.2.11 要求
    core >= 1.0——那等于推翻已验收的 M1–M3。RAGAS 与 LangGraph 都是 CLAUDE.md §2
    锁定的选型，两者不能二选一，所以隔离这个我们**永不使用**的 Vertex AI 代码路径。

    这个补丁是**条件式**的：真模块一旦存在（ragas 修复或依赖变化），它自动不生效。
    """
    import importlib.machinery
    import importlib.util
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"

    # 已经在 sys.modules 里就什么都不做——不管那是上游的真模块，还是本函数
    # 上一次装的占位模块。绝不覆盖已存在的实现。
    if name not in sys.modules:
        try:
            already_available = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError, AttributeError):
            already_available = False

        if not already_available:
            module = types.ModuleType(name)
            module.ChatVertexAI = type("ChatVertexAI", (), {})
            module.__doc__ = "占位模块，见 src/eval/ragas_runner._ensure_ragas_importable"
            # 补上 __spec__，否则后续任何 find_spec 都会抛 ValueError
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = module

    import langchain_community.llms as community_llms

    if not hasattr(community_llms, "VertexAI"):
        community_llms.VertexAI = type("VertexAI", (), {})


DEFAULT_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)


class LocalBGEEmbeddings(Embeddings):
    """把本地 BGEEmbedder 包成 LangChain 的 Embeddings 接口给 RAGAS 用。

    存在的理由只有一个：让 RAGAS 算 answer_relevancy 时用的向量也来自本地模型，
    语料一个字都不外发。查询侧走 is_query=True，BGE 的指令前缀才生效。
    """

    def __init__(self, embedder: Any | None = None):
        self.embedder = embedder or rag.BGEEmbedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedder.encode(list(texts), is_query=False)
        return [[float(x) for x in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.embedder.encode([text], is_query=True)[0]
        return [float(x) for x in vector]


def build_ragas_dataset(
    samples: Sequence[GoldenSample], traces: Mapping[str, AgentTrace]
) -> tuple[list[dict], list[dict]]:
    """把黄金集 + 轨迹整理成 RAGAS 要的行，并挑出该排除的样本。

    返回 (可评测的行, 被排除的样本及原因)。两者相加必然等于总样本数——
    每条样本要么被评、要么被明确排除，不许凭空消失。
    """
    rows: list[dict] = []
    excluded: list[dict] = []

    for sample in samples:
        trace = traces.get(sample.id)
        if trace is None:
            excluded.append({"id": sample.id, "reason": "没有对应的运行轨迹"})
            continue

        answer = (trace.answer or "").strip()
        contexts = [c.text for c in trace.retrieved_chunks]

        if not answer:
            excluded.append(
                {
                    "id": sample.id,
                    "reason": f"最终答案为空（stop_reason={trace.stop_reason}），无从评价答案质量",
                    "stop_reason": trace.stop_reason,
                    "failure_tag": sample.failure_tag,
                }
            )
            continue
        if not contexts:
            excluded.append(
                {
                    "id": sample.id,
                    "reason": "本轮没有检索内容（如纯算术题），faithfulness / context_recall 无从谈起",
                    "stop_reason": trace.stop_reason,
                    "failure_tag": sample.failure_tag,
                }
            )
            continue

        rows.append(
            {
                "id": sample.id,
                "user_input": sample.question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": sample.reference_answer,
            }
        )

    return rows, excluded


def build_run_config():
    """统一的 RunConfig：超时与并发都从 config 来。

    必须显式传给 LangchainLLMWrapper——它构造时会默认塞一个 RunConfig()（timeout=180）
    并**覆写** ChatOpenAI 的 request_timeout。只设 ChatOpenAI 的 timeout 是无效的，
    这一点是被 tests/test_ragas_runner.py 里那条断言逼出来的。
    """
    _ensure_ragas_importable()
    from ragas.run_config import RunConfig

    return RunConfig(
        timeout=int(config.JUDGE_TIMEOUT_S),
        max_workers=config.RAGAS_MAX_WORKERS,
    )


def build_judge_llm(run_config=None):
    """RAGAS 的裁判 LLM：DeepSeek（OpenAI 兼容端点）。"""
    _ensure_ragas_importable()
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    # bypass_n=True 至关重要：RAGAS 为自洽投票会请求 n=3，而 DeepSeek 只支持 n=1，
    # 会直接 400 拒绝。开启后 wrapper 改为发 n 次单生成请求，语义等价、能跑通。
    # 不开的代价实测过：136 个评分任务里 49 个被 400 打掉，均值只覆盖侥幸成功的部分。
    return LangchainLLMWrapper(
        run_config=run_config or build_run_config(),
        bypass_n=True,
        langchain_llm=ChatOpenAI(
            model=config.JUDGE_MODEL_NAME,
            base_url=config.BASE_URL,
            api_key=config.get_api_key(),
            temperature=0.0,
            timeout=config.JUDGE_TIMEOUT_S,
            max_retries=3,
        ),
    )


def build_judge_embeddings(embedder: Any | None = None):
    """RAGAS 的 embedding：本地 BGE，绝不走 API。"""
    _ensure_ragas_importable()
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(LocalBGEEmbeddings(embedder))


def _resolve_metrics(metric_names: Iterable[str]):
    _ensure_ragas_importable()
    from ragas import metrics as ragas_metrics

    resolved = []
    for name in metric_names:
        obj = getattr(ragas_metrics, name, None)
        if obj is None:
            raise ValueError(f"RAGAS 里没有这个指标：{name}")
        resolved.append(obj)
    return resolved


def _evaluate_rows(
    rows: list[dict], metric_names: Sequence[str] | None = None
) -> tuple[dict[str, float], dict[str, int]]:
    """真正调用 RAGAS，返回 (各指标均值, 各指标**真正打出分的行数**)。

    第二个返回值不是可有可无的：RAGAS 里单个评分任务失败（裁判 400、超时）时
    那一行是 NaN，而 pandas 的 mean() 默认跳过 NaN。只报"提交了几行"会把
    失败的行当成打过分的，等于虚报分母。

    测试里整体打桩，所以 CI 不需要装 ragas、不发请求。
    """
    _ensure_ragas_importable()
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset

    names = list(metric_names or DEFAULT_METRICS)
    dataset = EvaluationDataset.from_list(
        [{k: v for k, v in row.items() if k != "id"} for row in rows]
    )
    run_config = build_run_config()
    result = evaluate(
        dataset=dataset,
        metrics=_resolve_metrics(names),
        llm=build_judge_llm(run_config),
        embeddings=build_judge_embeddings(),
        run_config=run_config,
    )
    frame = result.to_pandas()
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in names:
        if name not in frame.columns:
            continue
        column = frame[name]
        counts[name] = int(column.notna().sum())
        scores[name] = float(column.mean()) if counts[name] else float("nan")
    return scores, counts


def run_ragas(
    samples: Sequence[GoldenSample],
    traces: Mapping[str, AgentTrace],
    metric_names: Sequence[str] | None = None,
) -> dict:
    """跑 RAGAS 并返回带分母与排除说明的结果。"""
    names = list(metric_names or DEFAULT_METRICS)
    rows, excluded = build_ragas_dataset(samples, traces)
    scores, counts = _evaluate_rows(rows, names) if rows else ({}, {})

    # 有任何一个指标没在全部提交行上打出分，就必须显式标出来——
    # 那意味着该指标的均值覆盖面小于分母，不能当作全量结果讲。
    incomplete = any(counts.get(name, 0) < len(rows) for name in names)

    return {
        "judge_model": config.JUDGE_MODEL_NAME,
        "embedding_model": config.EMBEDDING_MODEL,
        "metrics": names,
        "scores": scores,
        "scored_counts": counts,
        "has_incomplete_metric": bool(incomplete),
        "n_submitted": len(rows),
        "n_excluded": len(excluded),
        "total": len(samples),
        "evaluated_ids": [r["id"] for r in rows],
        "excluded": excluded,
    }
