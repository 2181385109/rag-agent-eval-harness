"""四个自定义指标（CLAUDE.md §6）。

本模块只做**纯计算**：输入黄金集样本 + 已跑好的轨迹，输出指标。
不跑 Agent、不发请求，因此可以在 CI 里用构造轨迹完整验证指标逻辑本身。

指标的精确口径写在 tests/test_metrics.py 的断言里——那里是定义，这里是实现。
口径要改先改测试。

三条贯穿始终的原则：
  1. **无定义就返回 None，不要当 0。** 纯算术题没有应检索文档，把它算成
     recall=0 会凭空拉低指标；开放题在 M3 无法规则判定，硬判对错就是编数。
  2. **分母如实上报。** 每个指标都带 (n 实际计入 / total 总题数)，
     报告里必须把分母写出来，不许拿全集稀释或抬高。
  3. **严格与宽松口径并列。** 工具序列比对给严格（有序）和宽松（集合）两版，
     检索召回给"全部检索并集"和"仅首次检索"两版，让人看见差多少，
     而不是我替你挑一个好看的。
"""

from __future__ import annotations

import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from src.agent.trace import AgentTrace
from src.eval.datasets import GoldenSample


# ------------------------------------------------------------------ 文本比对
def normalize(text: str) -> str:
    """全角转半角、去空格、转小写。

    为的是不让"０．２５"和"0.25"、"Stratify"和"stratify"这种纯书写差异
    被误判成答错。这是降低误伤，不是放宽正确性标准。
    """
    folded = unicodedata.normalize("NFKC", text or "")
    return "".join(folded.split()).lower()


def key_hits(key: str, text: str) -> bool:
    """一个 answer_key 是否命中文本。key 内用 `|` 分隔可接受的写法变体。"""
    pool = normalize(text)
    return any(normalize(alt) in pool for alt in key.split("|") if alt.strip())


# ------------------------------------------------------------------ 聚合结果
@dataclass
class MetricResult:
    """一个聚合指标。

    value: 指标值；n: 实际计入的题数（分母）；total: 黄金集总题数。
    n 与 total 不等时，报告里必须写清楚——这是"分母诚实"的落点。
    """

    name: str
    value: float | None
    n: int
    total: int
    detail: dict[str, float | bool | None] = field(default_factory=dict)

    @property
    def denominator_note(self) -> str:
        return f"n={self.n}/{self.total}"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "n": self.n,
            "total": self.total,
            "detail": self.detail,
        }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


# =========================================================== 工具调用准确率
def collapse_repeats(sequence: Iterable[str]) -> list[str]:
    """折叠连续重复：['retrieve','retrieve','calc'] -> ['retrieve','calc']。

    模型把一次检索拆成连着两次调用，是检索策略的差异，不是工具选错了。
    但 ['retrieve','calc','retrieve'] 不会被折叠——那是真的来回切换。
    """
    out: list[str] = []
    for name in sequence:
        if not out or out[-1] != name:
            out.append(name)
    return out


def tool_match(sample: GoldenSample, trace: AgentTrace) -> bool:
    """严格口径：折叠连续重复后，与标注的工具序列**逐项相等**。"""
    return collapse_repeats(trace.tool_sequence) == list(sample.expected_tool)


def tool_set_match(sample: GoldenSample, trace: AgentTrace) -> bool:
    """宽松口径：只看用了哪些工具，不看顺序。"""
    return set(trace.tool_sequence) == set(sample.expected_tool)


def tool_accuracy(
    samples: Sequence[GoldenSample], traces: Mapping[str, AgentTrace]
) -> MetricResult:
    """工具调用准确率（严格口径）。全部题目都参与。"""
    detail: dict[str, float | bool | None] = {}
    hits: list[float] = []
    for s in samples:
        trace = traces.get(s.id)
        if trace is None:
            detail[s.id] = None
            continue
        ok = tool_match(s, trace)
        detail[s.id] = ok
        hits.append(1.0 if ok else 0.0)
    return MetricResult("tool_accuracy", _mean(hits), len(hits), len(samples), detail)


def tool_set_accuracy(
    samples: Sequence[GoldenSample], traces: Mapping[str, AgentTrace]
) -> MetricResult:
    """工具调用准确率（顺序不敏感的宽松口径），与严格口径并列上报。"""
    detail: dict[str, float | bool | None] = {}
    hits: list[float] = []
    for s in samples:
        trace = traces.get(s.id)
        if trace is None:
            detail[s.id] = None
            continue
        ok = tool_set_match(s, trace)
        detail[s.id] = ok
        hits.append(1.0 if ok else 0.0)
    return MetricResult("tool_set_accuracy", _mean(hits), len(hits), len(samples), detail)


# ================================================================= recall@k
def _first_retrieve_doc_ids(trace: AgentTrace) -> list[str]:
    """只取第一次 retrieve 调用返回的 doc_id。"""
    for step in trace.steps:
        for result in step.results:
            if result.name == "retrieve" and result.retrieved:
                return [c.doc_id for c in result.retrieved]
    return []


def recall_at_k(sample: GoldenSample, trace: AgentTrace) -> float | None:
    """单题 recall@k：标注的应检索文档，有多少比例出现在**全部检索结果**里。

    用并集而不是单次结果，衡量的是"Agent 最终有没有看到证据"。
    没有标注应检索文档的题返回 None——对它无定义，不能当 0。
    """
    if not sample.expected_doc_ids:
        return None
    retrieved = set(trace.retrieved_doc_ids)
    hit = sum(1 for d in sample.expected_doc_ids if d in retrieved)
    return hit / len(sample.expected_doc_ids)


def recall_at_k_first_call(sample: GoldenSample, trace: AgentTrace) -> float | None:
    """更严格的一版：只看第一次检索的 top-k。

    注意口径：衡量的是"检索器在 Agent **首次**查询上的表现"，
    不是"在原始问题文本上的表现"——Agent 在发出第一次检索前就已经改写过 query 了。
    两版并列上报，差值反映的是**多次检索**（而非查询改写）捞回了多少。
    """
    if not sample.expected_doc_ids:
        return None
    retrieved = set(_first_retrieve_doc_ids(trace))
    hit = sum(1 for d in sample.expected_doc_ids if d in retrieved)
    return hit / len(sample.expected_doc_ids)


def aggregate_recall(
    samples: Sequence[GoldenSample],
    traces: Mapping[str, AgentTrace],
    first_call_only: bool = False,
) -> MetricResult:
    """recall@k 聚合。**分母只算有 expected_doc_ids 的题**（CLAUDE.md §6 修订记录）。"""
    fn = recall_at_k_first_call if first_call_only else recall_at_k
    name = "recall_at_k_first_call" if first_call_only else "recall_at_k"

    detail: dict[str, float | bool | None] = {}
    scores: list[float] = []
    for s in samples:
        trace = traces.get(s.id)
        value = None if trace is None else fn(s, trace)
        detail[s.id] = value
        if value is not None:
            scores.append(value)
    return MetricResult(name, _mean(scores), len(scores), len(samples), detail)


# ============================================================== 任务成功率
def judge_closed(sample: GoldenSample, answer: str) -> bool | None:
    """闭合题规则判定：answer_keys 必须**全部**命中最终答案。

    开放题返回 None——M3 没有裁判层，硬判就是编数，留给 M5。
    """
    if sample.answer_type != "closed":
        return None
    return all(key_hits(key, answer) for key in sample.answer_keys)


def task_success_rate(
    samples: Sequence[GoldenSample], traces: Mapping[str, AgentTrace]
) -> MetricResult:
    """任务成功率。M3 只统计闭合题，开放题计入 total 但不计入 n。"""
    detail: dict[str, float | bool | None] = {}
    hits: list[float] = []
    for s in samples:
        trace = traces.get(s.id)
        verdict = None if trace is None else judge_closed(s, trace.answer)
        detail[s.id] = verdict
        if verdict is not None:
            hits.append(1.0 if verdict else 0.0)
    return MetricResult("task_success_rate", _mean(hits), len(hits), len(samples), detail)


# ============================================================== 多轮一致性
@dataclass
class ConsistencyResult:
    """同题多次运行的稳定性。

    success_agreement: 每题里"多数派判定"占 K 次的比例，再按题平均。
                       1.0 表示每题 K 次判定完全一致；0.5 表示对半摇摆。
    tool_agreement:    同理，但看的是走过的工具序列是否稳定。
    success_variance:  **同题 K 次判定之间**的方差（0/1 序列的总体方差），再按题平均。
                       全部一致时为 0；K 次里三对两错时为 0.24（即 0.6*0.4）。
                       注意不是"题与题之间"的方差——CLAUDE.md §6 要的是同题重复的波动。
    """

    runs: int
    n_questions: int
    success_agreement: float | None
    tool_agreement: float | None
    success_variance: float | None
    detail: dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "runs": self.runs,
            "n_questions": self.n_questions,
            "success_agreement": self.success_agreement,
            "tool_agreement": self.tool_agreement,
            "success_variance": self.success_variance,
            "detail": self.detail,
        }


def _majority_share(values: Sequence) -> float:
    return Counter(values).most_common(1)[0][1] / len(values)


def consistency(
    samples: Sequence[GoldenSample], runs: Mapping[str, Sequence[AgentTrace]]
) -> ConsistencyResult:
    """多轮一致性：同题跑 K 次，看判定结果和工具路径稳不稳。

    答案对不对（success）和走的路径稳不稳（tool）是两件事，分开报：
    模型可能次次答对但路径乱跳，也可能路径固定却答案摇摆。
    """
    k_values = {len(v) for v in runs.values()}
    if not k_values:
        raise ValueError("没有任何运行结果")
    if min(k_values) < 2:
        raise ValueError("多轮一致性至少需要每题 2 次运行")

    success_shares: list[float] = []
    tool_shares: list[float] = []
    success_rates: list[float] = []
    success_variances: list[float] = []
    detail: dict[str, dict] = {}

    for s in samples:
        traces = runs.get(s.id)
        if not traces:
            continue
        seqs = [tuple(collapse_repeats(t.tool_sequence)) for t in traces]
        tool_share = _majority_share(seqs)
        tool_shares.append(tool_share)

        entry: dict = {"tool_agreement": tool_share, "tool_sequences": [list(x) for x in seqs]}

        verdicts = [judge_closed(s, t.answer) for t in traces]
        if all(v is not None for v in verdicts):
            share = _majority_share(verdicts)
            success_shares.append(share)
            outcomes = [1.0 if v else 0.0 for v in verdicts]
            rate = sum(outcomes) / len(outcomes)
            var = statistics.pvariance(outcomes) if len(outcomes) > 1 else 0.0
            success_rates.append(rate)
            success_variances.append(var)
            entry["success_agreement"] = share
            entry["success_rate"] = rate
            entry["success_variance"] = var
        else:
            entry["success_agreement"] = None  # 开放题，M3 无法规则判定

        detail[s.id] = entry

    return ConsistencyResult(
        runs=max(k_values),
        n_questions=len(detail),
        success_agreement=_mean(success_shares),
        tool_agreement=_mean(tool_shares),
        success_variance=_mean(success_variances),
        detail=detail,
    )
