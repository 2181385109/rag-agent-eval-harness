"""四个自定义指标的验收口径（CLAUDE.md §6）。

这些测试就是指标的**定义**：口径要改，先改这里的断言，再改实现。
全部用手工构造的轨迹，不跑 Agent、不发请求。
"""

from __future__ import annotations

import pytest

from src.agent.trace import AgentTrace, RetrievedChunk, ToolCall, ToolResult, TraceStep
from src.eval import metrics
from src.eval.datasets import GoldenSample


def make_sample(**overrides) -> GoldenSample:
    base = dict(
        id="q1",
        question="问题？",
        reference_answer="答案",
        answer_keys=["0.25"],
        expected_tool=["retrieve"],
        expected_doc_ids=["d1"],
        answer_type="closed",
        failure_tag=None,
    )
    base.update(overrides)
    return GoldenSample(**base)


def make_trace(answer: str = "", tools: list[str] | None = None, retrieved=None) -> AgentTrace:
    """retrieved: 每次 retrieve 调用返回的 doc_id 列表，按调用顺序。"""
    tools = tools or []
    retrieved = list(retrieved or [])
    steps, ri = [], 0
    for i, name in enumerate(tools):
        hits = []
        if name == "retrieve" and ri < len(retrieved):
            hits = [
                RetrievedChunk(doc_id=d, source="s.md", score=1.0 - j * 0.01, text="t")
                for j, d in enumerate(retrieved[ri])
            ]
            ri += 1
        steps.append(
            TraceStep(
                index=i,
                tool_calls=[ToolCall(id=f"c{i}", name=name, args={})],
                results=[ToolResult(call_id=f"c{i}", name=name, ok=True, retrieved=hits)],
            )
        )
    steps.append(TraceStep(index=len(tools), thought=answer))
    return AgentTrace(question="问题？", answer=answer, steps=steps)


# ============================================================ 文本归一化
@pytest.mark.parametrize(
    "answer,key,hit",
    [
        ("阈值是 0.25", "0.25", True),
        ("阈值是０．２５", "0.25", True),          # 全角数字
        ("PSI ＞ 0.25 触发", "0.25", True),
        ("答案是 0.3", "0.25", False),
        ("用 Stratify 分层切分", "分层|stratify", True),   # 变体命中其一即可
        ("按标签分层", "分层|stratify", True),
        ("随机切分", "分层|stratify", False),
        ("目标 80%", "0.8|80%", True),
        ("目标 0.80", "0.8|80%", True),
    ],
)
def test_answer_key_matching(answer, key, hit):
    assert metrics.key_hits(key, answer) is hit


# ======================================================== 工具调用准确率
def test_tool_sequence_exact_match():
    s = make_sample(expected_tool=["retrieve"])
    assert metrics.tool_match(s, make_trace(tools=["retrieve"])) is True


def test_consecutive_duplicate_tools_are_collapsed():
    """模型连调两次 retrieve 视同一次——它只是把检索拆细了，工具选得没错。"""
    s = make_sample(expected_tool=["retrieve"])
    assert metrics.tool_match(s, make_trace(tools=["retrieve", "retrieve"])) is True


def test_extra_tool_counts_as_wrong():
    """纯算术题却先检索了一次，算错——这正是要抓的偏差。"""
    s = make_sample(expected_tool=["calc"], expected_doc_ids=[])
    assert metrics.tool_match(s, make_trace(tools=["retrieve", "calc"])) is False


def test_wrong_order_counts_as_wrong_under_strict_rule():
    s = make_sample(expected_tool=["retrieve", "calc"])
    assert metrics.tool_match(s, make_trace(tools=["calc", "retrieve"])) is False


def test_set_match_ignores_order():
    """顺序不敏感的宽松口径，与严格口径并列上报，让人看见两者差多少。"""
    s = make_sample(expected_tool=["retrieve", "calc"])
    trace = make_trace(tools=["calc", "retrieve"])
    assert metrics.tool_set_match(s, trace) is True
    assert metrics.tool_match(s, trace) is False


def test_no_tool_when_one_expected_is_wrong():
    s = make_sample(expected_tool=["retrieve"])
    assert metrics.tool_match(s, make_trace(tools=[])) is False


def test_tool_accuracy_aggregate():
    samples = [
        make_sample(id="a", expected_tool=["retrieve"]),
        make_sample(id="b", expected_tool=["calc"], expected_doc_ids=[]),
    ]
    traces = {"a": make_trace(tools=["retrieve"]), "b": make_trace(tools=["retrieve", "calc"])}
    r = metrics.tool_accuracy(samples, traces)
    assert r.value == 0.5
    assert r.n == 2 and r.total == 2


# ================================================================ recall@k
def test_recall_full_hit():
    s = make_sample(expected_doc_ids=["d1", "d2"])
    t = make_trace(tools=["retrieve"], retrieved=[["d1", "d2", "d9"]])
    assert metrics.recall_at_k(s, t) == 1.0


def test_recall_partial_hit():
    s = make_sample(expected_doc_ids=["d1", "d2"])
    t = make_trace(tools=["retrieve"], retrieved=[["d1", "d9"]])
    assert metrics.recall_at_k(s, t) == 0.5


def test_recall_is_none_without_annotation():
    """纯算术题没有应检索文档，recall 对它无定义——不能当 0 拉低均值。"""
    s = make_sample(expected_doc_ids=[], expected_tool=["calc"])
    assert metrics.recall_at_k(s, make_trace(tools=["calc"])) is None


def test_recall_union_covers_multiple_retrieve_calls():
    """Agent 多次检索时，只要它最终见到了证据就算召回到。"""
    s = make_sample(expected_doc_ids=["d1", "d2"])
    t = make_trace(tools=["retrieve", "retrieve"], retrieved=[["d1"], ["d2"]])
    assert metrics.recall_at_k(s, t) == 1.0


def test_recall_first_call_only_is_stricter():
    """只看第一次检索，衡量的是检索器本身的质量，不含 Agent 改写查询的功劳。"""
    s = make_sample(expected_doc_ids=["d1", "d2"])
    t = make_trace(tools=["retrieve", "retrieve"], retrieved=[["d1"], ["d2"]])
    assert metrics.recall_at_k_first_call(s, t) == 0.5


def test_recall_aggregate_reports_denominator():
    """分母必须是有标注的题数，不是全集——CLAUDE.md §6 修订记录明写。"""
    samples = [
        make_sample(id="a", expected_doc_ids=["d1"]),
        make_sample(id="b", expected_doc_ids=["d2"]),
        make_sample(id="c", expected_doc_ids=[], expected_tool=["calc"]),
    ]
    traces = {
        "a": make_trace(tools=["retrieve"], retrieved=[["d1"]]),
        "b": make_trace(tools=["retrieve"], retrieved=[["zz"]]),
        "c": make_trace(tools=["calc"]),
    }
    r = metrics.aggregate_recall(samples, traces)
    assert r.value == 0.5
    assert r.n == 2, "分母应为有标注的 2 题"
    assert r.total == 3, "总题数仍如实上报"


# ============================================================ 任务成功率
def test_closed_success_needs_all_keys():
    s = make_sample(answer_keys=["0.25", "显著漂移"])
    assert metrics.judge_closed(s, "PSI 超过 0.25 即显著漂移") is True
    assert metrics.judge_closed(s, "PSI 超过 0.25") is False


def test_open_question_is_not_rule_judged():
    """开放题在 M3 无法自动判定，返回 None 而不是硬判对错。"""
    s = make_sample(answer_type="open", answer_keys=[])
    assert metrics.judge_closed(s, "任意回答") is None


def test_task_success_aggregate_excludes_open():
    samples = [
        make_sample(id="a", answer_keys=["0.25"]),
        make_sample(id="b", answer_keys=["96"]),
        make_sample(id="c", answer_type="open", answer_keys=[]),
    ]
    traces = {
        "a": make_trace(answer="阈值 0.25"),
        "b": make_trace(answer="没提到那个数"),
        "c": make_trace(answer="一段开放回答"),
    }
    r = metrics.task_success_rate(samples, traces)
    assert r.value == 0.5
    assert r.n == 2 and r.total == 3


# ============================================================== 多轮一致性
def test_consistency_all_same():
    s = make_sample(answer_keys=["0.25"])
    runs = [make_trace(answer="是 0.25", tools=["retrieve"]) for _ in range(5)]
    r = metrics.consistency([s], {"q1": runs})
    assert r.success_agreement == 1.0
    assert r.tool_agreement == 1.0
    assert r.runs == 5


def test_consistency_detects_flapping():
    """5 次里 3 次答对 2 次答错 -> 一致比例 0.6，方差非零。"""
    s = make_sample(answer_keys=["0.25"])
    runs = [make_trace(answer=a) for a in ["0.25", "0.25", "0.25", "错", "错"]]
    r = metrics.consistency([s], {"q1": runs})
    assert r.success_agreement == pytest.approx(0.6)
    assert r.success_variance > 0


def test_consistency_tracks_tool_sequence_separately():
    """答案对不对和走的路径稳不稳定是两件事，分开报。"""
    s = make_sample(answer_keys=["0.25"])
    runs = [
        make_trace(answer="0.25", tools=["retrieve"]),
        make_trace(answer="0.25", tools=["retrieve", "calc"]),
    ]
    r = metrics.consistency([s], {"q1": runs})
    assert r.success_agreement == 1.0
    assert r.tool_agreement == 0.5


def test_consistency_requires_at_least_two_runs():
    s = make_sample()
    with pytest.raises(ValueError):
        metrics.consistency([s], {"q1": [make_trace(answer="x")]})
