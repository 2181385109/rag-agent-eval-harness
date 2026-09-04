"""轨迹 schema 的验收口径。

轨迹是评测流水线的原料：M3 的工具调用准确率读 tool_sequence，
recall@k 读 retrieved_doc_ids，任务成功率读 answer。
这些派生属性的语义必须钉死。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent.trace import AgentTrace, RetrievedChunk, ToolCall, ToolResult, TraceStep


def _chunk(doc_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(doc_id=doc_id, source="a.md", score=score, text="正文")


def _step(index: int, tool: str, doc_ids: list[str] | None = None) -> TraceStep:
    return TraceStep(
        index=index,
        tool_calls=[ToolCall(id=f"c{index}", name=tool, args={})],
        results=[
            ToolResult(
                call_id=f"c{index}",
                name=tool,
                ok=True,
                observation="ok",
                retrieved=[_chunk(d) for d in (doc_ids or [])],
            )
        ],
    )


def test_tool_sequence_preserves_call_order():
    trace = AgentTrace(
        question="q",
        answer="a",
        steps=[_step(0, "retrieve"), _step(1, "calc"), _step(2, "retrieve")],
    )
    assert trace.tool_sequence == ["retrieve", "calc", "retrieve"]


def test_retrieved_doc_ids_dedupe_but_keep_rank_order():
    """recall@k 关心的是排名，所以顺序要保，重复要去。"""
    trace = AgentTrace(
        question="q",
        answer="a",
        steps=[_step(0, "retrieve", ["d1", "d2"]), _step(1, "retrieve", ["d2", "d3"])],
    )
    assert trace.retrieved_doc_ids == ["d1", "d2", "d3"]


def test_no_tool_call_yields_empty_sequence():
    """模型直接回答不调工具，是合法轨迹，也是"该调没调"这类失败的判定依据。"""
    trace = AgentTrace(question="q", answer="a", steps=[TraceStep(index=0, thought="直接答")])
    assert trace.tool_sequence == []
    assert trace.retrieved_doc_ids == []


def test_failed_tool_call_still_counts_in_sequence():
    """选错工具也要计入序列——CLAUDE.md §5 要求如实记录失败。"""
    step = TraceStep(
        index=0,
        tool_calls=[ToolCall(id="c0", name="browse_web", args={})],
        results=[ToolResult(call_id="c0", name="browse_web", ok=False, observation="", error="未知工具")],
    )
    trace = AgentTrace(question="q", answer="", steps=[step], stop_reason="answered")
    assert trace.tool_sequence == ["browse_web"]
    assert trace.has_failure is True


def test_stop_reason_is_constrained():
    with pytest.raises(ValidationError):
        AgentTrace(question="q", answer="a", stop_reason="随便写的")


def test_trace_round_trips_through_json():
    """轨迹要能落盘再读回来，报告和回归基线都靠它。"""
    trace = AgentTrace(
        question="q", answer="a", steps=[_step(0, "retrieve", ["d1"])], model="deepseek-chat"
    )
    restored = AgentTrace.model_validate_json(trace.model_dump_json())
    assert restored == trace
    assert restored.retrieved_doc_ids == ["d1"]


def test_token_usage_accumulates():
    trace = AgentTrace(question="q", answer="a", prompt_tokens=100, completion_tokens=30)
    assert trace.total_tokens == 130
