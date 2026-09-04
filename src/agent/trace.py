"""Agent 执行轨迹的 pydantic schema（CLAUDE.md §5）。

轨迹是评测流水线的原料，不是日志：
  - tool_sequence      -> M3 的工具调用准确率
  - retrieved_doc_ids  -> M3 的 recall@k
  - answer             -> M3 的任务成功率、M4 的 RAGAS、M5 的裁判
所以这里的字段语义一旦定下就不能随便改，改了要连带改指标定义。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StopReason = Literal["answered", "max_steps", "error"]


class ToolCall(BaseModel):
    """模型请求调用的一个工具。"""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = ""  # 模型原样吐出的参数串，解析失败时靠它复盘


class RetrievedChunk(BaseModel):
    """一次检索命中的语料片段。"""

    doc_id: str
    source: str
    score: float
    text: str


class ToolResult(BaseModel):
    """工具执行结果。失败也是结果，如实记录，不在 Agent 侧纠正。"""

    call_id: str
    name: str
    ok: bool
    observation: str = ""
    error: str | None = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)


class TraceStep(BaseModel):
    """ReAct 的一轮：模型说了什么 -> 要调什么工具 -> 工具返回了什么。"""

    index: int
    thought: str = ""  # 模型这一轮的自然语言输出
    tool_calls: list[ToolCall] = Field(default_factory=list)
    results: list[ToolResult] = Field(default_factory=list)


class AgentTrace(BaseModel):
    """一道题的完整轨迹。"""

    question: str
    answer: str = ""
    steps: list[TraceStep] = Field(default_factory=list)
    stop_reason: StopReason = "answered"
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tool_sequence(self) -> list[str]:
        """按调用顺序列出工具名（含选错的、调失败的）。"""
        return [call.name for step in self.steps for call in step.tool_calls]

    @property
    def retrieved_doc_ids(self) -> list[str]:
        """按检索排名去重后的 doc_id 列表，供 recall@k 使用。"""
        seen: dict[str, None] = {}
        for step in self.steps:
            for result in step.results:
                for chunk in result.retrieved:
                    seen.setdefault(chunk.doc_id, None)
        return list(seen)

    @property
    def retrieved_chunks(self) -> list[RetrievedChunk]:
        """展平的命中片段，供 M4 的 RAGAS 当 contexts 用。"""
        return [c for s in self.steps for r in s.results for c in r.retrieved]

    @property
    def has_failure(self) -> bool:
        """本轮是否出现过工具失败（选错工具、参数非法、执行报错）。"""
        return any(not r.ok for s in self.steps for r in s.results)
