"""一次运行 = 一行 JSONL 的 schema（PERF_SPEC B1）。

字段一旦定下就是分析脚本的输入契约，改了要连带改 tests/test_stability_analyze.py。
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, computed_field

from src.eval import metrics


class Latency(BaseModel):
    """一次运行的延迟分解，单位秒，全部 wall-clock（perf_counter）。

    llm_s      : 所有 chat_completion 调用的耗时之和（含网络往返与服务端排队，
                 本机无法把它们与模型推理时间拆开——见 LIMITATIONS.md）
    retrieve_s : 所有 retrieve 工具调用的耗时之和（BGE 编码查询 + FAISS 搜索）
    tool_s     : 其它工具（calc）的耗时之和
    overhead_s : total 减去三段，即 LangGraph 编排 / 消息拼装 / 记录轨迹的开销
    """

    total_s: float
    llm_s: float = 0.0
    retrieve_s: float = 0.0
    tool_s: float = 0.0
    n_llm_calls: int = 0
    n_retrieve_calls: int = 0
    n_tool_calls: int = 0
    llm_call_s: list[float] = Field(default_factory=list)  # 逐次 LLM 调用耗时

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overhead_s(self) -> float:
        return max(0.0, self.total_s - self.llm_s - self.retrieve_s - self.tool_s)


class Tokens(BaseModel):
    prompt: int = 0
    completion: int = 0
    total: int = 0
    # DeepSeek 特有：命中服务端 KV 缓存的 prompt token 数。同题重复调用时后几次
    # 大概率命中缓存，延迟与计费都会比第一次低——这是重复测量的一个已知混杂因素，
    # 如实记下来，分析时能看出 run_index=0 与之后几次的差别。
    cache_hit: int | None = None
    cache_miss: int | None = None


class RunRecord(BaseModel):
    """同一道题的第 run_index 次运行。"""

    run_id: str  # 本批次 id（= 原始产物文件名里的时间戳）
    sample_id: str
    run_index: int  # 0..k-1，pass 编号
    k: int
    timestamp_utc: str
    answer_type: str
    question: str
    answer: str
    tool_sequence: list[str]
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    stop_reason: str
    n_steps: int
    latency: Latency
    tokens: Tokens
    error: str | None = None
    # 模型响应里自带的版本线索（DeepSeek 不暴露权重版本，能记的只有这两项）
    response_model: str | None = None
    system_fingerprint: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_sequence_collapsed(self) -> list[str]:
        return metrics.collapse_repeats(self.tool_sequence)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def answer_sha1(self) -> str:
        """答案指纹，逐次裁判分靠它与运行记录配对（同 judge.answer_fingerprint）。"""
        return hashlib.sha1((self.answer or "").encode("utf-8")).hexdigest()[:12]

    @property
    def ok(self) -> bool:
        return self.error is None
