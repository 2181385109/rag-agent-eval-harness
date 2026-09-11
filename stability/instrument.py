"""给被测 Agent 挂计时探针，不改 Agent 代码（PERF_SPEC B1：每次运行记三段延迟）。

graph.py 里两处对外调用：
  - `llm.chat_completion(...)`   模块属性，运行期查找 -> 换掉模块上的函数即可计时
  - `self.toolbox.run(...)`      实例方法 -> 换掉该实例的 run 属性即可计时

探针只**旁路记录**耗时与 usage，不改任何返回值，Agent 的行为与轨迹不受影响。
用 with 语句包住一次 agent.run()，退出时恢复原函数。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.agent import llm


@dataclass
class PhaseTimer:
    llm_s: float = 0.0
    retrieve_s: float = 0.0
    tool_s: float = 0.0
    n_llm_calls: int = 0
    n_retrieve_calls: int = 0
    n_tool_calls: int = 0
    llm_call_s: list[float] = field(default_factory=list)
    cache_hit_tokens: int | None = None
    cache_miss_tokens: int | None = None
    response_model: str | None = None
    system_fingerprint: str | None = None

    def add_usage(self, resp) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        hit = _usage_field(usage, "prompt_cache_hit_tokens")
        miss = _usage_field(usage, "prompt_cache_miss_tokens")
        if hit is not None:
            self.cache_hit_tokens = (self.cache_hit_tokens or 0) + int(hit)
        if miss is not None:
            self.cache_miss_tokens = (self.cache_miss_tokens or 0) + int(miss)
        if self.response_model is None:
            self.response_model = getattr(resp, "model", None)
        if self.system_fingerprint is None:
            self.system_fingerprint = getattr(resp, "system_fingerprint", None)


def _usage_field(usage, name: str):
    """DeepSeek 在 usage 里多带的字段，openai SDK 把它们放在 model_extra。"""
    value = getattr(usage, name, None)
    if value is None:
        extra = getattr(usage, "model_extra", None) or {}
        value = extra.get(name)
    return value


@contextmanager
def instrumented(agent):
    """在 with 块内给 agent 挂探针，产出一个 PhaseTimer。"""
    timer = PhaseTimer()
    original_chat = llm.chat_completion
    original_run = agent.toolbox.run

    def timed_chat(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            resp = original_chat(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            timer.llm_s += dt
            timer.n_llm_calls += 1
            timer.llm_call_s.append(dt)
        timer.add_usage(resp)
        return resp

    def timed_run(name, args, call_id=""):
        t0 = time.perf_counter()
        try:
            return original_run(name, args, call_id=call_id)
        finally:
            dt = time.perf_counter() - t0
            if name == "retrieve":
                timer.retrieve_s += dt
                timer.n_retrieve_calls += 1
            else:
                timer.tool_s += dt
                timer.n_tool_calls += 1

    llm.chat_completion = timed_chat
    agent.toolbox.run = timed_run
    try:
        yield timer
    finally:
        llm.chat_completion = original_chat
        agent.toolbox.run = original_run
