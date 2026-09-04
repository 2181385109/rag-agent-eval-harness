"""构造假 ChatCompletion 的小工具，供 scripted_llm 使用。

形状照抄 OpenAI/DeepSeek 的返回结构，只保留 Agent 真正读的字段。
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def _usage(prompt: int = 10, completion: int = 5):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


def answer_message(text: str):
    """模型直接给出最终答案（没有 tool_calls）。"""
    msg = SimpleNamespace(role="assistant", content=text, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=_usage()
    )


def tool_call_message(calls: list[tuple[str, dict[str, Any]]], content: str = ""):
    """模型要求调工具。calls 形如 [("retrieve", {"query": "..."})]。"""
    tool_calls = [
        SimpleNamespace(
            id=f"call_{i}",
            type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
        )
        for i, (name, args) in enumerate(calls)
    ]
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=_usage()
    )


def malformed_tool_call(name: str, arguments: str):
    """参数不是合法 JSON 的工具调用——模型真的会犯这种错，必须被如实记录。"""
    tool_calls = [
        SimpleNamespace(
            id="call_bad", type="function", function=SimpleNamespace(name=name, arguments=arguments)
        )
    ]
    msg = SimpleNamespace(role="assistant", content="", tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=_usage()
    )
