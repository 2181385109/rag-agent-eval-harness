"""共享 fixture。

原则（CLAUDE.md §7）：默认路径下的测试一律不碰真实 API——CI 上没有 key，
而且真实调用既费钱又不确定。凡需要 LLM 的地方都在这里打桩。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def stub_chat(monkeypatch):
    """把 llm.chat 换成可控桩，返回 (设置回答的函数, 调用记录 list)。"""
    from src.agent import llm

    calls: list[dict] = []
    box = {"reply": "STUB_ANSWER"}

    def _fake_chat(messages, **kwargs):
        calls.append({"messages": messages, "kwargs": kwargs})
        return box["reply"]

    monkeypatch.setattr(llm, "chat", _fake_chat)

    def _set(reply: str) -> None:
        box["reply"] = reply

    return _set, calls
