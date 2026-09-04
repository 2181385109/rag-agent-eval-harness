"""真实打通 DeepSeek 的冒烟测试。

默认不跑（会花钱、要网络）。本机手动验证用：
    pytest -m live
CI 上不跑（CLAUDE.md §7）。
"""
from __future__ import annotations

import pytest

from src import config
from src.agent import graph, llm

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not config.has_api_key(), reason="未设置 DEEPSEEK_API_KEY"),
]


def test_llm_chat_round_trip():
    out = llm.chat(
        [{"role": "user", "content": "只回复两个字：收到"}],
        max_tokens=16,
    )
    assert isinstance(out, str) and out.strip(), "DeepSeek 返回了空内容"


def test_hello_graph_end_to_end():
    answer = graph.ask("用一句话解释什么是过拟合。")
    assert isinstance(answer, str)
    assert len(answer.strip()) > 5
