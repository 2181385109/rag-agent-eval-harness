"""hello-world LangGraph 图的验收口径：图能编译、状态能流转、LLM 全打桩。"""
from __future__ import annotations

import pytest

from src.agent import graph


def test_graph_compiles():
    app = graph.build_hello_graph()
    assert app is not None


def test_invoke_produces_answer(stub_chat):
    set_reply, calls = stub_chat
    set_reply("过拟合是模型把训练集噪声也学进去了。")

    app = graph.build_hello_graph()
    state = app.invoke({"question": "什么是过拟合？"})

    assert state["question"] == "什么是过拟合？"
    assert state["answer"] == "过拟合是模型把训练集噪声也学进去了。"
    assert len(calls) == 1, "hello 图对一个问题只应调用一次 LLM"


def test_question_reaches_the_model(stub_chat):
    _set, calls = stub_chat
    graph.build_hello_graph().invoke({"question": "什么是 PSI？"})

    messages = calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "什么是 PSI？"}


def test_ask_helper_returns_str(stub_chat):
    set_reply, _calls = stub_chat
    set_reply("答案")
    assert graph.ask("随便问") == "答案"


def test_empty_question_rejected(stub_chat):
    with pytest.raises(ValueError):
        graph.ask("   ")
