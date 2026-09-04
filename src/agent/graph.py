"""M1：hello-world LangGraph 状态图。

只有一个节点（问 → 答），目的是把 LangGraph 的状态流转和 DeepSeek 调用打通。
M2 会在同一套状态机上扩成 ReAct 循环（规划 → 选工具 → 调工具 → 观察 → 回答）。

命令行冒烟：
    python -m src.agent.graph "什么是过拟合？"
"""

from __future__ import annotations

import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent import llm

SYSTEM_PROMPT = "你是一个严谨的中文助手。回答要简洁、准确，不确定就说不确定。"


class HelloState(TypedDict, total=False):
    """M1 的最小状态。M2 会加 tool_calls / retrieved_chunks 等轨迹字段。"""

    question: str
    answer: str


def _answer_node(state: HelloState) -> HelloState:
    """唯一节点：把问题交给 DeepSeek，把回答写回状态。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["question"]},
    ]
    # 注意：这里走 llm.chat 而不是直接建客户端，测试才能在一处打桩。
    return {"answer": llm.chat(messages)}


def build_hello_graph():
    """编译 START -> answer -> END 的最小状态图。"""
    builder = StateGraph(HelloState)
    builder.add_node("answer", _answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def ask(question: str) -> str:
    """便捷入口：问一句，拿回答。"""
    if not question or not question.strip():
        raise ValueError("question 不能为空")
    state = build_hello_graph().invoke({"question": question.strip()})
    return state["answer"]


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 cp936，直接 print 中文会花屏
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    question = " ".join(argv).strip() or "用一句话解释什么是过拟合。"
    print(f"Q: {question}")
    print(f"A: {ask(question)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
