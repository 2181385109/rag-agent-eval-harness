"""被测系统：用 LangGraph 显式状态图实现的 ReAct Agent（CLAUDE.md §5）。

状态图只有两个节点，循环就在它们之间：

    START -> agent --(模型要求调工具)--> tools --+
                |                                |
                +--(模型给出最终答案 / 步数用尽)--> END
                                                 |
                     agent <----------------------+

两条纪律：
  1. **如实记录失败。** 模型选错工具、少给参数、吐出非法 JSON，都原样记进轨迹并
     计入指标，Agent 侧绝不偷偷纠正——这些失败正是评测集要抓的东西。
  2. **循环必须有硬上限。** config.MAX_AGENT_STEPS 限住 LLM 调用次数，
     模型打转时能停下来，一道题不会无限烧 token。

工具调用走 OpenAI 兼容的 function calling 而不是解析文本里的 "Action:"：
工具名和参数是结构化拿到的，"工具调用准确率"这个指标就不会被解析误差污染。

命令行：
    python -m src.agent.graph "什么是 PSI？"
    python -m src.agent.graph --trace "0.25 和 0.1 差多少？"   # 打印完整轨迹
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src import config
from src.agent import llm, tools as tools_mod
from src.agent.trace import AgentTrace, ToolCall, TraceStep

SYSTEM_PROMPT = (
    "你是一个严谨的中文问答助手，可以使用工具。\n"
    "涉及知识库内容、概念定义、项目细节的问题，先调用 retrieve 检索，"
    "再依据检索到的内容作答，不要凭记忆编造。\n"
    "需要算数时调用 calc，不要自己心算。\n"
    "检索不到依据时，明确说不知道。回答简洁准确。"
)


class AgentState(TypedDict, total=False):
    """在状态图节点之间流转的全部状态。"""

    question: str
    messages: list[dict[str, Any]]  # OpenAI 格式的对话历史
    steps: list[TraceStep]
    answer: str
    stop_reason: str
    prompt_tokens: int
    completion_tokens: int


def _assistant_message(content: str, raw_tool_calls) -> dict[str, Any]:
    """把模型返回的 assistant 消息转成可回传的 dict。"""
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if raw_tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in raw_tool_calls
        ]
    return msg


class ReActAgent:
    """被测 Agent。一次 run() 产出一道题的完整轨迹。"""

    def __init__(
        self,
        toolbox: tools_mod.ToolBox,
        model: str | None = None,
        max_steps: int | None = None,
    ):
        self.toolbox = toolbox
        self.model = model or config.MODEL_NAME
        self.max_steps = max_steps or config.MAX_AGENT_STEPS
        self.graph = self._build_graph()

    # ------------------------------------------------------------- 节点
    def _agent_node(self, state: AgentState) -> AgentState:
        """问模型：给最终答案，还是要调工具。"""
        resp = llm.chat_completion(
            state["messages"],
            model=self.model,
            tools=self.toolbox.schemas(),
        )
        message = resp.choices[0].message
        raw_calls = getattr(message, "tool_calls", None) or []
        content = message.content or ""

        step = TraceStep(index=len(state["steps"]), thought=content)
        for raw in raw_calls:
            args, err = tools_mod.parse_arguments(raw.function.arguments)
            step.tool_calls.append(
                ToolCall(
                    id=raw.id,
                    name=raw.function.name,
                    args=args,
                    raw_arguments=raw.function.arguments,
                )
            )
            if err:
                # 参数就没解析出来，工具压根不用执行，直接记一条失败。
                step.results.append(
                    tools_mod.ToolResult(
                        call_id=raw.id, name=raw.function.name, ok=False, error=err
                    )
                )

        usage = getattr(resp, "usage", None)
        return {
            "messages": state["messages"] + [_assistant_message(content, raw_calls)],
            "steps": state["steps"] + [step],
            "answer": "" if raw_calls else content,
            "prompt_tokens": state.get("prompt_tokens", 0) + getattr(usage, "prompt_tokens", 0),
            "completion_tokens": state.get("completion_tokens", 0)
            + getattr(usage, "completion_tokens", 0),
        }

    def _tools_node(self, state: AgentState) -> AgentState:
        """执行模型点名的工具，把观察结果按 tool 角色回传。"""
        step = state["steps"][-1]
        already_failed = {r.call_id for r in step.results}
        messages = list(state["messages"])

        for call in step.tool_calls:
            if call.id in already_failed:
                result = next(r for r in step.results if r.call_id == call.id)
            else:
                result = self.toolbox.run(call.name, call.args, call_id=call.id)
                step.results.append(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result.observation if result.ok else (result.error or "工具执行失败"),
                }
            )

        return {"messages": messages, "steps": state["steps"]}

    def _route(self, state: AgentState) -> str:
        """模型没要求调工具就收工；步数用尽也收工。"""
        last = state["steps"][-1]
        if not last.tool_calls:
            return END
        if len(state["steps"]) >= self.max_steps:
            return END
        return "tools"

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._tools_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", self._route, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")
        # recursion_limit 是 LangGraph 的兜底；真正的上限由 _route 里的 max_steps 决定。
        return builder.compile()

    # -------------------------------------------------------------- 入口
    def run(self, question: str) -> AgentTrace:
        if not question or not question.strip():
            raise ValueError("question 不能为空")
        question = question.strip()

        initial: AgentState = {
            "question": question,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "steps": [],
            "answer": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        final = self.graph.invoke(initial, {"recursion_limit": 2 * self.max_steps + 5})

        steps = final["steps"]
        exhausted = bool(steps) and bool(steps[-1].tool_calls) and len(steps) >= self.max_steps
        return AgentTrace(
            question=question,
            answer=final.get("answer", ""),
            steps=steps,
            stop_reason="max_steps" if exhausted else "answered",
            model=self.model,
            prompt_tokens=final.get("prompt_tokens", 0),
            completion_tokens=final.get("completion_tokens", 0),
        )


def build_agent(
    toolbox: tools_mod.ToolBox, model: str | None = None, max_steps: int | None = None
) -> ReActAgent:
    return ReActAgent(toolbox, model=model, max_steps=max_steps)


def build_default_agent(model: str | None = None) -> ReActAgent:
    """用真实 BGE + FAISS 组一个 Agent；索引已落盘就直接读，没有就现建。"""
    from src.agent import rag

    try:
        retriever = rag.Retriever.load()
    except FileNotFoundError:
        retriever = rag.Retriever.build()
    return build_agent(tools_mod.ToolBox(retriever), model=model)


def ask(question: str) -> str:
    """便捷入口：问一句，拿最终答案（丢弃轨迹）。"""
    return build_default_agent().run(question).answer


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="跑一次被测 Agent")
    parser.add_argument("question", nargs="*", help="要问的问题")
    parser.add_argument("--trace", action="store_true", help="打印完整轨迹")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    question = " ".join(args.question).strip() or "用一句话解释什么是过拟合。"
    trace = build_default_agent().run(question)

    print(f"Q: {trace.question}")
    print(f"A: {trace.answer}")
    print()
    print(
        f"工具序列: {trace.tool_sequence or '（未调用工具）'}   "
        f"停止原因: {trace.stop_reason}   token: {trace.total_tokens}"
    )

    if args.trace:
        for step in trace.steps:
            print(f"{'-' * 60}")
            print(f"step {step.index}  thought: {step.thought[:120]}")
            for call in step.tool_calls:
                print(f"  -> {call.name}({call.args})")
            for result in step.results:
                status = "ok" if result.ok else f"FAILED: {result.error}"
                print(f"  <- {result.name} {status}")
                for hit in result.retrieved:
                    print(f"       [{hit.score:.4f}] {hit.doc_id}  {hit.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
