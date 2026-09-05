"""ReAct 状态图的验收口径。

LLM 全部走 scripted_llm 打桩：这里测的是**循环控制和轨迹记录**是否正确，
而不是模型答得好不好——后者是 M3 之后评测集的事。
"""
from __future__ import annotations

import pytest

from src import config
from src.agent import graph, rag, tools
from tests.helpers import answer_message, malformed_tool_call, tool_call_message


@pytest.fixture
def agent(tiny_corpus, fake_embedder):
    box = tools.ToolBox(rag.Retriever.build(tiny_corpus, embedder=fake_embedder))
    return graph.build_agent(box)


# ------------------------------------------------------------ 基本回合
def test_direct_answer_without_tools(agent, scripted_llm):
    script, calls = scripted_llm
    script(answer_message("过拟合是模型把噪声也学进去了。"))

    trace = agent.run("什么是过拟合？")

    assert trace.answer == "过拟合是模型把噪声也学进去了。"
    assert trace.tool_sequence == []
    assert trace.stop_reason == "answered"
    assert len(calls) == 1


def test_retrieve_then_answer(agent, scripted_llm):
    script, calls = scripted_llm
    script(
        tool_call_message([("retrieve", {"query": "GroupKFold 防数据泄漏"})]),
        answer_message("GroupKFold 保证同组样本不跨折。"),
    )

    trace = agent.run("GroupKFold 是干什么的？")

    assert trace.tool_sequence == ["retrieve"]
    assert trace.retrieved_doc_ids, "检索结果必须进轨迹，否则 recall@k 算不出来"
    assert trace.answer == "GroupKFold 保证同组样本不跨折。"
    assert len(calls) == 2


def test_calc_then_answer(agent, scripted_llm):
    script, _calls = scripted_llm
    script(
        tool_call_message([("calc", {"expression": "0.25 * 4"})]),
        answer_message("结果是 1.0。"),
    )

    trace = agent.run("0.25 乘以 4 等于多少？")

    assert trace.tool_sequence == ["calc"]
    assert trace.steps[0].results[0].observation == "1.0"


def test_two_tools_in_sequence(agent, scripted_llm):
    script, _calls = scripted_llm
    script(
        tool_call_message([("retrieve", {"query": "PSI 阈值"})]),
        tool_call_message([("calc", {"expression": "0.25 - 0.1"})]),
        answer_message("差值 0.15。"),
    )

    trace = agent.run("PSI 的两个经验阈值差多少？")
    assert trace.tool_sequence == ["retrieve", "calc"]


# ------------------------------------------------------- 循环控制与失败
def test_max_steps_stops_the_loop(agent, scripted_llm):
    """模型一直要求调工具时必须能停下来，否则一道题能无限烧 token。"""
    script, calls = scripted_llm
    script(*[tool_call_message([("calc", {"expression": "1+1"})])] * (config.MAX_AGENT_STEPS + 5))

    trace = agent.run("永远不结束的问题")

    assert trace.stop_reason == "max_steps"
    assert len(trace.steps) <= config.MAX_AGENT_STEPS
    assert len(calls) <= config.MAX_AGENT_STEPS


def test_malformed_tool_arguments_are_recorded_and_loop_continues(agent, scripted_llm):
    """模型吐出非法 JSON 参数是真实失败，如实记录、继续跑，不许在 Agent 侧偷偷修正。"""
    script, _calls = scripted_llm
    script(
        malformed_tool_call("calc", "{这不是合法 JSON"),
        answer_message("抱歉，我算不了。"),
    )

    trace = agent.run("算个数")

    assert trace.tool_sequence == ["calc"]
    assert trace.steps[0].results[0].ok is False
    assert trace.has_failure is True
    assert trace.answer == "抱歉，我算不了。"


def test_wrong_tool_choice_is_not_corrected(agent, scripted_llm):
    """选错工具要原样留在轨迹里——这正是评测集要抓的失败。"""
    script, _calls = scripted_llm
    script(
        tool_call_message([("calc", {"expression": "1+1"})]),
        answer_message("好了。"),
    )
    trace = agent.run("什么是 PSI？")   # 明明该 retrieve，模型却用了 calc
    assert trace.tool_sequence == ["calc"]


# ------------------------------------------------------------ 协议细节
def test_tool_schemas_are_sent_to_the_model(agent, scripted_llm):
    script, calls = scripted_llm
    script(answer_message("答"))
    agent.run("随便问")
    sent = calls[0]["kwargs"].get("tools")
    assert sent, "必须把工具 schema 传给模型，否则 function calling 无从谈起"
    assert {s["function"]["name"] for s in sent} == {"retrieve", "calc"}


def test_tool_results_go_back_as_tool_role_messages(agent, scripted_llm):
    script, calls = scripted_llm
    script(
        tool_call_message([("calc", {"expression": "2+2"})]),
        answer_message("4"),
    )
    agent.run("2+2")

    second_turn = calls[1]["messages"]
    tool_msgs = [m for m in second_turn if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_0"
    assert tool_msgs[0]["content"] == "4"


def test_token_usage_is_accumulated(agent, scripted_llm):
    script, _calls = scripted_llm
    script(
        tool_call_message([("calc", {"expression": "1+1"})]),
        answer_message("2"),
    )
    trace = agent.run("1+1")
    # helpers 里每条假响应记 10 + 5，两轮就是 30
    assert trace.total_tokens == 30


def test_question_is_required(agent):
    with pytest.raises(ValueError):
        agent.run("   ")


def test_trace_is_serializable(agent, scripted_llm):
    script, _calls = scripted_llm
    script(
        tool_call_message([("retrieve", {"query": "过拟合"})]),
        answer_message("答案"),
    )
    trace = agent.run("什么是过拟合？")
    assert trace.model_dump_json()


# ------------------------------------------------- 温度（多轮一致性专测用）
def test_agent_defaults_to_config_temperature(agent, scripted_llm):
    """主评测锁 temp=0 保证可复现——默认不传就是走 config。"""
    script, calls = scripted_llm
    script(answer_message("答"))
    agent.run("问题")
    assert calls[0]["kwargs"].get("temperature") in (None, config.DEFAULT_TEMPERATURE)


def test_agent_temperature_is_passed_through(tiny_corpus, fake_embedder, scripted_llm):
    """一致性专测要在 temp>0 下跑，温度必须真的传到模型调用上。"""
    script, calls = scripted_llm
    script(answer_message("答"))

    box = tools.ToolBox(rag.Retriever.build(tiny_corpus, embedder=fake_embedder))
    graph.build_agent(box, temperature=0.7).run("问题")

    assert calls[0]["kwargs"]["temperature"] == 0.7
