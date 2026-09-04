"""两个工具的验收口径。

calc 的安全性单独重点测：工具是把模型输出直接喂给解释器的地方，
用 eval 就等于把任意代码执行权交给了 LLM 的输出。
"""
from __future__ import annotations

import pytest

from src.agent import rag, tools


# ------------------------------------------------------------------ calc
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("1+1", "2"),
        ("2 * 3 + 4", "10"),
        ("10 / 4", "2.5"),
        ("2 ** 10", "1024"),
        ("-5 + 3", "-2"),
        ("(1 + 2) * (3 + 4)", "21"),
        ("7 // 2", "3"),
        ("7 % 3", "1"),
        ("abs(-3)", "3"),
        ("round(3.14159, 2)", "3.14"),
        ("max(1, 9, 5)", "9"),
        ("sqrt(16)", "4.0"),
    ],
)
def test_calc_arithmetic(expr, expected):
    assert tools.calc(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "[x for x in range(10)]",
        "lambda: 1",
        "os.getenv('DEEPSEEK_API_KEY')",
        "globals()",
        "exec('a=1')",
        "1 if True else 2",
        "print('hi')",
        "a = 1",
    ],
)
def test_calc_rejects_non_arithmetic(expr):
    """白名单之外的一切 AST 节点都必须被拒——包括看起来无害的推导式和赋值。"""
    with pytest.raises(ValueError):
        tools.calc(expr)


def test_calc_rejects_huge_exponent():
    """9**9**9 会把进程卡死，属于拒绝服务，必须在算之前就拦下。"""
    with pytest.raises(ValueError):
        tools.calc("9 ** 9 ** 9")


def test_calc_rejects_division_by_zero_gracefully():
    with pytest.raises(ValueError):
        tools.calc("1 / 0")


def test_calc_rejects_empty():
    with pytest.raises(ValueError):
        tools.calc("   ")


def test_calc_never_leaks_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-never-appear")
    with pytest.raises(ValueError) as exc:
        tools.calc("os.environ['DEEPSEEK_API_KEY']")
    assert "sk-" not in str(exc.value)


# --------------------------------------------------------------- ToolBox
def test_toolbox_exposes_exactly_two_tools(tiny_corpus, fake_embedder):
    box = tools.ToolBox(rag.Retriever.build(tiny_corpus, embedder=fake_embedder))
    names = {s["function"]["name"] for s in box.schemas()}
    # CLAUDE.md §5：必须是两个工具，单工具没法评"工具调用准确率"
    assert names == {"retrieve", "calc"}


def test_tool_schemas_are_openai_shaped(tiny_corpus, fake_embedder):
    box = tools.ToolBox(rag.Retriever.build(tiny_corpus, embedder=fake_embedder))
    for schema in box.schemas():
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert fn["parameters"]["required"]


def test_run_retrieve_returns_chunks(tiny_corpus, fake_embedder):
    box = tools.ToolBox(rag.Retriever.build(tiny_corpus, embedder=fake_embedder))
    result = box.run("retrieve", {"query": "GroupKFold 防止数据泄漏"})
    assert result.ok
    assert result.retrieved, "retrieve 必须把命中的 chunk 记进轨迹，供 recall@k 使用"
    assert result.retrieved[0].doc_id
    assert result.observation.strip()


def test_run_calc_returns_observation(tiny_corpus, fake_embedder):
    box = tools.ToolBox(rag.Retriever.build(tiny_corpus, embedder=fake_embedder))
    result = box.run("calc", {"expression": "12 * 12"})
    assert result.ok and result.observation == "144"
    assert result.retrieved == []


def test_unknown_tool_is_recorded_not_raised(tiny_corpus, fake_embedder):
    """模型选了不存在的工具是一种真实失败，要如实记录、计入指标，不能崩掉整轮。"""
    box = tools.ToolBox(rag.Retriever.build(tiny_corpus, embedder=fake_embedder))
    result = box.run("browse_web", {"url": "x"})
    assert result.ok is False
    assert result.error


def test_missing_argument_is_recorded_not_raised(tiny_corpus, fake_embedder):
    box = tools.ToolBox(rag.Retriever.build(tiny_corpus, embedder=fake_embedder))
    result = box.run("calc", {})
    assert result.ok is False
    assert result.error


def test_retrieve_without_retriever_fails_cleanly():
    box = tools.ToolBox(retriever=None)
    result = box.run("retrieve", {"query": "任意"})
    assert result.ok is False
