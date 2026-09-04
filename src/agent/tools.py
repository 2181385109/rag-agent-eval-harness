"""Agent 的两个工具（CLAUDE.md §5）。

必须是两个：只有一个工具的话，"工具调用准确率"这个指标无从谈起——
模型没得选，也就没有"选错"这种失败可抓。

  retrieve(query)      走 RAG 检索 corpus/
  calc(expression)     算术求值，制造"该用哪个工具"的判定点

安全说明：calc 绝不用 eval/exec。工具是把**模型输出**直接喂给解释器的地方，
用 eval 等于把任意代码执行权交给 LLM 的输出——语料里一句提示词注入就能拿到
os.environ 里的 API key。这里改用 AST 白名单：只放行算术节点，其余一律拒绝。
"""

from __future__ import annotations

import ast
import json
import math
import operator
from typing import Any, Callable

from src import config
from src.agent.trace import ToolResult

# ---------------------------------------------------------------- calc 白名单
_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": math.pow,
}

# 幂运算是唯一能用极短表达式把进程算死的地方（9**9**9），单独设限。
MAX_EXPONENT = 64
MAX_EXPRESSION_LENGTH = 200


def calc(expression: str) -> str:
    """对一个纯算术表达式求值，返回字符串结果。

    非算术的一切（属性访问、下标、推导式、赋值、lambda、函数调用白名单之外的名字）
    都会抛 ValueError。
    """
    if not expression or not expression.strip():
        raise ValueError("表达式不能为空")
    expression = expression.strip()
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"表达式过长（上限 {MAX_EXPRESSION_LENGTH} 字符）")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误：{exc.msg}") from exc

    try:
        value = _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ValueError("除零") from exc
    except (TypeError, OverflowError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"求值失败：{exc}") from exc

    return _format(value)


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只允许数字常量")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不允许的运算符：{type(node.op).__name__}")
        if isinstance(node.op, ast.Pow):
            _check_exponent(node.right)
        return op(_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不允许的一元运算符：{type(node.op).__name__}")
        return op(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("只允许调用白名单里的数学函数")
        if node.keywords:
            raise ValueError("不允许关键字参数")
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])

    raise ValueError(f"不允许的表达式成分：{type(node).__name__}")


def _check_exponent(right: ast.AST) -> None:
    """指数必须是字面常量且不超过上限。

    这条同时挡下了 9 ** 9 ** 9：它的外层指数是一个 BinOp 而不是常量，
    在算之前就被拒掉，不会把进程卡死。
    """
    if not isinstance(right, ast.Constant) or isinstance(right.value, bool):
        raise ValueError("指数必须是数字常量")
    if not isinstance(right.value, (int, float)) or abs(right.value) > MAX_EXPONENT:
        raise ValueError(f"指数超出上限 {MAX_EXPONENT}")


def _format(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("结果不是数字")
    if isinstance(value, int):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError("结果不是有限数字")
    # 收掉浮点噪声：0.25 - 0.1 不该报成 0.15000000000000002
    return str(round(value, 10))


# ----------------------------------------------------------------- 工具箱
RETRIEVE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "从本地知识库检索相关段落。回答任何涉及文档内容、概念定义、"
            "项目细节的问题时都应先调用它，不要凭记忆作答。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索用的查询语句"},
            },
            "required": ["query"],
        },
    },
}

CALC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": (
            "计算一个算术表达式，支持 + - * / // % **、括号，以及 "
            "abs/round/min/max/sum/sqrt/log/exp/floor/ceil。只做数学，不查资料。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "例如 (0.25 - 0.1) * 100"},
            },
            "required": ["expression"],
        },
    },
}


class ToolBox:
    """把工具的 schema、执行、结果记录三件事收在一起。

    执行永远不抛异常：模型选错工具、少给参数、给了非法 JSON，都是真实的失败样本，
    要如实变成一条 ok=False 的 ToolResult 计入轨迹和指标，而不是让整轮崩掉，
    更不能在 Agent 侧偷偷替模型纠正（CLAUDE.md §5）。
    """

    def __init__(self, retriever=None, top_k: int | None = None):
        self.retriever = retriever
        self.top_k = top_k or config.RETRIEVE_TOP_K

    def schemas(self) -> list[dict]:
        return [RETRIEVE_SCHEMA, CALC_SCHEMA]

    def run(self, name: str, args: dict[str, Any], call_id: str = "") -> ToolResult:
        try:
            if name == "retrieve":
                return self._run_retrieve(args, call_id)
            if name == "calc":
                return self._run_calc(args, call_id)
            return ToolResult(
                call_id=call_id,
                name=name,
                ok=False,
                error=f"未知工具：{name}。可用工具：retrieve, calc",
            )
        except Exception as exc:  # 兜底：工具层绝不把异常抛回 Agent 循环
            return ToolResult(call_id=call_id, name=name, ok=False, error=str(exc))

    def _run_retrieve(self, args: dict[str, Any], call_id: str) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                call_id=call_id, name="retrieve", ok=False, error="缺少必需参数 query（字符串）"
            )
        if self.retriever is None:
            return ToolResult(
                call_id=call_id, name="retrieve", ok=False, error="检索器未初始化，无法检索"
            )

        hits = self.retriever.retrieve(query, k=self.top_k)
        blocks = [f"[{h.doc_id} | {h.source}]{chr(10)}{h.text}" for h in hits]
        return ToolResult(
            call_id=call_id,
            name="retrieve",
            ok=True,
            observation=(chr(10) * 2).join(blocks) if blocks else "（没有检索到相关内容）",
            retrieved=hits,
        )

    def _run_calc(self, args: dict[str, Any], call_id: str) -> ToolResult:
        expression = args.get("expression")
        if not isinstance(expression, str):
            return ToolResult(
                call_id=call_id, name="calc", ok=False, error="缺少必需参数 expression（字符串）"
            )
        try:
            return ToolResult(
                call_id=call_id, name="calc", ok=True, observation=calc(expression)
            )
        except ValueError as exc:
            return ToolResult(call_id=call_id, name="calc", ok=False, error=str(exc))


def parse_arguments(raw: str) -> tuple[dict[str, Any], str | None]:
    """解析模型吐出的工具参数串。

    返回 (参数字典, 错误说明)。模型确实会吐出非法 JSON，这是要计入指标的失败，
    所以这里不抛异常，把错误原样带出去。
    """
    if not raw or not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"工具参数不是合法 JSON：{exc.msg}"
    if not isinstance(parsed, dict):
        return {}, "工具参数必须是 JSON 对象"
    return parsed, None
