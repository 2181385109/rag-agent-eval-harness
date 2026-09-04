"""黄金集的 schema 与加载器（CLAUDE.md §6）。

schema 在 M3 有三处修订，理由见 CLAUDE.md §6 的"schema 修订记录"：
  1. expected_tool 从单字符串改为**有序列表**（多工具题需要表达顺序）；
  2. 新增 answer_keys，闭合题规则判定所需的必现片段；
  3. expected_doc_ids 允许为空（纯算术题、幻觉诱饵题），这类题不参与 recall@k。

校验放在 schema 层而不是靠人眼：标注一旦写错（标了不存在的工具、闭合题忘了
写 answer_keys），加载时就直接报错，不会带着坏标注跑出一份看着挺像样的指标。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src import config

# Agent 实际挂载的工具（src/agent/tools.py）。标注超出这个集合即为标注错误。
KNOWN_TOOLS = ("retrieve", "calc")

AnswerType = Literal["closed", "open"]


class GoldenSample(BaseModel):
    """黄金集里的一条。"""

    id: str
    question: str
    reference_answer: str
    answer_keys: list[str] = Field(default_factory=list)
    expected_tool: list[str]
    expected_doc_ids: list[str] = Field(default_factory=list)
    answer_type: AnswerType
    failure_tag: str | None = None

    @field_validator("expected_tool")
    @classmethod
    def _validate_tools(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("expected_tool 不能为空：不标注就无法判定工具选得对不对")
        unknown = [t for t in value if t not in KNOWN_TOOLS]
        if unknown:
            raise ValueError(f"标注了 Agent 没有挂载的工具 {unknown}，可用：{list(KNOWN_TOOLS)}")
        return value

    @model_validator(mode="after")
    def _validate_answer_keys(self) -> "GoldenSample":
        if self.answer_type == "closed" and not self.answer_keys:
            raise ValueError(
                f"{self.id}: 闭合题走规则判定，必须给 answer_keys，否则无从判定对错"
            )
        if self.answer_type == "open" and self.answer_keys:
            raise ValueError(
                f"{self.id}: 开放题交 M5 裁判层判定，不应再塞 answer_keys，"
                "两套口径并存会让指标含义模糊"
            )
        return self

    @property
    def is_scored_for_recall(self) -> bool:
        """是否参与 recall@k（有标注应检索文档的题才参与）。"""
        return bool(self.expected_doc_ids)

    @property
    def is_rule_judged(self) -> bool:
        """是否参与规则判定的任务成功率（M3 只能自动判闭合题）。"""
        return self.answer_type == "closed"


def load_golden_set(path: Path | str | None = None) -> list[GoldenSample]:
    """从 JSONL 读取黄金集，逐行校验，id 重复即报错。"""
    target = Path(path) if path is not None else config.GOLDEN_SET_PATH
    if not target.exists():
        raise FileNotFoundError(f"黄金集不存在：{target}")

    samples: list[GoldenSample] = []
    seen: set[str] = set()
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{lineno} 不是合法 JSON：{exc.msg}") from exc
        sample = GoldenSample(**payload)
        if sample.id in seen:
            raise ValueError(f"{target}:{lineno} id 重复：{sample.id}")
        seen.add(sample.id)
        samples.append(sample)

    if not samples:
        raise ValueError(f"{target} 里没有任何样本")
    return samples
