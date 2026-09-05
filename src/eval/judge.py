"""LLM-as-Judge 与人工标注对照（CLAUDE.md §9 M5）。

本模块的第一步产物是**人工标注表**：闭合题已由规则判定，开放题需要人来打分，
再和自动裁判对照算 Cohen's kappa。

一条设计纪律：**标注表绝不写入自动裁判的结论**。人先看到机器打的分再标，
kappa 度量的就不是"人机是否独立地看法一致"，而是确认偏误。所以
build_labeling_sheet 显式忽略任何传进来的裁判分数。

命令行：
    python -m src.eval.judge --sheet     # 生成待人工标注的表 + 空白骨架
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from src import config
from src.agent.trace import AgentTrace
from src.eval.datasets import GoldenSample, load_golden_set

# 三级有序标度。为什么不用二值：开放题常见"方向对但要点有遗漏"，
# 二值会把它硬塞进对或错，人机分歧被人为放大。
LABEL_SCALE: dict[int, str] = {
    0: "错误 / 编造 / 答非所问：与参考答案矛盾，或凭空捏造了知识库里没有的内容",
    1: "部分正确：方向对、无编造，但要点有遗漏或有轻微不准确",
    2: "正确且有据：要点齐全，结论与参考答案一致，且未编造",
}

SHEET_FILENAME = "human_labeling_sheet.md"


def _open_samples(samples: Sequence[GoldenSample]) -> list[GoldenSample]:
    """只有开放题需要人工标注；闭合题已由 answer_keys 规则判定。"""
    return [s for s in samples if s.answer_type == "open"]


def build_labeling_sheet(
    samples: Sequence[GoldenSample],
    traces: Mapping[str, AgentTrace],
    judge_scores: Mapping[str, int] | None = None,  # 刻意忽略，见模块 docstring
) -> str:
    """生成给人看的标注表（Markdown）。"""
    del judge_scores  # 绝不写进表里：人必须独立于机器判定

    targets = _open_samples(samples)
    lines = [
        "# 人工标注表（开放题）",
        "",
        f"共 **{len(targets)}** 道开放题。闭合题已由 `answer_keys` 规则判定，不需要人工打分。",
        "",
        "## 怎么标",
        "",
        "给每道题的 **Agent 答案**打一个分，判据如下：",
        "",
        "| 分数 | 判据 |",
        "|---|---|",
    ]
    for level, description in LABEL_SCALE.items():
        lines.append(f"| **{level}** | {description} |")

    lines += [
        "",
        "几点提醒：",
        "",
        "- 判的是 **Agent 答案**相对 **参考答案** 的质量，不是判问题出得好不好。",
        "- 标了 `hallucination_bait` 的题，**正确行为是明说「知识库里没有」**。"
        "如果 Agent 编出了一个看似合理的答案，那是 0 分，不是 2 分。",
        "- 请**先标完再看自动裁判的结果**。先看机器分再标，"
        "kappa 度量的就不是人机独立看法是否一致，而是确认偏误。",
        f"- 样本量只有 {len(targets)} 条，算出来的 kappa 会很不稳定"
        "（置信区间宽），报告里会如实标注这一点，不要把它当硬结论。",
        "",
        "标完把分数告诉我（形如 `cap_007=2, cap_008=1, ...`），"
        f"或直接编辑 `{config.HUMAN_LABELS_PATH.name}` 里的 `human_score` 字段。",
        "",
        "---",
        "",
        "## 逐题",
        "",
    ]

    for index, sample in enumerate(targets, start=1):
        trace = traces.get(sample.id)
        answer = (trace.answer if trace else "") or "（本轮没有产出答案）"
        sources = []
        if trace:
            seen: dict[str, None] = {}
            for chunk in trace.retrieved_chunks:
                seen.setdefault(f"{chunk.doc_id}  {chunk.source}", None)
            sources = list(seen)

        lines += [
            f"### {index}. `{sample.id}`"
            + (f"  ⚠ `{sample.failure_tag}`" if sample.failure_tag else ""),
            "",
            f"**问题**：{sample.question}",
            "",
            "**参考答案**：",
            "",
            f"> {sample.reference_answer}",
            "",
            "**Agent 答案**：",
            "",
        ]
        lines += [f"> {line}" if line.strip() else ">" for line in answer.splitlines()]
        lines += [
            "",
            f"<details><summary>检索到的片段（{len(sources)} 块）</summary>",
            "",
        ]
        lines += [f"- `{item}`" for item in sources] or ["- （未检索）"]
        lines += [
            "",
            "</details>",
            "",
            "**我的评分**： `___`   （0 / 1 / 2）",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


def write_labeling_sheet(
    samples: Sequence[GoldenSample],
    traces: Mapping[str, AgentTrace],
    path: Path | None = None,
) -> Path:
    target = path or (config.REPORTS_DIR / SHEET_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_labeling_sheet(samples, traces), encoding="utf-8")
    return target


def write_label_skeleton(
    samples: Sequence[GoldenSample],
    traces: Mapping[str, AgentTrace],
    path: Path | None = None,
) -> Path:
    """生成 human_labels.jsonl 骨架，human_score 留空等人填。

    已经填过的分数不会被覆盖——重新生成骨架不应该把人的劳动成果抹掉。
    """
    target = path or config.HUMAN_LABELS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, int | None] = {}
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["id"]] = row.get("human_score")

    with target.open("w", encoding="utf-8") as fh:
        for sample in _open_samples(samples):
            trace = traces.get(sample.id)
            payload = {
                "id": sample.id,
                "question": sample.question,
                "reference_answer": sample.reference_answer,
                "answer": (trace.answer if trace else "") or "",
                "failure_tag": sample.failure_tag,
                "human_score": existing.get(sample.id),
                "note": "",
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target


def load_human_labels(path: Path | None = None) -> dict[str, int]:
    """读回已标注的分数，未标注的（human_score 为 null）跳过。"""
    target = Path(path) if path is not None else config.HUMAN_LABELS_PATH
    if not target.exists():
        raise FileNotFoundError(f"人工标注文件不存在：{target}")

    labels: dict[str, int] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score = row.get("human_score")
        if score is None:
            continue
        if score not in LABEL_SCALE:
            raise ValueError(f"{row['id']} 的 human_score={score} 非法，只能是 0/1/2")
        labels[row["id"]] = int(score)
    return labels


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="裁判层与人工标注")
    parser.add_argument("--sheet", action="store_true", help="生成人工标注表与空白骨架")
    parser.add_argument(
        "--traces",
        type=str,
        default=None,
        help="轨迹文件（默认 reports/traces_latest.jsonl）",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    from src.eval.report import TRACES_FILENAME, load_traces

    if args.sheet:
        samples = load_golden_set()
        traces = load_traces(args.traces or (config.REPORTS_DIR / TRACES_FILENAME))
        sheet = write_labeling_sheet(samples, traces)
        skeleton = write_label_skeleton(samples, traces)
        print(f"标注表：{sheet}")
        print(f"空白骨架：{skeleton}")
        print(f"待标注开放题：{len(_open_samples(samples))} 条")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
