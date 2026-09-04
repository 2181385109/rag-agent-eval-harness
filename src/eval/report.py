"""跑评测并出报告（CLAUDE.md §6 / §9 M3）。

职责分工：
  datasets.py  黄金集 schema 与加载
  metrics.py   纯计算，不碰 Agent
  report.py    编排（跑 Agent 拿轨迹）+ 汇总 + 落盘

报告里记的不只是指标，还有**复现所需的全部上下文**：模型名、embedding 模型、
top-k、切分参数、黄金集条数、git commit。没有这些，一个数字过两周就没法复现，
也就违背了本项目的第一红线。

命令行：
    python -m src.eval.report --limit 5      # 先用小子集跑通（省钱，§10）
    python -m src.eval.report                # 全量，含多轮一致性
    python -m src.eval.report --no-consistency
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src import config
from src.agent.trace import AgentTrace
from src.eval import metrics
from src.eval.datasets import GoldenSample, load_golden_set


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_samples(
    agent, samples: Sequence[GoldenSample], repeats: int = 1, progress: bool = True
) -> dict[str, list[AgentTrace]]:
    """对每条样本跑 repeats 次，返回 id -> 轨迹列表。

    单条跑挂了不整轮中断：记一条 stop_reason="error" 的空轨迹，
    让它如实计入指标（失败就是失败），并把错误写进 answer 便于复盘。
    """
    results: dict[str, list[AgentTrace]] = {}
    total = len(samples) * repeats
    done = 0
    for sample in samples:
        runs: list[AgentTrace] = []
        for _ in range(repeats):
            try:
                runs.append(agent.run(sample.question))
            except Exception as exc:  # 网络抖动、超时等
                runs.append(
                    AgentTrace(
                        question=sample.question,
                        answer=f"[运行失败] {exc}",
                        stop_reason="error",
                        model=config.MODEL_NAME,
                    )
                )
            done += 1
            if progress:
                print(f"  [{done}/{total}] {sample.id}", file=sys.stderr, flush=True)
        results[sample.id] = runs
    return results


def build_report(
    samples: Sequence[GoldenSample],
    traces: dict[str, AgentTrace],
    consistency_runs: dict[str, list[AgentTrace]] | None = None,
    notes: str = "",
) -> dict:
    """把轨迹汇总成一份完整报告（纯函数，可在 CI 里用构造轨迹验证）。"""
    recall = metrics.aggregate_recall(samples, traces)
    recall_first = metrics.aggregate_recall(samples, traces, first_call_only=True)
    success = metrics.task_success_rate(samples, traces)
    tool = metrics.tool_accuracy(samples, traces)
    tool_set = metrics.tool_set_accuracy(samples, traces)

    consistency = None
    if consistency_runs:
        subset = [s for s in samples if s.id in consistency_runs]
        consistency = metrics.consistency(subset, consistency_runs).as_dict()

    prompt_tokens = sum(t.prompt_tokens for t in traces.values())
    completion_tokens = sum(t.completion_tokens for t in traces.values())

    per_question = []
    for s in samples:
        t = traces.get(s.id)
        per_question.append(
            {
                "id": s.id,
                "answer_type": s.answer_type,
                "failure_tag": s.failure_tag,
                "expected_tool": s.expected_tool,
                "actual_tool": metrics.collapse_repeats(t.tool_sequence) if t else None,
                "tool_ok": tool.detail.get(s.id),
                "expected_doc_ids": s.expected_doc_ids,
                "retrieved_doc_ids": t.retrieved_doc_ids if t else None,
                "recall_at_k": recall.detail.get(s.id),
                "recall_at_k_first_call": recall_first.detail.get(s.id),
                "success": success.detail.get(s.id),
                "stop_reason": t.stop_reason if t else None,
                "answer": t.answer if t else None,
            }
        )

    return {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "model": config.MODEL_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "use_query_instruction": config.USE_QUERY_INSTRUCTION,
            "retrieve_top_k": config.RETRIEVE_TOP_K,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "max_agent_steps": config.MAX_AGENT_STEPS,
            "temperature": config.DEFAULT_TEMPERATURE,
            "golden_set_size": len(samples),
            "notes": notes,
        },
        "metrics": {
            "recall_at_k": recall.as_dict(),
            "recall_at_k_first_call": recall_first.as_dict(),
            "task_success_rate": success.as_dict(),
            "tool_accuracy": tool.as_dict(),
            "tool_set_accuracy": tool_set.as_dict(),
            "consistency": consistency,
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "per_question": per_question,
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_markdown(report: dict) -> str:
    """人看的版本。每个指标都带分母，这是硬要求。"""
    meta, m = report["meta"], report["metrics"]
    lines = [
        "# 评测报告",
        "",
        f"- 时间（UTC）：{meta['timestamp_utc']}",
        f"- git commit：`{meta['git_commit']}`",
        f"- 被测模型：`{meta['model']}`（temperature={meta['temperature']}）",
        f"- Embedding：`{meta['embedding_model']}`（查询指令前缀={meta['use_query_instruction']}）",
        f"- 检索 top-k：{meta['retrieve_top_k']}；切分 {meta['chunk_size']}/{meta['chunk_overlap']}",
        f"- 黄金集：{meta['golden_set_size']} 条",
        "",
        "## 指标",
        "",
        "| 指标 | 值 | 分母 | 说明 |",
        "|---|---|---|---|",
    ]

    rows = [
        ("检索召回率 recall@k", "recall_at_k", "全部检索结果的并集"),
        ("recall@k（仅首次检索）", "recall_at_k_first_call", "只算第一次检索；与上一行的差＝多次检索捞回了多少"),
        ("任务成功率", "task_success_rate", "闭合题规则判定；开放题待 M5 裁判"),
        ("工具调用准确率（严格）", "tool_accuracy", "折叠连续重复后逐项比对"),
        ("工具调用准确率（宽松）", "tool_set_accuracy", "只看用了哪些工具，不看顺序"),
    ]
    for label, key, note in rows:
        item = m[key]
        lines.append(
            f"| {label} | {_fmt(item['value'])} | n={item['n']}/{item['total']} | {note} |"
        )

    c = m.get("consistency")
    if c:
        lines += [
            "",
            "## 多轮一致性",
            "",
            f"- 同题重复次数 K = {c['runs']}，覆盖 {c['n_questions']} 题",
            f"- 判定一致比例：{_fmt(c['success_agreement'])}（1.0 表示每题 K 次判定完全一致）",
            f"- 判定结果方差：{_fmt(c['success_variance'])}（同题 K 次 0/1 判定的方差，按题平均）",
            f"- 工具路径一致比例：{_fmt(c['tool_agreement'])}",
        ]

    tok = report["tokens"]
    lines += [
        "",
        f"## Token 消耗",
        "",
        f"- prompt {tok['prompt']} + completion {tok['completion']} = **{tok['total']}**",
        "",
        "## 逐题明细",
        "",
        "| id | 题型 | 期望工具 | 实际工具 | 工具 | recall | 成功 | 停止原因 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for q in report["per_question"]:
        def mark(v):
            return "—" if v is None else ("✔" if v else "✘")

        lines.append(
            f"| {q['id']} | {q['answer_type']} | {'+'.join(q['expected_tool'])} | "
            f"{'+'.join(q['actual_tool']) if q['actual_tool'] else '（无）'} | {mark(q['tool_ok'])} | "
            f"{_fmt(q['recall_at_k'])} | {mark(q['success'])} | {q['stop_reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict, directory: Path | None = None) -> dict[str, Path]:
    """落盘：时间戳快照 + latest.json + report.md。"""
    target = directory or config.REPORTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    # 去掉 : - 和时区偏移里的 +，文件名对 URL / shell 友好
    stamp = report["meta"]["timestamp_utc"].replace(":", "").replace("-", "")
    stamp = stamp.split("+")[0].rstrip("Z") + "Z"

    snapshot = target / f"eval_{stamp}.json"
    latest = target / "latest.json"
    markdown = target / "report.md"

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    snapshot.write_text(payload, encoding="utf-8")
    latest.write_text(payload, encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    return {"snapshot": snapshot, "latest": latest, "markdown": markdown}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="跑黄金集评测并出报告")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（调试省钱用）")
    parser.add_argument("--no-consistency", action="store_true", help="跳过多轮一致性")
    parser.add_argument(
        "--consistency-runs", type=int, default=config.CONSISTENCY_RUNS, help="同题重复次数 K"
    )
    parser.add_argument(
        "--consistency-size",
        type=int,
        default=config.CONSISTENCY_SUBSET_SIZE,
        help="参与一致性的题数（控制成本）",
    )
    parser.add_argument("--notes", type=str, default="", help="写进报告的备注")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    from src.agent import graph

    samples = load_golden_set()
    if args.limit:
        samples = samples[: args.limit]

    print(f"黄金集 {len(samples)} 条，构建 Agent（加载 BGE + FAISS）...", file=sys.stderr)
    agent = graph.build_default_agent()

    print("主评测：每题跑 1 次", file=sys.stderr)
    main_runs = run_samples(agent, samples, repeats=1)
    traces = {sid: runs[0] for sid, runs in main_runs.items()}

    consistency_runs = None
    if not args.no_consistency and args.consistency_size > 0:
        subset = samples[: args.consistency_size]
        print(
            f"多轮一致性：{len(subset)} 题 × {args.consistency_runs} 次", file=sys.stderr
        )
        consistency_runs = run_samples(agent, subset, repeats=args.consistency_runs)

    report = build_report(samples, traces, consistency_runs, notes=args.notes)
    paths = write_report(report)

    print()
    print(render_markdown(report))
    print()
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
