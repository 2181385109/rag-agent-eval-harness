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
from typing import Mapping, Sequence

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
    consistency_runs_hot: dict[str, list[AgentTrace]] | None = None,
    consistency_temperature: float | None = None,
    ragas: dict | None = None,
    ragas_baseline: dict | None = None,
    notes: str = "",
) -> dict:
    """把轨迹汇总成一份完整报告（纯函数，可在 CI 里用构造轨迹验证）。"""
    recall = metrics.aggregate_recall(samples, traces)
    recall_first = metrics.aggregate_recall(samples, traces, first_call_only=True)
    success = metrics.task_success_rate(samples, traces)
    tool = metrics.tool_accuracy(samples, traces)
    tool_set = metrics.tool_set_accuracy(samples, traces)

    def _consistency(runs, temperature):
        if not runs:
            return None
        subset = [s for s in samples if s.id in runs]
        payload = metrics.consistency(subset, runs).as_dict()
        payload["temperature"] = temperature
        return payload

    # 两档并存：temp=0 是主评测的可复现基线，temp>0 才测得出鲁棒性。
    consistency = _consistency(consistency_runs, config.DEFAULT_TEMPERATURE)
    consistency_hot = _consistency(consistency_runs_hot, consistency_temperature)

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

    # 已发现的失败模式：跑满步数没收口、或最终答案为空。
    # 这类样本进不了 M4 的 faithfulness/relevancy 均值——没有答案就无从评价答案，
    # 混进去会让均值失去意义。单独列出，如实说明。
    anomalies = []
    for s_ in samples:
        t = traces.get(s_.id)
        if t is None:
            continue
        empty = not (t.answer or "").strip()
        if t.stop_reason != "answered" or empty:
            anomalies.append(
                {
                    "id": s_.id,
                    "failure_tag": s_.failure_tag,
                    "stop_reason": t.stop_reason,
                    "empty_answer": empty,
                    "n_tool_calls": len(t.tool_sequence),
                    "exclude_from_generation_metrics": empty,
                    "note": (
                        "跑满 MAX_AGENT_STEPS 仍未给出答案"
                        if t.stop_reason == "max_steps" and empty
                        else ("最终答案为空" if empty else f"非正常结束：{t.stop_reason}")
                    ),
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
            "consistency_hot": consistency_hot,
            "ragas": ragas,
            "ragas_baseline": ragas_baseline,
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "anomalies": anomalies,
        "per_question": per_question,
    }


def load_ragas_baseline(path: Path | str) -> dict | None:
    """从上一份报告 JSON 里取出 RAGAS 段，用作换裁判前的对照基线。

    只取 judge_model 与 scores：对照的意义在于"同一批轨迹换个裁判差多少"，
    分母与排除清单由当前这次运行自己负责。
    """
    target = Path(path)
    if not target.exists():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    block = (payload.get("metrics") or {}).get("ragas")
    if not block:
        return None
    return {"judge_model": block.get("judge_model"), "scores": block.get("scores") or {}}


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

    cold, hot = m.get("consistency"), m.get("consistency_hot")
    if cold or hot:
        lines += [
            "",
            "## 多轮一致性",
            "",
            "两档并存，互不替代：`temp=0` 是主评测的可复现基线（近乎确定，一致比例天然接近 1.0，"
            "信息量有限）；`temp>0` 才测得出模型在采样噪声下的鲁棒性。",
            "",
            "| 档位 | temperature | K | 覆盖题数 | 判定一致比例 | 判定结果方差 | 工具路径一致比例 |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, c in (("可复现基线", cold), ("鲁棒性专测", hot)):
            if not c:
                continue
            lines.append(
                f"| {label} | {c.get('temperature')} | {c['runs']} | {c['n_questions']} | "
                f"{_fmt(c['success_agreement'])} | {_fmt(c['success_variance'])} | "
                f"{_fmt(c['tool_agreement'])} |"
            )
        lines.append("")
        lines.append("> 判定一致比例 1.0 表示每题 K 次判定完全一致；方差是同题 K 次 0/1 判定的方差，按题平均。")

    rg = m.get("ragas")
    if rg:
        lines += [
            "",
            "## RAGAS（生成质量）",
            "",
            f"- 裁判模型：`{rg.get('judge_model')}`；embedding：`{rg.get('embedding_model')}`（本地，不外发）",
            f"- 提交评测：**{rg.get('n_submitted')}/{rg.get('total')}** 条"
            f"（排除 {rg.get('n_excluded')} 条：无答案或无检索内容，见下节）",
            "",
            "| 指标 | 值 | 实际打分行数 |",
            "|---|---|---|",
        ]
        counts = rg.get("scored_counts") or {}
        submitted = rg.get("n_submitted") or 0
        for name, value in (rg.get("scores") or {}).items():
            got = counts.get(name)
            flag = "" if got is None or got == submitted else "  ⚠"
            lines.append(f"| {name} | {_fmt(value)} | {got}/{submitted}{flag} |")
        if rg.get("has_incomplete_metric"):
            lines += [
                "",
                "> ⚠ **有指标未在全部提交行上打出分**（裁判调用失败或超时，RAGAS 会把该行留空，"
                "而均值默认跳过空值）。带 ⚠ 的指标覆盖面小于分母，不能当作全量结果引用。",
            ]
        excluded = rg.get("excluded") or []
        if excluded:
            lines.append("")
            lines.append("被排除的样本：" + "、".join(
                f"`{e['id']}`（{e.get('reason', '')}）" for e in excluded
            ))

    base = m.get("ragas_baseline")
    if rg and base and base.get("scores"):
        lines += [
            "",
            "### 裁判敏感性对照",
            "",
            f"同一批轨迹，换裁判模型重判一次："
            f"`{base.get('judge_model')}` → `{rg.get('judge_model')}`。"
            "差值反映的是**评分标准本身有多依赖裁判模型**，与被测 Agent 无关。",
            "",
            "| 指标 | " + f"{base.get('judge_model')} | {rg.get('judge_model')} | 差值 |",
            "|---|---|---|---|",
        ]
        for name, new_value in (rg.get("scores") or {}).items():
            old_value = (base.get("scores") or {}).get(name)
            if old_value is None:
                lines.append(f"| {name} | — | {_fmt(new_value)} | — |")
                continue
            delta = new_value - old_value
            lines.append(
                f"| {name} | {_fmt(old_value)} | {_fmt(new_value)} | {delta:+.3f} |"
            )

    anomalies = report.get("anomalies") or []
    if anomalies:
        lines += [
            "",
            "## 已发现的失败模式",
            "",
            "以下样本未正常收口。**它们如实计入检索与工具类指标，但答案为空者不参与"
            "生成类指标（M4 的 faithfulness / answer_relevancy）的均值**——"
            "没有答案就无从评价答案，混进均值只会让指标失去意义。",
            "",
            "| id | failure_tag | 停止原因 | 空答案 | 工具调用次数 | 说明 |",
            "|---|---|---|---|---|---|",
        ]
        for a in anomalies:
            lines.append(
                f"| {a['id']} | {a['failure_tag'] or '—'} | {a['stop_reason']} | "
                f"{'是' if a['empty_answer'] else '否'} | {a['n_tool_calls']} | {a['note']} |"
            )

    tok = report["tokens"]
    lines += [
        "",
        "## Token 消耗",
        "",
        f"- Agent 侧：prompt {tok['prompt']} + completion {tok['completion']} = **{tok['total']}**",
        "- 注：**不含 RAGAS 裁判与多轮一致性重复运行的开销**——"
        "这两部分走的是独立调用，本计数只覆盖主评测每题一次的 Agent 运行。",
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


TRACES_FILENAME = "traces_latest.jsonl"
CONSISTENCY_COLD_FILENAME = "traces_consistency_cold.jsonl"
CONSISTENCY_HOT_FILENAME = "traces_consistency_hot.jsonl"


def save_traces(
    traces: Mapping[str, AgentTrace], path: Path, merge: bool = False
) -> Path:
    """把轨迹整份落盘。

    M4 的 RAGAS 要 chunk 原文、M5 的裁判要最终答案——都得基于**同一批**轨迹算，
    否则每加一个指标就得重跑一次 Agent：既烧钱，指标之间也对不上同一次运行。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # merge=True：只跑了子集（--limit）时，把已有轨迹保留下来再覆盖同名条目。
    # 轨迹是真金白银跑出来的，不能被一次子集运行顺手截断。
    payload: dict[str, AgentTrace] = {}
    if merge and path.exists():
        payload.update(load_traces(path))
    payload.update(traces)

    with path.open("w", encoding="utf-8") as fh:
        for sample_id, trace in payload.items():
            fh.write(json.dumps({"id": sample_id, "trace": trace.model_dump()}, ensure_ascii=False) + chr(10))
    return path


def load_traces(path: Path | str) -> dict[str, AgentTrace]:
    """读回落盘的轨迹，用于不重跑 Agent 就补算指标。"""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"轨迹文件不存在：{target}")
    out: dict[str, AgentTrace] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["id"]] = AgentTrace.model_validate(row["trace"])
    return out


def save_trace_runs(runs: Mapping[str, Sequence[AgentTrace]], path: Path) -> Path:
    """把"同题多次运行"的轨迹落盘（一致性用）。

    一次全量评测被中途掐断时，主评测轨迹还在、一致性的 K×N 次运行却白跑了。
    重复运行比主评测更贵，必须能存下来续用。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample_id, traces in runs.items():
            payload = {"id": sample_id, "runs": [t.model_dump() for t in traces]}
            fh.write(json.dumps(payload, ensure_ascii=False) + chr(10))
    return path


def load_trace_runs(path: Path | str) -> dict[str, list[AgentTrace]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"一致性轨迹文件不存在：{target}")
    out: dict[str, list[AgentTrace]] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["id"]] = [AgentTrace.model_validate(t) for t in row["runs"]]
    return out


def load_trace_runs_if_present(path: Path | str) -> dict[str, list[AgentTrace]] | None:
    """文件不存在就返回 None——没跑过一致性是合法状态，报告如实留空。"""
    try:
        return load_trace_runs(path)
    except FileNotFoundError:
        return None


def write_report(
    report: dict,
    directory: Path | None = None,
    traces: Mapping[str, AgentTrace] | None = None,
    consistency_runs: Mapping[str, Sequence[AgentTrace]] | None = None,
    consistency_runs_hot: Mapping[str, Sequence[AgentTrace]] | None = None,
) -> dict[str, Path]:
    """落盘：时间戳快照 + latest.json + report.md（+ 轨迹）。"""
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

    paths = {"snapshot": snapshot, "latest": latest, "markdown": markdown}
    if traces:
        paths["traces"] = save_traces(traces, target / TRACES_FILENAME, merge=True)
    if consistency_runs:
        paths["consistency_cold"] = save_trace_runs(
            consistency_runs, target / CONSISTENCY_COLD_FILENAME
        )
    if consistency_runs_hot:
        paths["consistency_hot"] = save_trace_runs(
            consistency_runs_hot, target / CONSISTENCY_HOT_FILENAME
        )
    return paths


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
    parser.add_argument(
        "--consistency-temperature",
        type=float,
        default=config.CONSISTENCY_TEMPERATURE,
        help="鲁棒性专测的温度（主评测始终锁 temp=0）",
    )
    parser.add_argument(
        "--no-hot-consistency", action="store_true", help="跳过 temp>0 的鲁棒性专测"
    )
    parser.add_argument("--ragas", action="store_true", help="附带跑 RAGAS 生成质量指标")
    parser.add_argument(
        "--from-traces",
        type=str,
        default=None,
        help="复用已落盘的轨迹重算指标，不重跑 Agent（省钱；如只想补 RAGAS）",
    )
    parser.add_argument(
        "--compare-ragas",
        type=str,
        default=None,
        help="上一份报告 JSON 的路径，用其 RAGAS 分数做换裁判前的对照基线",
    )
    parser.add_argument("--notes", type=str, default="", help="写进报告的备注")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    from src.agent import graph

    samples = load_golden_set()
    if args.limit:
        samples = samples[: args.limit]

    if args.from_traces:
        print(f"复用已落盘轨迹：{args.from_traces}（不重跑 Agent）", file=sys.stderr)
        traces = load_traces(args.from_traces)
        missing = [s.id for s in samples if s.id not in traces]
        if missing:
            raise SystemExit(f"轨迹文件缺少这些样本，无法复用：{missing}")
        traces = {s.id: traces[s.id] for s in samples}
        agent = None

        # 一致性那两档也一并读回：否则复用轨迹补算指标时，报告会平白丢掉一致性。
        trace_dir = Path(args.from_traces).parent
        consistency_runs = load_trace_runs_if_present(trace_dir / CONSISTENCY_COLD_FILENAME)
        consistency_runs_hot = load_trace_runs_if_present(trace_dir / CONSISTENCY_HOT_FILENAME)
        for label, loaded in (("temp=0", consistency_runs), ("temp>0", consistency_runs_hot)):
            if loaded:
                print(f"  复用 {label} 一致性轨迹：{len(loaded)} 题", file=sys.stderr)
    else:
        print(f"黄金集 {len(samples)} 条，构建 Agent（加载 BGE + FAISS）...", file=sys.stderr)
        agent = graph.build_default_agent()
        consistency_runs = None
        consistency_runs_hot = None
        print("主评测：每题跑 1 次", file=sys.stderr)
        main_runs = run_samples(agent, samples, repeats=1)
        traces = {sid: runs[0] for sid, runs in main_runs.items()}

    if agent is not None and not args.no_consistency and args.consistency_size > 0:
        subset = samples[: args.consistency_size]

        print(
            f"多轮一致性（可复现基线 temp={config.DEFAULT_TEMPERATURE}）："
            f"{len(subset)} 题 × {args.consistency_runs} 次",
            file=sys.stderr,
        )
        consistency_runs = run_samples(agent, subset, repeats=args.consistency_runs)

        if not args.no_hot_consistency:
            print(
                f"多轮一致性（鲁棒性专测 temp={args.consistency_temperature}）："
                f"{len(subset)} 题 × {args.consistency_runs} 次",
                file=sys.stderr,
            )
            hot_agent = graph.build_default_agent(temperature=args.consistency_temperature)
            consistency_runs_hot = run_samples(hot_agent, subset, repeats=args.consistency_runs)

    ragas_result = None
    if args.ragas:
        from src.eval import ragas_runner

        print("RAGAS：基于同一批轨迹算生成质量指标（裁判走 DeepSeek）", file=sys.stderr)
        ragas_result = ragas_runner.run_ragas(samples, traces)

    ragas_baseline = load_ragas_baseline(args.compare_ragas) if args.compare_ragas else None
    if ragas_baseline:
        print(
            f"裁判敏感性对照基线：{ragas_baseline['judge_model']}", file=sys.stderr
        )

    report = build_report(
        samples,
        traces,
        consistency_runs=consistency_runs,
        consistency_runs_hot=consistency_runs_hot,
        consistency_temperature=args.consistency_temperature,
        ragas=ragas_result,
        ragas_baseline=ragas_baseline,
        notes=args.notes,
    )
    paths = write_report(
        report,
        traces=traces,
        consistency_runs=consistency_runs,
        consistency_runs_hot=consistency_runs_hot,
    )

    print()
    print(render_markdown(report))
    print()
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
