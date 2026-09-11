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
    agreement: dict | None = None,
    judge_scores: Mapping[str, int] | None = None,
    judge_backfill: dict | None = None,
    judge_revision: dict | None = None,
    notes: str = "",
) -> dict:
    """把轨迹汇总成一份完整报告（纯函数，可在 CI 里用构造轨迹验证）。

    judge_scores 给了就用来回填开放题的任务成功率；judge_backfill 是它的溯源信息
    （来自哪个文件、哪个裁判模型、哪些行验过指纹），随指标一起落盘——
    一个由两种判据拼出来的均值，不带溯源就没法复现。
    """
    recall = metrics.aggregate_recall(samples, traces)
    recall_first = metrics.aggregate_recall(samples, traces, first_call_only=True)
    success = metrics.task_success_rate(samples, traces, judge_scores=judge_scores)
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
                "success_source": success.meta.get("source", {}).get(s.id),
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

    success_block = success.as_dict()
    if judge_backfill:
        success_block["backfill"] = judge_backfill

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
            "task_success_rate": success_block,
            "tool_accuracy": tool.as_dict(),
            "tool_set_accuracy": tool_set.as_dict(),
            "consistency": consistency,
            "consistency_hot": consistency_hot,
            "ragas": ragas,
            "ragas_baseline": ragas_baseline,
            "agreement": agreement,
            "judge_revision": judge_revision,
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "anomalies": anomalies,
        "per_question": per_question,
    }


def snapshot_filename(timestamp_utc: str) -> str:
    """报告快照的文件名。去掉 : - 和时区偏移里的 +，对 URL / shell 友好。

    抽成函数是为了让「写快照」和「解析复用来源」用同一套命名规则——
    两处各写一遍，早晚会对不上。
    """
    stamp = timestamp_utc.replace(":", "").replace("-", "")
    stamp = stamp.split("+")[0].rstrip("Z") + "Z"
    return f"eval_{stamp}.json"


def verify_ragas_reuse(
    path: Path | str, traces: Mapping[str, AgentTrace]
) -> dict:
    """核验"复用 RAGAS 前提轨迹未变"这句话，而不是嘴上说说。

    RAGAS 的输入是 question/answer/contexts。question 由同一份 golden set 决定
    （文件没变就不会变），contexts 由 doc_id 经同一份 corpus/index 确定性求出
    （corpus 自 M2 后未再改动，见 git log -- corpus/）——所以只需比对
    **answer 文本 + retrieved_doc_ids 集合**：两者都对得上，answer 与它引用的
    doc_id 集合都没变，contexts 就必然没变，不必去重新拼原文比对。

    指纹口径与 judge.answer_fingerprint 一致（sha1 前 12 位），
    但输入多了 retrieved_doc_ids——只比对答案文本不够，同一个答案配上
    不同的检索结果，RAGAS 实际吃到的 context 照样变了。

    对照的一侧是**基线快照的 per_question 段**（那次真实运行落盘的答案与
    检索结果），另一侧是**当前驱动本次报告的 traces**，不是重新跑一次 Agent——
    纯离线，不产生任何 API 调用。
    """
    from src.eval import judge

    baseline = json.loads(Path(path).read_text(encoding="utf-8"))
    ragas_block = (baseline.get("metrics") or {}).get("ragas") or {}
    evaluated_ids = ragas_block.get("evaluated_ids") or []
    by_id = {q["id"]: q for q in baseline.get("per_question", [])}

    def _fp(answer, doc_ids) -> str:
        payload = (answer or "") + "||" + ",".join(sorted(doc_ids or []))
        return judge.answer_fingerprint(payload)

    rows = []
    all_match = bool(evaluated_ids)
    for sid in evaluated_ids:
        base_row = by_id.get(sid)
        trace = traces.get(sid)
        if base_row is None or trace is None:
            rows.append(
                {"id": sid, "baseline_fp": None, "current_fp": None, "match": False}
            )
            all_match = False
            continue
        base_fp = _fp(base_row.get("answer"), base_row.get("retrieved_doc_ids"))
        cur_fp = _fp(trace.answer, trace.retrieved_doc_ids)
        match = base_fp == cur_fp
        all_match = all_match and match
        rows.append(
            {"id": sid, "baseline_fp": base_fp, "current_fp": cur_fp, "match": match}
        )

    return {
        "baseline_source": str(path),
        "baseline_timestamp": (baseline.get("meta") or {}).get("timestamp_utc"),
        "n_checked": len(rows),
        "all_match": all_match,
        "mismatched": [r["id"] for r in rows if not r["match"]],
        "rows": rows,
    }


def load_ragas_reuse(path: Path | str) -> dict | None:
    """从上一份报告里**原样取回** RAGAS 段。

    只在"同一批轨迹"下成立：RAGAS 的输入就是 question/answer/contexts，
    轨迹没变，重算一遍是花 40 分钟得到同一件事。但复用必须**留痕**——
    报告里会写明这段不是本次算的，以及它来自哪一份快照。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    block = (data.get("metrics") or {}).get("ragas")
    if not block:
        return None
    block = dict(block)
    # 若来源本身也是复用的，先清掉旧标记，免得层层套娃看不出真正的出处
    block.pop("reused_from", None)
    block.pop("reused_from_timestamp", None)

    source = Path(path)
    stamp = (data.get("meta") or {}).get("timestamp_utc")
    # 记录**当初算出这批数的那次运行的快照名**，而不是 latest.json——
    # 后者每跑一次就被覆盖，标记指过去就成了指向它自己，会误导后来的人。
    if stamp:
        canonical = source.parent / snapshot_filename(stamp)
        if canonical.exists():
            source = canonical
    block["reused_from"] = str(source)
    block["reused_from_timestamp"] = stamp
    return block


def load_agreement_if_present() -> dict | None:
    """人工标注与裁判分数都在，就算一致率。缺任一边就返回 None（不猜、不补）。

    纯离线计算，不产生任何 API 调用——所以 `--from-traces` 重出报告时它照样在。
    """
    from src.eval import judge

    try:
        human = judge.load_human_labels()
        auto = judge.load_judge_scores()
    except FileNotFoundError:
        return None
    if not human or not auto:
        return None
    stats = judge.agreement_stats(human, auto)
    # 解读随报告一起落盘：读 JSON 快照的人也该看到这些限定，而不是只看到数字。
    stats["interpretation"] = judge.load_agreement_interpretation()
    return stats


def load_judge_backfill(
    traces: Mapping[str, AgentTrace], path: Path | str | None = None
) -> tuple[dict[str, int], dict] | tuple[None, None]:
    """读回裁判分，供任务成功率回填开放题。返回 (可用的分数, 溯源信息)。

    这里做一件 load_judge_scores 不做的事：**核对分和轨迹对不对得上**。
    裁判分是给某一批答案打的，而报告可能是用另一批轨迹重算的
    （`--from-traces` 就是这么用的）。旧分套新轨迹不会报错，只会静默地
    产出一个错的成功率——所以逐条比对答案指纹：

      verified   指纹对得上，可用；
      stale      指纹对不上，**不采用**（答案已变，这个分作废）；
      unverified 打分行没存指纹（本功能之前写的老文件），无从核对；
                 采用，但在报告里标出来，别让读的人以为它被验过。

    纯离线，不产生任何 API 调用。
    """
    from src.eval import judge

    target = Path(path) if path else (config.REPORTS_DIR / judge.JUDGE_SCORES_FILENAME)
    try:
        rows = judge.load_judge_rows(target)
    except FileNotFoundError:
        return None, None
    if not rows:
        return None, None

    scores: dict[str, int] = {}
    verified, unverified, stale = [], [], []
    for row in rows:
        sid, score = row["id"], row.get("judge_score")
        if score is None:
            continue
        trace = traces.get(sid)
        want = row.get("answer_sha1")
        if want is None:
            unverified.append(sid)
        elif trace is None or judge.answer_fingerprint(trace.answer) != want:
            stale.append(sid)
            continue
        else:
            verified.append(sid)
        scores[sid] = int(score)

    models = sorted({row.get("judge_model") for row in rows if row.get("judge_model")})
    # 判据版本是任务成功率口径的一部分：同一批答案换一套判据就是换了量尺，
    # 分数不再与旧分可比（闸 B 据此判 incomparable）。老文件没这个字段，记 None。
    rubrics = sorted({row.get("rubric_version") for row in rows if row.get("rubric_version")})
    provenance = {
        "source": str(target),
        "judge_model": models[0] if len(models) == 1 else models,
        "rubric_version": (rubrics[0] if len(rubrics) == 1 else rubrics) or None,
        "open_threshold": config.OPEN_SUCCESS_THRESHOLD,
        "n_rows": len(rows),
        "verified": verified,
        "unverified": unverified,
        "stale": stale,
    }
    return scores, provenance


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
    return {
        "judge_model": block.get("judge_model"),
        "scores": block.get("scores") or {},
        "scored_counts": block.get("scored_counts") or {},
        "n_submitted": block.get("n_submitted"),
    }


def build_judge_revision(
    samples: Sequence[GoldenSample],
    traces: Mapping[str, AgentTrace],
    baseline_path: Path | str,
    current_scores: Mapping[str, int] | None = None,
) -> dict | None:
    """裁判判据改前 / 改后的对照块。

    判据一改，**两个指标同时动**：开放题的裁判分既进 kappa，也进任务成功率。
    只报改后的数字，等于让读者无从判断这个变化是判据带来的还是模型带来的——
    所以两版都算、都落盘，差值和归因写在旁边。

    输入是同一批轨迹、同一份人工标注，唯一的变量就是那份判据。
    纯离线：两边的分都是已落盘的，这里不再调任何 API。
    """
    from src.eval import judge

    base = Path(baseline_path)
    if not base.exists():
        return None

    before_rows = judge.load_judge_rows(base)
    after_rows = judge.load_judge_rows()
    before = {r["id"]: int(r["judge_score"]) for r in before_rows if r.get("judge_score") is not None}
    after = dict(current_scores or judge.load_judge_scores())
    if not before or not after:
        return None

    def _version(rows: Sequence[Mapping]) -> str:
        seen = sorted({r.get("rubric_version") for r in rows if r.get("rubric_version")})
        # 没标版本的就是加这个字段之前打的分，如实说"未标注"而不是替它编一个。
        return "、".join(seen) if seen else "未标注（判据版本字段之前）"

    try:
        human = judge.load_human_labels()
    except FileNotFoundError:
        human = {}

    def _success(scores: Mapping[str, int]) -> dict:
        r = metrics.task_success_rate(samples, traces, judge_scores=scores)
        return {"value": r.value, "n": r.n, "total": r.total}

    changes = []
    for sid in sorted(set(before) | set(after)):
        b, a = before.get(sid), after.get(sid)
        if b == a:
            continue
        row = {"id": sid, "before": b, "after": a}
        if sid in human:
            row["human"] = human[sid]
        changes.append(row)

    block = {
        "baseline_source": str(base),
        "rubric_before": _version(before_rows),
        "rubric_after": _version(after_rows),
        "n_scored": len(after),
        "score_changes": changes,
        "task_success": {"before": _success(before), "after": _success(after)},
    }
    if human:
        block["agreement"] = {
            "before": judge.agreement_stats(human, before),
            "after": judge.agreement_stats(human, after),
        }
    return block


def _render_judge_revision(report: dict) -> list[str]:
    """把改前 / 改后对照渲染成表。数字全部来自上面那个纯函数，这里只排版。"""
    block = (report.get("metrics") or {}).get("judge_revision")
    if not block:
        return []

    out = [
        "",
        "## 裁判判据修订对照（改前 / 改后）",
        "",
        f"- 判据：`{block['rubric_before']}` → `{block['rubric_after']}`",
        f"- 改前的分留档在 `{Path(block['baseline_source']).name}`；"
        f"轨迹、人工标注、被测模型三者**均未变动**，唯一的变量是判据本身。",
        "",
    ]

    sc = block["task_success"]
    ag = block.get("agreement") or {}
    rows = [
        (
            "任务成功率",
            sc["before"]["value"],
            sc["after"]["value"],
            f"n={sc['after']['n']}/{sc['after']['total']}",
        )
    ]
    if ag:
        b, a = ag["before"], ag["after"]
        rows += [
            ("完全一致率（人↔裁判）", b["exact_agreement"], a["exact_agreement"], f"n={a['n']}"),
            ("相邻一致率（差 ≤1 档）", b["adjacent_agreement"], a["adjacent_agreement"], f"n={a['n']}"),
            ("Cohen's kappa（unweighted）", b["kappa"], a["kappa"], f"n={a['n']}"),
            ("Cohen's kappa（quadratic）", b["kappa_quadratic"], a["kappa_quadratic"], f"n={a['n']}"),
        ]

    out += ["| 量 | 改前 | 改后 | 差 | 分母 |", "|---|---|---|---|---|"]
    for label, before, after, denom in rows:
        delta = "—" if before is None or after is None else f"{after - before:+.3f}"
        out.append(f"| {label} | {_fmt(before)} | {_fmt(after)} | {delta} | {denom} |")

    changes = block.get("score_changes") or []
    out += ["", f"### 改判的题（{len(changes)} 道）", ""]
    if not changes:
        out.append("无——判据收紧后没有任何一题改档。")
    else:
        out += ["| id | 改前 | 改后 | 人工 |", "|---|---|---|---|"]
        for row in changes:
            human = row.get("human")
            out.append(
                f"| `{row['id']}` | {row['before']} | {row['after']} | "
                f"{'—' if human is None else human} |"
            )
    return out


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _render_success_rule(report: dict) -> list[str]:
    """任务成功率的口径自述。

    这个数是两把尺子拼出来的（闭合题规则 + 开放题裁判），
    不把拼法、门槛、失败题和裁判分的溯源写在数字旁边，
    "任务成功率 0.972" 就是个无法回溯的漂亮数字——第一红线不允许。
    """
    sr = report["metrics"]["task_success_rate"]
    meta = sr.get("meta") or {}
    backfill = sr.get("backfill")
    if not backfill:
        return []

    by_id = {row["id"]: row for row in report.get("per_question", [])}
    failed = [
        sid for sid, ok in (sr.get("detail") or {}).items() if ok is False
    ]

    def _label(sid: str) -> str:
        src = (meta.get("source") or {}).get(sid)
        kind = {"rule": "闭合/规则", "judge": "开放/裁判"}.get(src, "?")
        return f"`{sid}`（{kind}）"

    out = [
        "",
        "### 任务成功率的判定口径",
        "",
        f"- 闭合题 **{meta.get('n_rule_judged')}** 题：`answer_keys` 全部命中才算成功。",
        f"- 开放题 **{meta.get('n_judge_judged')}** 题：裁判（`{backfill.get('judge_model')}`）"
        f"打分 **≥ {meta.get('open_threshold')}** 才算成功——"
        f"三级标度里 1 分是「方向对但要点有遗漏」，**不计成功**。",
        f"- 裁判分来源：`{Path(backfill.get('source', '')).name}`。",
    ]
    unjudged = meta.get("unjudged") or []
    if unjudged:
        out.append(
            f"- ⚠ 未判定（计入 total 不计入 n）：{', '.join(f'`{x}`' for x in unjudged)}"
        )
    stale = backfill.get("stale") or []
    if stale:
        out.append(
            f"- ⚠ **已作废**（打分时的答案与本批轨迹对不上，不予采用）："
            f"{', '.join(f'`{x}`' for x in stale)}"
        )
    unverified = backfill.get("unverified") or []
    if unverified:
        out.append(
            f"- ⚠ 未核验（打分行没存答案指纹，无从机器确认它是给这批轨迹打的）："
            f"{len(unverified)} 行——重跑一次 `python -m src.eval.judge --score` 即可补上。"
        )
    if failed:
        out.append("")
        out.append("未通过的题：")
        for sid in failed:
            row = by_id.get(sid) or {}
            note = ""
            if (meta.get("source") or {}).get(sid) == "judge":
                note = "裁判未给满分"
            elif row.get("answer_type") == "closed":
                note = "answer_keys 未全部命中"
            out.append(f"- {_label(sid)} {note}")
    return out


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

    sr = m["task_success_rate"]
    sr_meta = sr.get("meta") or {}
    sr_backfill = sr.get("backfill")
    success_note = (
        f"闭合题 {sr_meta.get('n_rule_judged')} 题走 answer_keys；"
        f"开放题 {sr_meta.get('n_judge_judged')} 题走裁判分，"
        f"满 {sr_meta.get('open_threshold')} 分才算成功"
        if sr_backfill
        else "闭合题规则判定；开放题未回填裁判分，不计入分母"
    )

    rows = [
        ("检索召回率 recall@k", "recall_at_k", "全部检索结果的并集"),
        ("recall@k（仅首次检索）", "recall_at_k_first_call", "只算第一次检索；与上一行的差＝多次检索捞回了多少"),
        ("任务成功率", "task_success_rate", success_note),
        ("工具调用准确率（严格）", "tool_accuracy", "折叠连续重复后逐项比对"),
        ("工具调用准确率（宽松）", "tool_set_accuracy", "只看用了哪些工具，不看顺序"),
    ]
    for label, key, note in rows:
        item = m[key]
        lines.append(
            f"| {label} | {_fmt(item['value'])} | n={item['n']}/{item['total']} | {note} |"
        )

    lines += _render_success_rule(report)

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
        ]
        if rg.get("reused_from"):
            verification = rg.get("reuse_verification")
            if verification and verification.get("all_match"):
                lines.append(
                    f"- ✅ **已验证轨迹指纹一致（{verification['n_checked']} 条），复用成立**："
                    f"分数原样取自 `{rg['reused_from']}`（{rg.get('reused_from_timestamp')}）。"
                    "核验口径：answer 文本 + retrieved_doc_ids 集合的指纹逐题比对"
                    "（corpus 自 M2 后未再变动，doc_id 相同即 context 必然相同）。"
                )
            elif verification:
                lines.append(
                    f"- ⚠ **轨迹指纹核验未通过**（{len(verification['mismatched'])}/"
                    f"{verification['n_checked']} 条不一致：{verification['mismatched']}）："
                    f"分数取自 `{rg['reused_from']}`，但复用前提不成立，这份分数**不应采用**，"
                    "需要对当前轨迹重跑 RAGAS。"
                )
            else:
                lines.append(
                    f"- ⚠ **本节为复用，不是本次重算**：分数原样取自 `{rg['reused_from']}`"
                    f"（{rg.get('reused_from_timestamp')}），前提是轨迹未变（未机器核验）。"
                )
        lines += [
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
            "| 指标 | "
            + f"{base.get('judge_model')} | {rg.get('judge_model')} | 差值 | 打分行数 |",
            "|---|---|---|---|---|",
        ]
        base_counts = base.get("scored_counts") or {}
        new_counts = rg.get("scored_counts") or {}
        base_n = base.get("n_submitted")
        new_n = rg.get("n_submitted")
        incomparable = []

        for name, new_value in (rg.get("scores") or {}).items():
            old_value = (base.get("scores") or {}).get(name)
            b_cnt, n_cnt = base_counts.get(name), new_counts.get(name)
            span = (
                f"{b_cnt}/{base_n} → {n_cnt}/{new_n}"
                if b_cnt is not None and n_cnt is not None
                else "—"
            )
            if old_value is None:
                lines.append(f"| {name} | — | {_fmt(new_value)} | — | {span} |")
                continue

            # 两次打分覆盖的行不同，均值就不是在同一批样本上算的，差值没有可比性。
            same_span = b_cnt is not None and n_cnt is not None and b_cnt == n_cnt
            delta = f"{new_value - old_value:+.3f}" if same_span else "**不可比**"
            if not same_span:
                incomparable.append(name)
            lines.append(
                f"| {name} | {_fmt(old_value)} | {_fmt(new_value)} | {delta} | {span} |"
            )

        if incomparable:
            lines += [
                "",
                f"> ⚠ **{'、'.join(incomparable)} 的差值不可比**：两次运行打分成功的行数不同，"
                "均值是在不同子集上算的。看着像「换裁判没影响」的 0.000 差值，"
                "很可能只是两个不同样本集碰巧接近。要得到可比的对照，"
                "必须两次都打满同样的行数。",
            ]

    agreement = m.get("agreement")
    if agreement:
        from src.eval.judge import render_agreement_markdown

        lines += ["", render_agreement_markdown(agreement).rstrip()]

    lines += _render_judge_revision(report)

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
        "| id | 题型 | 期望工具 | 实际工具 | 工具 | recall | 成功 | 判据 | 停止原因 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    # 「判据」这一列是回填后新增的：同一列 ✔ 现在可能来自两种判定，
    # 不标出来就分不清哪个 ✔ 是规则判的、哪个是裁判判的。
    source_label = {"rule": "规则", "judge": "裁判"}
    for q in report["per_question"]:
        def mark(v):
            return "—" if v is None else ("✔" if v else "✘")

        lines.append(
            f"| {q['id']} | {q['answer_type']} | {'+'.join(q['expected_tool'])} | "
            f"{'+'.join(q['actual_tool']) if q['actual_tool'] else '（无）'} | {mark(q['tool_ok'])} | "
            f"{_fmt(q['recall_at_k'])} | {mark(q['success'])} | "
            f"{source_label.get(q.get('success_source'), '—')} | {q['stop_reason']} |"
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
    snapshot = target / snapshot_filename(report["meta"]["timestamp_utc"])
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


# ===========================================================================
# 回归门禁（M6，CLAUDE.md §7）
#
# 两道闸，管的是两件不同的事：
#
#   闸 A「口径闸」：输入冻结（tests/fixtures 里的固定子集 + 固定轨迹），
#       重算指标必须与基线**逐位相等**。输入没变、数字变了，只可能是
#       指标代码的口径变了——不论变高变低都得拦。这是信贷项目里
#       "同一批样本 PSI 必须复现"的同一思路。
#
#   闸 B「回归闸」：拿本次真实运行的报告与上一份快照比，
#       关键指标跌超 config.METRIC_DROP_TOLERANCE 即 fail。
#       真实运行有模型噪声，所以这道闸用容差而不是等值。
#
# 两道闸都**离线**：闸 A 读 fixture，闸 B 读已提交进 git 的报告 JSON，
# 全程不碰 DeepSeek API（CI 上没有 key，也刻意不放）。
# ===========================================================================

GATE_FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
GATE_SAMPLES_PATH = GATE_FIXTURE_DIR / "gate_samples.jsonl"
GATE_TRACES_PATH = GATE_FIXTURE_DIR / "gate_traces.jsonl"
# 裁判分现在是任务成功率的输入之一，所以它也必须被冻结进 fixture。
# 让闸 A 去读 reports/judge_scores.jsonl 就等于让门禁的输入随手一跑就变——
# 那道闸测的就不再是"口径有没有变"了。
GATE_JUDGE_SCORES_PATH = GATE_FIXTURE_DIR / "gate_judge_scores.jsonl"
GATE_BASELINE_FILENAME = "gate_baseline.json"

# 闸 A 盯的指标：全部可离线重算的自定义指标。
# RAGAS 类指标要调裁判，进不了闸 A，由闸 B 在报告层面看。
GATE_EXACT_METRICS = (
    "recall_at_k",
    "recall_at_k_first_call",
    "task_success_rate",
    "tool_accuracy",
    "tool_set_accuracy",
)


def load_gate_fixture(
    samples_path: Path | str | None = None,
    traces_path: Path | str | None = None,
    judge_scores_path: Path | str | None = None,
):
    """读回冻结的门禁子集，返回 (样本, 轨迹, 裁判分)。

    刻意含已知失败样本——输入全是满分的门禁形同虚设。
    裁判分也在冻结之列：任务成功率的开放题那一半靠它判定。
    """
    from src.eval import judge
    from src.eval.datasets import GoldenSample

    sp = Path(samples_path) if samples_path else GATE_SAMPLES_PATH
    tp = Path(traces_path) if traces_path else GATE_TRACES_PATH
    jp = Path(judge_scores_path) if judge_scores_path else GATE_JUDGE_SCORES_PATH
    samples = [
        GoldenSample.model_validate_json(line)
        for line in sp.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    traces = load_traces(tp)

    # fixture 的裁判分同样要过指纹核对：冻结轨迹若被改过而分没跟着改，
    # 门禁就会拿着对不上的分算出一个"稳定"的数字，那正是本机制要防的。
    scores, _ = load_judge_backfill(traces, jp)
    return samples, traces, (scores or {})


def compute_gate_metrics(samples, traces, judge_scores=None) -> dict:
    """闸 A 的被测量：从冻结输入重算出的那几个指标。"""
    rep = build_report(samples, traces, judge_scores=judge_scores)
    return {
        name: {
            "value": rep["metrics"][name]["value"],
            "n": rep["metrics"][name]["n"],
            "total": rep["metrics"][name]["total"],
        }
        for name in GATE_EXACT_METRICS
    }


def write_gate_baseline(path: Path | None = None) -> Path:
    """重算并写入闸 A 的基线。

    **这不是日常操作**：输入是冻结的，基线变动只可能因为口径变了。
    执行它等于宣布"我确实改了指标定义"，必须同时在报告/PR 里说明改了什么、为什么。
    """
    samples, traces, judge_scores = load_gate_fixture()
    target = Path(path) if path else (config.REPORTS_DIR / GATE_BASELINE_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "fixture_ids": [s.id for s in samples],
            "judge_scored_ids": sorted(judge_scores),
            "open_success_threshold": config.OPEN_SUCCESS_THRESHOLD,
            "note": "闸 A 基线：输入冻结，重算必须逐位相等。改动它=改了指标口径。",
        },
        "metrics": compute_gate_metrics(samples, traces, judge_scores),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_gate_baseline(path: Path | None = None) -> dict:
    target = Path(path) if path else (config.REPORTS_DIR / GATE_BASELINE_FILENAME)
    if not target.exists():
        raise FileNotFoundError(
            f"闸 A 基线不存在：{target}。首次建立请跑 "
            f"`python -m src.eval.report --write-gate-baseline`"
        )
    return json.loads(target.read_text(encoding="utf-8"))


def check_exact_gate(baseline: dict | None = None) -> list[dict]:
    """闸 A：冻结输入下重算，逐项与基线比对。返回 findings。"""
    base = baseline if baseline is not None else load_gate_baseline()
    samples, traces, judge_scores = load_gate_fixture()
    current = compute_gate_metrics(samples, traces, judge_scores)

    findings = []
    for name in GATE_EXACT_METRICS:
        want = (base.get("metrics") or {}).get(name)
        got = current.get(name)
        if want is None:
            findings.append({"metric": name, "status": "missing_baseline", "current": got})
            continue
        same = (
            want.get("value") == got.get("value")
            and want.get("n") == got.get("n")
            and want.get("total") == got.get("total")
        )
        findings.append(
            {
                "metric": name,
                "status": "ok" if same else "changed",
                "baseline": want,
                "current": got,
            }
        )
    return findings


def _metric_value(report: Mapping, name: str):
    """从报告里取一个指标值。faithfulness 这类在 RAGAS 段里，路径不同。"""
    metrics = report.get("metrics") or {}
    item = metrics.get(name)
    if isinstance(item, Mapping) and "value" in item:
        return item["value"]
    ragas = metrics.get("ragas") or {}
    scores = ragas.get("scores") or {}
    return scores.get(name)


def _metric_n(report: Mapping, name: str):
    """取一个指标自己的分母。RAGAS 段里的指标没有这个字段，返回 None。"""
    item = (report.get("metrics") or {}).get(name)
    if isinstance(item, Mapping):
        return item.get("n")
    return None


def _metric_rubric(report: Mapping, name: str):
    """取算这个指标时用的裁判判据版本；不依赖裁判分的指标返回 None。"""
    item = (report.get("metrics") or {}).get(name)
    if isinstance(item, Mapping):
        return ((item.get("backfill") or {}) or {}).get("rubric_version")
    return None


def check_regression_gate(
    current: Mapping,
    baseline: Mapping,
    tolerance: float | None = None,
    metrics: Sequence[str] = config.GATED_METRICS,
) -> list[dict]:
    """闸 B：本次报告 vs 上一份快照，关键指标跌超容差即 dropped。

    两份报告的黄金集规模不同就不比——那是在不同题目集上算的均值，
    差值没有意义（同 §RAGAS 的分母陷阱）。如实标 incomparable，不硬凑。

    **单个指标自己的分母变了同样不比。** 任务成功率从"26 道闭合题"扩到
    "36 道全集"时，题数没变、指标名没变，差值却完全来自口径而非模型——
    只看 golden_set_size 会把这种变动当成一次普通波动放过去。
    这是"闸 A 拦口径"在报告层面的对应物。

    **裁判判据换了版本同样不比。** 任务成功率的开放题那一半由裁判分判定，
    换判据就是换量尺：同一批答案、同一个模型，分数照样会动。
    判据 v1（结论级）-> v2（要点级）那次，两题掉档、成功率跌 0.056，
    按容差判会报 dropped，但被测系统一点没变。

    这三条都有同一个代价：**它们也能被用来给一次真实的变差打掩护。**
    对策不在这个函数里——口径变动必须同时改 fixture 基线（闸 A 逐位比对、涨了也拦）、
    改 RUBRIC_VERSION 或 config 里的阈值，全都会出现在 diff 和提交历史里。
    这里的职责只是**不把口径差值伪装成模型差值**，而不是替人把关口径该不该改。
    """
    tol = config.METRIC_DROP_TOLERANCE if tolerance is None else tolerance
    cur_size = (current.get("meta") or {}).get("golden_set_size")
    base_size = (baseline.get("meta") or {}).get("golden_set_size")
    comparable = cur_size == base_size

    findings = []
    for name in metrics:
        old = _metric_value(baseline, name)
        new = _metric_value(current, name)
        if old is None or new is None:
            findings.append(
                {"metric": name, "status": "missing", "baseline": old, "current": new}
            )
            continue
        old_n, new_n = _metric_n(baseline, name), _metric_n(current, name)
        denominator_moved = (
            old_n is not None and new_n is not None and old_n != new_n
        )
        old_rubric, new_rubric = _metric_rubric(baseline, name), _metric_rubric(current, name)
        rubric_moved = old_rubric != new_rubric

        if not comparable or denominator_moved or rubric_moved:
            if not comparable:
                note = f"黄金集规模不同（{base_size} -> {cur_size}），均值不在同一批题上"
            elif denominator_moved:
                note = f"该指标分母变了（n={old_n} -> n={new_n}），差值来自口径而非模型"
            else:
                note = (
                    f"裁判判据变了（{old_rubric or '未标注'} -> {new_rubric or '未标注'}），"
                    "量尺换了，差值不归因于被测系统"
                )
            findings.append(
                {
                    "metric": name,
                    "status": "incomparable",
                    "baseline": old,
                    "current": new,
                    "delta": new - old,
                    "note": note,
                }
            )
            continue
        delta = new - old
        findings.append(
            {
                "metric": name,
                "status": "dropped" if delta < -tol else "ok",
                "baseline": old,
                "current": new,
                "delta": delta,
            }
        )
    return findings


def find_previous_snapshot(directory: Path | None = None, current: Path | None = None):
    """找可作基线的上一份快照：按文件名时间戳排序后，最新的那个非 current。

    文件名里的时间戳是零填充的定长格式，字典序即时间序。
    """
    target = Path(directory) if directory else config.REPORTS_DIR
    cur = Path(current).name if current else None
    snapshots = sorted(p for p in target.glob("eval_*.json") if p.name != cur)
    return snapshots[-1] if snapshots else None


def run_gates() -> int:
    """跑两道闸并打印结论。返回值即进程退出码：有 changed / dropped 就非 0。"""
    failed = False

    print("闸 A（口径闸）：冻结输入重算，必须与基线逐位相等")
    for f in check_exact_gate():
        if f["status"] == "ok":
            print(f"  ok        {f['metric']} = {f['current']['value']}")
        else:
            failed = True
            print(f"  {f['status'].upper():9} {f['metric']}: 基线 {f.get('baseline')} -> 现在 {f['current']}")

    latest = config.REPORTS_DIR / "latest.json"
    print()
    if not latest.exists():
        print("闸 B（回归闸）：跳过——还没有 latest.json")
    else:
        current_report = json.loads(latest.read_text(encoding="utf-8"))
        prev_name = snapshot_filename(current_report["meta"]["timestamp_utc"])
        previous = find_previous_snapshot(current=config.REPORTS_DIR / prev_name)
        if previous is None:
            print("闸 B（回归闸）：跳过——除本次外没有可比的历史快照")
        else:
            baseline_report = json.loads(previous.read_text(encoding="utf-8"))
            print(f"闸 B（回归闸）：latest.json vs {previous.name}，容差 {config.METRIC_DROP_TOLERANCE}")
            for f in check_regression_gate(current_report, baseline_report):
                if f["status"] == "dropped":
                    failed = True
                delta = f.get("delta")
                shown = f"{delta:+.4f}" if isinstance(delta, float) else "—"
                print(f"  {f['status']:12} {f['metric']}: {f.get('baseline')} -> {f.get('current')}  ({shown})")

    # 闸 C（稳定性闸，PERF_SPEC B4）：同一输入重复 k 次的成功率标准差与轨迹自洽率，
    # 外加"summary.json 的数字必须能由原始产物重算出来"。逻辑在 stability/gate.py。
    from stability import gate as stability_gate

    print()
    gate_c_ok, lines = stability_gate.run_gate_c()
    for line in lines:
        print(line)
    if not gate_c_ok:
        failed = True

    print()
    print("门禁结果：" + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


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
    parser.add_argument(
        "--reuse-ragas",
        type=str,
        default=None,
        help="从上一份报告 JSON 原样取回 RAGAS 段（同一批轨迹下省 40 分钟；报告里会标注复用）",
    )
    parser.add_argument(
        "--gate", action="store_true", help="只跑回归门禁（离线，不调 API），不重跑评测"
    )
    parser.add_argument(
        "--write-gate-baseline",
        action="store_true",
        help="重写闸 A 基线。仅在**有意变更指标口径**时执行，并须在 PR/报告里说明",
    )
    parser.add_argument(
        "--judge-scores",
        type=str,
        default=None,
        help="裁判打分文件（默认 reports/judge_scores.jsonl），用于回填开放题的任务成功率",
    )
    parser.add_argument(
        "--compare-judge",
        type=str,
        default=None,
        help="改判据前那份 judge_scores.jsonl 的路径；报告里出改前/改后对照（离线）",
    )
    parser.add_argument(
        "--no-judge-backfill",
        action="store_true",
        help="不回填开放题：任务成功率退回只算闭合题（分母会缩小，报告里会写明）",
    )
    parser.add_argument("--notes", type=str, default="", help="写进报告的备注")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.write_gate_baseline:
        path = write_gate_baseline()
        print(f"闸 A 基线已重写：{path}")
        print("注意：输入是冻结的，基线变动只可能因为指标口径变了——请在提交信息里写明改了什么。")
        return 0

    if args.gate:
        return run_gates()

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

    if ragas_result is None and args.reuse_ragas:
        ragas_result = load_ragas_reuse(args.reuse_ragas)
        if ragas_result:
            verification = verify_ragas_reuse(args.reuse_ragas, traces)
            ragas_result["reuse_verification"] = verification
            status = "一致" if verification["all_match"] else "不一致"
            print(
                f"复用 RAGAS 段：{args.reuse_ragas}（未重算；轨迹指纹核验：{status}，"
                f"n={verification['n_checked']}）",
                file=sys.stderr,
            )
            for row in verification["rows"]:
                mark = "OK" if row["match"] else "DIFF"
                print(
                    f"  {row['id']:10s} base={row['baseline_fp']} "
                    f"current={row['current_fp']}  {mark}",
                    file=sys.stderr,
                )
            if not verification["all_match"]:
                print(
                    "  ⚠ 指纹对不上，复用前提不成立——按规则应改为重算 RAGAS，"
                    "不能带着这份复用分继续。",
                    file=sys.stderr,
                )

    ragas_baseline = load_ragas_baseline(args.compare_ragas) if args.compare_ragas else None
    if ragas_baseline:
        print(
            f"裁判敏感性对照基线：{ragas_baseline['judge_model']}", file=sys.stderr
        )

    agreement = load_agreement_if_present()
    if agreement:
        print(f"自动↔人工一致率：n={agreement['n']}", file=sys.stderr)

    judge_scores, judge_backfill = (None, None)
    if not args.no_judge_backfill:
        judge_scores, judge_backfill = load_judge_backfill(traces, args.judge_scores)
    if judge_backfill:
        print(
            f"任务成功率回填开放题：{len(judge_scores)} 条裁判分"
            f"（验过指纹 {len(judge_backfill['verified'])}、"
            f"未存指纹 {len(judge_backfill['unverified'])}、"
            f"已作废 {len(judge_backfill['stale'])}），"
            f"门槛 ≥{config.OPEN_SUCCESS_THRESHOLD}",
            file=sys.stderr,
        )
        if judge_backfill["stale"]:
            print(
                f"  ⚠ 这些题的裁判分与当前轨迹对不上，已丢弃："
                f"{judge_backfill['stale']}（重跑 judge --score 可修）",
                file=sys.stderr,
            )
    else:
        print("没有可用的裁判分：开放题不计入任务成功率", file=sys.stderr)

    judge_revision = None
    if args.compare_judge:
        judge_revision = build_judge_revision(
            samples, traces, args.compare_judge, judge_scores
        )
        if judge_revision:
            sc = judge_revision["task_success"]
            print(
                f"判据修订对照：{judge_revision['rubric_before']} -> "
                f"{judge_revision['rubric_after']}，改判 "
                f"{len(judge_revision['score_changes'])} 题；"
                f"任务成功率 {sc['before']['value']:.3f} -> {sc['after']['value']:.3f}",
                file=sys.stderr,
            )
        else:
            print(f"⚠ 对照基线读不到或为空：{args.compare_judge}", file=sys.stderr)

    report = build_report(
        samples,
        traces,
        consistency_runs=consistency_runs,
        consistency_runs_hot=consistency_runs_hot,
        consistency_temperature=args.consistency_temperature,
        ragas=ragas_result,
        ragas_baseline=ragas_baseline,
        agreement=agreement,
        judge_scores=judge_scores,
        judge_backfill=judge_backfill,
        judge_revision=judge_revision,
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
