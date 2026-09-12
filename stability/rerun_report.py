"""对照重跑报告：把一份另存的主评测快照与基线快照逐题对照，脚本生成 markdown（CLAUDE.md §13.1）。

    python -m stability.rerun_report --snapshot reports/eval_20260912T074455Z_full.json
        [--baseline reports/eval_20260906T092835Z.json] [--baseline-traces reports/traces_20260906T092835Z.jsonl]
        [--out reports/report_20260912T074455Z_full.md]

只做四件事，全部是从两份 JSON 算出来的事实，不含归因：
  1. 头部指标并列（value / n / total），分母不同就标出来，差值照列。
  2. RAGAS 原始口径与交集口径并列。交集口径只有两份快照都存了逐题分数才算，
     缺任一侧就写「无法重算」——不估算、不拿全量均值近似。
  3. 三条事实：faithfulness 分母与 tool_accuracy 的耦合；检索次数与 recall 的逐题对照；
     独立性折价范围（被测 / 裁判 / RAGAS 三条链路是否同一后端）。
  4. 逐题判定变化表。

基线的原始检索次数只在基线轨迹文件（与基线快照同名的 reports/traces_{name}.jsonl，gitignore）存在
且与基线快照逐题核对一致时才有，否则标「未记录」——基线快照只存了折叠后的工具序列，猜不出次数；
不用 traces_latest.jsonl：它跟着 latest.json 走，换了 latest 就是另一次运行的轨迹。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import config

UNRECORDED = "未记录"
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
HEADLINE = ("task_success_rate", "tool_accuracy", "recall_at_k", "recall_at_k_first_call", "faithfulness")
NO_CONTEXT_REASON_PREFIX = "本轮没有检索内容"

# 固定措辞：事实陈述，不含判断
FAITHFULNESS_DENOMINATOR_COUPLING = (
    "faithfulness 的分母由被测系统的检索行为决定（本轮无检索内容的题被排除），"
    "因此与 tool_accuracy 存在耦合：同一行为变化同时抬高 tool_accuracy、缩小 faithfulness 分母。"
)
INDEPENDENCE_SCOPE_EXTENDED = (
    "RAGAS 链路的响应 model 与 system_fingerprint 与被测、裁判链路相同，"
    "因此 RAGAS 四项指标与开放题判定同样无法确认独立性。"
)
RAGAS_INTERSECTION_UNAVAILABLE = "无法重算：快照未存逐题 RAGAS 分数（只有均值、scored_counts 与 evaluated_ids）。"


# ================================================================== 读取
def load_snapshot(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pq(snapshot: Mapping) -> dict[str, dict]:
    return {q["id"]: q for q in snapshot.get("per_question") or []}


def _ragas(snapshot: Mapping) -> dict:
    return (snapshot.get("metrics") or {}).get("ragas") or {}


def _metric(snapshot: Mapping, name: str) -> tuple[float | None, int | None, int | None]:
    """(value, n, total)。faithfulness 取自 RAGAS 段：n = 实际打出分的行数，total = 提交行数。"""
    m = snapshot.get("metrics") or {}
    if name in RAGAS_METRICS:
        rg = _ragas(snapshot)
        return (rg.get("scores") or {}).get(name), (rg.get("scored_counts") or {}).get(name), rg.get("n_submitted")
    blk = m.get(name) or {}
    return blk.get("value"), blk.get("n"), blk.get("total")


# ================================================================== 1. 头部指标
def headline_rows(new: Mapping, base: Mapping) -> list[dict]:
    rows = []
    for name in HEADLINE:
        bv, bn, bt = _metric(base, name)
        nv, nn, nt = _metric(new, name)
        comparable = (bn, bt) == (nn, nt) and bv is not None and nv is not None
        rows.append(
            {
                "metric": name,
                "base_value": bv, "base_n": bn, "base_total": bt,
                "new_value": nv, "new_n": nn, "new_total": nt,
                "diff": (nv - bv) if (bv is not None and nv is not None) else None,
                "comparable": comparable,
            }
        )
    return rows


# ================================================================== 2. RAGAS 交集口径
def _ragas_per_question(snapshot: Mapping) -> dict[str, dict] | None:
    """逐题 RAGAS 分数。两种存法都认：metrics.ragas.per_question[id][metric]，或 per_question[i].ragas[metric]。"""
    rg = _ragas(snapshot)
    if isinstance(rg.get("per_question"), Mapping) and rg["per_question"]:
        return {k: dict(v) for k, v in rg["per_question"].items()}
    found = {q["id"]: dict(q["ragas"]) for q in snapshot.get("per_question") or [] if isinstance(q.get("ragas"), Mapping)}
    return found or None


def ragas_intersection(new: Mapping, base: Mapping) -> dict:
    b_rg, n_rg = _ragas(base), _ragas(new)
    common = sorted(set(b_rg.get("evaluated_ids") or []) & set(n_rg.get("evaluated_ids") or []))
    original = {
        side: {m: {"value": (rg.get("scores") or {}).get(m), "n": (rg.get("scored_counts") or {}).get(m)} for m in RAGAS_METRICS}
        for side, rg in (("base", b_rg), ("new", n_rg))
    }
    b_pq, n_pq = _ragas_per_question(base), _ragas_per_question(new)
    out: dict[str, Any] = {
        "n_common": len(common),
        "common_ids": common,
        "per_question_available": {"base": b_pq is not None, "new": n_pq is not None},
        "original": original,
        "base_reused_from": b_rg.get("reused_from"),
        "base_judge_model": b_rg.get("judge_model"),
        "new_judge_model": n_rg.get("judge_model"),
    }
    if b_pq is None or n_pq is None:
        out["status"] = RAGAS_INTERSECTION_UNAVAILABLE
        return out
    scores: dict[str, dict] = {}
    for m in RAGAS_METRICS:
        b_vals = [b_pq[i][m] for i in common if i in b_pq and b_pq[i].get(m) is not None]
        n_vals = [n_pq[i][m] for i in common if i in n_pq and n_pq[i].get(m) is not None]
        scores[m] = {
            "base": statistics.fmean(b_vals) if b_vals else None, "base_n": len(b_vals),
            "new": statistics.fmean(n_vals) if n_vals else None, "new_n": len(n_vals),
        }
    out["status"] = "recomputed"
    out["scores"] = scores
    return out


# ================================================================== 3a. 分母耦合
def _n_retrieve_new(q: Mapping) -> int | None:
    if q.get("n_retrieve_calls") is not None:
        return int(q["n_retrieve_calls"])
    seq = q.get("tool_sequence_raw")
    return sum(1 for x in seq if x == "retrieve") if seq is not None else None


def _n_retrieve_from_traces(traces: Mapping | None, sid: str) -> int | None:
    if traces is None or sid not in traces:
        return None
    return sum(1 for x in traces[sid].tool_sequence if x == "retrieve")


def faithfulness_denominator_coupling(new: Mapping, base: Mapping, base_traces: Mapping | None = None) -> dict:
    """本次因「无检索内容」被排除、而基线进入了 RAGAS 分母的题，及其 tool_ok 变化。"""
    b_pq, n_pq = _pq(base), _pq(new)
    base_eval = set(_ragas(base).get("evaluated_ids") or [])
    rows = []
    for e in _ragas(new).get("excluded") or []:
        sid, reason = e.get("id"), str(e.get("reason") or "")
        if not reason.startswith(NO_CONTEXT_REASON_PREFIX) or sid not in base_eval:
            continue
        rows.append(
            {
                "id": sid,
                "tool_ok_base": (b_pq.get(sid) or {}).get("tool_ok"),
                "tool_ok_new": (n_pq.get(sid) or {}).get("tool_ok"),
                "n_retrieve_new": _n_retrieve_new(n_pq.get(sid) or {}),
                "n_retrieve_base": _n_retrieve_from_traces(base_traces, sid),
                "excluded_reason": reason,
            }
        )
    rows.sort(key=lambda r: r["id"])
    bt, nt = _metric(base, "tool_accuracy"), _metric(new, "tool_accuracy")
    bf, nf = _metric(base, "faithfulness"), _metric(new, "faithfulness")
    return {
        "ids": [r["id"] for r in rows],
        "rows": rows,
        "tool_accuracy_base": bt[0], "tool_accuracy_new": nt[0],
        "faithfulness_n_base": bf[1], "faithfulness_n_new": nf[1],
        "statement": FAITHFULNESS_DENOMINATOR_COUPLING if rows else None,
    }


# ================================================================== 3b. 检索次数 vs recall
def _expected_groups(q: Mapping) -> list[list[str]]:
    groups = []
    for g in q.get("expected_doc_ids") or []:
        groups.append([g] if isinstance(g, str) else list(g))
    return groups


def _hit_in(doc_ids: Sequence[str] | None, groups: Sequence[Sequence[str]]) -> bool | None:
    if not groups or doc_ids is None:
        return None
    return all(any(d in doc_ids for d in g) for g in groups)


def _first_retrieve_from_trace(trace) -> list[str]:
    for step in getattr(trace, "steps", []) or []:
        for r in getattr(step, "results", []) or []:
            if getattr(r, "name", None) == "retrieve" and getattr(r, "retrieved", None):
                return [c.doc_id for c in r.retrieved]
    return []


def retrieval_vs_recall(new: Mapping, base: Mapping, base_traces: Mapping | None = None) -> dict:
    b_pq, n_pq = _pq(base), _pq(new)
    new_counts = {sid: _n_retrieve_new(q) for sid, q in n_pq.items()}
    base_counts = {sid: _n_retrieve_from_traces(base_traces, sid) for sid in b_pq} if base_traces is not None else {}
    nc = [c for c in new_counts.values() if c is not None]
    bc = [c for c in base_counts.values() if c is not None]
    rows = []
    direction = {"up": [], "down": [], "same": [], "unknown": []}
    for sid in sorted(n_pq):
        q, bq = n_pq[sid], b_pq.get(sid)
        if bq is None:
            continue
        rb, rn = bq.get("recall_at_k"), q.get("recall_at_k")
        cb, cn = base_counts.get(sid), new_counts.get(sid)
        if cb is None or cn is None:
            direction["unknown"].append(sid)
        elif cn > cb:
            direction["up"].append(sid)
        elif cn < cb:
            direction["down"].append(sid)
        else:
            direction["same"].append(sid)
        if rb == rn:
            continue
        groups = _expected_groups(q)
        first_new = q.get("first_retrieve_doc_ids")
        first_base = _first_retrieve_from_trace(base_traces[sid]) if base_traces is not None and sid in base_traces else None
        rows.append(
            {
                "id": sid,
                "recall_base": rb, "recall_new": rn,
                "n_retrieve_base": cb, "n_retrieve_new": cn,
                "union_hit_base": _hit_in(bq.get("retrieved_doc_ids"), groups),
                "union_hit_new": _hit_in(q.get("retrieved_doc_ids"), groups),
                "first_hit_new": _hit_in(first_new, groups) if first_new is not None else None,
                "first_same_as_base": (first_new == first_base) if (first_new is not None and first_base is not None) else None,
            }
        )
    return {
        "median_new": statistics.median(nc) if nc else None,
        "median_base": statistics.median(bc) if bc else None,
        "n_new": len(nc), "n_base": len(bc),
        "base_counts_source": "基线轨迹文件" if base_traces is not None else UNRECORDED,
        "direction_counts": {k: len(v) for k, v in direction.items()},
        "direction_ids": direction,
        "rows": rows,
    }


# ================================================================== 3c. 独立性范围
def independence_scope(new: Mapping) -> dict:
    served = (new.get("meta") or {}).get("served") or {}
    chains, pairs, unrecorded = {}, set(), []
    for chain in ("agent", "judge", "ragas"):
        blk = served.get(chain) or {}
        models, fps = blk.get("response_model_counts") or {}, blk.get("system_fingerprint_counts") or {}
        recorded = bool(models) and blk.get("status") != UNRECORDED and UNRECORDED not in models and UNRECORDED not in fps
        chains[chain] = {
            "requested_model": blk.get("requested_model"), "n_calls": blk.get("n_calls"),
            "response_model_counts": models, "system_fingerprint_counts": fps, "recorded": recorded,
        }
        if not recorded:
            unrecorded.append(chain)
            continue
        # 各链路内 model 与指纹分别只有一种取值时，才能把它当成一个 (model, fp) 对
        if len(models) == 1 and len(fps) == 1:
            pairs.add((next(iter(models)), next(iter(fps))))
        else:
            pairs.update((m, f) for m in models for f in fps)
    single = (len(pairs) == 1) if not unrecorded else None
    return {
        "chains": chains,
        "pairs": sorted([list(p) for p in pairs]),
        "unrecorded_chains": unrecorded,
        "single_backend": single,
        "statement": INDEPENDENCE_SCOPE_EXTENDED if single else None,
    }


# ================================================================== 4. 逐题判定变化
def verdict_changes(new: Mapping, base: Mapping) -> list[dict]:
    b_pq, n_pq = _pq(base), _pq(new)
    rows = []
    for sid in sorted(n_pq):
        q, bq = n_pq[sid], b_pq.get(sid)
        if bq is None:
            continue
        if bq.get("success") == q.get("success") and bq.get("tool_ok") == q.get("tool_ok"):
            continue
        rows.append(
            {
                "id": sid, "answer_type": q.get("answer_type"), "success_source": q.get("success_source"),
                "success_base": bq.get("success"), "success_new": q.get("success"),
                "tool_ok_base": bq.get("tool_ok"), "tool_ok_new": q.get("tool_ok"),
                "actual_tool_base": bq.get("actual_tool"), "actual_tool_new": q.get("actual_tool"),
            }
        )
    return rows


# ================================================================== 基线轨迹与基线快照配对
def default_baseline_traces(baseline_snapshot: Path | str) -> Path:
    """基线轨迹的默认位置：与基线快照同名的 traces_{name}.jsonl（eval_{name}.json -> traces_{name}.jsonl）。

    不能默认 traces_latest.jsonl：它跟着 latest.json 走，latest 一换它就是另一次运行的轨迹，
    拿来当基线的检索次数就是张冠李戴（2026-09-12 实际发生过一次，靠 sha256 对比才发现）。
    """
    p = Path(baseline_snapshot)
    stem = p.stem
    name = stem[len("eval_"):] if stem.startswith("eval_") else stem
    return p.parent / f"traces_{name}.jsonl"


def traces_match_snapshot(traces: Mapping, snapshot: Mapping) -> dict:
    """逐题核对轨迹与快照是否同一次运行：答案原文与 retrieved_doc_ids 都要相等。"""
    pq = _pq(snapshot)
    mismatched, missing = [], []
    n_checked = 0
    for sid, q in pq.items():
        t = traces.get(sid)
        if t is None:
            missing.append(sid)
            continue
        n_checked += 1
        if getattr(t, "answer", None) != q.get("answer") or list(getattr(t, "retrieved_doc_ids", []) or []) != list(q.get("retrieved_doc_ids") or []):
            mismatched.append(sid)
    return {"n_checked": n_checked, "mismatched": sorted(mismatched), "missing": sorted(missing)}


# ================================================================== 汇总 / 渲染
def build(new: Mapping, base: Mapping, *, base_traces: Mapping | None, new_name: str, base_name: str,
          base_traces_name: str | None = None) -> dict:
    check: dict = {"given": base_traces is not None, "used": False, "n_checked": 0, "mismatched": [], "missing": []}
    if base_traces is not None:
        check.update(traces_match_snapshot(base_traces, base))
        check["used"] = not check["mismatched"] and not check["missing"]
        if not check["used"]:
            base_traces = None  # 对不上就不用：宁可标「未记录」，不拿别的运行冒充基线
    return {
        "new_name": new_name, "base_name": base_name, "base_traces_name": base_traces_name,
        "base_traces_check": check,
        "new_meta": {k: (new.get("meta") or {}).get(k) for k in ("git_commit", "timestamp_utc", "started_utc", "finished_utc", "model", "requested_model", "judge_requested_model")},
        "base_meta": {
            **{k: (base.get("meta") or {}).get(k) for k in ("git_commit", "timestamp_utc", "model")},
            # 基线 meta 没有裁判请求名，从回填溯源里取；基线没有任何响应字段
            "judge_requested_model": (((base.get("metrics") or {}).get("task_success_rate") or {}).get("backfill") or {}).get("judge_model"),
            "served_recorded": bool((base.get("meta") or {}).get("served")),
        },
        "headline": headline_rows(new, base),
        "ragas": ragas_intersection(new, base),
        "denominator_coupling": faithfulness_denominator_coupling(new, base, base_traces),
        "retrieval_vs_recall": retrieval_vs_recall(new, base, base_traces),
        "independence": independence_scope(new),
        "verdict_changes": verdict_changes(new, base),
    }


def _f(v, d: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def _fn(v, n, t) -> str:
    return "—" if v is None else f"{_f(v)} (n={n}/{t})"


def _served_cell(ind: Mapping) -> str:
    """三条链路的响应模型名按次计数并列；未记录的链路写「未记录」，不回填请求名。"""
    parts = []
    for chain in ("agent", "judge", "ragas"):
        c = (ind.get("chains") or {}).get(chain) or {}
        if c.get("recorded"):
            parts.append(", ".join(f"`{k}`×{v}" for k, v in c["response_model_counts"].items()))
        else:
            parts.append(UNRECORDED)
    return " / ".join(parts)


def render_markdown(s: Mapping) -> str:
    L: list[str] = []
    nm, bm = s["new_meta"], s["base_meta"]
    L += [
        "# 主评测对照重跑报告（脚本生成，勿手改）", "",
        f"> 由 `python -m stability.rerun_report` 从 `{s['new_name']}` 与基线 `{s['base_name']}` 生成。",
        "> 本报告只列两份快照算出来的事实；差异的归因不在本报告范围内。", "",
        "| | 基线 | 本次 |", "|---|---|---|",
        f"| 快照 | `{s['base_name']}` | `{s['new_name']}` |",
        f"| git commit | {bm.get('git_commit')} | {nm.get('git_commit')} |",
        f"| 时间（UTC） | {bm.get('timestamp_utc')} | {nm.get('started_utc') or nm.get('timestamp_utc')} → {nm.get('finished_utc') or '—'} |",
        f"| 请求模型名（被测 / 裁判） | `{bm.get('model')}` / `{bm.get('judge_requested_model') or '—'}` | `{nm.get('requested_model') or nm.get('model')}` / `{nm.get('judge_requested_model')}` |",
        f"| 响应模型名（被测 / 裁判 / RAGAS，按次计数） | {UNRECORDED if not bm.get('served_recorded') else '已记录'} | {_served_cell(s['independence'])}（详见「事实 3」） |",
        "",
    ]

    # 1. 头部指标
    L += ["## 头部指标", "", "| 指标 | 基线 | 本次 | 差 | 可比性 |", "|---|---|---|---|---|"]
    for r in s["headline"]:
        comp = "分母相同" if r["comparable"] else ("分母不同（闸 B 口径：incomparable）" if r["diff"] is not None else "—")
        L.append(f"| {r['metric']} | {_fn(r['base_value'], r['base_n'], r['base_total'])} | {_fn(r['new_value'], r['new_n'], r['new_total'])} | {('—' if r['diff'] is None else f'{r['diff']:+.3f}')} | {comp} |")
    L.append("")

    # 2. RAGAS
    rg = s["ragas"]
    L += ["## RAGAS：原始口径与交集口径并列", "",
          f"基线 RAGAS 段：judge_model 请求名 `{rg.get('base_judge_model')}`" + (f"，复用自 `{rg['base_reused_from']}`" if rg.get("base_reused_from") else "") +
          f"；本次：judge_model 请求名 `{rg.get('new_judge_model')}`。两侧 evaluated_ids 交集 n={rg['n_common']}。", "",
          f"| 指标 | 原始口径 · 基线 (n={rg['original']['base']['faithfulness']['n']}) | 原始口径 · 本次 (n={rg['original']['new']['faithfulness']['n']}) | 交集口径 n={rg['n_common']} · 基线 | 交集口径 n={rg['n_common']} · 本次 |",
          "|---|---|---|---|---|"]
    for m in RAGAS_METRICS:
        ob, on = rg["original"]["base"][m], rg["original"]["new"][m]
        if rg["status"] == "recomputed":
            ib, inn = rg["scores"][m]["base"], rg["scores"][m]["new"]
            L.append(f"| {m} | {_f(ob['value'])} (n={ob['n']}) | {_f(on['value'])} (n={on['n']}) | {_f(ib)} (n={rg['scores'][m]['base_n']}) | {_f(inn)} (n={rg['scores'][m]['new_n']}) |")
        else:
            L.append(f"| {m} | {_f(ob['value'])} (n={ob['n']}) | {_f(on['value'])} (n={on['n']}) | 无法重算 | 无法重算 |")
    L.append("")
    if rg["status"] != "recomputed":
        avail = rg["per_question_available"]
        L.append(f"交集口径：**{rg['status']}** 逐题分数存在与否：基线 {'有' if avail['base'] else '无'}、本次 {'有' if avail['new'] else '无'}。不估算，不用全量均值近似；原始口径保留如上。")
    else:
        L.append("交集口径为两份快照逐题分数在共同题上的均值，与原始口径并列，不替换原始口径。")
    L.append("")

    # 3a. 分母耦合
    dc = s["denominator_coupling"]
    L += ["## 事实 1：faithfulness 分母与 tool_accuracy 的耦合", ""]
    if dc["ids"]:
        L += [f"本次因「无检索内容」被排除、而基线进入了 RAGAS 分母的题：{', '.join(dc['ids'])}。", "",
              "| 题 | tool_ok 基线→本次 | retrieve 次数 基线→本次 | 本次排除原因 |", "|---|---|---|---|"]
        for r in dc["rows"]:
            L.append(f"| {r['id']} | {r['tool_ok_base']}→{r['tool_ok_new']} | {UNRECORDED if r['n_retrieve_base'] is None else r['n_retrieve_base']}→{r['n_retrieve_new']} | {r['excluded_reason']} |")
        L += ["", f"同一批题上：tool_accuracy {_f(dc['tool_accuracy_base'])} → {_f(dc['tool_accuracy_new'])}；faithfulness 分母 n {dc['faithfulness_n_base']} → {dc['faithfulness_n_new']}。", "",
              f"**{dc['statement']}**", ""]
    else:
        L += ["本次没有「无检索内容」导致的新增排除，faithfulness 分母未因检索行为变化而缩小。", ""]

    # 3b. 检索次数 vs recall
    rv = s["retrieval_vs_recall"]
    chk = s.get("base_traces_check") or {}
    L += ["## 事实 2：检索次数与 recall 的逐题对照", ""]
    if chk.get("given") and not chk.get("used"):
        L += [f"基线轨迹文件与基线快照不一致（核对 {chk.get('n_checked')} 题：答案或 retrieved_doc_ids 不同 {len(chk.get('mismatched') or [])} 题，"
              f"缺失 {len(chk.get('missing') or [])} 题），**未采用**，基线原始检索次数按「{UNRECORDED}」处理。", ""]
    elif chk.get("used"):
        L += [f"基线轨迹文件已与基线快照逐题核对一致（{chk.get('n_checked')} 题：答案与 retrieved_doc_ids 相同）。", ""]
    L += [
          f"retrieve 调用次数中位数：基线 {_f(rv['median_base'], 1)}（n={rv['n_base']}，来源：{rv['base_counts_source']}）→ 本次 {_f(rv['median_new'], 1)}（n={rv['n_new']}）。"
          f"逐题方向：增加 {rv['direction_counts']['up']}、不变 {rv['direction_counts']['same']}、减少 {rv['direction_counts']['down']}、未知 {rv['direction_counts']['unknown']}。", ""]
    if rv["rows"]:
        L += ["recall@k 有变化的题：", "",
              "| 题 | recall 基线→本次 | retrieve 次数 基线→本次 | 标注块在并集内 基线→本次 | 本次首次检索命中标注块 | 首次检索 doc_id 与基线相同 |",
              "|---|---|---|---|---|---|"]
        for r in rv["rows"]:
            yn = lambda v: "—" if v is None else ("是" if v else "否")
            L.append(f"| {r['id']} | {_f(r['recall_base'], 2)}→{_f(r['recall_new'], 2)} | {UNRECORDED if r['n_retrieve_base'] is None else r['n_retrieve_base']}→{r['n_retrieve_new']} | {yn(r['union_hit_base'])}→{yn(r['union_hit_new'])} | {yn(r['first_hit_new'])} | {yn(r['first_same_as_base'])} |")
        downs = [r["id"] for r in rv["rows"] if r["n_retrieve_base"] is not None and r["n_retrieve_new"] is not None and r["n_retrieve_new"] < r["n_retrieve_base"] and (r["recall_new"] or 0) < (r["recall_base"] or 0)]
        L += ["", f"recall 下降且 retrieve 次数减少的题：{', '.join(downs) if downs else '无'}（{len(downs)}/{len(rv['rows'])}）。", ""]
    else:
        L += ["recall@k 逐题无变化。", ""]

    # 3c. 独立性范围
    ind = s["independence"]
    L += ["## 事实 3：独立性折价范围", "", "| 链路 | 请求模型名 | 响应 model（逐条计数） | system_fingerprint（逐条计数） | 调用数 |", "|---|---|---|---|---|"]
    for chain, c in ind["chains"].items():
        if c["recorded"]:
            ms = ", ".join(f"`{k}`×{v}" for k, v in c["response_model_counts"].items())
            fs = ", ".join(f"`{k}`×{v}" for k, v in c["system_fingerprint_counts"].items())
        else:
            ms = fs = UNRECORDED
        L.append(f"| {chain} | `{c['requested_model']}` | {ms} | {fs} | {c['n_calls'] if c['n_calls'] is not None else '—'} |")
    L.append("")
    if ind["single_backend"]:
        L += [f"三条链路的 (响应 model, 指纹) 只有一种取值：{ind['pairs'][0][0]} / {ind['pairs'][0][1]}。", "", f"**{ind['statement']}**", ""]
    elif ind["single_backend"] is None:
        L += [f"链路 {', '.join(ind['unrecorded_chains'])} 未记录响应字段，无法判断三条链路是否同一后端。", ""]
    else:
        L += [f"三条链路的 (响应 model, 指纹) 取值：{ind['pairs']}。", ""]

    # 4. 逐题判定变化
    vc = s["verdict_changes"]
    L += ["## 逐题判定变化（success 或 tool_ok 有变化的题）", ""]
    if vc:
        L += ["| 题 | 类型 | 判定来源 | success 基线→本次 | tool_ok 基线→本次 | 工具序列（折叠）基线→本次 |", "|---|---|---|---|---|---|"]
        for r in vc:
            L.append(f"| {r['id']} | {r['answer_type']} | {r['success_source'] or '—'} | {r['success_base']}→{r['success_new']} | {r['tool_ok_base']}→{r['tool_ok_new']} | {r['actual_tool_base']}→{r['actual_tool_new']} |")
        t2f = [r["id"] for r in vc if r["success_base"] and r["success_new"] is False]
        f2t = [r["id"] for r in vc if r["success_new"] and r["success_base"] is False]
        L += ["", f"success True→False {len(t2f)} 题：{', '.join(t2f) or '无'}；False→True {len(f2t)} 题：{', '.join(f2t) or '无'}。", ""]
    else:
        L += ["无。", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="对照重跑报告（离线，脚本生成）")
    parser.add_argument("--snapshot", required=True, help="另存的主评测快照，如 reports/eval_20260912T074455Z_full.json")
    parser.add_argument("--baseline", default=str(config.REPORTS_DIR / "eval_20260906T092835Z.json"))
    parser.add_argument("--baseline-traces", default=None,
                        help="基线轨迹（gitignore）；缺省为与基线快照同名的 traces_{name}.jsonl；不存在或与基线快照对不上则基线原始检索次数标「未记录」")
    parser.add_argument("--out", default=None, help="输出 markdown；缺省 reports/report_{name}.md（name 取自快照文件名 eval_{name}.json）")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    snap_path, base_path = Path(args.snapshot), Path(args.baseline)
    new, base = load_snapshot(snap_path), load_snapshot(base_path)
    traces_path = Path(args.baseline_traces) if args.baseline_traces else default_baseline_traces(base_path)
    base_traces = None
    if traces_path.exists():
        from src.eval import report as report_mod

        base_traces = report_mod.load_traces(traces_path)
    stem = snap_path.stem
    name = stem[len("eval_"):] if stem.startswith("eval_") else stem
    out = Path(args.out) if args.out else (snap_path.parent / f"report_{name}.md")

    def _rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(config.PROJECT_ROOT).as_posix()
        except ValueError:
            return p.name

    summary = build(new, base, base_traces=base_traces, new_name=_rel(snap_path), base_name=_rel(base_path),
                    base_traces_name=_rel(traces_path) if base_traces is not None else None)
    md = render_markdown(summary)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
