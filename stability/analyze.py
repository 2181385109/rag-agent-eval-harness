"""原始运行记录 -> 指标汇总（summary.json）+ 报告（report.md）。PERF_SPEC B2。

本模块只做**纯计算**：不跑 Agent、不发请求。答案相似度要用 BGE 编码，
embedder 由调用方注入（测试注入假的；CI 上不算，如实标"未计算"）。

口径定义在 tests/test_stability_analyze.py，这里是实现。

命令行：
    python -m stability.analyze                      # 取 stability/raw/ 里最新的 run_*.jsonl
    python -m stability.analyze --raw stability/raw/run_20260912T010203Z.jsonl
    python -m stability.analyze --judge stability/raw/judge_20260912T020304Z.jsonl
    python -m stability.analyze --no-embed           # 跳过答案相似度（不加载 BGE）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src import config
from src.eval import metrics
from src.eval.datasets import GoldenSample, load_golden_set
from stability.records import RunRecord

STABILITY_DIR = config.PROJECT_ROOT / "stability"
RAW_DIR = STABILITY_DIR / "raw"
SUMMARY_PATH = STABILITY_DIR / "summary.json"
REPORT_PATH = STABILITY_DIR / "report.md"

PHASES = ("llm", "retrieve", "tool", "overhead")


# ------------------------------------------------------------------ 读写
def load_rows(path: Path | str) -> list[RunRecord]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"原始运行记录不存在：{target}")
    rows: list[RunRecord] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(RunRecord.model_validate_json(line))
    return rows


def load_run_meta(raw_path: Path | str) -> dict:
    """原始产物旁边的 *.meta.json（环境、模型、k、日期）。没有就返回空 dict。"""
    meta_path = Path(raw_path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_judge_rows(path: Path | str | None) -> list[dict] | None:
    if path is None:
        return None
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"逐次裁判分文件不存在：{target}")
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_raw(directory: Path | None = None) -> Path:
    files = sorted((directory or RAW_DIR).glob("run_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"{directory or RAW_DIR} 里没有 run_*.jsonl，先跑 python -m stability.run_repeat")
    return files[-1]


def sha256_of(path: Path | str) -> str:
    """文本产物的指纹，**按行结束符归一后**算。

    Windows 上 Python 写出的是 CRLF，git 入库归一成 LF，Linux CI 检出的是 LF——
    按原始字节算指纹会让同一份产物在 CI 上对不上。归一到 LF 后算，指纹只随内容变。
    """
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _relative(path: Path | str | None) -> str | None:
    """仓库内路径记成相对路径（可移植，也不带本机用户名）；仓库外原样记。"""
    if path is None:
        return None
    p = Path(path).resolve()
    try:
        return p.relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


# ------------------------------------------------------------------ 小工具
def _by_sample(rows: Sequence[RunRecord]) -> dict[str, list[RunRecord]]:
    out: dict[str, list[RunRecord]] = {}
    for r in rows:
        out.setdefault(r.sample_id, []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r.run_index)
    return out


def _pct(values: Sequence[float]) -> dict:
    """P50/P95/P99 + 均值/最大，numpy 默认线性插值；空集全部 None，绝不落成 0。"""
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    arr = np.asarray(values, dtype="float64")
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


# ================================================================== 延迟
def latency_section(rows: Sequence[RunRecord]) -> dict:
    """端到端与三段分解。只对 error 为空的行算分位数；错误率的分母是全部行。"""
    ok = [r for r in rows if r.ok]
    n_error = len(rows) - len(ok)

    phase_values = {p: [getattr(r.latency, f"{p}_s") for r in ok] for p in PHASES}
    shares: dict[str, list[float]] = {p: [] for p in PHASES}
    for r in ok:
        if r.latency.total_s > 0:
            for p in PHASES:
                shares[p].append(getattr(r.latency, f"{p}_s") / r.latency.total_s)

    return {
        "note": "wall-clock 秒；分位数为 numpy.percentile 线性插值；只统计 error 为空的行",
        "e2e": _pct([r.latency.total_s for r in ok]),
        "phases": {p: _pct(phase_values[p]) for p in PHASES},
        "share_median": {p: (float(statistics.median(shares[p])) if shares[p] else None) for p in PHASES},
        "llm_calls_per_run": _pct([float(r.latency.n_llm_calls) for r in ok]),
        "single_llm_call": _pct([s for r in ok for s in r.latency.llm_call_s]),
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_error": n_error,
        "error_rate": (n_error / len(rows)) if rows else None,
    }


# ================================================================== 稳定性
def _verdict(sample: GoldenSample, row: RunRecord, judge_lookup: Mapping[tuple[str, int], int]) -> bool | None:
    """单次运行的任务成功判定。闭合题走规则；开放题只认逐次裁判分；报错行算失败。"""
    if not row.ok:
        return False
    if sample.answer_type == "closed":
        return metrics.judge_closed(sample, row.answer)
    score = judge_lookup.get((sample.id, row.run_index))
    return metrics.judge_open(sample, score)


def _index_judge_rows(rows: Sequence[RunRecord], judge_rows: Sequence[Mapping] | None):
    """把逐次裁判分按 (sample_id, run_index) 建索引，指纹对不上的丢弃并记下。"""
    lookup: dict[tuple[str, int], int] = {}
    stale: list[dict] = []
    if not judge_rows:
        return lookup, stale
    fp = {(r.sample_id, r.run_index): r.answer_sha1 for r in rows}
    for jr in judge_rows:
        key = (jr["sample_id"], int(jr["run_index"]))
        if fp.get(key) != jr.get("answer_sha1"):
            stale.append({"sample_id": key[0], "run_index": key[1]})
            continue
        lookup[key] = int(jr["judge_score"])
    return lookup, stale


def stability_section(
    rows: Sequence[RunRecord],
    samples: Sequence[GoldenSample],
    judge_rows: Sequence[Mapping] | None = None,
) -> dict:
    """轨迹自洽率 / 判定自洽率 / 成功率跨次标准差。答案相似度另见 answer_similarity_section。"""
    groups = _by_sample(rows)
    sample_map = {s.id: s for s in samples}
    judge_lookup, stale = _index_judge_rows(rows, judge_rows)

    # ---- 轨迹自洽率：严格口径（逐项相等）为主，折叠版并列
    traj_hits: list[float] = []
    traj_collapsed_hits: list[float] = []
    traj_bad: list[str] = []
    traj_detail: dict[str, dict] = {}
    for sid in sorted(groups):
        rs = groups[sid]
        if len(rs) < 2:
            continue
        strict = len({tuple(r.tool_sequence) for r in rs}) == 1
        collapsed = len({tuple(r.tool_sequence_collapsed) for r in rs}) == 1
        traj_hits.append(1.0 if strict else 0.0)
        traj_collapsed_hits.append(1.0 if collapsed else 0.0)
        if not strict:
            traj_bad.append(sid)
        traj_detail[sid] = {
            "consistent": strict,
            "consistent_collapsed": collapsed,
            "sequences": [r.tool_sequence for r in rs],
        }

    # ---- 判定自洽率
    verdict_hits: list[float] = []
    verdict_bad: list[str] = []
    unjudged: list[str] = []
    source_counts = {"rule": 0, "judge": 0}
    verdict_detail: dict[str, dict] = {}
    for sid in sorted(groups):
        rs = groups[sid]
        sample = sample_map.get(sid)
        if sample is None or len(rs) < 2:
            continue
        verdicts = [_verdict(sample, r, judge_lookup) for r in rs]
        if any(v is None for v in verdicts):
            unjudged.append(sid)
            continue
        consistent = len(set(verdicts)) == 1
        verdict_hits.append(1.0 if consistent else 0.0)
        if not consistent:
            verdict_bad.append(sid)
        source_counts["rule" if sample.answer_type == "closed" else "judge"] += 1
        verdict_detail[sid] = {"consistent": consistent, "verdicts": verdicts}

    # ---- 成功率跨次标准差：按 pass 算成功率，再对 k 个成功率取标准差
    k = max((r.k for r in rows), default=0)
    rates: list[float] = []
    n_per_pass: list[int] = []
    for pass_idx in range(k):
        outcomes: list[float] = []
        for sid, rs in groups.items():
            sample = sample_map.get(sid)
            if sample is None:
                continue
            for r in rs:
                if r.run_index != pass_idx:
                    continue
                v = _verdict(sample, r, judge_lookup)
                if v is not None:
                    outcomes.append(1.0 if v else 0.0)
        n_per_pass.append(len(outcomes))
        rates.append(float(sum(outcomes) / len(outcomes)) if outcomes else float("nan"))
    valid_rates = [x for x in rates if not np.isnan(x)]
    std_sample = float(np.std(valid_rates, ddof=1)) if len(valid_rates) >= 2 else None
    std_pop = float(np.std(valid_rates, ddof=0)) if len(valid_rates) >= 2 else None

    return {
        "trajectory_consistency": {
            "rule": "k 次运行的工具调用序列逐项完全一致的题占比（不折叠连续重复）",
            "value": _mean(traj_hits),
            "collapsed_value": _mean(traj_collapsed_hits),
            "n": len(traj_hits),
            "total": len(groups),
            "inconsistent_ids": traj_bad,
            "detail": traj_detail,
        },
        "verdict_consistency": {
            "rule": (
                "k 次运行的任务成功判定全部相同的题占比；闭合题走 answer_keys 规则，"
                f"开放题走逐次裁判分（>= {config.OPEN_SUCCESS_THRESHOLD} 为成功），"
                "没有逐次裁判分的开放题不计入分母；报错的运行计为失败"
            ),
            "value": _mean(verdict_hits),
            "n": len(verdict_hits),
            "total": len(groups),
            "source_counts": source_counts,
            "inconsistent_ids": verdict_bad,
            "unjudged_ids": unjudged,
            "stale_judge_rows": stale,
            "detail": verdict_detail,
        },
        "success_rate_per_pass": {
            "rule": "第 r 次 pass 的成功率 = 该 pass 里有判定的题的成功比例；std_sample 为 k 个成功率的样本标准差（ddof=1）",
            "rates": [None if np.isnan(x) else x for x in rates],
            "n_per_pass": n_per_pass,
            "std_sample": std_sample,
            "std_population": std_pop,
            "k": k,
        },
    }


# ================================================================== 答案相似度
def answer_similarity_section(
    rows: Sequence[RunRecord], samples: Sequence[GoldenSample], embedder
) -> dict:
    """同题 k 个答案两两余弦相似度。报错行不参与；不足两个有效答案的题跳过。"""
    groups = _by_sample(rows)
    known = {s.id for s in samples}
    per_q: dict[str, dict] = {}

    for sid in sorted(groups):
        if sid not in known:
            continue
        answers = [r.answer for r in groups[sid] if r.ok and (r.answer or "").strip()]
        if len(answers) < 2:
            continue
        vecs = np.asarray(embedder.encode(answers, is_query=False), dtype="float64")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        sims = [float(np.dot(vecs[i], vecs[j])) for i, j in combinations(range(len(answers)), 2)]
        per_q[sid] = {"mean": float(np.mean(sims)), "min": float(np.min(sims)), "n_pairs": len(sims)}

    if not per_q:
        return {"mean": None, "min": None, "min_id": None, "n": 0, "total": len(groups), "per_question": {}}
    min_id = min(per_q, key=lambda s: per_q[s]["min"])
    return {
        "rule": "同题 k 个答案两两余弦相似度（BGE 向量）；mean = 题内均值的均值，min = 题内最小值的最小值",
        "embedding_model": getattr(embedder, "model_name", type(embedder).__name__),
        "mean": float(np.mean([v["mean"] for v in per_q.values()])),
        "min": per_q[min_id]["min"],
        "min_id": min_id,
        "n": len(per_q),
        "total": len(groups),
        "per_question": per_q,
    }


# ================================================================== 成本
def cost_section(rows: Sequence[RunRecord], samples: Sequence[GoldenSample]) -> dict:
    groups = _by_sample(rows)
    totals = [r.tokens.total for r in rows]
    per_question_mean = {sid: _mean([r.tokens.total for r in rs]) for sid, rs in groups.items()}
    cache_hits = [r.tokens.cache_hit for r in rows if r.tokens.cache_hit is not None]
    prompts = [r.tokens.prompt for r in rows]
    return {
        "n_runs": len(rows),
        "n_questions": len(groups),
        "total_tokens_all_runs": int(sum(totals)),
        "total_prompt_tokens": int(sum(prompts)),
        "total_completion_tokens": int(sum(r.tokens.completion for r in rows)),
        "mean_tokens_per_run": _mean(totals),
        # 一轮完整评测 = 每题取 k 次均值再求和（k 次都跑完时 = 总量 / k）
        "tokens_per_full_round": float(sum(v for v in per_question_mean.values() if v is not None)),
        "cache_hit_tokens_total": int(sum(cache_hits)) if cache_hits else None,
        "cache_hit_share_of_prompt": (float(sum(cache_hits) / sum(prompts)) if cache_hits and sum(prompts) else None),
        "by_run_index": {
            str(i): {
                "mean_tokens": _mean([r.tokens.total for r in rows if r.run_index == i]),
                "mean_cache_hit": _mean([r.tokens.cache_hit for r in rows if r.run_index == i and r.tokens.cache_hit is not None]),
                "p50_total_s": (_pct([r.latency.total_s for r in rows if r.run_index == i and r.ok])["p50"]),
            }
            for i in sorted({r.run_index for r in rows})
        },
    }


# ================================================================== 对照主评测快照
def main_snapshot_comparison(
    rows: Sequence[RunRecord],
    samples: Sequence[GoldenSample],
    judge_rows: Sequence[Mapping] | None,
    latest_path: Path | str | None = None,
) -> dict | None:
    """逐题对照：本次 k 轮的判定 vs 主评测快照 reports/latest.json 的判定。

    主评测是单次运行；这里是 k 次。两边判定口径相同（闭合题规则、开放题裁判 >=2），
    差异只可能来自被测模型的行为变化或裁判波动。列出所有"主评测通过、本次 k 次里
    至少一次不通过"或反过来的题，附主评测快照的时间与 commit，让漂移可追溯。
    本节**不进可复现闸**：latest.json 会随下次主评测更新，它不是本次原始产物。
    """
    target = Path(latest_path) if latest_path else (config.REPORTS_DIR / "latest.json")
    if not target.exists():
        return None
    report = json.loads(target.read_text(encoding="utf-8"))
    main = {q["id"]: q for q in report.get("per_question", [])}
    groups = _by_sample(rows)
    sample_map = {s.id: s for s in samples}
    judge_lookup, _ = _index_judge_rows(rows, judge_rows)

    changed: list[dict] = []
    agree = 0
    compared = 0
    for sid in sorted(groups):
        sample = sample_map.get(sid)
        q = main.get(sid)
        if sample is None or q is None or q.get("success") is None:
            continue
        verdicts = [_verdict(sample, r, judge_lookup) for r in groups[sid]]
        if any(v is None for v in verdicts):
            continue
        compared += 1
        if all(v == q["success"] for v in verdicts):
            agree += 1
        else:
            changed.append(
                {
                    "id": sid,
                    "answer_type": sample.answer_type,
                    "main_success": q["success"],
                    "run_verdicts": verdicts,
                    "main_tool_sequence": q.get("actual_tool"),
                    "run_tool_sequences": [r.tool_sequence_collapsed for r in groups[sid]],
                }
            )
    main_metric = report.get("metrics", {}).get("task_success_rate", {})
    return {
        "snapshot": target.relative_to(config.PROJECT_ROOT).as_posix(),
        "snapshot_timestamp_utc": report.get("meta", {}).get("timestamp_utc"),
        "snapshot_git_commit": report.get("meta", {}).get("git_commit"),
        "snapshot_task_success_rate": main_metric.get("value"),
        "snapshot_n": main_metric.get("n"),
        "n_compared": compared,
        "n_all_runs_agree": agree,
        "changed": changed,
    }


# ================================================================== 汇总
def build_summary(
    rows: Sequence[RunRecord],
    samples: Sequence[GoldenSample],
    *,
    run_meta: Mapping | None = None,
    judge_rows: Sequence[Mapping] | None = None,
    embedder=None,
    raw_path: Path | str | None = None,
    judge_path: Path | str | None = None,
) -> dict:
    run_meta = dict(run_meta or {})
    k = int(run_meta.get("k") or max((r.k for r in rows), default=0))
    stability = stability_section(rows, samples, judge_rows=judge_rows)
    stability["answer_similarity"] = (
        answer_similarity_section(rows, samples, embedder) if embedder is not None else None
    )
    stability["main_snapshot_comparison"] = main_snapshot_comparison(rows, samples, judge_rows)
    return {
        "meta": {
            "analyzed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_file": _relative(raw_path),
            "raw_sha256": sha256_of(raw_path) if raw_path and Path(raw_path).exists() else None,
            "judge_rows_given": judge_rows is not None,
            "judge_file": _relative(judge_path),
            "judge_sha256": sha256_of(judge_path) if judge_path and Path(judge_path).exists() else None,
            "n_rows": len(rows),
            "n_questions": len({r.sample_id for r in rows}),
            "k": k,
            "served_models": sorted({r.response_model for r in rows if r.response_model}),
            "system_fingerprints": sorted({r.system_fingerprint for r in rows if r.system_fingerprint}),
            "run_meta": run_meta,
        },
        "latency": latency_section(rows),
        "stability": stability,
        "cost": cost_section(rows, samples),
    }


# ================================================================== 报告
def _f(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _s(value) -> str:
    return "—" if value is None else f"{value:.2f}s"


def render_markdown(summary: dict) -> str:
    m = summary["meta"]
    rm = m.get("run_meta", {})
    env = rm.get("env", {})
    lat = summary["latency"]
    st = summary["stability"]
    cost = summary["cost"]
    k = m["k"]
    n_rows = m["n_rows"]
    n_q = m.get("n_questions")
    tag = f"n={n_rows}（{n_q} 题 × k={k}），temperature={rm.get('temperature', '—')}，model={rm.get('model', '—')}"

    lines = [
        "# 稳定性与延迟分解报告（脚本生成，勿手改）",
        "",
        f"> 本文件由 `python -m stability.analyze` 从 `{m.get('raw_file')}` 生成；",
        f"> 原始产物 sha256 = `{m.get('raw_sha256')}`；分析时间 {m['analyzed_at_utc']}。",
        "> 每个数字旁都带样本量 n、重复次数 k 与环境标识（PERF_SPEC §0.3）。",
        "",
        "## 环境与随机性来源",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 被测模型 | 请求 `{rm.get('model', '—')}`；响应 model 字段：{m.get('served_models') or '—'}；system_fingerprint：{m.get('system_fingerprints') or '—'} |",
        f"| temperature | {rm.get('temperature', '—')} |",
        f"| seed | {rm.get('seed', '—')} |",
        f"| embedding | {rm.get('embedding_model', '—')}（query 指令前缀：{rm.get('use_query_instruction', '—')}） |",
        f"| 检索参数 | top_k={rm.get('retrieve_top_k', '—')}，chunk={rm.get('chunk_size', '—')}/{rm.get('chunk_overlap', '—')}，max_steps={rm.get('max_agent_steps', '—')} |",
        f"| git commit | {rm.get('git_commit', '—')} |",
        f"| 运行时间 | {rm.get('started_utc', '—')} → {rm.get('finished_utc', '—')} |",
        f"| 顺序 | {rm.get('ordering', '—')} |",
        f"| 硬件 | {env.get('cpu', '—')} × {env.get('cpu_count', '—')} 核，RAM {env.get('ram_gb', '—')} GB |",
        f"| 软件 | {env.get('os', '—')}；Python {env.get('python', '—')}；openai {env.get('openai', '—')}；sentence-transformers {env.get('sentence_transformers', '—')} |",
        f"| 网络 | 压测客户端与 DeepSeek API 之间为公网；LLM 延迟含网络往返与服务端排队 |",
        "",
        "## 延迟分解",
        "",
        f"{tag}；有效行 n_ok={lat['n_ok']}，报错行 n_error={lat['n_error']}，错误率 {_f(lat['error_rate'])}。",
        "",
        "| 段 | n | P50 | P95 | P99 | 均值 | 最大 | 占比中位数 |",
        "|---|---|---|---|---|---|---|---|",
        f"| 端到端 | {lat['e2e']['n']} | {_s(lat['e2e']['p50'])} | {_s(lat['e2e']['p95'])} | {_s(lat['e2e']['p99'])} | {_s(lat['e2e']['mean'])} | {_s(lat['e2e']['max'])} | 1.00 |",
    ]
    names = {"llm": "LLM 调用", "retrieve": "检索（BGE+FAISS）", "tool": "工具（calc）", "overhead": "编排开销"}
    for p in PHASES:
        ph = lat["phases"][p]
        lines.append(
            f"| {names[p]} | {ph['n']} | {_s(ph['p50'])} | {_s(ph['p95'])} | {_s(ph['p99'])} | {_s(ph['mean'])} | {_s(ph['max'])} | {_f(lat['share_median'][p], 2)} |"
        )
    sc = lat["single_llm_call"]
    lines += [
        "",
        f"单次 LLM 调用（n={sc['n']} 次）：P50 {_s(sc['p50'])} / P95 {_s(sc['p95'])} / P99 {_s(sc['p99'])}；"
        f"每次运行 LLM 调用次数 P50 {_f(lat['llm_calls_per_run']['p50'], 1)}、最大 {_f(lat['llm_calls_per_run']['max'], 0)}。",
        "",
        f"**结论**：LLM 调用占端到端延迟的 {_f(lat['share_median']['llm'] * 100 if lat['share_median']['llm'] is not None else None, 1)}%（占比中位数），"
        f"检索占 {_f(lat['share_median']['retrieve'] * 100 if lat['share_median']['retrieve'] is not None else None, 1)}%，"
        f"工具占 {_f(lat['share_median']['tool'] * 100 if lat['share_median']['tool'] is not None else None, 1)}%，"
        f"编排开销占 {_f(lat['share_median']['overhead'] * 100 if lat['share_median']['overhead'] is not None else None, 1)}%。",
        "",
        "## 稳定性（三个层级，逐级放宽）",
        "",
        "| 指标 | 值 | 分母 | 口径 |",
        "|---|---|---|---|",
    ]
    tj = st["trajectory_consistency"]
    vc = st["verdict_consistency"]
    sim = st.get("answer_similarity")
    sp = st["success_rate_per_pass"]
    lines += [
        f"| 轨迹自洽率（严格） | **{_f(tj['value'])}** | n={tj['n']}/{tj['total']}，k={k} | {tj['rule']} |",
        f"| 轨迹自洽率（折叠连续重复） | {_f(tj['collapsed_value'])} | n={tj['n']}/{tj['total']}，k={k} | 参考口径 |",
        f"| 判定自洽率 | **{_f(vc['value'])}** | n={vc['n']}/{vc['total']}（规则 {vc['source_counts']['rule']}、裁判 {vc['source_counts']['judge']}），k={k} | {vc['rule']} |",
    ]
    if sim is None:
        lines.append("| 答案相似度 | 未计算 | — | 本次分析未加载 embedding（--no-embed 或 CI） |")
    elif sim["n"] == 0:
        lines.append(f"| 答案相似度 | — | n=0/{sim['total']} | 没有任何题有 ≥2 个有效答案 |")
    else:
        lines.append(
            f"| 答案相似度 | 均值 **{_f(sim['mean'])}**，最小 **{_f(sim['min'])}**（{sim['min_id']}） | n={sim['n']}/{sim['total']} 题，k={k} | {sim['rule']}；模型 {sim['embedding_model']} |"
        )
    lines += [
        f"| 成功率跨次标准差 | **{_f(sp['std_sample'])}**（样本，ddof=1）；{_f(sp['std_population'])}（总体） | k={k}，各 pass 分母 {sp['n_per_pass']} | 各 pass 成功率 {[None if x is None else round(x, 3) for x in sp['rates']]} |",
        "",
    ]
    if tj["inconsistent_ids"]:
        lines.append(f"轨迹不自洽的题：{', '.join(tj['inconsistent_ids'])}。")
        for sid in tj["inconsistent_ids"]:
            seqs = tj["detail"][sid]["sequences"]
            lines.append(f"- `{sid}`：{seqs}")
    else:
        lines.append("轨迹不自洽的题：无。")
    if vc["inconsistent_ids"]:
        lines.append(f"判定不自洽的题：{', '.join(vc['inconsistent_ids'])}。")
        for sid in vc["inconsistent_ids"]:
            lines.append(f"- `{sid}`：{vc['detail'][sid]['verdicts']}")
    else:
        lines.append("判定不自洽的题：无。")
    if vc["unjudged_ids"]:
        lines.append(f"未计入判定自洽率的开放题（无逐次裁判分）：{', '.join(vc['unjudged_ids'])}。")
    if vc["stale_judge_rows"]:
        lines.append(f"⚠ 指纹对不上而被丢弃的裁判分：{vc['stale_judge_rows']}。")
    lines += [
        "",
        "### 解读",
        "",
    ]
    if tj["value"] is not None and vc["value"] is not None:
        if tj["value"] < vc["value"]:
            lines.append(
                f"轨迹自洽率（{_f(tj['value'])}）低于判定自洽率（{_f(vc['value'])}）："
                "Agent 走的路径不稳，但结果凑到了同一个判定上——路径层面的非确定性被判定层面掩盖了。"
            )
        elif tj["value"] == vc["value"]:
            lines.append(f"轨迹自洽率与判定自洽率相等（{_f(tj['value'])}），路径与结果的稳定性在本次样本上没有分层。")
        else:
            lines.append(
                f"轨迹自洽率（{_f(tj['value'])}）高于判定自洽率（{_f(vc['value'])}）："
                "路径稳定但结果摇摆——同样的工具序列下答案措辞变化足以翻转判定。"
            )
    if tj["value"] is not None and tj["value"] < 1.0:
        lines.append("temperature=0 **没有**给出确定性输出：轨迹自洽率 < 1.0 不是 bug，它就是结论本身（PERF_SPEC B3；见 LIMITATIONS.md）。")
    cmp_ = st.get("main_snapshot_comparison")
    if cmp_:
        rates = [x for x in sp["rates"] if x is not None]
        lines += [
            "",
            "### 对照主评测快照（逐题，同一判定口径）",
            "",
            f"主评测快照 `{cmp_['snapshot']}`（{cmp_['snapshot_timestamp_utc']}，commit {cmp_['snapshot_git_commit']}）为单次运行，"
            f"任务成功率 {_f(cmp_['snapshot_task_success_rate'])}（n={cmp_['snapshot_n']}）；"
            f"本次 k={k} 轮各 pass 成功率均值 {_f(float(np.mean(rates)) if rates else None)}。"
            f"可对照 {cmp_['n_compared']} 题中，{cmp_['n_all_runs_agree']} 题 k 次判定与快照全部一致。",
            "",
        ]
        if cmp_["changed"]:
            lines += ["| 题 | 类型 | 快照判定 | 本次 k 次判定 | 快照工具序列 | 本次工具序列（折叠） |", "|---|---|---|---|---|---|"]
            for c in cmp_["changed"]:
                lines.append(
                    f"| {c['id']} | {c['answer_type']} | {c['main_success']} | {c['run_verdicts']} | {c['main_tool_sequence']} | {c['run_tool_sequences']} |"
                )
            lines += [
                "",
                "判定口径两边相同，差异只能来自被测模型的行为变化或裁判波动——这正是回归闸（闸 B）要抓的漂移；"
                "要确认，需重跑一次主评测（`python -m src.eval.report`）让闸 B 正式比对。",
            ]
        else:
            lines.append("没有任何题的判定与快照不同。")
    lines += [
        "",
        "## 成本",
        "",
        f"| 项 | 值 | 分母 |",
        f"|---|---|---|",
        f"| 每次运行平均 token | {_f(cost['mean_tokens_per_run'], 0)} | n={cost['n_runs']} 次运行 |",
        f"| 全部 {k} 轮总 token | {cost['total_tokens_all_runs']}（prompt {cost['total_prompt_tokens']} / completion {cost['total_completion_tokens']}） | n={cost['n_runs']} |",
        f"| 跑完一轮完整评测的 token | {_f(cost['tokens_per_full_round'], 0)} | {cost['n_questions']} 题，每题取 k={k} 次均值 |",
        f"| 命中服务端缓存的 prompt token | {cost['cache_hit_tokens_total'] if cost['cache_hit_tokens_total'] is not None else '—'}（占 prompt 的 {_f(cost['cache_hit_share_of_prompt'])}） | n={cost['n_runs']} |",
        "",
        "按 pass 看缓存与延迟（run_index=0 是首次，之后大概率命中 DeepSeek 服务端 KV 缓存）：",
        "",
        "| pass | 平均 token | 平均缓存命中 | 端到端 P50 |",
        "|---|---|---|---|",
    ]
    for idx, row in cost["by_run_index"].items():
        lines.append(f"| {idx} | {_f(row['mean_tokens'], 0)} | {_f(row['mean_cache_hit'], 0)} | {_s(row['p50_total_s'])} |")
    lines += [
        "",
        "## 门禁（闸 C）",
        "",
        f"- 成功率跨次标准差 ≤ {config.STABILITY_MAX_SUCCESS_RATE_STD}：当前 {_f(sp['std_sample'])}",
        f"- 轨迹自洽率 ≥ {config.STABILITY_MIN_TRAJECTORY_CONSISTENCY}：当前 {_f(tj['value'])}",
        "",
        "阈值在 `src/config.py`，首次由本次测量结果确定后固定；未达标只记录、不调低（PERF_SPEC §1）。",
        "",
    ]
    return "\n".join(lines)


# ================================================================== CLI
def write_outputs(summary: dict, summary_path: Path = SUMMARY_PATH, report_path: Path = REPORT_PATH) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_markdown(summary), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="原始运行记录 -> 稳定性 / 延迟报告")
    parser.add_argument("--raw", type=str, default=None, help="run_*.jsonl 路径（默认取最新）")
    parser.add_argument("--judge", type=str, default=None, help="逐次裁判分 judge_*.jsonl（可选）")
    parser.add_argument("--no-embed", action="store_true", help="不算答案相似度（不加载 BGE）")
    parser.add_argument("--summary", type=str, default=str(SUMMARY_PATH))
    parser.add_argument("--report", type=str, default=str(REPORT_PATH))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    raw = Path(args.raw) if args.raw else latest_raw()
    rows = load_rows(raw)
    samples = load_golden_set()
    run_meta = load_run_meta(raw)
    judge_rows = load_judge_rows(args.judge)

    embedder = None
    if not args.no_embed:
        from src.agent.rag import BGEEmbedder

        embedder = BGEEmbedder()
        print(f"答案相似度：加载 {embedder.model_name}", file=sys.stderr)

    summary = build_summary(
        rows, samples, run_meta=run_meta, judge_rows=judge_rows, embedder=embedder,
        raw_path=raw, judge_path=args.judge,
    )
    write_outputs(summary, Path(args.summary), Path(args.report))
    print(render_markdown(summary))
    print(f"summary: {args.summary}\nreport:  {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
