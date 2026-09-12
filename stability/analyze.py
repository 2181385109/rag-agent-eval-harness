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
        "by_cache": latency_by_cache(ok),
        "pass0_observation": pass0_observation(ok),
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_error": n_error,
        "error_rate": (n_error / len(rows)) if rows else None,
    }


def latency_by_cache(ok: Sequence[RunRecord]) -> dict:
    """按运行级 prompt_cache_hit_tokens == 0 / > 0 分两组报端到端分位数。

    注意分组依据是**整次运行内所有 LLM 调用的缓存命中之和**（09-11 的记录只有这一级）。
    一次运行通常含 ≥2 次调用，第二次起会命中第一次写入的系统提示词前缀，
    所以"== 0"组只可能包含单次调用的运行。逐次调用的缓存命中自 09-12 起才记录。
    未命中组为空时 headline_group 退到 hit，并如实标出 n=0，不能拿 0 冒充。
    """
    miss = [r for r in ok if r.tokens.cache_hit == 0]
    hit = [r for r in ok if r.tokens.cache_hit is not None and r.tokens.cache_hit > 0]
    unknown = [r for r in ok if r.tokens.cache_hit is None]
    miss_stats = _pct([r.latency.total_s for r in miss])
    hit_stats = _pct([r.latency.total_s for r in hit])
    return {
        "rule": "按 tokens.cache_hit（整次运行内各 LLM 调用的 prompt_cache_hit_tokens 之和）== 0 / > 0 分组；端到端 wall-clock 秒",
        "miss": miss_stats,
        "hit": hit_stats,
        "unknown_n": len(unknown),
        "headline_group": "miss" if miss_stats["n"] > 0 else "hit",
        "caveat": (
            "分组依据是运行级的和：同一运行内第二次调用起会命中第一次写入的前缀，"
            "所以含 ≥2 次 LLM 调用的运行几乎必然 > 0；逐次调用的缓存命中 2026-09-12 起才逐条记录"
        ),
    }


SINGLE_CALL_CLOSE_TOLERANCE = 0.10  # 各 pass 单次 LLM 调用 P50 相对 pass 0 的偏差 ≤ 10% 视为"接近"


def pass0_observation(ok: Sequence[RunRecord]) -> dict | None:
    """pass 0（无跨轮缓存）与后续 pass 的端到端 P50 对照，并补两项检查：

    1. 单次 LLM 调用延迟（llm_call_s 逐条）按 pass 的 P50 是否接近（阈值见常量）；
    2. 把轨迹长度固定在众数（n_llm_calls 的众数）后，pass 0 的端到端 P50 是否仍低于全部后续 pass。

    两项同时成立（单次接近 + 控制长度后差距消失）才把标签改成
    「延迟差异主要由轨迹长度解释，非缓存效应」；否则保留「未解释现象」。只陈述，不解释。
    """
    from collections import Counter

    by_pass: dict[int, list[RunRecord]] = {}
    for r in ok:
        by_pass.setdefault(r.run_index, []).append(r)
    if 0 not in by_pass or len(by_pass) < 2:
        return None

    modal_calls = Counter(r.latency.n_llm_calls for r in ok).most_common(1)[0][0]
    per_pass: dict[str, dict] = {}
    for i in sorted(by_pass):
        rs = by_pass[i]
        calls = [x for r in rs for x in r.latency.llm_call_s]
        at_modal = [r.latency.total_s for r in rs if r.latency.n_llm_calls == modal_calls]
        per_pass[str(i)] = {
            "n": len(rs),
            "e2e_p50": float(np.percentile([r.latency.total_s for r in rs], 50)),
            "e2e_p50_at_modal_calls": (float(np.percentile(at_modal, 50)) if at_modal else None),
            "n_at_modal_calls": len(at_modal),
            "single_call_p50": (float(np.percentile(calls, 50)) if calls else None),
            "single_call_p95": (float(np.percentile(calls, 95)) if calls else None),
            "n_calls": len(calls),
            "mean_n_llm_calls": float(np.mean([r.latency.n_llm_calls for r in rs])),
            "mean_tokens": float(np.mean([r.tokens.total for r in rs])),
            "mean_cache_hit": _mean([r.tokens.cache_hit for r in rs if r.tokens.cache_hit is not None]),
        }

    p0 = per_pass["0"]
    later = {k_: v for k_, v in per_pass.items() if k_ != "0"}
    faster = p0["e2e_p50"] < min(v["e2e_p50"] for v in later.values())
    base = p0["single_call_p50"]
    single_close = (
        base is not None and base > 0
        and all(v["single_call_p50"] is not None and abs(v["single_call_p50"] - base) / base <= SINGLE_CALL_CLOSE_TOLERANCE for v in later.values())
    )
    controlled = [v["e2e_p50_at_modal_calls"] for v in later.values() if v["e2e_p50_at_modal_calls"] is not None]
    gap_persists = (
        p0["e2e_p50_at_modal_calls"] is not None and bool(controlled)
        and p0["e2e_p50_at_modal_calls"] < min(controlled)
    )
    if not faster:
        label = "符合预期方向（pass 0 不快于后续 pass）"
    elif single_close and not gap_persists:
        label = "延迟差异主要由轨迹长度解释，非缓存效应"
    else:
        label = "未解释现象"
    return {
        "pass0_p50": p0["e2e_p50"],
        "pass0_n": p0["n"],
        "later_p50_by_pass": {k_: v["e2e_p50"] for k_, v in later.items()},
        "later_p50_min": min(v["e2e_p50"] for v in later.values()),
        "later_p50_max": max(v["e2e_p50"] for v in later.values()),
        "pass0_faster_than_all_later": faster,
        "modal_n_llm_calls": modal_calls,
        "per_pass": per_pass,
        "single_call_close": single_close,
        "single_call_close_rule": f"各 pass 单次 LLM 调用 P50 相对 pass 0 偏差 ≤ {SINGLE_CALL_CLOSE_TOLERANCE:.0%}",
        "length_controlled_gap_persists": gap_persists,
        "label": label,
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
    # 分路统计：规则判定（闭合题）与裁判判定（开放题）各自的自洽率与分母，
    # 外加裁判路每题 k 次裁判分的标准差（总体标准差 ddof=0）。
    by_source: dict[str, dict] = {
        "rule": {"hits": [], "inconsistent_ids": []},
        "judge": {"hits": [], "inconsistent_ids": [], "score_std_per_question": {}, "scores_per_question": {}},
    }
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
        source = "rule" if sample.answer_type == "closed" else "judge"
        source_counts[source] += 1
        by_source[source]["hits"].append(1.0 if consistent else 0.0)
        if not consistent:
            by_source[source]["inconsistent_ids"].append(sid)
        if source == "judge":
            scores = [judge_lookup[(sid, r.run_index)] for r in rs]
            by_source["judge"]["scores_per_question"][sid] = scores
            by_source["judge"]["score_std_per_question"][sid] = float(np.std(scores, ddof=0))
        verdict_detail[sid] = {"consistent": consistent, "verdicts": verdicts}

    # 判定翻转的位置：偏离多数派的 pass 编号。只陈述位置；k=5 下判不了是否与运行顺序相关。
    flip_positions: dict[str, dict] = {}
    for sid in verdict_bad:
        verdicts = verdict_detail[sid]["verdicts"]
        majority = max(set(verdicts), key=verdicts.count)
        deviating = [i for i, v in enumerate(verdicts) if v != majority]
        k_ = len(verdicts)
        flip_positions[sid] = {
            "majority": majority,
            "deviating_passes": deviating,
            "all_in_last_two": all(i >= k_ - 2 for i in deviating),
        }

    judge_stds = list(by_source["judge"]["score_std_per_question"].values())
    by_source_out = {
        "rule": {
            "value": _mean(by_source["rule"]["hits"]),
            "n": len(by_source["rule"]["hits"]),
            "inconsistent_ids": by_source["rule"]["inconsistent_ids"],
        },
        "judge": {
            "value": _mean(by_source["judge"]["hits"]),
            "n": len(by_source["judge"]["hits"]),
            "inconsistent_ids": by_source["judge"]["inconsistent_ids"],
            "scores_per_question": by_source["judge"]["scores_per_question"],
            "score_std": {
                "rule": "每题 k 次裁判分（0/1/2）的总体标准差（ddof=0）；mean/max 为跨题的均值与最大值",
                "per_question": by_source["judge"]["score_std_per_question"],
                "mean": _mean(judge_stds),
                "max": (max(judge_stds) if judge_stds else None),
            },
        },
    }

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
            "by_source": by_source_out,
            "flip_positions": flip_positions,
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


# ================================================================== 模型归属 / 裁判独立性
# 事实记录（措辞限定：只写「请求名已弃用、实际服务模型为 X」，不写"静默替换"或"厂商未告知"）。
MODEL_DEPRECATION_NOTE = (
    "DeepSeek 官方已于 2026-07-24 弃用 deepseek-chat / deepseek-reasoner 两个模型名；"
    "现由旧名路由至 deepseek-flash（V4.1 Flash）。本仓库 src/config.py 仍硬编码旧名。"
)
JUDGE_INDEPENDENCE_UNCONFIRMED = (
    "本次运行无法确认裁判与被测模型的独立性，开放题判定结果（n=10）应据此折价。"
)
JUDGE_INDEPENDENCE_CONFIRMED = "被测与裁判的响应模型 / 指纹不同，独立性成立。"


def load_probe(path: Path | str | None) -> dict | None:
    if path is None:
        return None
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"模型探针文件不存在：{target}")
    return json.loads(target.read_text(encoding="utf-8"))


def latest_probe(directory: Path | None = None) -> Path | None:
    files = sorted((directory or RAW_DIR).glob("model_probe_*.json"))
    return files[-1] if files else None


def judge_cache_facts(judge_rows: Sequence[Mapping] | None) -> dict:
    """裁判链路有没有走缓存：客户端一层看代码，服务端一层看记录。"""
    rows = list(judge_rows or [])
    with_hit = [r for r in rows if r.get("prompt_cache_hit_tokens") is not None]
    return {
        "client_side_cache": "none",
        "client_side_note": "src/eval/judge.py DeepSeekJudge.__call__ 每次直接 client.chat.completions.create，无本地缓存层",
        "server_cache_recorded": bool(with_hit),
        "n_rows": len(rows),
        "n_rows_with_cache_hit": len(with_hit),
        "cache_hit_tokens_total": (int(sum(int(r["prompt_cache_hit_tokens"]) for r in with_hit)) if with_hit else None),
        "note": (
            "裁判记录含服务端 prompt_cache_hit_tokens，可判断是否命中"
            if with_hit
            else "裁判记录未存 usage（2026-09-12 起 judge_runs.py 才记录），服务端缓存是否命中无法确认"
        ),
    }


def served_model_section(
    rows: Sequence[RunRecord],
    run_meta: Mapping,
    judge_rows: Sequence[Mapping] | None,
    probe: Mapping | None,
) -> dict:
    """请求模型名 vs 服务端响应模型名；裁判与被测是否落在同一后端。"""
    from collections import Counter

    requested = run_meta.get("model")
    served = Counter(r.response_model for r in rows if r.response_model)
    fps = Counter(r.system_fingerprint for r in rows if r.system_fingerprint)
    mismatch = bool(served) and any(m != requested for m in served)

    # 单次运行内逐条响应模型是否一致（逐条记录 2026-09-12 起才有）
    recorded = [r for r in rows if r.response_models]
    consistent = sum(1 for r in recorded if len({m for m in r.response_models}) == 1)
    per_run = {
        "n_recorded_per_call": len(recorded),
        "n_consistent": consistent,
        "n_inconsistent": len(recorded) - consistent,
        "n_unrecorded": len(rows) - len(recorded),
        "note": "旧格式（2026-09-11）只存每次运行首条响应，无法判断运行内一致性；逐条记录自 2026-09-12 起",
    }

    # 裁判链路
    jrows = list(judge_rows or [])
    judge_requested = sorted({r.get("judge_model") for r in jrows if r.get("judge_model")})
    judge_served = Counter(r.get("response_model") for r in jrows if r.get("response_model"))
    judge_fps = Counter(r.get("system_fingerprint") for r in jrows if r.get("system_fingerprint"))

    probe_rows = list((probe or {}).get("results", []))
    probe_by_name = {p["requested_model"]: p for p in probe_rows}
    probe_same_backend = None
    if len(probe_rows) >= 2:
        pairs = {(p.get("response_model"), p.get("system_fingerprint")) for p in probe_rows}
        probe_same_backend = len(pairs) == 1

    # 判定：只要能拿到的证据里被测与裁判的 (响应模型, 指纹) 相同，就无法确认独立
    same_by_records = bool(judge_served) and set(judge_served) == set(served) and set(judge_fps) == set(fps)
    if judge_served:
        unconfirmed = same_by_records
        evidence = "records"
    elif probe_same_backend is not None:
        unconfirmed = probe_same_backend
        evidence = "probe"
    else:
        unconfirmed = None
        evidence = "none"
    # 独立性只有被证据**证明不同**才算成立；证据显示相同、或根本没有证据，都按"无法确认"处理。
    verdict = JUDGE_INDEPENDENCE_CONFIRMED if unconfirmed is False else JUDGE_INDEPENDENCE_UNCONFIRMED

    return {
        "requested_model": requested,
        "served_models": dict(served),
        "system_fingerprints": dict(fps),
        "mismatch": mismatch,
        "deprecation_note": MODEL_DEPRECATION_NOTE,
        "per_run_consistency": per_run,
        "judge_independence": {
            "judge_requested_model": (judge_requested[0] if len(judge_requested) == 1 else judge_requested) or None,
            "judge_served_recorded": bool(judge_served),
            "judge_served_models": dict(judge_served),
            "judge_system_fingerprints": dict(judge_fps),
            "probe_file_used": probe is not None,
            "probed_at_utc": (probe or {}).get("probed_at_utc"),
            "probe": {
                name: {
                    "response_model": p.get("response_model"),
                    "system_fingerprint": p.get("system_fingerprint"),
                    "http_status": p.get("http_status"),
                }
                for name, p in probe_by_name.items()
            },
            "probe_same_backend": probe_same_backend,
            "models_listed_by_endpoint": (probe or {}).get("models_listed_by_endpoint"),
            "evidence": evidence,
            "verdict": verdict,
        },
        "judge_cache": judge_cache_facts(judge_rows),
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


# ================================================================== 多步检索行为对照
def _first_retrieve_docs_from_trace(trace) -> list[str]:
    for step in trace.steps:
        for r in step.results:
            if r.name == "retrieve" and r.retrieved:
                return [c.doc_id for c in r.retrieved]
    return []


def retrieval_persistence_section(
    rows: Sequence[RunRecord],
    samples: Sequence[GoldenSample],
    snapshot_traces: Mapping[str, object],
    snapshot_report: Mapping,
    judge_rows: Sequence[Mapping] | None = None,
    top_k: int | None = None,
) -> dict:
    """逐题对比快照与本次的 retrieve 调用次数；标出"首次检索相同但判定不同"的题。

    首次检索的比较口径：快照取第一次 retrieve 返回的 doc_id 列表；本次运行记录只存
    去重并集（按检索顺序），取其前 top_k 个作为首次检索（每次检索恒返回 top_k 块时二者等价）。
    结论句只填数字，不做归因。
    """
    from collections import Counter

    k_top = top_k or config.RETRIEVE_TOP_K
    groups = _by_sample(rows)
    sample_map = {s.id: s for s in samples}
    snap_success = {q["id"]: q.get("success") for q in snapshot_report.get("per_question", [])}
    judge_lookup, _ = _index_judge_rows(rows, judge_rows)

    per_q: dict[str, dict] = {}
    snap_counts: list[int] = []
    cur_counts_runs: list[int] = []
    cur_medians: list[float] = []
    same_first_diff_verdict: list[str] = []
    failed_after_miss_no_retry: list[str] = []
    failed_after_miss_no_retry_any: list[str] = []  # 不要求首次检索与快照相同
    direction: dict[str, list[str]] = {"decreased": [], "unchanged": [], "increased": []}
    for sid in sorted(groups):
        sample = sample_map.get(sid)
        trace = snapshot_traces.get(sid)
        if sample is None or trace is None:
            continue
        rs = groups[sid]
        snap_n = sum(1 for x in trace.tool_sequence if x == "retrieve")
        cur_n = [sum(1 for x in r.tool_sequence if x == "retrieve") for r in rs]
        snap_first = _first_retrieve_docs_from_trace(trace)
        cur_first = [r.retrieved_doc_ids[:k_top] for r in rs]
        identical = bool(snap_first) and all(f == snap_first for f in cur_first)
        verdicts = [_verdict(sample, r, judge_lookup) for r in rs]
        snap_v = snap_success.get(sid)
        verdict_differs = snap_v is not None and any(v is not None and v != snap_v for v in verdicts)
        expected = {d for g in sample.expected_doc_ids for d in g}
        first_hit = bool(expected & set(snap_first)) if expected else None
        # 失败的 pass 里，首次检索未命中标注证据且没有再检索
        failed_no_retry = bool(expected) and first_hit is False and any(
            v is False and n == 1 for v, n in zip(verdicts, cur_n)
        )
        snap_counts.append(snap_n)
        cur_counts_runs.extend(cur_n)
        cur_medians.append(float(np.median(cur_n)))
        if identical and verdict_differs:
            same_first_diff_verdict.append(sid)
            if failed_no_retry:
                failed_after_miss_no_retry.append(sid)
        if verdict_differs and failed_no_retry:
            failed_after_miss_no_retry_any.append(sid)
        med = float(np.median(cur_n))
        direction["decreased" if med < snap_n else ("increased" if med > snap_n else "unchanged")].append(sid)
        per_q[sid] = {
            "snapshot_retrieves": snap_n,
            "current_retrieves": cur_n,
            "current_median": float(np.median(cur_n)),
            "first_retrieval_identical": identical,
            "snapshot_first_retrieval": snap_first,
            "first_retrieval_hits_expected": first_hit,
            "snapshot_success": snap_v,
            "current_verdicts": verdicts,
        }

    snap_med = float(np.median(snap_counts)) if snap_counts else None
    cur_med = float(np.median(cur_counts_runs)) if cur_counts_runs else None
    cur_med_q = float(np.median(cur_medians)) if cur_medians else None
    if snap_med is None or cur_med is None:
        conclusion = "没有可对照的题（快照轨迹与本次运行记录没有交集）。"
    else:
        verb = "降至" if cur_med < snap_med else "变为"
        conclusion = (
            f"同一代码与同一索引下，retrieve 调用次数中位数由 {snap_med:.1f} {verb} {cur_med:.1f}"
            f"；{len(failed_after_miss_no_retry)} 题因首次检索未命中后未再检索而失败。"
        )
    return {
        "rule": (
            "retrieve 调用次数 = 工具序列里 retrieve 的出现次数（不折叠）；快照为单次运行，本次为 k 次；"
            f"首次检索 = 快照第一次 retrieve 的 doc_id 列表 vs 本次去重并集的前 {k_top} 个"
        ),
        "n_questions": len(per_q),
        "snapshot_median": snap_med,
        "snapshot_distribution": dict(sorted(Counter(snap_counts).items())),
        "current_median_runs": cur_med,
        "current_median_of_question_medians": cur_med_q,
        "current_distribution_runs": dict(sorted(Counter(cur_counts_runs).items())),
        "n_current_runs": len(cur_counts_runs),
        "same_first_retrieval_different_verdict": same_first_diff_verdict,
        "failed_after_first_miss_without_retry": failed_after_miss_no_retry,
        "failed_after_first_miss_without_retry_any_first": failed_after_miss_no_retry_any,
        "direction_counts": {k_: len(v) for k_, v in direction.items()},
        "direction_ids": direction,
        "conclusion": conclusion,
        "per_question": per_q,
    }


def load_snapshot_traces(path: Path | str | None = None):
    """快照轨迹（reports/traces_latest.jsonl，gitignore）；不存在返回 None。"""
    from src.eval import report as report_mod

    target = Path(path) if path else (config.REPORTS_DIR / "traces_latest.jsonl")
    if not target.exists():
        return None
    return report_mod.load_traces(target)


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
    probe: Mapping | None = None,
    probe_path: Path | str | None = None,
) -> dict:
    run_meta = dict(run_meta or {})
    k = int(run_meta.get("k") or max((r.k for r in rows), default=0))
    stability = stability_section(rows, samples, judge_rows=judge_rows)
    stability["answer_similarity"] = (
        answer_similarity_section(rows, samples, embedder) if embedder is not None else None
    )
    stability["main_snapshot_comparison"] = main_snapshot_comparison(rows, samples, judge_rows)
    snap_traces = load_snapshot_traces()
    latest = config.REPORTS_DIR / "latest.json"
    stability["retrieval_persistence"] = (
        retrieval_persistence_section(
            rows, samples, snap_traces, json.loads(latest.read_text(encoding="utf-8")), judge_rows=judge_rows
        )
        if snap_traces is not None and latest.exists()
        else None
    )
    return {
        "meta": {
            "analyzed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_file": _relative(raw_path),
            "raw_sha256": sha256_of(raw_path) if raw_path and Path(raw_path).exists() else None,
            "judge_rows_given": judge_rows is not None,
            "judge_file": _relative(judge_path),
            "judge_sha256": sha256_of(judge_path) if judge_path and Path(judge_path).exists() else None,
            "probe_file": _relative(probe_path),
            "probe_sha256": sha256_of(probe_path) if probe_path and Path(probe_path).exists() else None,
            "n_rows": len(rows),
            "n_questions": len({r.sample_id for r in rows}),
            "k": k,
            "served_models": sorted({r.response_model for r in rows if r.response_model}),
            "system_fingerprints": sorted({r.system_fingerprint for r in rows if r.system_fingerprint}),
            "run_meta": run_meta,
        },
        "model_attribution": served_model_section(rows, run_meta, judge_rows, probe),
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
    attr0 = summary.get("model_attribution") or {}
    served_names = "/".join(sorted(attr0.get("served_models", {}))) or "未记录"
    tag = (
        f"n={n_rows}（{n_q} 题 × k={k}），temperature={rm.get('temperature', '—')}，"
        f"model=请求 {rm.get('model', '—')} / 响应 {served_names}"
    )

    attr = summary.get("model_attribution") or {}
    served_str = ", ".join(f"`{k_}`×{v}" for k_, v in sorted(attr.get("served_models", {}).items())) or "—"
    fp_str = ", ".join(f"`{k_}`×{v}" for k_, v in sorted(attr.get("system_fingerprints", {}).items())) or "—"
    lines = [
        "# 稳定性与延迟分解报告（脚本生成，勿手改）",
        "",
    ]
    if attr.get("mismatch"):
        lines += [
            "> **告警：请求模型名与服务端响应模型名不一致。**",
            f"> 请求 `{attr.get('requested_model')}`，全部 {m['n_rows']} 次运行的响应 `model` 字段为 {served_str}"
            f"（system_fingerprint {fp_str}）。",
            f"> {attr.get('deprecation_note')}",
            "> 本报告中所有「被测模型」的数字，实际服务模型均为上述响应模型名。",
            "",
        ]
    lines += [
        f"> 本文件由 `python -m stability.analyze` 从 `{m.get('raw_file')}` 生成；",
        f"> 原始产物 sha256 = `{m.get('raw_sha256')}`；分析时间 {m['analyzed_at_utc']}。",
        "> 每个数字旁都带样本量 n、重复次数 k 与环境标识（PERF_SPEC §0.3）。",
        "",
        "## 环境与随机性来源",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 请求模型名（config.MODEL_NAME） | `{rm.get('model', '—')}` |",
        f"| 实际响应模型名（响应 `model` 字段，按行计数） | {served_str}{'  **← 与请求名不一致**' if attr.get('mismatch') else ''} |",
        f"| system_fingerprint（按行计数） | {fp_str} |",
        f"| 单次运行内响应模型一致性 | 逐条记录 {attr.get('per_run_consistency', {}).get('n_recorded_per_call', 0)} 次运行：一致 {attr.get('per_run_consistency', {}).get('n_consistent', 0)}、不一致 {attr.get('per_run_consistency', {}).get('n_inconsistent', 0)}；未逐条记录 {attr.get('per_run_consistency', {}).get('n_unrecorded', 0)} 次（{attr.get('per_run_consistency', {}).get('note', '')}） |",
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
    ]
    bc = lat.get("by_cache") or {}
    if bc:
        lines += ["### 按服务端缓存命中拆分（端到端）", ""]
        if bc.get("headline_group") == "miss":
            lines += [
                f"分组口径：{bc.get('rule')}。",
                "",
                "| 组 | n | P50 | P95 | P99 | 均值 |",
                "|---|---|---|---|---|---|",
                f"| 未命中（cache_hit == 0） | {bc['miss']['n']} | {_s(bc['miss']['p50'])} | {_s(bc['miss']['p95'])} | {_s(bc['miss']['p99'])} | {_s(bc['miss']['mean'])} |",
                f"| 命中（cache_hit > 0） | {bc['hit']['n']} | {_s(bc['hit']['p50'])} | {_s(bc['hit']['p95'])} | {_s(bc['hit']['p99'])} | {_s(bc['hit']['mean'])} |",
                "",
                f"主结论按未命中组：P50 {_s(bc['miss']['p50'])} / P95 {_s(bc['miss']['p95'])} / P99 {_s(bc['miss']['p99'])}（n={bc['miss']['n']}）。",
                "",
            ]
        else:
            lines += [
                f"本批数据无法按缓存命中拆分：{bc['hit']['n']} 次有效运行的运行级 cache_hit 全部 > 0，未命中组 n=0。"
                f"{bc.get('caveat')}。上文端到端分位数即为全部有效行。",
                "",
            ]
    obs = lat.get("pass0_observation")
    if obs:
        pp = obs["per_pass"]
        lines += [
            "### 按 pass 看单次 LLM 调用延迟、轨迹长度与 token",
            "",
            f"轨迹长度众数 n_llm_calls = {obs['modal_n_llm_calls']}；「控制长度」列只取 n_llm_calls 等于众数的运行。",
            "",
            "| pass | n | 端到端 P50 | 端到端 P50（控制长度） | n（控制长度） | 单次 LLM 调用 P50 | 单次 P95 | 调用数 | 平均 n_llm_calls | 平均 token | 平均缓存命中 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i, v in pp.items():
            lines.append(
                f"| {i} | {v['n']} | {_s(v['e2e_p50'])} | {_s(v['e2e_p50_at_modal_calls'])} | {v['n_at_modal_calls']} | {_s(v['single_call_p50'])} | {_s(v['single_call_p95'])} | {v['n_calls']} | {_f(v['mean_n_llm_calls'], 2)} | {_f(v['mean_tokens'], 0)} | {_f(v['mean_cache_hit'], 0)} |"
            )
        lines += [
            "",
            f"pass 0（无跨轮缓存）端到端 P50 {_s(obs['pass0_p50'])}，后续 pass 的 P50 为 "
            + "、".join(f"pass {i} {_s(v)}" for i, v in obs["later_p50_by_pass"].items())
            + f"；pass 0 {'低于' if obs['pass0_faster_than_all_later'] else '不低于'}全部后续 pass。"
            f"单次 LLM 调用 P50 各 pass {'接近' if obs['single_call_close'] else '不接近'}（{obs['single_call_close_rule']}）；"
            f"控制轨迹长度后 pass 0 {'仍' if obs['length_controlled_gap_persists'] else '不再'}低于全部后续 pass。"
            f"—— **{obs['label']}**。",
            "",
        ]
    lines += [
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
        f"| 判定自洽率（合并） | **{_f(vc['value'])}** | n={vc['n']}/{vc['total']}（规则 {vc['source_counts']['rule']}、裁判 {vc['source_counts']['judge']}），k={k} | {vc['rule']} |",
        f"| 判定自洽率 · 规则路（闭合题） | **{_f(vc['by_source']['rule']['value'])}** | n={vc['by_source']['rule']['n']}，k={k} | answer_keys 全命中；不一致：{vc['by_source']['rule']['inconsistent_ids'] or '无'} |",
        f"| 判定自洽率 · 裁判路（开放题） | **{_f(vc['by_source']['judge']['value'])}** | n={vc['by_source']['judge']['n']}，k={k} | 裁判分 ≥ {config.OPEN_SUCCESS_THRESHOLD}；不一致：{vc['by_source']['judge']['inconsistent_ids'] or '无'} |",
        f"| 裁判分跨次标准差（裁判路） | 均值 **{_f(vc['by_source']['judge']['score_std']['mean'])}**，最大 **{_f(vc['by_source']['judge']['score_std']['max'])}** | n={vc['by_source']['judge']['n']} 题，每题 k={k} 次 | {vc['by_source']['judge']['score_std']['rule']} |",
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
        fp = vc.get("flip_positions", {})
        for sid in vc["inconsistent_ids"]:
            pos = fp.get(sid, {})
            lines.append(
                f"- `{sid}`：{vc['detail'][sid]['verdicts']}；偏离多数派的 pass：{pos.get('deviating_passes')}"
                f"{'（全部落在最后两个 pass）' if pos.get('all_in_last_two') else ''}"
            )
        last_two = [sid for sid in vc["inconsistent_ids"] if fp.get(sid, {}).get("all_in_last_two")]
        lines.append(
            f"翻转位置与运行顺序的关系：{len(last_two)}/{len(vc['inconsistent_ids'])} 题的偏离全部落在最后两个 pass；"
            f"k={k} 不足以判定翻转是否与运行顺序相关，需更大的 k。"
        )
    else:
        lines.append("判定不自洽的题：无。")
    if vc["unjudged_ids"]:
        lines.append(f"未计入判定自洽率的开放题（无逐次裁判分）：{', '.join(vc['unjudged_ids'])}。")
    if vc["stale_judge_rows"]:
        lines.append(f"⚠ 指纹对不上而被丢弃的裁判分：{vc['stale_judge_rows']}。")
    js = vc["by_source"]["judge"]
    if js["scores_per_question"]:
        lines.append("裁判路逐题 k 次裁判分（0/1/2）与标准差：")
        for sid, scores in js["scores_per_question"].items():
            lines.append(f"- `{sid}`：{scores}，std={_f(js['score_std']['per_question'][sid])}")
    ind = attr.get("judge_independence") or {}
    jc = attr.get("judge_cache") or {}
    lines += [
        "",
        "### 裁判独立性核验",
        "",
        "| 链路 | 请求模型名 | 响应 model | system_fingerprint | 证据 |",
        "|---|---|---|---|---|",
        f"| 被测（Agent） | `{attr.get('requested_model')}` | {served_str} | {fp_str} | 本次 {m['n_rows']} 行运行记录 |",
    ]
    jreq = ind.get("judge_requested_model")
    if ind.get("judge_served_recorded"):
        jm = ", ".join(f"`{k_}`×{v}" for k_, v in sorted(ind.get("judge_served_models", {}).items()))
        jf = ", ".join(f"`{k_}`×{v}" for k_, v in sorted(ind.get("judge_system_fingerprints", {}).items()))
        lines.append(f"| 裁判（judge_runs） | `{jreq}` | {jm} | {jf} | 本次裁判记录 |")
    else:
        lines.append(f"| 裁判（judge_runs） | `{jreq}` | 未记录 | 未记录 | 本次裁判记录未存响应字段（2026-09-12 起才记录） |")
    for name, pr in (ind.get("probe") or {}).items():
        lines.append(f"| 探针 | `{name}` | `{pr.get('response_model')}` | `{pr.get('system_fingerprint')}` | `{m.get('probe_file')}`（{ind.get('probed_at_utc')}，HTTP {pr.get('http_status')}） |")
    lines += [
        "",
        f"探针同一时刻两个请求名是否落在同一 (响应 model, 指纹)：{ind.get('probe_same_backend')}；"
        f"端点 `/models` 列出的模型：{ind.get('models_listed_by_endpoint')}。",
        "",
        f"**{ind.get('verdict')}**（判断依据：{ind.get('evidence')}）",
        "",
        f"裁判链路缓存：客户端 {jc.get('client_side_cache')}（{jc.get('client_side_note')}）；"
        f"服务端：{jc.get('note')}"
        + (f"，命中 token 合计 {jc.get('cache_hit_tokens_total')}（n={jc.get('n_rows_with_cache_hit')}/{jc.get('n_rows')} 行）" if jc.get("server_cache_recorded") else "")
        + "。",
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
    rp = st.get("retrieval_persistence")
    if rp:
        lines += [
            "",
            "### 多步检索行为对照（快照单次 vs 本次 k 次）",
            "",
            f"口径：{rp['rule']}。可对照 n={rp['n_questions']} 题。",
            "",
            "| | 快照（2026-09-06，单次） | 本次（k 次运行） |",
            "|---|---|---|",
            f"| retrieve 调用次数中位数 | {_f(rp['snapshot_median'], 1)}（n={rp['n_questions']} 题） | {_f(rp['current_median_runs'], 1)}（n={rp['n_current_runs']} 次运行）；题内中位数再取中位数 {_f(rp['current_median_of_question_medians'], 1)} |",
            f"| 次数分布（次数: 题/运行数） | {rp['snapshot_distribution']} | {rp['current_distribution_runs']} |",
            "",
            "| 题 | 快照 retrieve 次数 | 本次各 pass 次数 | 本次中位数 | 首次检索 doc_id 相同 | 快照判定 | 本次判定 |",
            "|---|---|---|---|---|---|---|",
        ]
        for sid, q in rp["per_question"].items():
            lines.append(
                f"| {sid} | {q['snapshot_retrieves']} | {q['current_retrieves']} | {_f(q['current_median'], 1)} | "
                f"{'是' if q['first_retrieval_identical'] else '否'} | {q['snapshot_success']} | {q['current_verdicts']} |"
            )
        same = rp["same_first_retrieval_different_verdict"]
        lines += [
            "",
            f"首次检索 doc_id 相同但判定不同的题（n={len(same)}）：{', '.join(same) if same else '无'}。"
            "这类题的检索输入没有变化，差异出在首次检索之后的行为。",
        ]
        for sid in same:
            q = rp["per_question"][sid]
            lines.append(
                f"- `{sid}`：快照 {q['snapshot_retrieves']} 次 retrieve、判定 {q['snapshot_success']}；本次 {q['current_retrieves']} 次、判定 {q['current_verdicts']}；"
                f"首次检索命中标注证据：{q['first_retrieval_hits_expected']}"
            )
        dc = rp["direction_counts"]
        lines += [
            "",
            f"逐题方向（本次题内中位数 vs 快照次数）：减少 {dc['decreased']} 题 {rp['direction_ids']['decreased']}；"
            f"不变 {dc['unchanged']} 题；增加 {dc['increased']} 题 {rp['direction_ids']['increased']}。",
            "",
            f"**{rp['conclusion']}**",
            f"（「因首次检索未命中后未再检索而失败」的判定条件：有标注证据、快照首次检索未命中、失败的 pass 里 retrieve 次数为 1，"
            f"且首次检索 doc_id 与快照相同；命中：{rp['failed_after_first_miss_without_retry'] or '无'}。"
            f"放宽「首次检索相同」这一条后命中：{rp['failed_after_first_miss_without_retry_any_first'] or '无'}。）",
        ]

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
                "两边判定口径相同（判定代码与黄金集在两个 commit 间逐字相同）。快照未记录响应模型名；"
                f"本次响应模型名见顶部。差异的归因不在本报告范围内；正式比对需重跑主评测（`python -m src.eval.report`）走闸 B。",
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
    parser.add_argument("--probe", type=str, default=None, help="模型探针 model_probe_*.json（默认取最新；--no-probe 不用）")
    parser.add_argument("--no-probe", action="store_true", help="不读取模型探针文件")
    parser.add_argument("--no-embed", action="store_true", help="不算答案相似度（不加载 BGE）")
    parser.add_argument("--summary", type=str, default=str(SUMMARY_PATH))
    parser.add_argument("--report", type=str, default=str(REPORT_PATH))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    raw = Path(args.raw) if args.raw else latest_raw()
    rows = load_rows(raw)
    samples = load_golden_set()
    run_meta = load_run_meta(raw)
    judge_rows = load_judge_rows(args.judge)
    probe_path = None if args.no_probe else (Path(args.probe) if args.probe else latest_probe())
    probe = load_probe(probe_path)

    embedder = None
    if not args.no_embed:
        from src.agent.rag import BGEEmbedder

        embedder = BGEEmbedder()
        print(f"答案相似度：加载 {embedder.model_name}", file=sys.stderr)

    summary = build_summary(
        rows, samples, run_meta=run_meta, judge_rows=judge_rows, embedder=embedder,
        raw_path=raw, judge_path=args.judge, probe=probe, probe_path=probe_path,
    )
    write_outputs(summary, Path(args.summary), Path(args.report))
    print(render_markdown(summary))
    print(f"summary: {args.summary}\nreport:  {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
