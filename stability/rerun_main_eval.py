"""主评测对照重跑：同一代码、同一磁盘索引，单次运行，另存快照，不碰 latest.json。

    python -m stability.rerun_main_eval --name 20260912_rerun
    python -m stability.rerun_main_eval --ragas        # 名字默认 {启动时刻}_full，附带 RAGAS

产物（都不覆盖既有文件）：
    reports/eval_{name}.json              与 src.eval.report 同一 build_report 产出的快照，
                                          外加逐题 raw 工具序列 / retrieve 次数 / 首次检索 doc_id /
                                          查询词 / 响应 model / system_fingerprint
    reports/traces_{name}.jsonl           完整轨迹（gitignore）
    reports/judge_scores_{name}.jsonl     开放题裁判分（同一裁判、同一判据 v2），带响应字段

与 `python -m src.eval.report` 的差别：不写 latest.json / report.md；不跑多轮一致性；
给 Agent 挂了计时探针以记录响应 model；裁判分写到独立文件而不是 judge_scores.jsonl；
`--ragas` 时给 RAGAS 的 ChatOpenAI 也挂探针（见 ragas_probe）。
指标计算走的是同一个 report.build_report，RAGAS 走同一个 ragas_runner.run_ragas。

meta.served 里被测 / 裁判 / RAGAS 三条链路的响应 model 与 system_fingerprint **逐条**记录，
探针拿不到就写「未记录」——不回填请求名，不推测。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.agent.trace import AgentTrace
from src.eval import judge, report
from src.eval.datasets import load_golden_set
from stability.instrument import instrumented

UNRECORDED = "未记录"


def _first_retrieve(trace: AgentTrace) -> list[str]:
    for step in trace.steps:
        for r in step.results:
            if r.name == "retrieve" and r.retrieved:
                return [c.doc_id for c in r.retrieved]
    return []


def _queries(trace: AgentTrace) -> list[str | None]:
    return [c.args.get("query") for s in trace.steps for c in s.tool_calls if c.name == "retrieve"]


@contextmanager
def ragas_probe():
    """给 ragas_runner.build_judge_llm 里构造的 ChatOpenAI 换成会记录响应字段的子类。

    build_judge_llm 在函数体内 `from langchain_openai import ChatOpenAI`，运行期取模块属性，
    所以 with 块内临时替换 `langchain_openai.ChatOpenAI` 即可，不改 src/（与 instrument.py
    替换 llm.chat_completion 是同一手法）。子类只旁路记录，不改任何返回值。

    取值直接来自 openai SDK 的原始响应对象（response.model / response.system_fingerprint），
    缺失就记 None。**不用** langchain 的 llm_output["model_name"]：它在响应缺 model 字段时
    会回填请求名，那样"响应名 == 请求名"就分不清是真的还是回填的。
    """
    import langchain_openai
    from langchain_openai import ChatOpenAI

    records: list[dict] = []

    class RecordingChatOpenAI(ChatOpenAI):
        def _create_chat_result(self, response, generation_info=None):
            if isinstance(response, dict):
                usage = response.get("usage") or {}
                records.append(
                    {
                        "response_model": response.get("model"),
                        "system_fingerprint": response.get("system_fingerprint"),
                        "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
                        "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
                    }
                )
            else:
                usage = getattr(response, "usage", None)
                records.append(
                    {
                        "response_model": getattr(response, "model", None),
                        "system_fingerprint": getattr(response, "system_fingerprint", None),
                        "prompt_cache_hit_tokens": judge._usage_field(usage, "prompt_cache_hit_tokens"),
                        "prompt_cache_miss_tokens": judge._usage_field(usage, "prompt_cache_miss_tokens"),
                    }
                )
            return super()._create_chat_result(response, generation_info)

    original = langchain_openai.ChatOpenAI
    langchain_openai.ChatOpenAI = RecordingChatOpenAI
    try:
        yield records
    finally:
        langchain_openai.ChatOpenAI = original


def _counts(values) -> dict[str, int]:
    """逐条值 -> {值: 次数}；None（探针没拿到）计在「未记录」下，不丢。"""
    return {(v if v is not None else UNRECORDED): n for v, n in sorted(Counter(values).items(), key=lambda kv: str(kv[0]))}


def _chain_block(requested: str, per_call: list[dict], by_id: dict[str, list[dict]] | None = None) -> dict:
    models = [c.get("response_model") for c in per_call]
    fps = [c.get("system_fingerprint") for c in per_call]
    block = {
        "requested_model": requested,
        "n_calls": len(per_call),
        "response_model_counts": _counts(models),
        "system_fingerprint_counts": _counts(fps),
        "calls": per_call,  # 逐条
    }
    if by_id is not None:
        block["calls_by_question"] = by_id
    return block


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="主评测对照重跑（单次，另存快照）")
    parser.add_argument("--name", default=None, help="快照名后缀，如 20260912_rerun；缺省为 {启动时刻 UTC}_full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ragas", action="store_true", help="附带跑 RAGAS（同一批轨迹，裁判走 config.JUDGE_MODEL_NAME）")
    parser.add_argument("--out-dir", default=None, help="产物目录，缺省 reports/（调试时可指向临时目录）")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started_dt = datetime.now(timezone.utc)
    started = started_dt.isoformat(timespec="seconds")
    name = args.name or f"{started_dt:%Y%m%dT%H%M%SZ}_full"
    out_dir = Path(args.out_dir) if args.out_dir else config.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"eval_{name}.json"
    out_traces = out_dir / f"traces_{name}.jsonl"
    out_judge = out_dir / f"judge_scores_{name}.jsonl"
    for p in (out_json, out_traces, out_judge):
        if p.exists():
            raise SystemExit(f"已存在，不覆盖：{p}")

    from src.agent import graph

    samples = load_golden_set()
    if args.limit:
        samples = samples[: args.limit]
    print(f"{len(samples)} 题 × 1 次，构建 Agent（加载 BGE + FAISS）...", file=sys.stderr)
    agent = graph.build_default_agent()

    traces: dict[str, AgentTrace] = {}
    served: dict[str, dict] = {}
    for i, s in enumerate(samples, start=1):
        with instrumented(agent) as timer:
            try:
                trace = agent.run(s.question)
            except Exception as exc:  # 与 report.run_samples 同样的兜底：失败也入账
                trace = AgentTrace(question=s.question, answer=f"[运行失败] {exc}", stop_reason="error", model=config.MODEL_NAME)
        traces[s.id] = trace
        served[s.id] = {
            "response_models": timer.response_models,
            "system_fingerprints": timer.system_fingerprints,
            "llm_call_s": timer.llm_call_s,
            "llm_call_cache_hit": timer.llm_call_cache_hit,
        }
        print(f"  [{i}/{len(samples)}] {s.id} tools={trace.tool_sequence} served={sorted(set(m for m in timer.response_models if m))}", file=sys.stderr, flush=True)
    report.save_traces(traces, out_traces, merge=False)

    # 开放题裁判：同一裁判类、同一判据；逐次记录响应字段（含重试的那次）
    caller = judge.DeepSeekJudge()
    metas: list[tuple[str, dict]] = []

    def recording_call(messages, *, model=None, sample_id=None, **kw):
        reply = caller(messages, model=model, sample_id=sample_id, **kw)
        metas.append((sample_id, dict(caller.last_response_meta)))
        return reply

    recording_call.model = caller.model  # type: ignore[attr-defined]
    recording_call.params = caller.params  # type: ignore[attr-defined]
    print(f"开放题裁判：{judge.RUBRIC_VERSION}，{caller.model}", file=sys.stderr)
    rows = judge.score_samples(samples, traces, call=recording_call)
    for row in rows:
        last = next((m for sid, m in reversed(metas) if sid == row["id"]), {})
        row.update(last)
        row["judge_params"] = dict(caller.params)
    judge.write_judge_scores(rows, out_judge)
    judge_by_q: dict[str, list[dict]] = {}
    for sid, m in metas:
        judge_by_q.setdefault(sid, []).append(m)

    agent_calls = [
        {"id": sid, "response_model": m, "system_fingerprint": f}
        for sid, v in served.items()
        for m, f in zip(v["response_models"], v["system_fingerprints"])
    ]
    agent_by_q = {
        sid: [{"response_model": m, "system_fingerprint": f} for m, f in zip(v["response_models"], v["system_fingerprints"])]
        for sid, v in served.items()
    }
    judge_calls = [{"id": sid, **m} for sid, m in metas]

    def _assemble(ragas_result, ragas_records, finished: str | None) -> dict:
        scores, backfill = report.load_judge_backfill(traces, out_judge)
        rep = report.build_report(
            samples, traces, judge_scores=scores, judge_backfill=backfill, ragas=ragas_result,
            notes=f"对照重跑：单次运行，另存快照，不写 latest.json；不跑多轮一致性；started {started}"
            + ("；RAGAS 与主评测同一批轨迹" if ragas_result else ("；RAGAS 进行中（本文件为中间落盘）" if args.ragas else "；未跑 RAGAS")),
        )
        all_models = sorted({m for v in served.values() for m in v["response_models"] if m})
        all_fps = sorted({f for v in served.values() for f in v["system_fingerprints"] if f})
        meta = rep["meta"]
        meta["started_utc"] = started
        meta["finished_utc"] = finished
        meta["requested_model"] = config.MODEL_NAME
        meta["served_models"] = all_models
        meta["system_fingerprints"] = all_fps
        meta["judge_requested_model"] = caller.model
        meta["judge_params"] = dict(caller.params)
        meta["judge_served_models"] = sorted({r.get("response_model") for r in rows if r.get("response_model")})
        meta["judge_system_fingerprints"] = sorted({r.get("system_fingerprint") for r in rows if r.get("system_fingerprint")})
        meta["rerun_of"] = "reports/latest.json"
        if ragas_records is None:
            ragas_block = {"requested_model": config.JUDGE_MODEL_NAME, "status": "未跑 RAGAS" if not args.ragas else "进行中"}
        elif not ragas_records:
            ragas_block = {"requested_model": config.JUDGE_MODEL_NAME, "status": UNRECORDED, "n_calls": 0,
                           "note": "探针未捕获到任何响应（ChatOpenAI 未走 _create_chat_result？），不推测"}
        else:
            ragas_block = {"status": "recorded", **_chain_block(config.JUDGE_MODEL_NAME, ragas_records)}
        meta["served"] = {
            "agent": _chain_block(config.MODEL_NAME, agent_calls, agent_by_q),
            "judge": _chain_block(caller.model, judge_calls, judge_by_q),
            "ragas": ragas_block,
        }
        for q in rep["per_question"]:
            t = traces[q["id"]]
            q["tool_sequence_raw"] = t.tool_sequence
            q["n_retrieve_calls"] = sum(1 for x in t.tool_sequence if x == "retrieve")
            q["first_retrieve_doc_ids"] = _first_retrieve(t)
            q["retrieve_queries"] = _queries(t)
            q["n_steps"] = len(t.steps)
            q.update(served[q["id"]])
        return rep

    def _write(rep: dict) -> None:
        out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    # 先落一份不含 RAGAS 的快照：RAGAS 要跑几十分钟，中途挂掉也不丢前面的结果
    rep = _assemble(None, None, None if args.ragas else datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _write(rep)

    if args.ragas:
        from src.eval import ragas_runner

        print("RAGAS：基于同一批轨迹算生成质量指标（裁判走 DeepSeek，挂响应探针）", file=sys.stderr)
        with ragas_probe() as records:
            ragas_result = ragas_runner.run_ragas(samples, traces)
        rep = _assemble(ragas_result, list(records), datetime.now(timezone.utc).isoformat(timespec="seconds"))
        _write(rep)

    sr = rep["metrics"]["task_success_rate"]
    print(f"task_success_rate = {sr['value']:.4f} (n={sr['n']}/{sr['total']})  served={rep['meta']['served_models']} fp={rep['meta']['system_fingerprints']}")
    rg = rep["meta"]["served"]["ragas"]
    print(f"ragas served: {rg.get('status')} n_calls={rg.get('n_calls')} models={rg.get('response_model_counts')} fp={rg.get('system_fingerprint_counts')}")
    print(f"-> {out_json}\n-> {out_traces}\n-> {out_judge}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
