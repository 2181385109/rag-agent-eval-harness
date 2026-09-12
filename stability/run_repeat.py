"""重复运行：黄金集每题跑 k 次（temperature=0），一次运行落一行（PERF_SPEC B1）。

    python -m stability.run_repeat                      # 36 题 × k=5，约 18 万 token/轮
    python -m stability.run_repeat --ids cap_001,cap_007 --runs 2    # 调试
    python -m stability.run_repeat --limit 5            # 前 5 题
    python -m stability.run_repeat --resume stability/raw/run_20260912T010203Z.jsonl
                                                        # 续跑：已有的 (题, 次) 跳过

产物：
    stability/raw/run_{ts}.jsonl        一行一次运行（RunRecord）
    stability/raw/run_{ts}.meta.json    环境、模型、k、顺序、起止时间——所有随机性来源

三条纪律（PERF_SPEC §1）：
  - 每行跑完立刻落盘并 flush：中途掐断也不丢已花钱跑出的记录。
  - 报错的运行照样落一行（error 字段非空），不删、不重跑、不挑。
  - 产物文件名带时间戳，永不覆盖；改口径就是新文件。

顺序是 **pass-major**：先把所有题跑一遍（pass 0），再跑第二遍……
这样"第 r 次 pass 的成功率"是自然量，且同题两次调用之间隔了整整一轮，
减轻（但消除不了）DeepSeek 服务端 KV 缓存对后几次延迟的影响——
缓存命中量已逐行记录在 tokens.cache_hit 里，分析时按 pass 拆开看。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.agent.trace import AgentTrace
from src.eval.datasets import GoldenSample, load_golden_set
from stability.analyze import RAW_DIR
from stability.instrument import instrumented
from stability.records import Latency, RunRecord, Tokens


# ------------------------------------------------------------------ 环境
def _cpu_model() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / 1024**3, 1)
        if Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except Exception:
        pass
    return None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.PROJECT_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def environment() -> dict:
    """硬件 / 软件标识。刻意不记主机名、用户名、绝对路径（PERF_SPEC §5.3）。"""
    return {
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "openai": _pkg_version("openai"),
        "langgraph": _pkg_version("langgraph"),
        "sentence_transformers": _pkg_version("sentence-transformers"),
        "faiss_cpu": _pkg_version("faiss-cpu"),
        "torch": _pkg_version("torch"),
    }


def run_meta(k: int, sample_ids: list[str], run_id: str) -> dict:
    return {
        "run_id": run_id,
        "k": k,
        "n_questions": len(sample_ids),
        "sample_ids": sample_ids,
        "ordering": "pass-major（先把所有题跑完第 0 次，再第 1 次……）",
        "model": config.MODEL_NAME,
        "base_url": config.BASE_URL,
        "temperature": config.DEFAULT_TEMPERATURE,
        "max_tokens": config.DEFAULT_MAX_TOKENS,
        "request_timeout_s": config.REQUEST_TIMEOUT_S,
        "max_retries": config.MAX_RETRIES,
        "seed": "none — 本机侧无随机数：FAISS 精确内积搜索、规则判定均确定；唯一随机源是 API 侧解码（已发 temperature=0）",
        "embedding_model": config.EMBEDDING_MODEL,
        "use_query_instruction": config.USE_QUERY_INSTRUCTION,
        "retrieve_top_k": config.RETRIEVE_TOP_K,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "max_agent_steps": config.MAX_AGENT_STEPS,
        "git_commit": _git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "finished_utc": None,
        "interrupted": False,
        "env": environment(),
    }


# ------------------------------------------------------------------ 单次运行
def run_once(agent, sample: GoldenSample, run_index: int, k: int, run_id: str) -> RunRecord:
    """跑一次，无论成败都返回一条记录。"""
    t0 = time.perf_counter()
    error: str | None = None
    trace: AgentTrace | None = None
    with instrumented(agent) as timer:
        try:
            trace = agent.run(sample.question)
        except Exception as exc:  # 网络抖动、超时、限流……如实记录，不重试不掩盖
            error = f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - t0

    latency = Latency(
        total_s=total,
        llm_s=timer.llm_s,
        retrieve_s=timer.retrieve_s,
        tool_s=timer.tool_s,
        n_llm_calls=timer.n_llm_calls,
        n_retrieve_calls=timer.n_retrieve_calls,
        n_tool_calls=timer.n_tool_calls,
        llm_call_s=timer.llm_call_s,
        llm_call_cache_hit=timer.llm_call_cache_hit,
        llm_call_cache_miss=timer.llm_call_cache_miss,
    )
    tokens = Tokens(
        prompt=trace.prompt_tokens if trace else 0,
        completion=trace.completion_tokens if trace else 0,
        total=trace.total_tokens if trace else 0,
        cache_hit=timer.cache_hit_tokens,
        cache_miss=timer.cache_miss_tokens,
    )
    return RunRecord(
        run_id=run_id,
        sample_id=sample.id,
        run_index=run_index,
        k=k,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        answer_type=sample.answer_type,
        question=sample.question,
        answer=(trace.answer if trace else f"[运行失败] {error}"),
        tool_sequence=(trace.tool_sequence if trace else []),
        retrieved_doc_ids=(trace.retrieved_doc_ids if trace else []),
        stop_reason=(trace.stop_reason if trace else "error"),
        n_steps=(len(trace.steps) if trace else 0),
        latency=latency,
        tokens=tokens,
        error=error,
        response_model=timer.response_model,
        system_fingerprint=timer.system_fingerprint,
        response_models=timer.response_models,
        system_fingerprints=timer.system_fingerprints,
    )


# ------------------------------------------------------------------ 预热
def warm_up(agent) -> dict:
    """触发一次性初始化，不让它混进第一条记录。

    实测不预热时第一条记录的 retrieve 段是 11.8s（BGE 权重懒加载），
    之后稳定在 0.3s 左右——那 11.5s 是进程冷启动，不是检索延迟。
    同理 LLM 侧先发一次 1-token 请求，把 TLS 握手与连接池建立摊掉。
    预热的耗时与 token 都记进 meta，不丢。
    """
    from src.agent import llm

    t0 = time.perf_counter()
    agent.toolbox.retriever.retrieve("预热")
    retriever_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    resp = llm.chat_completion([{"role": "user", "content": "ping"}], max_tokens=1)
    llm_s = time.perf_counter() - t0
    usage = getattr(resp, "usage", None)
    return {
        "retriever_s": retriever_s,
        "llm_s": llm_s,
        "llm_tokens": getattr(usage, "total_tokens", None),
        "note": "预热不计入任何指标：一次 retrieve 触发 BGE 懒加载，一次 1-token 请求建立 HTTPS 连接",
    }


# ------------------------------------------------------------------ 批次
def _existing_pairs(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done.add((row["sample_id"], int(row["run_index"])))
    return done


def _write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def run_batch(agent, samples: list[GoldenSample], k: int, raw_path: Path, meta: dict) -> int:
    """pass-major 跑完全部 (题, 次)，逐行追加落盘。返回本次新增行数。"""
    meta_path = raw_path.with_suffix(".meta.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    warm = warm_up(agent)
    meta.setdefault("warmups", []).append(warm)
    print(f"预热：retrieve {warm['retriever_s']:.2f}s，llm {warm['llm_s']:.2f}s", file=sys.stderr)
    _write_meta(meta_path, meta)

    done = _existing_pairs(raw_path)
    todo = [(r, s) for r in range(k) for s in samples if (s.id, r) not in done]
    print(f"待跑 {len(todo)} 次（已存在 {len(done)} 次），落盘 {raw_path.name}", file=sys.stderr)

    written = 0
    try:
        with raw_path.open("a", encoding="utf-8") as fh:
            for i, (run_index, sample) in enumerate(todo, start=1):
                rec = run_once(agent, sample, run_index, k, meta["run_id"])
                fh.write(rec.model_dump_json() + "\n")
                fh.flush()
                written += 1
                status = "ERR " if rec.error else "ok  "
                print(
                    f"  [{i}/{len(todo)}] pass {run_index} {sample.id} {status}"
                    f"{rec.latency.total_s:6.2f}s  llm {rec.latency.llm_s:5.2f}s  "
                    f"retrieve {rec.latency.retrieve_s:4.2f}s  tokens {rec.tokens.total}"
                    f"  cache_hit {rec.tokens.cache_hit}  tools {rec.tool_sequence}",
                    file=sys.stderr, flush=True,
                )
    except KeyboardInterrupt:
        meta["interrupted"] = True
        print("已中断：已落盘的记录保留，可用 --resume 续跑", file=sys.stderr)
        raise
    finally:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta["n_rows"] = len(_existing_pairs(raw_path))
        _write_meta(meta_path, meta)
    return written


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="黄金集每题重复 k 次（temperature=0）")
    parser.add_argument("--runs", type=int, default=config.STABILITY_RUNS, help="重复次数 k")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--ids", type=str, default=None, help="只跑这些题，逗号分隔")
    parser.add_argument("--resume", type=str, default=None, help="续跑已有的 run_*.jsonl")
    parser.add_argument("--out-dir", type=str, default=str(RAW_DIR))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.runs < 2:
        raise SystemExit("--runs 至少为 2，否则谈不上稳定性")

    samples = load_golden_set()
    if args.ids:
        wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
        by_id = {s.id: s for s in samples}
        missing = [x for x in wanted if x not in by_id]
        if missing:
            raise SystemExit(f"黄金集里没有这些 id：{missing}")
        samples = [by_id[x] for x in wanted]
    if args.limit:
        samples = samples[: args.limit]

    if args.resume:
        raw_path = Path(args.resume)
        meta_path = raw_path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise SystemExit(f"续跑需要 {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["k"] != args.runs or meta["sample_ids"] != [s.id for s in samples]:
            raise SystemExit("续跑的 k 或题目集合与原批次不一致——那是新批次，不要用 --resume")
        meta["resumed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_path = Path(args.out_dir) / f"run_{run_id}.jsonl"
        meta = run_meta(args.runs, [s.id for s in samples], run_id)

    from src.agent import graph

    print(f"{len(samples)} 题 × k={args.runs}，temperature={config.DEFAULT_TEMPERATURE}，构建 Agent（加载 BGE + FAISS）...", file=sys.stderr)
    agent = graph.build_default_agent()
    n = run_batch(agent, samples, args.runs, raw_path, meta)
    print(f"完成：新增 {n} 行 -> {raw_path}", file=sys.stderr)
    print(f"下一步：python -m stability.analyze --raw {raw_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
