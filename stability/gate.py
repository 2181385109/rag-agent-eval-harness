"""闸 C：稳定性门禁（PERF_SPEC B4）。并入 src/eval/report.run_gates 的第三闸。

两组检查，全部离线：

  check_thresholds(summary)   阈值闸——成功率跨次标准差、轨迹自洽率、k
  check_reproducible(summary) 可复现闸——summary.json 的数字必须能由它引用的
                              原始产物重算出来（答案相似度除外：要加载 BGE，
                              CI 上不算，只核对记录了 embedding 模型名）

口径定义在 tests/test_stability_gate.py。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

from src import config
from stability import analyze


def _finding(check: str, status: str, value=None, threshold=None, note: str = "") -> dict:
    return {"check": check, "status": status, "value": value, "threshold": threshold, "note": note}


# ------------------------------------------------------------------ 阈值闸
def check_thresholds(summary: dict) -> list[dict]:
    """指标缺失也是 fail：拿不到数就放行的门禁形同虚设。"""
    st = summary.get("stability", {})
    k = summary.get("meta", {}).get("k")
    out: list[dict] = []

    out.append(
        _finding("k", "ok" if isinstance(k, int) and k >= 2 else "fail", k, 2, "重复次数 k 至少 2 才谈得上稳定性")
    )

    std = st.get("success_rate_per_pass", {}).get("std_sample")
    limit = config.STABILITY_MAX_SUCCESS_RATE_STD
    if std is None:
        out.append(_finding("success_rate_std", "fail", None, limit, "指标缺失（None）"))
    else:
        out.append(_finding("success_rate_std", "ok" if std <= limit else "fail", std, limit, "任务成功率跨 k 次的样本标准差 ≤ 阈值"))

    traj = st.get("trajectory_consistency", {}).get("value")
    floor = config.STABILITY_MIN_TRAJECTORY_CONSISTENCY
    if traj is None:
        out.append(_finding("trajectory_consistency", "fail", None, floor, "指标缺失（None）"))
    else:
        out.append(_finding("trajectory_consistency", "ok" if traj >= floor else "fail", traj, floor, "轨迹自洽率 ≥ 阈值"))
    return out


# ------------------------------------------------------------------ 可复现闸
NOT_RECOMPUTED = ("answer_similarity", "main_snapshot_comparison", "retrieval_persistence")


def _strip_similarity(section: dict) -> dict:
    """摘掉不在离线重算范围内的两段：答案相似度（要 BGE）、主评测快照对照（latest.json 会更新）。"""
    return {k: v for k, v in section.items() if k not in NOT_RECOMPUTED}


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def check_reproducible(
    summary: dict,
    *,
    samples: Sequence | None = None,
    raw_path: Path | str | None = None,
    judge_path: Path | str | None = None,
) -> list[dict]:
    """从 summary 引用的原始产物重算，逐段比对。"""
    meta = summary.get("meta", {})
    raw = Path(raw_path) if raw_path else (config.PROJECT_ROOT / meta.get("raw_file", ""))
    out: list[dict] = []

    if not meta.get("raw_file") and raw_path is None:
        return [_finding("raw_file", "fail", None, None, "summary.meta.raw_file 缺失，数字无从追溯")]
    if not raw.exists():
        return [_finding("raw_file", "fail", str(raw), None, "summary 引用的原始产物不存在")]

    digest = analyze.sha256_of(raw)
    out.append(
        _finding(
            "raw_sha256",
            "ok" if digest == meta.get("raw_sha256") else "fail",
            digest,
            meta.get("raw_sha256"),
            "原始产物的 sha256 必须与 summary 记录的一致（改过原始记录就对不上）",
        )
    )

    rows = analyze.load_rows(raw)
    if samples is None:
        from src.eval.datasets import load_golden_set

        samples = load_golden_set()
    judge_rows = None
    if judge_path is not None:
        judge_rows = analyze.load_judge_rows(judge_path)
    elif meta.get("judge_file"):
        jp = config.PROJECT_ROOT / meta["judge_file"]
        if jp.exists():
            judge_rows = analyze.load_judge_rows(jp)
        else:
            out.append(_finding("judge_file", "fail", str(jp), None, "summary 引用的逐次裁判分文件不存在"))

    probe = None
    probe_path = None
    if meta.get("probe_file"):
        probe_path = config.PROJECT_ROOT / meta["probe_file"]
        if probe_path.exists():
            probe = analyze.load_probe(probe_path)
        else:
            out.append(_finding("probe_file", "fail", str(probe_path), None, "summary 引用的模型探针文件不存在"))

    recomputed = analyze.build_summary(
        rows, samples, run_meta=meta.get("run_meta"), judge_rows=judge_rows, embedder=None,
        raw_path=raw, judge_path=(judge_path or (config.PROJECT_ROOT / meta["judge_file"] if meta.get("judge_file") else None)),
        probe=probe, probe_path=probe_path,
    )

    for section in ("latency", "cost", "model_attribution"):
        same = _canon(recomputed.get(section)) == _canon(summary.get(section))
        out.append(_finding(section, "ok" if same else "fail", None, None, f"{section} 段重算必须逐位相等"))

    same = _canon(_strip_similarity(recomputed["stability"])) == _canon(_strip_similarity(summary.get("stability", {})))
    out.append(_finding("stability", "ok" if same else "fail", None, None, "stability 段（答案相似度除外）重算必须逐位相等"))

    sim = summary.get("stability", {}).get("answer_similarity")
    if sim is not None and sim.get("n", 0) > 0:
        has_model = bool(sim.get("embedding_model"))
        out.append(_finding("answer_similarity_provenance", "ok" if has_model else "fail", sim.get("embedding_model"), None, "答案相似度不在离线重算范围，但必须记录所用 embedding 模型"))

    same_n = recomputed["meta"]["n_rows"] == meta.get("n_rows") and recomputed["meta"]["k"] == meta.get("k")
    out.append(_finding("meta_n_k", "ok" if same_n else "fail", (recomputed["meta"]["n_rows"], recomputed["meta"]["k"]), (meta.get("n_rows"), meta.get("k")), "n 与 k 必须与原始产物一致"))
    return out


# ------------------------------------------------------------------ 入口
def run_gate_c(summary_path: Path | None = None) -> tuple[bool, list[str]]:
    """跑闸 C，返回 (是否通过, 打印行)。summary 不存在时跳过（视为通过，但打印说明）。"""
    path = summary_path or analyze.SUMMARY_PATH
    lines: list[str] = []
    if not path.exists():
        lines.append("闸 C（稳定性闸）：跳过——还没有 stability/summary.json（尚未做首次测量）")
        return True, lines

    summary = json.loads(path.read_text(encoding="utf-8"))
    lines.append(
        f"闸 C（稳定性闸）：{path.relative_to(config.PROJECT_ROOT).as_posix()}，"
        f"std ≤ {config.STABILITY_MAX_SUCCESS_RATE_STD}，轨迹自洽率 ≥ {config.STABILITY_MIN_TRAJECTORY_CONSISTENCY}"
    )
    ok = True
    for f in check_thresholds(summary) + check_reproducible(summary):
        if f["status"] != "ok":
            ok = False
        shown = f["value"] if not isinstance(f["value"], float) else f"{f['value']:.4f}"
        lines.append(f"  {f['status']:9} {f['check']}: {shown}  （阈值/期望 {f['threshold']}）  {f['note']}")
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok, lines = run_gate_c()
    print("\n".join(lines))
    print("闸 C 结果：" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
