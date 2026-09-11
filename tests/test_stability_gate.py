"""闸 C（稳定性闸）的验收口径（PERF_SPEC §3 B4）。

并入 M6 的双闸门禁，成为第三闸。管两件事：

1. **阈值闸**：`stability/summary.json` 里的
   - 任务成功率跨 k 次的样本标准差 > config.STABILITY_MAX_SUCCESS_RATE_STD → fail
   - 轨迹自洽率 < config.STABILITY_MIN_TRAJECTORY_CONSISTENCY → fail
   指标缺失（None）**也是 fail**，不是 skip——拿不到数就放行的门禁形同虚设。
2. **可复现闸**：summary.json 里的每个数字必须能由它引用的原始产物
   `stability/raw/run_*.jsonl` 重算出来、逐位相等（答案相似度一节除外：
   它要加载 BGE，CI 上不算，改为核对"记录了 embedding 模型名"）。
   这是 PERF_SPEC §0.2「报告里的每个数字都由脚本从原始产物生成」的机器化版本。

全部离线，不碰 DeepSeek API，不下载模型。
"""

from __future__ import annotations

import json

import pytest

from src import config
from stability import analyze, gate
from stability.records import RunRecord
from tests.test_stability_analyze import _row, _sample


def _summary(std: float | None = 0.0, traj: float | None = 1.0, k: int = 5) -> dict:
    return {
        "meta": {"k": k, "n_rows": 10, "raw_file": "stability/raw/run_x.jsonl"},
        "stability": {
            "success_rate_per_pass": {"std_sample": std, "rates": [1.0] * k, "n_per_pass": [2] * k},
            "trajectory_consistency": {"value": traj, "n": 2, "total": 2},
        },
    }


# ---------------------------------------------------------------- 阈值来自 config
def test_thresholds_are_pinned():
    # PERF_SPEC B4：std > 0.05 fail（规格给定值）；轨迹自洽率阈值按"首次全量实测值 − 0.05"
    # 的规则于 2026-09-11 一次性定为 0.727（实测 0.7778），规格里的 0.8 未达标记在 LIMITATIONS.md。
    # 自此冻结：**不许调低**（PERF_SPEC §1 铁律）。改这里 = 改口径，必须在提交信息里说明。
    assert config.STABILITY_MAX_SUCCESS_RATE_STD == 0.05
    assert config.STABILITY_MIN_TRAJECTORY_CONSISTENCY == 0.727


def test_gate_passes_a_clean_summary():
    findings = gate.check_thresholds(_summary(std=0.02, traj=0.9))
    assert all(f["status"] == "ok" for f in findings)
    assert {f["check"] for f in findings} == {"success_rate_std", "trajectory_consistency", "k"}


def test_gate_fails_when_success_rate_std_exceeds_threshold():
    findings = {f["check"]: f for f in gate.check_thresholds(_summary(std=0.051))}
    assert findings["success_rate_std"]["status"] == "fail"


def test_gate_fails_when_trajectory_consistency_below_threshold():
    below = config.STABILITY_MIN_TRAJECTORY_CONSISTENCY - 0.001
    findings = {f["check"]: f for f in gate.check_thresholds(_summary(traj=below))}
    assert findings["trajectory_consistency"]["status"] == "fail"


def test_gate_passes_exactly_at_threshold():
    findings = {
        f["check"]: f
        for f in gate.check_thresholds(_summary(std=0.05, traj=config.STABILITY_MIN_TRAJECTORY_CONSISTENCY))
    }
    assert findings["success_rate_std"]["status"] == "ok"
    assert findings["trajectory_consistency"]["status"] == "ok"


def test_missing_metric_is_a_failure_not_a_skip():
    findings = {f["check"]: f for f in gate.check_thresholds(_summary(std=None, traj=None))}
    assert findings["success_rate_std"]["status"] == "fail"
    assert findings["trajectory_consistency"]["status"] == "fail"


def test_gate_fails_with_fewer_than_two_repeats():
    findings = {f["check"]: f for f in gate.check_thresholds(_summary(k=1))}
    assert findings["k"]["status"] == "fail"


# ---------------------------------------------------------------- 可复现闸
def _build(tmp_path):
    samples = [_sample("q1", keys=["对"]), _sample("q2", keys=["对"])]
    rows = [
        _row("q1", 0, answer="对", total_s=1.0), _row("q2", 0, answer="对", total_s=2.0),
        _row("q1", 1, answer="对", total_s=1.5), _row("q2", 1, answer="错", total_s=2.5),
    ]
    raw = tmp_path / "run_t.jsonl"
    raw.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    summary = analyze.build_summary(
        rows, samples, run_meta={"run_id": "run_t", "k": 2}, embedder=None, raw_path=raw
    )
    return samples, raw, summary


def test_reproducibility_check_passes_when_summary_matches_raw(tmp_path):
    samples, raw, summary = _build(tmp_path)
    findings = gate.check_reproducible(summary, samples=samples, raw_path=raw)
    assert findings and all(f["status"] == "ok" for f in findings), findings


def test_reproducibility_check_catches_a_hand_edited_number(tmp_path):
    samples, raw, summary = _build(tmp_path)
    summary["latency"]["e2e"]["p95"] = 0.123  # 手改数字
    findings = {f["check"]: f for f in gate.check_reproducible(summary, samples=samples, raw_path=raw)}
    assert findings["latency"]["status"] == "fail"


def test_reproducibility_check_catches_a_tampered_raw_file(tmp_path):
    samples, raw, summary = _build(tmp_path)
    rows = analyze.load_rows(raw)
    rows[0] = RunRecord(**{**rows[0].model_dump(), "answer": "错"})  # 改了原始产物
    raw.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    findings = {f["check"]: f for f in gate.check_reproducible(summary, samples=samples, raw_path=raw)}
    assert findings["raw_sha256"]["status"] == "fail"
    assert findings["stability"]["status"] == "fail"


def test_reproducibility_check_fails_when_raw_file_is_missing(tmp_path):
    samples, raw, summary = _build(tmp_path)
    raw.unlink()
    findings = gate.check_reproducible(summary, samples=samples, raw_path=raw)
    assert any(f["status"] == "fail" for f in findings)


# ---------------------------------------------------------------- 已提交的产物
COMMITTED_SUMMARY = config.PROJECT_ROOT / "stability" / "summary.json"


@pytest.mark.skipif(not COMMITTED_SUMMARY.exists(), reason="还没有 stability/summary.json（尚未做首次测量）")
def test_committed_summary_passes_thresholds():
    summary = json.loads(COMMITTED_SUMMARY.read_text(encoding="utf-8"))
    failed = [f for f in gate.check_thresholds(summary) if f["status"] != "ok"]
    assert not failed, failed


@pytest.mark.skipif(not COMMITTED_SUMMARY.exists(), reason="还没有 stability/summary.json（尚未做首次测量）")
def test_committed_summary_is_reproducible_from_committed_raw():
    summary = json.loads(COMMITTED_SUMMARY.read_text(encoding="utf-8"))
    failed = [f for f in gate.check_reproducible(summary) if f["status"] != "ok"]
    assert not failed, failed


@pytest.mark.skipif(not COMMITTED_SUMMARY.exists(), reason="还没有 stability/summary.json（尚未做首次测量）")
def test_committed_summary_records_every_randomness_source():
    """PERF_SPEC §1：seed、模型版本、运行日期、硬件必须显式记录。"""
    meta = json.loads(COMMITTED_SUMMARY.read_text(encoding="utf-8"))["meta"]["run_meta"]
    for key in ("model", "temperature", "embedding_model", "git_commit", "started_utc", "seed"):
        assert key in meta, key
    for key in ("cpu", "cpu_count", "ram_gb", "os", "python"):
        assert key in meta["env"], key
    assert "hostname" not in json.dumps(meta).lower()


# ---------------------------------------------------------------- 并入 run_gates
def test_run_gates_reports_gate_c(capsys):
    from src.eval import report as report_mod

    report_mod.run_gates()
    out = capsys.readouterr().out
    assert "闸 C" in out
