"""回归门禁的验收口径（CLAUDE.md §7，M6）。

两道闸管两件事：

- **闸 A（口径闸）**：输入冻结（`tests/fixtures/`），重算指标必须与基线逐位相等。
  输入没变、数字变了，只可能是指标代码的口径变了——涨了也要拦，不只是跌。
- **闸 B（回归闸）**：本次真实运行的报告 vs 上一份快照，关键指标跌超容差即 fail。
  真实运行带模型噪声，所以用容差而不是等值。

全部离线：闸 A 读 fixture，闸 B 读已提交进 git 的报告 JSON，不碰 DeepSeek API。
"""

from __future__ import annotations

import json

import pytest

from src import config
from src.eval import report as report_mod

# 冻结的子集。改这个列表 = 换了门禁的被测输入，必须是有意为之。
EXPECTED_FIXTURE_IDS = [
    "cap_001",
    "cap_005",
    "cap_007",
    "cap_010",
    "cap_026",
    "cap_030",
    "cap_033",
    "cap_034",
    "cap_035",
    "cap_036",
]


# ---------------------------------------------------------------- fixture 本身
def test_fixture_subset_is_frozen():
    samples, traces = report_mod.load_gate_fixture()
    assert [s.id for s in samples] == EXPECTED_FIXTURE_IDS
    assert set(traces) == set(EXPECTED_FIXTURE_IDS)


def test_fixture_contains_known_failures():
    """门禁的输入必须含失败样本。

    全是满分的子集，指标算错了也照样"绿"——那道闸就是摆设。
    """
    samples, traces = report_mod.load_gate_fixture()
    metrics = report_mod.compute_gate_metrics(samples, traces)
    assert metrics["recall_at_k"]["value"] < 1.0, "缺检索失败的样本"
    assert metrics["tool_accuracy"]["value"] < 1.0, "缺工具选错的样本"
    assert any(s.failure_tag == "hallucination_bait" for s in samples), "缺幻觉诱饵题"


def test_fixture_carries_no_corpus_text():
    """fixture 不复制 corpus 原文：门禁只用 doc_id，原文进 git 没必要。"""
    raw = report_mod.GATE_TRACES_PATH.read_text(encoding="utf-8")
    for row in (json.loads(line) for line in raw.splitlines() if line.strip()):
        for step in row["trace"]["steps"]:
            for res in step["results"]:
                for chunk in res["retrieved"]:
                    assert chunk["text"] == ""


# ------------------------------------------------------------------ 闸 A
def test_exact_gate_passes_against_committed_baseline():
    """当前代码在冻结输入上重算，必须与提交进 git 的基线完全一致。"""
    findings = report_mod.check_exact_gate()
    changed = [f for f in findings if f["status"] != "ok"]
    assert not changed, f"指标口径变了：{changed}"


def test_exact_gate_catches_a_changed_metric():
    """把基线改一个数，闸 A 必须拦下来——包括**变高**。"""
    baseline = report_mod.load_gate_baseline()
    baseline["metrics"]["tool_accuracy"]["value"] = 0.5  # 假装基线更低
    findings = {f["metric"]: f for f in report_mod.check_exact_gate(baseline)}
    assert findings["tool_accuracy"]["status"] == "changed"
    assert findings["recall_at_k"]["status"] == "ok"


def test_exact_gate_catches_changed_denominator():
    """值一样但分母变了同样是口径变动（recall 只在有标注的题上算）。"""
    baseline = report_mod.load_gate_baseline()
    baseline["metrics"]["recall_at_k"]["n"] += 1
    findings = {f["metric"]: f for f in report_mod.check_exact_gate(baseline)}
    assert findings["recall_at_k"]["status"] == "changed"


def test_exact_gate_flags_missing_baseline_entry():
    baseline = report_mod.load_gate_baseline()
    del baseline["metrics"]["task_success_rate"]
    findings = {f["metric"]: f for f in report_mod.check_exact_gate(baseline)}
    assert findings["task_success_rate"]["status"] == "missing_baseline"


# ------------------------------------------------------------------ 闸 B
def _report(size=36, success=0.9, tool=0.94, faithfulness=0.86):
    return {
        "meta": {"golden_set_size": size},
        "metrics": {
            "task_success_rate": {"value": success, "n": size, "total": size},
            "tool_accuracy": {"value": tool, "n": size, "total": size},
            "ragas": {"scores": {"faithfulness": faithfulness}},
        },
    }


def test_regression_gate_passes_when_flat():
    findings = report_mod.check_regression_gate(_report(), _report())
    assert {f["status"] for f in findings} == {"ok"}


def test_regression_gate_fails_on_drop_beyond_tolerance():
    current = _report(success=0.9 - config.METRIC_DROP_TOLERANCE - 0.01)
    findings = {f["metric"]: f for f in report_mod.check_regression_gate(current, _report())}
    assert findings["task_success_rate"]["status"] == "dropped"
    assert findings["tool_accuracy"]["status"] == "ok"


def test_regression_gate_tolerates_drop_within_tolerance():
    """小幅波动不该报警——真实运行本就有模型噪声，否则门禁天天误报。"""
    current = _report(success=0.9 - config.METRIC_DROP_TOLERANCE + 0.001)
    findings = {f["metric"]: f for f in report_mod.check_regression_gate(current, _report())}
    assert findings["task_success_rate"]["status"] == "ok"


def test_regression_gate_never_fails_on_improvement():
    current = _report(success=1.0, tool=1.0, faithfulness=0.99)
    assert {f["status"] for f in report_mod.check_regression_gate(current, _report())} == {"ok"}


def test_regression_gate_reads_faithfulness_from_ragas_block():
    """faithfulness 不在 metrics 顶层，取值路径不同，别取成 None 后静默跳过。"""
    current = _report(faithfulness=0.5)
    findings = {f["metric"]: f for f in report_mod.check_regression_gate(current, _report())}
    assert findings["faithfulness"]["status"] == "dropped"
    assert findings["faithfulness"]["baseline"] == 0.86


def test_regression_gate_marks_missing_metric():
    current = _report()
    del current["metrics"]["ragas"]
    findings = {f["metric"]: f for f in report_mod.check_regression_gate(current, _report())}
    assert findings["faithfulness"]["status"] == "missing"


def test_regression_gate_refuses_to_compare_different_golden_set_sizes():
    """题数不同就是在不同题目集上算的均值，差值没有意义（分母陷阱）。"""
    findings = report_mod.check_regression_gate(_report(size=40), _report(size=36))
    assert {f["status"] for f in findings} == {"incomparable"}
    assert all("40" in f["note"] and "36" in f["note"] for f in findings)


# -------------------------------------------------------- 快照选取 / 真实报告
def test_find_previous_snapshot_excludes_current(tmp_path):
    for name in ("eval_20260101T000000Z.json", "eval_20260202T000000Z.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    prev = report_mod.find_previous_snapshot(
        tmp_path, current=tmp_path / "eval_20260202T000000Z.json"
    )
    assert prev.name == "eval_20260101T000000Z.json"


def test_find_previous_snapshot_returns_none_when_alone(tmp_path):
    (tmp_path / "eval_20260101T000000Z.json").write_text("{}", encoding="utf-8")
    assert (
        report_mod.find_previous_snapshot(
            tmp_path, current=tmp_path / "eval_20260101T000000Z.json"
        )
        is None
    )


def test_committed_reports_have_not_regressed():
    """真正的回归断言：仓库里最新的报告相对上一份快照没有跌破容差。

    这条测试读的是已提交进 git 的 JSON，不跑评测、不调 API。
    有人提交了一份变差的报告，CI 就会在这里红。
    """
    latest = config.REPORTS_DIR / "latest.json"
    if not latest.exists():
        pytest.skip("还没有 latest.json")
    current = json.loads(latest.read_text(encoding="utf-8"))
    previous = report_mod.find_previous_snapshot(
        current=config.REPORTS_DIR / report_mod.snapshot_filename(current["meta"]["timestamp_utc"])
    )
    if previous is None:
        pytest.skip("除本次外没有可比的历史快照")

    findings = report_mod.check_regression_gate(
        current, json.loads(previous.read_text(encoding="utf-8"))
    )
    dropped = [f for f in findings if f["status"] == "dropped"]
    assert not dropped, f"关键指标相对 {previous.name} 跌破容差：{dropped}"


def test_run_gates_returns_zero_when_clean():
    assert report_mod.run_gates() == 0
