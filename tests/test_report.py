"""报告汇总链路的验收口径。

CLAUDE.md §7：CI 里不调真实 API，只验证流水线和指标计算逻辑本身。
所以这里用构造轨迹跑完整的 build_report -> render_markdown -> write_report。
"""

from __future__ import annotations

import json

import pytest

from src.eval import report as report_mod
from tests.test_metrics import make_sample, make_trace


@pytest.fixture
def tiny_eval():
    samples = [
        make_sample(id="a", expected_tool=["retrieve"], expected_doc_ids=["d1"], answer_keys=["0.25"]),
        make_sample(id="b", expected_tool=["calc"], expected_doc_ids=[], answer_keys=["7"]),
        make_sample(id="c", answer_type="open", answer_keys=[], expected_doc_ids=["d2"]),
    ]
    traces = {
        "a": make_trace(answer="阈值 0.25", tools=["retrieve"], retrieved=[["d1", "dx"]]),
        "b": make_trace(answer="等于 7", tools=["retrieve", "calc"]),
        "c": make_trace(answer="一段开放回答", tools=["retrieve"], retrieved=[["zz"]]),
    }
    return samples, traces


def test_report_carries_reproducibility_metadata(tiny_eval):
    """没有这些字段，指标过两周就没法复现——违背第一红线。"""
    samples, traces = tiny_eval
    meta = report_mod.build_report(samples, traces)["meta"]
    for key in (
        "timestamp_utc",
        "git_commit",
        "model",
        "embedding_model",
        "use_query_instruction",
        "retrieve_top_k",
        "chunk_size",
        "chunk_overlap",
        "temperature",
        "golden_set_size",
    ):
        assert key in meta, f"报告缺少复现所需字段：{key}"
    assert meta["golden_set_size"] == 3


def test_metrics_carry_their_denominators(tiny_eval):
    samples, traces = tiny_eval
    m = report_mod.build_report(samples, traces)["metrics"]

    # recall 只算有 expected_doc_ids 的 a、c 两题
    assert m["recall_at_k"]["n"] == 2
    assert m["recall_at_k"]["total"] == 3
    assert m["recall_at_k"]["value"] == pytest.approx(0.5)  # a 命中、c 未命中

    # 任务成功率只算闭合题 a、b
    assert m["task_success_rate"]["n"] == 2
    assert m["task_success_rate"]["total"] == 3
    assert m["task_success_rate"]["value"] == 1.0

    # 工具准确率全部题都算；b 多调了一次 retrieve 算错
    assert m["tool_accuracy"]["n"] == 3
    assert m["tool_accuracy"]["value"] == pytest.approx(2 / 3)


def test_strict_and_loose_tool_metrics_diverge_on_wrong_order():
    """两个口径只在"用对了工具但顺序反了"时才分岔——正是要让人看见的那个差值。"""
    samples = [make_sample(id="a", expected_tool=["retrieve", "calc"], answer_keys=["x"])]
    traces = {"a": make_trace(answer="x", tools=["calc", "retrieve"])}
    m = report_mod.build_report(samples, traces)["metrics"]
    assert m["tool_accuracy"]["value"] == 0.0
    assert m["tool_set_accuracy"]["value"] == 1.0


def test_per_question_detail_is_complete(tiny_eval):
    samples, traces = tiny_eval
    rows = report_mod.build_report(samples, traces)["per_question"]
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    row = rows[1]
    assert row["expected_tool"] == ["calc"]
    assert row["actual_tool"] == ["retrieve", "calc"]
    assert row["tool_ok"] is False


def test_consistency_section_included_when_runs_given(tiny_eval):
    samples, traces = tiny_eval
    runs = {"a": [make_trace(answer="0.25"), make_trace(answer="错")]}
    m = report_mod.build_report(samples, traces, consistency_runs=runs)["metrics"]
    assert m["consistency"]["runs"] == 2
    assert m["consistency"]["success_agreement"] == pytest.approx(0.5)


def test_consistency_absent_when_not_run(tiny_eval):
    samples, traces = tiny_eval
    assert report_mod.build_report(samples, traces)["metrics"]["consistency"] is None


def test_markdown_states_every_denominator(tiny_eval):
    samples, traces = tiny_eval
    md = report_mod.render_markdown(report_mod.build_report(samples, traces))
    assert "n=2/3" in md, "recall / 成功率的分母必须出现在报告正文里"
    assert "n=3/3" in md
    assert "逐题明细" in md


def test_write_report_creates_snapshot_latest_and_markdown(tiny_eval, tmp_path):
    samples, traces = tiny_eval
    report = report_mod.build_report(samples, traces)
    paths = report_mod.write_report(report, directory=tmp_path)

    assert paths["snapshot"].exists() and paths["latest"].exists() and paths["markdown"].exists()
    assert paths["snapshot"].name.startswith("eval_")
    restored = json.loads(paths["latest"].read_text(encoding="utf-8"))
    assert restored["metrics"]["recall_at_k"]["n"] == 2


def test_run_samples_records_failures_instead_of_crashing():
    """单题跑挂了不能中断整轮，要如实记成一次失败。"""

    class Boom:
        def run(self, question):
            raise RuntimeError("网络超时")

    samples = [make_sample(id="a")]
    runs = report_mod.run_samples(Boom(), samples, repeats=2, progress=False)
    assert len(runs["a"]) == 2
    assert all(t.stop_reason == "error" for t in runs["a"])
    assert "网络超时" in runs["a"][0].answer
