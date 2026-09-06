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


# ------------------------------------ 双温度一致性 + 异常样本（M3 增补）
def test_report_holds_both_consistency_passes(tiny_eval):
    """temp=0 的可复现基线与 temp>0 的鲁棒性专测并存报数，不互相替代。"""
    samples, traces = tiny_eval
    cold = {"a": [make_trace(answer="0.25"), make_trace(answer="0.25")]}
    hot = {"a": [make_trace(answer="0.25"), make_trace(answer="错")]}

    m = report_mod.build_report(
        samples, traces, consistency_runs=cold, consistency_runs_hot=hot,
        consistency_temperature=0.7,
    )["metrics"]

    assert m["consistency"]["success_agreement"] == 1.0
    assert m["consistency_hot"]["success_agreement"] == pytest.approx(0.5)
    assert m["consistency_hot"]["temperature"] == 0.7


def test_hot_consistency_absent_when_not_run(tiny_eval):
    samples, traces = tiny_eval
    m = report_mod.build_report(samples, traces)["metrics"]
    assert m["consistency_hot"] is None


def test_anomalies_flag_unanswered_runs():
    """跑到 max_steps 却没给出答案，是已发现的失败模式，必须单独列出来，
    而不是混进均值里被稀释——M4 的 faithfulness 要据此排除这类样本。"""
    samples = [
        make_sample(id="ok", answer_keys=["x"]),
        make_sample(id="bait", answer_type="open", answer_keys=[], expected_doc_ids=[]),
    ]
    good = make_trace(answer="x", tools=["retrieve"])
    stuck = make_trace(answer="", tools=["retrieve"])
    stuck.stop_reason = "max_steps"
    traces = {"ok": good, "bait": stuck}

    report = report_mod.build_report(samples, traces)
    anomalies = {a["id"]: a for a in report["anomalies"]}

    assert "bait" in anomalies and "ok" not in anomalies
    assert anomalies["bait"]["stop_reason"] == "max_steps"
    assert anomalies["bait"]["empty_answer"] is True
    assert anomalies["bait"]["exclude_from_generation_metrics"] is True


def test_markdown_lists_known_failure_modes():
    samples = [make_sample(id="bait", answer_type="open", answer_keys=[], expected_doc_ids=[])]
    stuck = make_trace(answer="", tools=["retrieve"])
    stuck.stop_reason = "max_steps"
    md = report_mod.render_markdown(report_mod.build_report(samples, {"bait": stuck}))
    assert "已发现的失败模式" in md
    assert "bait" in md


# --------------------------------------------- 轨迹落盘与复用（M4 前置）
def test_traces_are_persisted_alongside_the_report(tiny_eval, tmp_path):
    """轨迹必须落盘：M4 的 RAGAS、M5 的裁判都要基于**同一批**轨迹算，
    否则每加一个指标就要重跑一次 Agent——既烧钱，指标之间也对不上。"""
    samples, traces = tiny_eval
    report = report_mod.build_report(samples, traces)
    paths = report_mod.write_report(report, directory=tmp_path, traces=traces)

    assert paths["traces"].exists()
    restored = report_mod.load_traces(paths["traces"])
    assert set(restored) == set(traces)
    assert restored["a"].answer == traces["a"].answer
    assert restored["a"].retrieved_doc_ids == traces["a"].retrieved_doc_ids


def test_persisted_traces_keep_chunk_texts(tiny_eval, tmp_path):
    """RAGAS 判 faithfulness 要的是 chunk 原文，不是 doc_id——落盘不能丢正文。"""
    samples, traces = tiny_eval
    paths = report_mod.write_report(
        report_mod.build_report(samples, traces), directory=tmp_path, traces=traces
    )
    restored = report_mod.load_traces(paths["traces"])
    assert restored["a"].retrieved_chunks
    assert all(c.text for c in restored["a"].retrieved_chunks)


def test_report_without_traces_still_writes(tiny_eval, tmp_path):
    samples, traces = tiny_eval
    paths = report_mod.write_report(report_mod.build_report(samples, traces), directory=tmp_path)
    assert "traces" not in paths


def test_ragas_section_is_optional(tiny_eval):
    samples, traces = tiny_eval
    assert report_mod.build_report(samples, traces)["metrics"]["ragas"] is None


def test_ragas_section_records_denominator(tiny_eval):
    samples, traces = tiny_eval
    ragas_result = {
        "judge_model": "deepseek-chat",
        "scores": {"faithfulness": 0.9},
        "n_evaluated": 2,
        "n_excluded": 1,
        "total": 3,
        "excluded": [{"id": "c", "reason": "最终答案为空"}],
    }
    report = report_mod.build_report(samples, traces, ragas=ragas_result)
    md = report_mod.render_markdown(report)
    assert report["metrics"]["ragas"]["n_evaluated"] == 2
    assert "n=2/3" in md
    assert "faithfulness" in md


def test_saving_a_subset_never_truncates_an_existing_trace_file(tiny_eval, tmp_path):
    """复用轨迹 + --limit 时，绝不能把完整的轨迹文件覆盖成子集。

    真踩过：`--from-traces ... --limit 5` 把 36 条的轨迹文件写成了 5 条，
    后续想补算全量指标就得重跑 Agent。轨迹是花钱买来的，不能被顺手覆盖。
    """
    samples, traces = tiny_eval
    full = dict(traces)
    full["zz"] = make_trace(answer="额外一条", tools=["retrieve"])
    path = tmp_path / report_mod.TRACES_FILENAME
    report_mod.save_traces(full, path)
    assert len(report_mod.load_traces(path)) == 4

    subset = {"a": traces["a"]}
    report_mod.save_traces(subset, path, merge=True)

    restored = report_mod.load_traces(path)
    assert set(restored) == {"a", "b", "c", "zz"}, "已有轨迹被截断了"


# ---------------------------------------- 一致性轨迹落盘（可续跑，M4 增补）
def test_consistency_runs_round_trip(tmp_path):
    """一致性那两档也要能落盘读回。

    起因：一次全量评测被中途掐断后，主评测轨迹还在、一致性的 100 次运行却白跑了。
    轨迹是真金白银，K 次重复更贵，必须存下来。
    """
    runs = {
        "a": [make_trace(answer="x", tools=["retrieve"]), make_trace(answer="y", tools=["calc"])],
        "b": [make_trace(answer="z", tools=["retrieve"], retrieved=[["d1"]])] * 3,
    }
    path = tmp_path / "cons.jsonl"
    report_mod.save_trace_runs(runs, path)

    restored = report_mod.load_trace_runs(path)
    assert set(restored) == {"a", "b"}
    assert [t.answer for t in restored["a"]] == ["x", "y"]
    assert len(restored["b"]) == 3
    assert restored["b"][0].retrieved_doc_ids == ["d1"]


def test_write_report_persists_both_consistency_passes(tiny_eval, tmp_path):
    samples, traces = tiny_eval
    cold = {"a": [make_trace(answer="0.25"), make_trace(answer="0.25")]}
    hot = {"a": [make_trace(answer="0.25"), make_trace(answer="错")]}

    paths = report_mod.write_report(
        report_mod.build_report(samples, traces, consistency_runs=cold, consistency_runs_hot=hot),
        directory=tmp_path,
        traces=traces,
        consistency_runs=cold,
        consistency_runs_hot=hot,
    )

    assert paths["consistency_cold"].exists()
    assert paths["consistency_hot"].exists()
    assert len(report_mod.load_trace_runs(paths["consistency_hot"])["a"]) == 2


def test_missing_consistency_file_loads_as_none(tmp_path):
    """没跑过一致性时不应报错，返回 None 让报告如实留空。"""
    assert report_mod.load_trace_runs_if_present(tmp_path / "nope.jsonl") is None


# ------------------------------------------- 裁判敏感性对照（M5 前置）
def test_ragas_baseline_renders_side_by_side(tiny_eval):
    """同一批轨迹、两个裁判，分数并列——差值本身就是"换裁判影响多大"的证据。"""
    samples, traces = tiny_eval
    current = {
        "judge_model": "deepseek-reasoner",
        "scores": {"faithfulness": 0.80, "context_precision": 0.70},
        "scored_counts": {"faithfulness": 3, "context_precision": 3},
        "n_submitted": 3, "n_excluded": 0, "total": 3, "excluded": [],
        "has_incomplete_metric": False,
    }
    baseline = {
        "judge_model": "deepseek-chat",
        "scores": {"faithfulness": 0.86, "context_precision": 0.66},
        # 两侧打分行数一致，差值才可比——不一致时表格会标「不可比」
        "scored_counts": {"faithfulness": 3, "context_precision": 3},
        "n_submitted": 3,
    }
    report = report_mod.build_report(samples, traces, ragas=current, ragas_baseline=baseline)
    md = report_mod.render_markdown(report)

    assert report["metrics"]["ragas_baseline"]["judge_model"] == "deepseek-chat"
    assert "裁判敏感性" in md
    assert "deepseek-reasoner" in md and "deepseek-chat" in md
    assert "-0.060" in md or "−0.060" in md, "必须把差值算出来，不能让人自己减"


def test_ragas_baseline_is_optional(tiny_eval):
    samples, traces = tiny_eval
    assert report_mod.build_report(samples, traces)["metrics"]["ragas_baseline"] is None


def test_extract_ragas_block_from_previous_report(tmp_path):
    """从上一份报告里取出 RAGAS 段，用作对照基线。"""
    import json as _json

    path = tmp_path / "prev.json"
    path.write_text(
        _json.dumps({"metrics": {"ragas": {"judge_model": "deepseek-chat",
                                           "scores": {"faithfulness": 0.86}}}}),
        encoding="utf-8",
    )
    block = report_mod.load_ragas_baseline(path)
    assert block["judge_model"] == "deepseek-chat"
    assert block["scores"]["faithfulness"] == 0.86


def test_missing_previous_report_returns_none(tmp_path):
    assert report_mod.load_ragas_baseline(tmp_path / "nope.json") is None


def test_baseline_comparison_flags_non_comparable_rows(tiny_eval):
    """两次运行打分行数不同的指标，差值没有可比性，必须标出来。

    真踩过：换裁判后 faithfulness 因超时只打了 31/35 行，而基线是 35/35，
    差值 -0.000 看着像"换裁判没影响"，其实是在比两个不同的子集。
    """
    samples, traces = tiny_eval
    current = {
        "judge_model": "deepseek-reasoner",
        "scores": {"faithfulness": 0.86, "answer_relevancy": 0.82},
        "scored_counts": {"faithfulness": 31, "answer_relevancy": 35},
        "n_submitted": 35, "n_excluded": 0, "total": 36, "excluded": [],
        "has_incomplete_metric": True,
    }
    baseline = {
        "judge_model": "deepseek-chat",
        "scores": {"faithfulness": 0.86, "answer_relevancy": 0.84},
        "scored_counts": {"faithfulness": 35, "answer_relevancy": 35},
    }
    md = report_mod.render_markdown(
        report_mod.build_report(samples, traces, ragas=current, ragas_baseline=baseline)
    )
    assert "不可比" in md
    assert "31/35" in md and "35/35" in md


def test_baseline_extraction_keeps_scored_counts(tmp_path):
    import json as _json

    path = tmp_path / "prev.json"
    path.write_text(
        _json.dumps({"metrics": {"ragas": {
            "judge_model": "deepseek-chat",
            "scores": {"faithfulness": 0.86},
            "scored_counts": {"faithfulness": 35},
        }}}),
        encoding="utf-8",
    )
    block = report_mod.load_ragas_baseline(path)
    assert block["scored_counts"]["faithfulness"] == 35


# ===========================================================================
# M5：一致率进报告 + RAGAS 段复用
# ===========================================================================


def test_agreement_section_lands_in_markdown():
    """kappa 算出来了就得进报告，不能只活在终端里。"""
    from src.eval import judge

    samples = [make_sample(id="q1")]
    traces = {"q1": make_trace(answer="a", tools=["retrieve"], retrieved=[["d1"]])}
    ids = [f"q{i}" for i in range(10)]
    stats = judge.agreement_stats(
        dict(zip(ids, [0, 1, 1, 1, 1, 2, 2, 2, 2, 2])),
        dict(zip(ids, [0, 1, 2, 1, 1, 2, 2, 1, 2, 2])),
        bootstrap=50,
    )
    rep = report_mod.build_report(samples, traces, agreement=stats)
    assert rep["metrics"]["agreement"]["n"] == 10

    text = report_mod.render_markdown(rep)
    assert "自动↔人工一致率" in text
    assert "n=10" in text
    assert "混淆矩阵" in text


def test_reused_ragas_is_marked_as_reused(tmp_path):
    """复用上一次的 RAGAS 分数可以，但必须留痕。

    不标注就等于把 40 分钟前算的数字当成本次结果报出去——
    读报告的人无从分辨，那是变相编造。
    """
    prior = tmp_path / "prev.json"
    prior.write_text(
        json.dumps(
            {
                "meta": {"timestamp_utc": "2026-09-05T14:25:07+00:00"},
                "metrics": {
                    "ragas": {
                        "judge_model": "deepseek-reasoner",
                        "scores": {"faithfulness": 0.9},
                        "n_submitted": 35,
                        "total": 36,
                        "scored_counts": {"faithfulness": 35},
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    block = report_mod.load_ragas_reuse(prior)
    assert block["reused_from"] == str(prior)
    assert block["reused_from_timestamp"] == "2026-09-05T14:25:07+00:00"

    rep = report_mod.build_report([make_sample(id="q1")], {}, ragas=block)
    text = report_mod.render_markdown(rep)
    assert "复用" in text and str(prior) in text


def test_reuse_ragas_returns_none_when_absent(tmp_path):
    empty = tmp_path / "no_ragas.json"
    empty.write_text(json.dumps({"meta": {}, "metrics": {}}), encoding="utf-8")
    assert report_mod.load_ragas_reuse(empty) is None


def test_reuse_marker_points_at_the_stable_snapshot(tmp_path):
    """复用标记必须指向当初算出这批数的那次**快照**，不是 latest.json。

    latest.json 每跑一次就被覆盖：标记指过去，下一次运行后它就指向了自己，
    读报告的人无从追溯这批数究竟出自哪次运行。
    """
    stamp = "2026-09-05T14:25:07+00:00"
    payload = {
        "meta": {"timestamp_utc": stamp},
        "metrics": {"ragas": {"judge_model": "deepseek-reasoner", "scores": {"faithfulness": 0.9}}},
    }
    snapshot = tmp_path / report_mod.snapshot_filename(stamp)
    snapshot.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    latest = tmp_path / "latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    block = report_mod.load_ragas_reuse(latest)
    assert block["reused_from"] == str(snapshot)
    assert "latest.json" not in block["reused_from"]


def test_reuse_marker_falls_back_when_snapshot_missing(tmp_path):
    """快照不在（比如被清理过）就如实记录给定路径，不编一个不存在的文件名。"""
    latest = tmp_path / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "meta": {"timestamp_utc": "2026-09-05T14:25:07+00:00"},
                "metrics": {"ragas": {"scores": {"faithfulness": 0.9}}},
            }
        ),
        encoding="utf-8",
    )
    block = report_mod.load_ragas_reuse(latest)
    assert block["reused_from"] == str(latest)


def test_reuse_does_not_nest_markers(tmp_path):
    """从一份「已经是复用」的报告再复用，标记不该套娃。"""
    src = tmp_path / "latest.json"
    src.write_text(
        json.dumps(
            {
                "meta": {"timestamp_utc": "2026-09-06T00:00:00+00:00"},
                "metrics": {
                    "ragas": {
                        "scores": {"faithfulness": 0.9},
                        "reused_from": "reports/old.json",
                        "reused_from_timestamp": "2026-09-01T00:00:00+00:00",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    block = report_mod.load_ragas_reuse(src)
    assert block["reused_from_timestamp"] == "2026-09-06T00:00:00+00:00"
    assert "old.json" not in block["reused_from"]


def test_snapshot_filename_matches_what_write_report_uses(tmp_path):
    rep = report_mod.build_report([make_sample(id="q1")], {})
    paths = report_mod.write_report(rep, directory=tmp_path)
    assert paths["snapshot"].name == report_mod.snapshot_filename(rep["meta"]["timestamp_utc"])
