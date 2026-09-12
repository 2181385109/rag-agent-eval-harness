"""对照重跑报告（stability/rerun_report.py）的验收口径。

输入是两份主评测快照（JSON）；全部离线。约定：

  - 头部指标表：value/n/total 逐项并列；n 或 total 任一不同即标「分母不同」，差值照列不隐藏。
  - RAGAS 交集口径：只有**两份快照都存了逐题 RAGAS 分数**才重算；缺任一侧就写「无法重算」，
    不估算、不用全量均值近似。交集口径与原始口径并列，不替换。
  - faithfulness 分母耦合：列出「本次因无检索内容被排除、但基线进入了分母」的题，
    以及这些题的 tool_ok 变化；结论句是固定措辞的事实陈述。
  - 检索次数 vs recall：逐题列检索次数与 recall 的变化；基线原始次数只能来自基线轨迹文件，
    没有就写「未记录」，不用折叠序列去猜。
  - 独立性范围：被测 / 裁判 / RAGAS 三条链路的 (响应 model, 指纹) 只有一种取值时，
    固定措辞指出 RAGAS 四项指标同样无法确认独立性；任一链路未记录则不下这个结论。
"""

from __future__ import annotations

from stability import rerun_report as rr


def _metric(value, n, total, detail=None):
    return {"value": value, "n": n, "total": total, "detail": detail or {}}


def _snapshot(*, success, tool_ok, recall, retrieve_counts, ragas_scores, ragas_counts, evaluated, excluded,
              served=None, ragas_per_question=None, first_docs=None, retrieved=None, expected=None):
    ids = sorted(success)
    per_q = []
    for sid in ids:
        q = {
            "id": sid,
            "answer_type": "closed",
            "expected_tool": ["retrieve"],
            "actual_tool": ["retrieve"] if retrieve_counts.get(sid, 0) else ["calc"],
            "tool_ok": tool_ok[sid],
            "expected_doc_ids": (expected or {}).get(sid, [[f"doc_{sid}"]]),
            "retrieved_doc_ids": (retrieved or {}).get(sid, [f"doc_{sid}"] if recall.get(sid) else ["other"]),
            "recall_at_k": recall.get(sid),
            "success": success[sid],
        }
        if sid in retrieve_counts and retrieve_counts[sid] is not None:
            q["n_retrieve_calls"] = retrieve_counts[sid]
        if first_docs and sid in first_docs:
            q["first_retrieve_doc_ids"] = first_docs[sid]
        per_q.append(q)
    n_recall = sum(1 for v in recall.values() if v is not None)
    ragas = {
        "judge_model": "deepseek-reasoner",
        "scores": ragas_scores,
        "scored_counts": ragas_counts,
        "n_submitted": len(evaluated),
        "evaluated_ids": list(evaluated),
        "excluded": excluded,
        "total": len(ids),
    }
    if ragas_per_question is not None:
        ragas["per_question"] = ragas_per_question
    return {
        "meta": {"timestamp_utc": "2026-09-12T00:00:00+00:00", "git_commit": "abc1234", "model": "deepseek-chat",
                 "served": served or {}},
        "metrics": {
            "task_success_rate": _metric(sum(success.values()) / len(ids), len(ids), len(ids), success),
            "tool_accuracy": _metric(sum(tool_ok.values()) / len(ids), len(ids), len(ids), tool_ok),
            "recall_at_k": _metric(
                (sum(v for v in recall.values() if v is not None) / n_recall) if n_recall else None,
                n_recall, len(ids), recall),
            "recall_at_k_first_call": _metric(0.5, n_recall, len(ids), {}),
            "ragas": ragas,
        },
        "per_question": per_q,
    }


def _pair():
    """基线 4 题全进 RAGAS；本次 q3 检索 0 次被排除、tool_ok 由 False 变 True。"""
    base = _snapshot(
        success={"q1": True, "q2": True, "q3": True, "q4": True},
        tool_ok={"q1": True, "q2": True, "q3": False, "q4": True},
        recall={"q1": 1.0, "q2": 1.0, "q3": None, "q4": 1.0},
        retrieve_counts={"q1": None, "q2": None, "q3": None, "q4": None},  # 基线快照没有原始次数
        ragas_scores={"faithfulness": 0.8, "answer_relevancy": 0.7, "context_recall": 0.6, "context_precision": 0.5},
        ragas_counts={"faithfulness": 4, "answer_relevancy": 4, "context_recall": 4, "context_precision": 4},
        evaluated=["q1", "q2", "q3", "q4"],
        excluded=[],
    )
    new = _snapshot(
        success={"q1": True, "q2": False, "q3": True, "q4": True},
        tool_ok={"q1": True, "q2": True, "q3": True, "q4": True},
        recall={"q1": 1.0, "q2": 0.0, "q3": None, "q4": 1.0},
        retrieve_counts={"q1": 2, "q2": 1, "q3": 0, "q4": 2},
        ragas_scores={"faithfulness": 0.9, "answer_relevancy": 0.6, "context_recall": 0.7, "context_precision": 0.4},
        ragas_counts={"faithfulness": 3, "answer_relevancy": 3, "context_recall": 3, "context_precision": 3},
        evaluated=["q1", "q2", "q4"],
        excluded=[{"id": "q3", "reason": "本轮没有检索内容（如纯算术题），faithfulness / context_recall 无从谈起"}],
        served={
            "agent": {"requested_model": "deepseek-chat", "n_calls": 8,
                      "response_model_counts": {"deepseek-flash": 8}, "system_fingerprint_counts": {"fp1": 8}},
            "judge": {"requested_model": "deepseek-reasoner", "n_calls": 2,
                      "response_model_counts": {"deepseek-flash": 2}, "system_fingerprint_counts": {"fp1": 2}},
            "ragas": {"status": "recorded", "requested_model": "deepseek-reasoner", "n_calls": 40,
                      "response_model_counts": {"deepseek-flash": 40}, "system_fingerprint_counts": {"fp1": 40}},
        },
        first_docs={"q2": ["other"]},
    )
    return new, base


# ------------------------------------------------------------------ 头部指标
def test_headline_flags_denominator_change_and_keeps_diff():
    new, base = _pair()
    rows = {r["metric"]: r for r in rr.headline_rows(new, base)}
    f = rows["faithfulness"]
    assert (f["base_n"], f["new_n"]) == (4, 3)
    assert f["comparable"] is False and f["diff"] is not None  # 差值照列，只是标出分母不同
    s = rows["task_success_rate"]
    assert s["comparable"] is True and abs(s["diff"] - (-0.25)) < 1e-9


# ------------------------------------------------------------------ RAGAS 交集口径
def test_ragas_intersection_is_unavailable_without_per_question_scores():
    new, base = _pair()
    sec = rr.ragas_intersection(new, base)
    assert sec["n_common"] == 3
    assert sec["status"] == rr.RAGAS_INTERSECTION_UNAVAILABLE
    assert sec["per_question_available"] == {"base": False, "new": False}
    assert "scores" not in sec  # 不估算


def test_ragas_intersection_recomputes_only_when_both_sides_have_per_question():
    new, base = _pair()
    base["metrics"]["ragas"]["per_question"] = {
        "q1": {"faithfulness": 1.0}, "q2": {"faithfulness": 0.5}, "q3": {"faithfulness": 0.0}, "q4": {"faithfulness": 1.0}}
    new["metrics"]["ragas"]["per_question"] = {
        "q1": {"faithfulness": 0.5}, "q2": {"faithfulness": 0.5}, "q4": {"faithfulness": 1.0}}
    sec = rr.ragas_intersection(new, base)
    assert sec["status"] == "recomputed"
    assert sec["common_ids"] == ["q1", "q2", "q4"]
    assert abs(sec["scores"]["faithfulness"]["base"] - (2.5 / 3)) < 1e-9
    assert abs(sec["scores"]["faithfulness"]["new"] - (2.0 / 3)) < 1e-9
    # 原始口径原样保留在旁边
    assert sec["original"]["base"]["faithfulness"] == {"value": 0.8, "n": 4}
    assert sec["original"]["new"]["faithfulness"] == {"value": 0.9, "n": 3}


# ------------------------------------------------------------------ 分母耦合
def test_faithfulness_denominator_coupling_lists_ids_and_tool_ok_flip():
    new, base = _pair()
    sec = rr.faithfulness_denominator_coupling(new, base)
    assert sec["ids"] == ["q3"]
    assert sec["rows"][0] == {"id": "q3", "tool_ok_base": False, "tool_ok_new": True,
                              "n_retrieve_new": 0, "n_retrieve_base": None, "excluded_reason": new["metrics"]["ragas"]["excluded"][0]["reason"]}
    assert (sec["faithfulness_n_base"], sec["faithfulness_n_new"]) == (4, 3)
    assert abs(sec["tool_accuracy_base"] - 0.75) < 1e-9 and sec["tool_accuracy_new"] == 1.0
    assert sec["statement"] == rr.FAITHFULNESS_DENOMINATOR_COUPLING


def test_faithfulness_denominator_coupling_empty_when_no_new_exclusions():
    new, base = _pair()
    new["metrics"]["ragas"]["excluded"] = []
    sec = rr.faithfulness_denominator_coupling(new, base)
    assert sec["ids"] == [] and sec["statement"] is None


# ------------------------------------------------------------------ 检索次数 vs recall
def test_retrieval_vs_recall_marks_base_counts_unrecorded_without_traces():
    new, base = _pair()
    sec = rr.retrieval_vs_recall(new, base, base_traces=None)
    assert sec["median_base"] is None and sec["base_counts_source"] == rr.UNRECORDED
    assert sec["median_new"] == 1.5
    ids = [r["id"] for r in sec["rows"]]
    assert ids == ["q2"]  # 只有 q2 的 recall 变了
    assert sec["rows"][0]["n_retrieve_base"] is None and sec["rows"][0]["n_retrieve_new"] == 1


def test_retrieval_vs_recall_uses_base_traces_when_given():
    new, base = _pair()

    class _T:
        def __init__(self, seq, first):
            self.tool_sequence = seq
            self._first = first
            self.steps = []

    traces = {"q1": _T(["retrieve"], []), "q2": _T(["retrieve", "retrieve", "retrieve"], []),
              "q3": _T(["retrieve"], []), "q4": _T(["retrieve"], [])}
    sec = rr.retrieval_vs_recall(new, base, base_traces=traces)
    assert sec["median_base"] == 1.0 and sec["base_counts_source"] != rr.UNRECORDED
    row = sec["rows"][0]
    assert row["id"] == "q2" and row["n_retrieve_base"] == 3 and row["n_retrieve_new"] == 1
    assert row["recall_base"] == 1.0 and row["recall_new"] == 0.0
    assert sec["direction_counts"]["down"] >= 1


# ------------------------------------------------------------------ 独立性范围
def test_independence_scope_extends_to_ragas_when_all_chains_share_one_backend():
    new, _ = _pair()
    sec = rr.independence_scope(new)
    assert sec["single_backend"] is True
    assert sec["pairs"] == [["deepseek-flash", "fp1"]]
    assert sec["statement"] == rr.INDEPENDENCE_SCOPE_EXTENDED


def test_independence_scope_does_not_conclude_when_a_chain_is_unrecorded():
    new, _ = _pair()
    new["meta"]["served"]["ragas"] = {"status": rr.UNRECORDED, "requested_model": "deepseek-reasoner", "n_calls": 0}
    sec = rr.independence_scope(new)
    assert sec["single_backend"] is None and sec["statement"] is None
    assert "ragas" in sec["unrecorded_chains"]


# ------------------------------------------------------------------ 渲染
def test_markdown_shows_both_ragas_denominations_side_by_side_and_fixed_facts():
    new, base = _pair()
    md = rr.render_markdown(rr.build(new, base, base_traces=None, new_name="eval_x.json", base_name="eval_y.json"))
    assert "原始口径" in md and "交集口径 n=3" in md
    assert rr.RAGAS_INTERSECTION_UNAVAILABLE in md
    assert rr.FAITHFULNESS_DENOMINATOR_COUPLING in md
    assert rr.INDEPENDENCE_SCOPE_EXTENDED in md
    assert rr.UNRECORDED in md  # 基线原始检索次数没轨迹就标未记录
    # 头部并列打印响应模型名（按次计数），基线没记录就写未记录
    assert "| 响应模型名（被测 / 裁判 / RAGAS，按次计数） | 未记录 | `deepseek-flash`×8 / `deepseek-flash`×2 / `deepseek-flash`×40（详见「事实 3」） |" in md
    for bad in ("造假", "变笨", "静默换模型", "厂商未告知"):
        assert bad not in md


# ------------------------------------------------------------------ 基线轨迹必须与基线快照配对
class _Trace:
    def __init__(self, answer, doc_ids, seq=("retrieve",)):
        self.answer = answer
        self.retrieved_doc_ids = list(doc_ids)
        self.tool_sequence = list(seq)
        self.steps = []


def test_traces_match_snapshot_reports_mismatched_ids():
    new, base = _pair()
    bq = {q["id"]: q for q in base["per_question"]}
    for q in base["per_question"]:
        q["answer"] = f"答 {q['id']}"
    good = {sid: _Trace(f"答 {sid}", bq[sid]["retrieved_doc_ids"]) for sid in bq}
    assert rr.traces_match_snapshot(good, base) == {"n_checked": 4, "mismatched": [], "missing": []}
    bad = dict(good)
    bad["q2"] = _Trace("别的答案", bq["q2"]["retrieved_doc_ids"])
    del bad["q4"]
    assert rr.traces_match_snapshot(bad, base) == {"n_checked": 3, "mismatched": ["q2"], "missing": ["q4"]}


def test_build_refuses_baseline_traces_that_do_not_match_the_baseline_snapshot():
    """traces_latest.jsonl 被别的运行覆盖后，默认路径指向的就不再是基线的轨迹——
    对不上就按「未记录」处理并写明原因，不能拿另一批运行的检索次数冒充基线。"""
    new, base = _pair()
    for q in base["per_question"]:
        q["answer"] = f"答 {q['id']}"
    bq = {q["id"]: q for q in base["per_question"]}
    wrong = {sid: _Trace("另一次运行的答案", bq[sid]["retrieved_doc_ids"], seq=("retrieve", "retrieve")) for sid in bq}
    s = rr.build(new, base, base_traces=wrong, new_name="eval_x.json", base_name="eval_y.json")
    assert s["retrieval_vs_recall"]["base_counts_source"] == rr.UNRECORDED
    assert s["retrieval_vs_recall"]["median_base"] is None
    assert s["base_traces_check"]["used"] is False and s["base_traces_check"]["mismatched"]
    md = rr.render_markdown(s)
    assert "与基线快照不一致" in md and rr.UNRECORDED in md


def test_default_baseline_traces_path_is_tied_to_the_baseline_snapshot_name(tmp_path):
    assert rr.default_baseline_traces(tmp_path / "eval_20260906T092835Z.json") == tmp_path / "traces_20260906T092835Z.jsonl"
    assert rr.default_baseline_traces(tmp_path / "eval_20260912T074455Z_full.json") == tmp_path / "traces_20260912T074455Z_full.jsonl"
