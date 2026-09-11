"""稳定性 / 延迟分解指标的验收口径（PERF_SPEC §3 B2）。

这里是定义，stability/analyze.py 是实现。口径要改先改这里。

全部离线：输入是构造出来的运行记录，不碰 DeepSeek API、不加载 BGE
（答案相似度用确定性假 embedder）。

约定的口径：
  - 延迟分位数用 numpy.percentile 的默认线性插值；只对 error 为空的行算，n 如实上报。
  - 三段占比 = 该段耗时 / 端到端耗时，按行算后取**中位数**。
  - 轨迹自洽率：k 次运行的工具序列**逐项完全一致**的题占比（不折叠连续重复；
    折叠版另列一栏，只作参考）。
  - 判定自洽率：k 次运行的任务成功判定全部相同的题占比。闭合题走规则；
    开放题只有给了逐次裁判分才参与，否则**不计入分母**（不许当 0 或当 1）。
  - 答案相似度：同题 k 个答案两两余弦相似度，题内取均值与最小值；
    全局报"题内均值的均值"与"题内最小值的最小值"。
  - 成功率跨次标准差：第 r 次 pass 的成功率 = 该 pass 里有判定的题的成功比例，
    得到 k 个成功率后取**样本标准差**（n-1）；总体标准差并列上报。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.eval.datasets import GoldenSample
from stability import analyze
from stability.records import RunRecord


# ------------------------------------------------------------------ 构造工具
def _sample(sid: str, answer_type: str = "closed", keys=None) -> GoldenSample:
    return GoldenSample(
        id=sid,
        question=f"问题 {sid}",
        reference_answer="参考",
        answer_keys=list(keys or (["对"] if answer_type == "closed" else [])),
        expected_tool=["retrieve"],
        expected_doc_ids=[],
        answer_type=answer_type,
    )


def _row(
    sid: str,
    run_index: int,
    *,
    answer: str = "答案：对",
    tools=("retrieve",),
    total_s: float = 10.0,
    llm_s: float = 8.0,
    retrieve_s: float = 1.0,
    tool_s: float = 0.5,
    tokens: int = 1000,
    error: str | None = None,
    k: int = 3,
) -> RunRecord:
    return RunRecord(
        run_id="run_test",
        sample_id=sid,
        run_index=run_index,
        k=k,
        timestamp_utc="2026-09-12T00:00:00+00:00",
        answer_type="closed",
        question=f"问题 {sid}",
        answer=answer,
        tool_sequence=list(tools),
        retrieved_doc_ids=[],
        stop_reason="error" if error else "answered",
        n_steps=len(tools) + 1,
        latency={
            "total_s": total_s,
            "llm_s": llm_s,
            "retrieve_s": retrieve_s,
            "tool_s": tool_s,
            "n_llm_calls": len(tools) + 1,
            "n_retrieve_calls": sum(1 for t in tools if t == "retrieve"),
            "n_tool_calls": sum(1 for t in tools if t != "retrieve"),
            "llm_call_s": [llm_s / (len(tools) + 1)] * (len(tools) + 1),
        },
        tokens={"prompt": tokens - 100, "completion": 100, "total": tokens},
        error=error,
    )


class FakeEmbedder:
    """确定性假 embedder：同文本同向量；A 与 B 正交，A 与 A2 很接近。"""

    def encode(self, texts, is_query: bool = False):
        table = {
            "A": [1.0, 0.0, 0.0],
            "A2": [0.98, 0.2, 0.0],
            "B": [0.0, 1.0, 0.0],
        }
        out = np.array([table.get(t, [0.0, 0.0, 1.0]) for t in texts], dtype="float32")
        return out / np.linalg.norm(out, axis=1, keepdims=True)


# ------------------------------------------------------------------ 记录 schema
def test_record_roundtrips_through_jsonl(tmp_path):
    row = _row("cap_001", 0)
    path = tmp_path / "run_x.jsonl"
    path.write_text(row.model_dump_json() + "\n", encoding="utf-8")
    loaded = analyze.load_rows(path)
    assert loaded == [row]


def test_record_derives_collapsed_sequence_and_overhead():
    row = _row(
        "cap_001", 0, tools=("retrieve", "retrieve", "calc"),
        total_s=10, llm_s=8, retrieve_s=1, tool_s=0.5,
    )
    assert row.tool_sequence_collapsed == ["retrieve", "calc"]
    assert row.latency.overhead_s == pytest.approx(0.5)
    assert row.answer_sha1  # 指纹必须存在：裁判分回填靠它配对


# ------------------------------------------------------------------ 延迟
def test_latency_percentiles_use_only_ok_rows_and_report_n():
    rows = [_row("q1", i, total_s=t) for i, t in enumerate([1.0, 2.0, 3.0, 4.0, 5.0])]
    rows.append(_row("q1", 5, total_s=100.0, error="boom"))  # 报错行不进分位数
    lat = analyze.latency_section(rows)
    assert lat["e2e"]["n"] == 5
    assert lat["e2e"]["p50"] == pytest.approx(3.0)
    assert lat["e2e"]["p95"] == pytest.approx(np.percentile([1, 2, 3, 4, 5], 95))
    assert lat["e2e"]["p99"] == pytest.approx(np.percentile([1, 2, 3, 4, 5], 99))
    assert lat["n_error"] == 1
    assert lat["error_rate"] == pytest.approx(1 / 6)


def test_phase_share_is_median_of_per_row_shares():
    rows = [
        _row("q1", 0, total_s=10, llm_s=8, retrieve_s=1, tool_s=0.5),  # llm 0.8
        _row("q1", 1, total_s=10, llm_s=6, retrieve_s=3, tool_s=0.5),  # llm 0.6
        _row("q1", 2, total_s=10, llm_s=9, retrieve_s=0.5, tool_s=0.2),  # llm 0.9
    ]
    lat = analyze.latency_section(rows)
    assert lat["share_median"]["llm"] == pytest.approx(0.8)
    assert lat["share_median"]["retrieve"] == pytest.approx(0.1)
    assert lat["share_median"]["tool"] == pytest.approx(0.05)
    assert 0 < lat["share_median"]["overhead"] < 1
    assert lat["phases"]["llm"]["p50"] == pytest.approx(8.0)


def test_latency_section_with_no_ok_rows_is_none_not_zero():
    rows = [_row("q1", 0, error="x"), _row("q1", 1, error="y")]
    lat = analyze.latency_section(rows)
    assert lat["e2e"]["n"] == 0
    assert lat["e2e"]["p50"] is None
    assert lat["error_rate"] == 1.0


# ------------------------------------------------------------------ 轨迹自洽率
def test_trajectory_consistency_is_exact_sequence_identity():
    samples = [_sample("q1"), _sample("q2"), _sample("q3")]
    rows = [
        # q1：三次完全一致
        _row("q1", 0, tools=("retrieve",)),
        _row("q1", 1, tools=("retrieve",)),
        _row("q1", 2, tools=("retrieve",)),
        # q2：一次多调了 retrieve —— 严格口径不一致，折叠后一致
        _row("q2", 0, tools=("retrieve",)),
        _row("q2", 1, tools=("retrieve", "retrieve")),
        _row("q2", 2, tools=("retrieve",)),
        # q3：真的换了路径
        _row("q3", 0, tools=("retrieve",)),
        _row("q3", 1, tools=("calc",)),
        _row("q3", 2, tools=("retrieve",)),
    ]
    st = analyze.stability_section(rows, samples)
    traj = st["trajectory_consistency"]
    assert traj["value"] == pytest.approx(1 / 3)
    assert traj["collapsed_value"] == pytest.approx(2 / 3)
    assert traj["n"] == 3 and traj["total"] == 3
    assert traj["inconsistent_ids"] == ["q2", "q3"]


def test_trajectory_consistency_skips_questions_with_fewer_than_two_runs():
    samples = [_sample("q1"), _sample("q2")]
    rows = [_row("q1", 0), _row("q1", 1), _row("q2", 0)]
    st = analyze.stability_section(rows, samples)
    assert st["trajectory_consistency"]["n"] == 1
    assert st["trajectory_consistency"]["total"] == 2


# ------------------------------------------------------------------ 判定自洽率
def test_verdict_consistency_closed_uses_rule_and_open_excluded_without_judge():
    samples = [_sample("q1", keys=["对"]), _sample("q2", keys=["对"]), _sample("q3", "open")]
    rows = [
        _row("q1", 0, answer="对"), _row("q1", 1, answer="对"), _row("q1", 2, answer="对"),
        _row("q2", 0, answer="对"), _row("q2", 1, answer="错"), _row("q2", 2, answer="对"),
        _row("q3", 0, answer="x"), _row("q3", 1, answer="y"), _row("q3", 2, answer="z"),
    ]
    st = analyze.stability_section(rows, samples)
    v = st["verdict_consistency"]
    assert v["value"] == pytest.approx(0.5)
    assert v["n"] == 2 and v["total"] == 3  # 开放题没裁判分：不计入分母
    assert v["unjudged_ids"] == ["q3"]
    assert v["inconsistent_ids"] == ["q2"]


def test_verdict_consistency_open_uses_per_run_judge_scores_matched_by_fingerprint():
    samples = [_sample("q3", "open")]
    rows = [_row("q3", 0, answer="x"), _row("q3", 1, answer="y"), _row("q3", 2, answer="z")]
    judge_rows = [
        {
            "sample_id": "q3",
            "run_index": i,
            "answer_sha1": rows[i].answer_sha1,
            "judge_score": s,
        }
        for i, s in enumerate([2, 2, 1])
    ]
    st = analyze.stability_section(rows, samples, judge_rows=judge_rows)
    v = st["verdict_consistency"]
    assert v["n"] == 1 and v["value"] == 0.0  # 2,2,1 -> 成功,成功,失败 -> 不一致
    assert v["source_counts"] == {"rule": 0, "judge": 1}


def test_judge_score_with_stale_fingerprint_is_discarded_not_used():
    samples = [_sample("q3", "open")]
    rows = [_row("q3", 0, answer="x"), _row("q3", 1, answer="y")]
    judge_rows = [
        {"sample_id": "q3", "run_index": 0, "answer_sha1": rows[0].answer_sha1, "judge_score": 2},
        {"sample_id": "q3", "run_index": 1, "answer_sha1": "deadbeefdead", "judge_score": 2},
    ]
    st = analyze.stability_section(rows, samples, judge_rows=judge_rows)
    assert st["verdict_consistency"]["n"] == 0
    assert st["verdict_consistency"]["stale_judge_rows"] == [{"sample_id": "q3", "run_index": 1}]


# ------------------------------------------------------------------ 成功率跨次标准差
def test_success_rate_per_pass_std_is_sample_std_over_pass_rates():
    samples = [_sample("q1", keys=["对"]), _sample("q2", keys=["对"])]
    rows = [
        _row("q1", 0, answer="对"), _row("q2", 0, answer="对"),  # pass0: 1.0
        _row("q1", 1, answer="对"), _row("q2", 1, answer="错"),  # pass1: 0.5
        _row("q1", 2, answer="对"), _row("q2", 2, answer="对"),  # pass2: 1.0
    ]
    st = analyze.stability_section(rows, samples)
    sp = st["success_rate_per_pass"]
    assert sp["rates"] == [1.0, 0.5, 1.0]
    assert sp["std_sample"] == pytest.approx(np.std([1.0, 0.5, 1.0], ddof=1))
    assert sp["std_population"] == pytest.approx(np.std([1.0, 0.5, 1.0], ddof=0))
    assert sp["n_per_pass"] == [2, 2, 2]


def test_success_rate_per_pass_std_is_none_with_single_pass():
    samples = [_sample("q1", keys=["对"])]
    rows = [_row("q1", 0, answer="对", k=1)]
    st = analyze.stability_section(rows, samples)
    assert st["success_rate_per_pass"]["std_sample"] is None


# ------------------------------------------------------------------ 答案相似度
def test_answer_similarity_reports_mean_of_means_and_min_of_mins():
    samples = [_sample("q1"), _sample("q2")]
    rows = [
        _row("q1", 0, answer="A"), _row("q1", 1, answer="A"), _row("q1", 2, answer="A2"),
        _row("q2", 0, answer="A"), _row("q2", 1, answer="B"),
    ]
    sim = analyze.answer_similarity_section(rows, samples, FakeEmbedder())
    v = np.array([0.98, 0.2, 0.0])
    a_a2 = float(np.dot([1, 0, 0], v / np.linalg.norm(v)))
    # q1：三对 (A,A)=1, (A,A2), (A,A2) -> 均值 (1+2*a_a2)/3，最小 a_a2；q2：一对 (A,B)=0
    assert sim["per_question"]["q1"]["mean"] == pytest.approx((1 + 2 * a_a2) / 3)
    assert sim["per_question"]["q1"]["min"] == pytest.approx(a_a2)
    assert sim["per_question"]["q2"]["min"] == pytest.approx(0.0, abs=1e-6)
    assert sim["mean"] == pytest.approx(((1 + 2 * a_a2) / 3 + 0.0) / 2)
    assert sim["min"] == pytest.approx(0.0, abs=1e-6)
    assert sim["min_id"] == "q2"
    assert sim["n"] == 2


def test_answer_similarity_excludes_error_rows():
    samples = [_sample("q1")]
    rows = [_row("q1", 0, answer="A"), _row("q1", 1, answer="[运行失败] x", error="x")]
    sim = analyze.answer_similarity_section(rows, samples, FakeEmbedder())
    assert sim["n"] == 0 and sim["mean"] is None


# ------------------------------------------------------------------ 成本
def test_cost_section_reports_per_run_mean_and_full_round_total():
    samples = [_sample("q1"), _sample("q2")]
    rows = [
        _row("q1", 0, tokens=1000), _row("q1", 1, tokens=3000),
        _row("q2", 0, tokens=2000), _row("q2", 1, tokens=2000),
    ]
    cost = analyze.cost_section(rows, samples)
    assert cost["total_tokens_all_runs"] == 8000
    assert cost["mean_tokens_per_run"] == pytest.approx(2000)
    # 一轮完整评测 = 每题取 k 次均值再求和 = 2000 + 2000
    assert cost["tokens_per_full_round"] == pytest.approx(4000)
    assert cost["n_runs"] == 4 and cost["n_questions"] == 2


# ------------------------------------------------------------------ 汇总 + 报告
def test_summary_carries_denominators_k_and_env_next_to_every_number():
    samples = [_sample("q1", keys=["对"])]
    rows = [_row("q1", i, answer="对") for i in range(3)]
    meta = {
        "run_id": "run_test", "k": 3, "model": "deepseek-chat", "temperature": 0.0,
        "env": {"cpu": "x", "python": "3.12"},
    }
    summary = analyze.build_summary(rows, samples, run_meta=meta, embedder=None)
    assert summary["meta"]["k"] == 3
    assert summary["meta"]["n_rows"] == 3
    assert summary["meta"]["run_meta"]["env"]["cpu"] == "x"
    assert summary["stability"]["answer_similarity"] is None  # 没给 embedder 就不编
    md = analyze.render_markdown(summary)
    assert "n=" in md and "k=3" in md
    assert "deepseek-chat" in md


def test_summary_is_deterministic_for_same_input():
    samples = [_sample("q1", keys=["对"])]
    rows = [_row("q1", i, answer="对", total_s=1 + i) for i in range(3)]
    meta = {"run_id": "run_test", "k": 3}
    a = analyze.build_summary(rows, samples, run_meta=meta, embedder=None)
    b = analyze.build_summary(rows, samples, run_meta=meta, embedder=None)
    a["meta"].pop("analyzed_at_utc")
    b["meta"].pop("analyzed_at_utc")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
