"""黄金集加载与 schema 校验的验收口径。

除了 schema 本身，这里还有一条**数据完整性**测试：黄金集里标注的每个
expected_doc_ids 都必须真实存在于当前语料切出来的 chunk 里。
语料一改、切分参数一动，这条就会红——这正是我们要的，因为那意味着
recall@k 的标注已经失效，指标不再可信。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.eval import datasets


def _sample(**overrides):
    base = dict(
        id="cap_001",
        question="问题？",
        reference_answer="答案。",
        answer_keys=["0.25"],
        expected_tool=["retrieve"],
        expected_doc_ids=["abc12345_0000"],
        answer_type="closed",
        failure_tag=None,
    )
    base.update(overrides)
    return base


# ------------------------------------------------------------------ schema
def test_valid_sample_parses():
    s = datasets.GoldenSample(**_sample())
    assert s.id == "cap_001"
    assert s.expected_tool == ["retrieve"]


def test_expected_tool_must_be_a_list():
    """schema 修订：单字符串不再接受，必须是有序列表。"""
    with pytest.raises(ValidationError):
        datasets.GoldenSample(**_sample(expected_tool="retrieve"))


def test_expected_tool_cannot_be_empty():
    with pytest.raises(ValidationError):
        datasets.GoldenSample(**_sample(expected_tool=[]))


def test_unknown_tool_rejected():
    """标注了 Agent 根本没挂的工具，是标注错误，必须当场报错。"""
    with pytest.raises(ValidationError):
        datasets.GoldenSample(**_sample(expected_tool=["browse_web"]))


def test_closed_question_requires_answer_keys():
    """闭合题走规则判定；没有 answer_keys 就无从判定对错。"""
    with pytest.raises(ValidationError):
        datasets.GoldenSample(**_sample(answer_type="closed", answer_keys=[]))


def test_open_question_must_not_have_answer_keys():
    """开放题交 M5 裁判，不该塞规则关键词，避免两套判定口径打架。"""
    with pytest.raises(ValidationError):
        datasets.GoldenSample(**_sample(answer_type="open", answer_keys=["x"]))


def test_open_question_with_empty_keys_is_fine():
    s = datasets.GoldenSample(**_sample(answer_type="open", answer_keys=[]))
    assert s.answer_type == "open"


def test_empty_expected_doc_ids_allowed():
    """纯算术题和幻觉诱饵题没有应检索文档，这是合法的。"""
    s = datasets.GoldenSample(**_sample(expected_doc_ids=[], expected_tool=["calc"]))
    assert s.expected_doc_ids == []


# ------------------------------------------------------------------ 加载器
def test_load_golden_set_returns_samples():
    samples = datasets.load_golden_set()
    assert len(samples) >= 30, "CLAUDE.md M3 要求 30–50 条"
    assert len(samples) <= 50


def test_ids_are_unique():
    ids = [s.id for s in datasets.load_golden_set()]
    assert len(ids) == len(set(ids))


def test_duplicate_id_rejected(tmp_path):
    import json

    path = tmp_path / "dup.jsonl"
    row = _sample()
    path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复"):
        datasets.load_golden_set(path)


def test_blank_lines_are_skipped(tmp_path):
    import json

    path = tmp_path / "blanks.jsonl"
    path.write_text(
        "\n" + json.dumps(_sample(), ensure_ascii=False) + "\n\n", encoding="utf-8"
    )
    assert len(datasets.load_golden_set(path)) == 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        datasets.load_golden_set(tmp_path / "nope.jsonl")


# -------------------------------------------------------------- 数据完整性
def test_every_expected_doc_id_exists_in_corpus():
    """标注引用的 chunk 必须真实存在。

    语料被改动、切分参数被调整时，这条会红——那说明 recall@k 的标注
    已经对不上语料了，指标不能再信，必须重新标注。
    """
    from src.agent import rag

    known = {c.doc_id for c in rag.build_chunks()}
    dangling = {
        s.id: [d for d in s.all_doc_ids if d not in known]
        for s in datasets.load_golden_set()
    }
    dangling = {k: v for k, v in dangling.items() if v}
    assert not dangling, f"这些题标注了不存在的 doc_id：{dangling}"


def test_dataset_has_both_tools_and_both_answer_types():
    """评测集必须能区分开这些维度，否则对应指标是常数，测不出东西。"""
    samples = datasets.load_golden_set()
    tools = {t for s in samples for t in s.expected_tool}
    assert tools == {"retrieve", "calc"}
    assert {s.answer_type for s in samples} == {"closed", "open"}
    assert any(len(s.expected_tool) > 1 for s in samples), "至少要有一道多工具题"
    assert any(s.failure_tag == "hallucination_bait" for s in samples), "至少要有一道幻觉诱饵"


# ------------------------------------------------ 证据组建模（M5 前置修订）
def test_bare_string_becomes_a_single_element_group():
    """单块证据仍写成裸字符串，解析后归一为一个单元素组——标注不必变啰嗦。"""
    s = datasets.GoldenSample(**_sample(expected_doc_ids=["abc12345_0000"]))
    assert s.expected_doc_ids == [["abc12345_0000"]]


def test_nested_list_is_kept_as_a_group():
    """组内是替代关系：任一命中即算该组召回。"""
    s = datasets.GoldenSample(
        **_sample(expected_doc_ids=[["a_0001", "a_0002"], ["b_0001"]])
    )
    assert s.expected_doc_ids == [["a_0001", "a_0002"], ["b_0001"]]


def test_mixed_form_is_normalized():
    s = datasets.GoldenSample(**_sample(expected_doc_ids=["a_0001", ["b_0001", "b_0002"]]))
    assert s.expected_doc_ids == [["a_0001"], ["b_0001", "b_0002"]]


def test_empty_group_rejected():
    """空组会让 recall 的分母凭空多一格却永远命不中，属于标注错误。"""
    with pytest.raises(ValidationError):
        datasets.GoldenSample(**_sample(expected_doc_ids=[[]]))


def test_all_doc_ids_flattens_groups():
    """完整性校验（doc_id 是否存在于语料）需要摊平后的全集。"""
    s = datasets.GoldenSample(**_sample(expected_doc_ids=[["a", "b"], ["c"]]))
    assert s.all_doc_ids == ["a", "b", "c"]


def test_no_groups_means_not_scored_for_recall():
    s = datasets.GoldenSample(**_sample(expected_doc_ids=[], expected_tool=["calc"]))
    assert s.expected_doc_ids == []
    assert s.is_scored_for_recall is False
    assert s.all_doc_ids == []
