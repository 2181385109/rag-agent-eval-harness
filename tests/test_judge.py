"""裁判层与人工标注表的验收口径（M5）。

kappa 的全部价值在于"自动判定"和"人工判定"是**独立**得出的。
所以这里守两件事：标注表必须给出足够的判断依据，且**不能泄露自动裁判的结论**——
人先看到机器打的分再标，kappa 就变成了确认偏误的度量。
"""

from __future__ import annotations

import json

import pytest

from src.eval import judge
from tests.test_metrics import make_sample, make_trace


@pytest.fixture
def open_samples():
    samples = [
        make_sample(
            id="cap_007",
            question="两个项目各自怎么防数据泄露？",
            reference_answer="信贷按标签分层切分；航班按 tail_id 做 GroupKFold。",
            answer_type="open",
            answer_keys=[],
            expected_doc_ids=[["c730fa5c_0019"], ["2da65ad3_0026"]],
        ),
        make_sample(
            id="cap_010",
            question="规格书里指定用 Redis 吗？",
            reference_answer="没有，未提及 Redis。",
            answer_type="open",
            answer_keys=[],
            expected_doc_ids=[],
            failure_tag="hallucination_bait",
        ),
        make_sample(id="cap_001", answer_keys=["0.25"]),  # 闭合题，不该进表
    ]
    traces = {
        "cap_007": make_trace(answer="信贷用分层切分，航班用 GroupKFold。", tools=["retrieve"]),
        "cap_010": make_trace(answer="知识库中未提及 Redis。", tools=["retrieve"]),
        "cap_001": make_trace(answer="阈值 0.25", tools=["retrieve"]),
    }
    return samples, traces


# ---------------------------------------------------------------- 评分标度
def test_scale_is_documented_and_ordinal():
    """标度要有序且有明确判据，人和机器才可能对齐。"""
    assert list(judge.LABEL_SCALE) == [0, 1, 2]
    assert all(text.strip() for text in judge.LABEL_SCALE.values())


# ---------------------------------------------------------------- 标注表
def test_sheet_covers_only_open_questions(open_samples):
    """闭合题走规则判定，不该占用人工标注的精力。"""
    samples, traces = open_samples
    sheet = judge.build_labeling_sheet(samples, traces)
    assert "cap_007" in sheet and "cap_010" in sheet
    assert "cap_001" not in sheet


def test_sheet_contains_everything_needed_to_judge(open_samples):
    samples, traces = open_samples
    sheet = judge.build_labeling_sheet(samples, traces)
    assert "两个项目各自怎么防数据泄露？" in sheet
    assert "信贷按标签分层切分" in sheet, "缺参考答案就无从判定"
    assert "信贷用分层切分，航班用 GroupKFold。" in sheet, "缺 Agent 答案就没得判"
    for level, description in judge.LABEL_SCALE.items():
        assert str(level) in sheet and description[:6] in sheet


def test_sheet_flags_hallucination_bait(open_samples):
    """幻觉诱饵题的正确行为是"说不知道"，不标出来人会误判成没答上。"""
    samples, traces = open_samples
    sheet = judge.build_labeling_sheet(samples, traces)
    assert "hallucination_bait" in sheet


def test_sheet_never_leaks_automatic_scores(open_samples):
    """人工标注必须独立于裁判结论，否则 kappa 度量的是确认偏误。"""
    samples, traces = open_samples
    sheet = judge.build_labeling_sheet(samples, traces, judge_scores={"cap_007": 2})
    assert "裁判" not in sheet.split("## 逐题")[-1] or "2" in sheet
    # 显式约定：sheet 生成函数忽略 judge_scores，绝不写进表里
    assert "judge_score" not in sheet


def test_sheet_warns_about_small_sample():
    """n=10 的 kappa 很不稳，表里必须提醒，避免把它当成硬结论。"""
    sheet = judge.build_labeling_sheet([], {})
    assert "kappa" in sheet.lower()


# ------------------------------------------------------- human_labels 骨架
def test_label_skeleton_has_blank_scores(open_samples, tmp_path):
    samples, traces = open_samples
    path = judge.write_label_skeleton(samples, traces, tmp_path / "human_labels.jsonl")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["id"] for r in rows] == ["cap_007", "cap_010"]
    assert all(r["human_score"] is None for r in rows), "分数必须留空等人来填"
    assert all(r["answer"] for r in rows)


def test_label_skeleton_does_not_overwrite_existing_scores(open_samples, tmp_path):
    """人已经标过的分不能被重新生成骨架覆盖掉。"""
    samples, traces = open_samples
    path = tmp_path / "human_labels.jsonl"
    judge.write_label_skeleton(samples, traces, path)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["human_score"] = 2
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    judge.write_label_skeleton(samples, traces, path)
    again = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert again[0]["human_score"] == 2, "已有的人工标注被覆盖了"


def test_load_human_labels_skips_unlabelled(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps({"id": "a", "human_score": 2}, ensure_ascii=False)
        + "\n"
        + json.dumps({"id": "b", "human_score": None}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    assert judge.load_human_labels(path) == {"a": 2}


def test_invalid_human_score_rejected(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps({"id": "a", "human_score": 5}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0/1/2"):
        judge.load_human_labels(path)
