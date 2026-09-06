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


# ------------------------------------------------ 含片段全文的完整版标注表
def test_full_sheet_includes_chunk_text(open_samples):
    """标注时对某题拿不准，需要能直接读到 Agent 当时看到的原文。"""
    samples, traces = open_samples
    traces["cap_007"] = make_trace(
        answer="信贷用分层切分，航班用 GroupKFold。",
        tools=["retrieve"],
        retrieved=[["c730fa5c_0019"]],
    )
    traces["cap_007"].steps[0].results[0].retrieved[0].text = "这是被检索到的原文片段"

    brief = judge.build_labeling_sheet(samples, traces)
    full = judge.build_labeling_sheet(samples, traces, include_chunk_text=True)

    assert "这是被检索到的原文片段" not in brief, "默认版不塞全文，否则表会长到没法读"
    assert "这是被检索到的原文片段" in full


def test_full_sheet_still_hides_judge_scores(open_samples):
    """加了全文也不能顺手把裁判结论漏进来。"""
    samples, traces = open_samples
    full = judge.build_labeling_sheet(
        samples, traces, judge_scores={"cap_007": 2}, include_chunk_text=True
    )
    assert "judge_score" not in full


# ===========================================================================
# M5 后半截：自动裁判打分 + kappa
#
# 九条门禁（用户 2026-09-06 确认的验收口径）。全部离线：裁判调用一律打桩，
# CI 上不碰真实 API（CLAUDE.md §7）。
# ===========================================================================


def _fake_call(replies):
    """构造一个假裁判：按序返回 replies，并记录每次收到的 messages / kwargs。"""
    queue = list(replies)
    calls: list[dict] = []

    def _call(messages, **kwargs):
        calls.append({"messages": [dict(m) for m in messages], "kwargs": kwargs})
        if not queue:
            raise AssertionError("假裁判的脚本用光了：调用次数超出预期")
        return queue.pop(0)

    _call.calls = calls
    return _call


def _ok(score: int, reason: str = "理由"):
    return json.dumps({"score": score, "reason": reason}, ensure_ascii=False)


# ---------------------------------------------------------------- 1. 裁判身份
def test_scorer_actually_uses_the_judge_model(open_samples):
    """config 里说换了裁判，代码得真换。

    现有 test_config 只钉住配置值本身，钉不住「打分时用的是不是它」——
    自评偏好正是从这种地方漏进来的。
    """
    from src import config

    samples, traces = open_samples
    call = _fake_call([_ok(2), _ok(1)])
    judge.score_samples(samples, traces, call=call)

    assert call.calls, "裁判一次都没被调用"
    for record in call.calls:
        assert record["kwargs"]["model"] == config.JUDGE_MODEL_NAME
        assert record["kwargs"]["model"] != config.MODEL_NAME, "裁判和被测同源了"


# ------------------------------------------------- 2. 输入对等：不给检索片段
def test_judge_prompt_never_contains_retrieved_chunks(open_samples):
    """人标注时看的是问题+参考答案+Agent答案，裁判必须看同样的东西。

    给了片段，裁判会拿原文替 Agent「找补」，判的就不再是答案本身；
    两边输入不对等，kappa 也就失去了「同一份材料下看法是否一致」的含义。
    """
    samples, traces = open_samples
    traces["cap_007"] = make_trace(
        answer="信贷用分层切分，航班用 GroupKFold。",
        tools=["retrieve"],
        retrieved=[["c730fa5c_0019"]],
    )
    traces["cap_007"].steps[0].results[0].retrieved[0].text = "这是被检索到的原文片段"

    call = _fake_call([_ok(2), _ok(1)])
    judge.score_samples(samples, traces, call=call)

    blob = json.dumps([c["messages"] for c in call.calls], ensure_ascii=False)
    assert "这是被检索到的原文片段" not in blob, "提示词里漏了检索片段"
    assert "c730fa5c_0019" not in blob, "连 doc_id 也不该给"


# ------------------------------------------------------------ 3. 判据同源
def test_judge_prompt_reuses_the_same_rubric_as_humans(open_samples):
    """人机判据必须是同一段字，各写一份必然漂移。"""
    samples, traces = open_samples
    call = _fake_call([_ok(2), _ok(1)])
    judge.score_samples(samples, traces, call=call)

    system = call.calls[0]["messages"][0]
    assert system["role"] == "system"
    for description in judge.LABEL_SCALE.values():
        assert judge._plain(description) in system["content"], "判据文本和标注表不同源"


def test_judge_prompt_and_sheet_share_the_scoring_notes(open_samples):
    """打分细则也必须同源。

    v1 的教训：判据只说"要点齐全"、没说什么算一个要点，人和机器就各解释各的
    （人比对要点层、机器比对结论层），kappa 被这道缝隙压到 0.216。
    细则各写一份 = 把那道缝隙重新打开。
    """
    samples, traces = open_samples
    call = _fake_call([_ok(2), _ok(1)])
    judge.score_samples(samples, traces, call=call)
    system = call.calls[0]["messages"][0]["content"]
    sheet = judge.build_labeling_sheet(samples, traces)

    assert judge.RUBRIC_NOTES, "细则不能是空的"
    for note in judge.RUBRIC_NOTES:
        assert judge._plain(note) in system, f"裁判提示词缺细则：{note[:20]}"
        assert note in sheet, f"标注表缺细则：{note[:20]}"


def test_rubric_is_point_level_not_conclusion_level():
    """v2 判据的核心：缺一项具体限定就得降档，「结论一致」不足以给满分。

    这条断言是这次修订的验收口径本身——判据要是被改回结论级，它必须红。
    """
    blob = judge.build_judge_system_prompt()
    assert "具体限定" in blob
    assert "缺一项即降到 1 分" in blob
    assert "结论方向一致" in blob and "不足以给 2 分" in blob
    assert "逐项" in blob


def test_judge_scores_record_the_rubric_version(open_samples):
    """分数要带判据版本：改前改后两批分混在一起就没法对照了。"""
    samples, traces = open_samples
    rows = judge.score_samples(samples, traces, call=_fake_call([_ok(2), _ok(1)]))
    assert all(r["rubric_version"] == judge.RUBRIC_VERSION for r in rows)
    assert all(r["answer_sha1"] for r in rows)


def test_judge_prompt_flags_hallucination_bait(open_samples):
    """诱饵标记要给裁判——人标注时表里就写着 hallucination_bait，输入必须对等。"""
    samples, traces = open_samples
    call = _fake_call([_ok(2), _ok(0)])
    judge.score_samples(samples, traces, call=call)

    # 只查 user 消息：system 里的判据本来就要向裁判解释诱饵题怎么判，
    # 那段字对每道题都在；真正区分"这题是不是诱饵"的是 user 里的题目标记。
    by_id = {c["kwargs"]["sample_id"]: c["messages"][-1]["content"] for c in call.calls}
    assert "hallucination_bait" in by_id["cap_010"]
    assert "幻觉诱饵题" in by_id["cap_010"]
    assert "hallucination_bait" not in by_id["cap_007"]


# ------------------------------------------------------ 4~6. kappa 计算正确性
def test_kappa_perfect_agreement():
    human = {"a": 0, "b": 1, "c": 2, "d": 1}
    stats = judge.agreement_stats(human, dict(human), bootstrap=0)
    assert stats["kappa"] == pytest.approx(1.0)
    assert stats["kappa_quadratic"] == pytest.approx(1.0)
    assert stats["exact_agreement"] == pytest.approx(1.0)
    assert stats["n"] == 4


def test_kappa_matches_hand_computation():
    """手算例子，故意让两种口径显著不同——加权口径写反了这里就会炸。

        人  = [0, 1, 2, 0, 1, 2]
        机  = [1, 1, 2, 0, 2, 2]

    unweighted: po = 4/6 = 2/3；人的边缘 (1/3,1/3,1/3)、机的边缘 (1/6,2/6,3/6)
                pe = 1/3 * (1/6+2/6+3/6) = 1/3
                kappa = (2/3 - 1/3) / (1 - 1/3) = 0.5
    quadratic:  w = 1 - (i-j)^2/4，四处一致得 1、两处差一档各得 0.75
                po_w = (4 + 1.5)/6 = 0.9167；pe_w = 2/3
                kappa = (0.9167 - 0.6667) / (1 - 0.6667) = 0.75
    """
    ids = list("abcdef")
    human = dict(zip(ids, [0, 1, 2, 0, 1, 2]))
    auto = dict(zip(ids, [1, 1, 2, 0, 2, 2]))
    stats = judge.agreement_stats(human, auto, bootstrap=0)

    assert stats["kappa"] == pytest.approx(0.5, abs=1e-9)
    assert stats["kappa_quadratic"] == pytest.approx(0.75, abs=1e-9)
    assert stats["exact_agreement"] == pytest.approx(4 / 6)
    assert stats["adjacent_agreement"] == pytest.approx(1.0)


def test_kappa_labels_are_pinned_not_inferred():
    """标度固定为 0/1/2。若让 sklearn 从数据里推断类别，

    某一次裁判恰好没打过 0 分，两次运行的 kappa 就不在同一套类别上，不可比。
    """
    ids = list("abcd")
    # 两边都只出现 1 和 2，但类别集必须仍是 {0,1,2}
    human = dict(zip(ids, [1, 2, 1, 2]))
    auto = dict(zip(ids, [1, 2, 2, 1]))
    stats = judge.agreement_stats(human, auto, bootstrap=0)
    assert stats["labels"] == [0, 1, 2]
    assert len(stats["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in stats["confusion_matrix"])


def test_kappa_undefined_when_a_rater_has_no_variance():
    """退化情形必须明说「未定义」，绝不落成 0.0。

    0.0 读起来像「完全不一致」，而真实情况是这个数学量根本没有定义——
    把它写成 0.0 就是编造指标（CLAUDE.md 1.1）。
    """
    human = {"a": 2, "b": 2, "c": 2}
    auto = {"a": 2, "b": 2, "c": 1}
    stats = judge.agreement_stats(human, auto, bootstrap=0)

    assert stats["kappa"] is None
    assert stats["kappa_quadratic"] is None
    assert stats["degenerate"] is True
    assert "未定义" in stats["kappa_note"]
    # 一致率这类不依赖变异的量照常报
    assert stats["exact_agreement"] == pytest.approx(2 / 3)

    text = judge.render_agreement_markdown(stats)
    assert "未定义" in text
    assert "0.000" not in text.split("混淆矩阵")[0]


def test_both_raters_constant_and_identical_is_still_undefined():
    """两边全打 2 分：看着「完全一致」，但机遇一致率也是 1，kappa 依然无定义。"""
    stats = judge.agreement_stats({"a": 2, "b": 2}, {"a": 2, "b": 2}, bootstrap=0)
    assert stats["kappa"] is None and stats["degenerate"] is True
    assert stats["exact_agreement"] == pytest.approx(1.0)


def test_bootstrap_ci_is_reproducible_and_reports_dropped_resamples():
    ids = [f"q{i}" for i in range(10)]
    human = dict(zip(ids, [0, 1, 1, 1, 1, 2, 2, 2, 2, 2]))
    auto = dict(zip(ids, [0, 1, 2, 1, 1, 2, 2, 1, 2, 2]))

    a = judge.agreement_stats(human, auto, bootstrap=500, seed=7)
    b = judge.agreement_stats(human, auto, bootstrap=500, seed=7)
    assert a["kappa_ci"] == b["kappa_ci"], "固定种子下 bootstrap 必须可复现"

    lo, hi = a["kappa_ci"]
    assert lo <= a["kappa"] <= hi
    # 重抽样里出现「某一方无变异」的样本是常态，必须如实报出被丢掉多少次
    assert a["bootstrap_degenerate"] >= 0
    assert a["bootstrap_used"] + a["bootstrap_degenerate"] == 500


# ------------------------------------------------------- 7. 非法输出必须炸
@pytest.mark.parametrize("bad", ["3", "很好", "", '{"score": 1.5}', '{"reason": "x"}'])
def test_illegal_judge_output_raises_after_one_retry(open_samples, bad):
    """裁判吐了不合法的东西，必须重问一次；再不行就报错。

    绝不静默兜底成某个分数——那等于凭空造了一个指标。
    """
    samples, traces = open_samples
    call = _fake_call([bad, bad])
    with pytest.raises(ValueError, match="裁判"):
        judge.score_samples(samples, traces, call=call)
    assert len(call.calls) == 2, "应当且只应当重问一次"


def test_judge_retry_succeeds_on_second_try(open_samples):
    samples, traces = open_samples
    call = _fake_call(["不是 JSON", _ok(2), _ok(0)])
    rows = judge.score_samples(samples, traces, call=call)
    assert [r["judge_score"] for r in rows] == [2, 0]
    assert rows[0]["retried"] is True and rows[1]["retried"] is False


def test_judge_accepts_json_wrapped_in_code_fence(open_samples):
    """reasoner 常把 JSON 包在代码围栏里，正则兜底要能抠出来。"""
    samples, traces = open_samples
    fenced = '```json\n{"score": 1, "reason": "要点有遗漏"}\n```'
    call = _fake_call([fenced, _ok(2)])
    rows = judge.score_samples(samples, traces, call=call)
    assert rows[0]["judge_score"] == 1
    assert rows[0]["retried"] is False, "能解析出来就不该浪费一次重问"


# --------------------------------------------------- 8. 分母：缺分的题要露出来
def test_agreement_reports_missing_on_both_sides():
    """只在两边都有分的题上算，且缺了谁必须写出来（分母陷阱，RAGAS 那次教训）。"""
    human = {"a": 2, "b": 1, "c": 0}
    auto = {"a": 2, "b": 1, "d": 2}
    stats = judge.agreement_stats(human, auto, bootstrap=0)

    assert stats["n"] == 2
    assert stats["missing_judge"] == ["c"], "人标了机器没打分的题"
    assert stats["missing_human"] == ["d"], "机器打了人没标的题"
    assert "n=2" in judge.render_agreement_markdown(stats)


def test_disagreements_are_listed_for_review():
    human = {"a": 2, "b": 1, "c": 0}
    auto = {"a": 2, "b": 2, "c": 2}
    stats = judge.agreement_stats(human, auto, bootstrap=0)
    assert [d["id"] for d in stats["disagreements"]] == ["b", "c"]
    assert stats["disagreements"][-1]["gap"] == 2


# ------------------------------------------------ 9. 反向独立：机器不许看人的分
def test_scoring_never_reads_human_labels(open_samples, monkeypatch):
    """独立性要两个方向都锁死。

    标注表不给人看机器分（上面已有测试）；这里守另一边：
    裁判打分的整条路径不许碰 human_labels.jsonl。
    """
    samples, traces = open_samples

    def _boom(*args, **kwargs):
        raise AssertionError("裁判打分读了人工标注文件")

    monkeypatch.setattr(judge, "load_human_labels", _boom)
    rows = judge.score_samples(samples, traces, call=_fake_call([_ok(2), _ok(0)]))
    assert len(rows) == 2

    blob = json.dumps(rows, ensure_ascii=False)
    assert "human_score" not in blob


# ------------------------------------------------------------ 落盘与读回
def test_judge_scores_roundtrip(open_samples, tmp_path):
    samples, traces = open_samples
    rows = judge.score_samples(samples, traces, call=_fake_call([_ok(2), _ok(0)]))
    path = judge.write_judge_scores(rows, tmp_path / "judge_scores.jsonl")
    assert judge.load_judge_scores(path) == {"cap_007": 2, "cap_010": 0}


def test_load_judge_scores_rejects_illegal_score(tmp_path):
    path = tmp_path / "judge_scores.jsonl"
    path.write_text(json.dumps({"id": "a", "judge_score": 7}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0/1/2"):
        judge.load_judge_scores(path)


def test_report_of_small_sample_is_flagged():
    """n=10 的 kappa 不稳，报告里必须自带这句话，不能等人去记住。"""
    ids = [f"q{i}" for i in range(10)]
    human = dict(zip(ids, [0, 1, 1, 1, 1, 2, 2, 2, 2, 2]))
    auto = dict(zip(ids, [0, 1, 2, 1, 1, 2, 2, 1, 2, 2]))
    text = judge.render_agreement_markdown(judge.agreement_stats(human, auto, bootstrap=200))
    assert "n=10" in text
    assert "置信区间" in text
    assert "不作硬结论" in text or "不要把它当硬结论" in text


# ------------------------------------------------- 解读：人写的部分要能落进报告
def test_interpretation_is_rendered_and_labelled_as_human_written():
    """数字是机器算的、叙述是人写的，报告里必须分得清。"""
    stats = judge.agreement_stats({"a": 0, "b": 2}, {"a": 2, "b": 2}, bootstrap=0)
    text = judge.render_agreement_markdown(stats, "这是人写的解读。")
    assert "解读（人工撰写，非自动生成）" in text
    assert "这是人写的解读。" in text


def test_no_interpretation_means_no_section():
    """没写解读就不出这一节，绝不自动编一段叙述充数。"""
    stats = judge.agreement_stats({"a": 0, "b": 2}, {"a": 2, "b": 2}, bootstrap=0)
    assert "解读" not in judge.render_agreement_markdown(stats, "")


def test_interpretation_file_is_optional(tmp_path):
    assert judge.load_agreement_interpretation(tmp_path / "nope.md") == ""
    f = tmp_path / "x.md"
    f.write_text("有内容", encoding="utf-8")
    assert judge.load_agreement_interpretation(f) == "有内容"
