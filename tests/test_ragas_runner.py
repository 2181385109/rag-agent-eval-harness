"""RAGAS 接入层的验收口径（M4）。

这里测的是**我们自己那部分**：喂给 RAGAS 的数据组织对不对、该排除的样本有没有
被排除、embedding 有没有老老实实走本地 BGE。RAGAS 内部怎么算 faithfulness
不是我们要测的东西。

ragas 本体在函数内部延迟导入，所以这些测试不需要装 ragas 也能跑——
CI 上跑的就是这条路径（CLAUDE.md §7：CI 不调真实 API）。
"""

from __future__ import annotations

import importlib.util

import pytest

from src import config
from src.eval import ragas_runner

# 注意：不能用 pytest.importorskip('ragas')——它会绕过 shim 直接 import，
# 撞上那个坏导入后把测试误跳过，守卫就形同虚设。只判断包有没有装。
_HAS_RAGAS = importlib.util.find_spec("ragas") is not None
requires_ragas = pytest.mark.skipif(not _HAS_RAGAS, reason="未安装 ragas")
from tests.test_metrics import make_sample, make_trace


@pytest.fixture
def samples_and_traces():
    samples = [
        make_sample(id="ok1", answer_keys=["x"], expected_doc_ids=["d1"]),
        make_sample(id="ok2", answer_type="open", answer_keys=[], expected_doc_ids=["d2"]),
        make_sample(id="empty", answer_type="open", answer_keys=[], expected_doc_ids=[]),
        make_sample(id="nocontext", answer_keys=["y"], expected_doc_ids=[]),
    ]
    stuck = make_trace(answer="", tools=["retrieve"], retrieved=[["d9"]])
    stuck.stop_reason = "max_steps"
    traces = {
        "ok1": make_trace(answer="答案一", tools=["retrieve"], retrieved=[["d1"]]),
        "ok2": make_trace(answer="答案二", tools=["retrieve"], retrieved=[["d2"]]),
        "empty": stuck,
        "nocontext": make_trace(answer="纯算术答案", tools=["calc"]),
    }
    return samples, traces


# ---------------------------------------------------------------- 数据组织
def test_dataset_rows_carry_the_four_required_fields(samples_and_traces):
    samples, traces = samples_and_traces
    rows, _ = ragas_runner.build_ragas_dataset(samples, traces)
    row = rows[0]
    for key in ("user_input", "response", "retrieved_contexts", "reference"):
        assert key in row, f"RAGAS 需要的字段缺失：{key}"
    assert row["response"] == "答案一"
    assert isinstance(row["retrieved_contexts"], list) and row["retrieved_contexts"]


def test_contexts_are_chunk_texts_not_doc_ids(samples_and_traces):
    """faithfulness 判的是"答案有没有被检索内容支撑"，喂 doc_id 毫无意义。"""
    samples, traces = samples_and_traces
    rows, _ = ragas_runner.build_ragas_dataset(samples, traces)
    assert all(c and not c.startswith("d") for c in rows[0]["retrieved_contexts"])


def test_reference_comes_from_golden_set(samples_and_traces):
    """context_recall 需要标准答案做参照，用黄金集里人工写的 reference_answer。"""
    samples, traces = samples_and_traces
    rows, _ = ragas_runner.build_ragas_dataset(samples, traces)
    assert rows[0]["reference"] == samples[0].reference_answer


# ------------------------------------------------------------ 排除与说明
def test_empty_answer_is_excluded_with_a_reason(samples_and_traces):
    """cap_036 那类跑满步数没给答案的样本：没有答案就无从评价答案，
    不能混进 faithfulness 均值稀释它——排除，并如实记下原因。"""
    samples, traces = samples_and_traces
    rows, excluded = ragas_runner.build_ragas_dataset(samples, traces)

    assert "empty" not in {r["id"] for r in rows}
    reasons = {e["id"]: e["reason"] for e in excluded}
    assert "empty" in reasons
    assert "答案" in reasons["empty"]


def test_no_context_sample_is_excluded(samples_and_traces):
    """没检索过的纯算术题没有 contexts，faithfulness / context_recall 无从谈起。"""
    samples, traces = samples_and_traces
    _rows, excluded = ragas_runner.build_ragas_dataset(samples, traces)
    assert "nocontext" in {e["id"] for e in excluded}


def test_normal_samples_are_kept(samples_and_traces):
    samples, traces = samples_and_traces
    rows, _ = ragas_runner.build_ragas_dataset(samples, traces)
    assert {r["id"] for r in rows} == {"ok1", "ok2"}


def test_excluded_ids_never_overlap_with_evaluated(samples_and_traces):
    samples, traces = samples_and_traces
    rows, excluded = ragas_runner.build_ragas_dataset(samples, traces)
    assert not ({r["id"] for r in rows} & {e["id"] for e in excluded})


def test_all_samples_accounted_for(samples_and_traces):
    """每条样本要么被评、要么被明确排除，不许凭空消失。"""
    samples, traces = samples_and_traces
    rows, excluded = ragas_runner.build_ragas_dataset(samples, traces)
    assert len(rows) + len(excluded) == len(samples)


# -------------------------------------------------------- 本地 embedding
def test_embeddings_adapter_uses_local_bge_not_an_api():
    """§1.7：除对 DeepSeek 的模型调用外不外发数据。语料绝不能被送去第三方做 embedding。"""

    class SpyEmbedder:
        def __init__(self):
            self.doc_calls, self.query_calls = [], []

        def encode(self, texts, is_query: bool = False):
            import numpy as np

            (self.query_calls if is_query else self.doc_calls).extend(texts)
            return np.ones((len(texts), 4), dtype="float32") / 2.0

    spy = SpyEmbedder()
    adapter = ragas_runner.LocalBGEEmbeddings(spy)

    vectors = adapter.embed_documents(["文档一", "文档二"])
    assert len(vectors) == 2 and len(vectors[0]) == 4
    assert all(isinstance(x, float) for x in vectors[0]), "LangChain 要的是 python float 列表"
    assert spy.doc_calls == ["文档一", "文档二"]

    adapter.embed_query("查询")
    assert spy.query_calls == ["查询"], "查询侧要走 is_query=True，BGE 的指令前缀才生效"


# ------------------------------------------------------------------ 元信息
def test_result_records_judge_model_for_reproducibility(monkeypatch, samples_and_traces):
    """裁判模型是哪一个，直接决定分数——不记下来这个数就不可复现。"""
    samples, traces = samples_and_traces

    def fake_evaluate(rows, metric_names=None):
        names = list(metric_names or ragas_runner.DEFAULT_METRICS)
        return {n: 0.5 for n in names}, {n: len(rows) for n in names}

    monkeypatch.setattr(ragas_runner, "_evaluate_rows", fake_evaluate)
    result = ragas_runner.run_ragas(samples, traces)

    assert result["judge_model"] == config.JUDGE_MODEL_NAME
    assert result["embedding_model"] == config.EMBEDDING_MODEL
    assert result["n_submitted"] == 2
    assert result["n_excluded"] == 2
    assert result["total"] == 4
    assert set(result["scores"]) == set(ragas_runner.DEFAULT_METRICS)


def test_run_ragas_reports_excluded_detail(monkeypatch, samples_and_traces):
    samples, traces = samples_and_traces
    monkeypatch.setattr(ragas_runner, "_evaluate_rows", lambda rows, metric_names=None: ({}, {}))
    result = ragas_runner.run_ragas(samples, traces)
    assert {e["id"] for e in result["excluded"]} == {"empty", "nocontext"}


# ------------------------------------------------ RAGAS 兼容性补丁（M4）
@requires_ragas
def test_ragas_is_importable_and_metrics_resolve():
    """守住那个 shim：ragas 能导入、四个指标都能解析出来。

    这条会在 ragas 或 langchain-community 版本变化把兼容性打破时立刻变红，
    而不是等到跑评测才发现。装了 ragas 才跑。
    """
    ragas_runner._ensure_ragas_importable()

    resolved = ragas_runner._resolve_metrics(ragas_runner.DEFAULT_METRICS)
    assert len(resolved) == len(ragas_runner.DEFAULT_METRICS)
    assert all(m is not None for m in resolved)


def test_shim_does_not_override_an_existing_module(monkeypatch):
    """真模块存在时补丁必须自动失效，不能盖掉上游的真实实现。"""
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    sentinel = types.ModuleType(name)
    sentinel.ChatVertexAI = "REAL_IMPLEMENTATION"
    monkeypatch.setitem(sys.modules, name, sentinel)

    ragas_runner._ensure_ragas_importable()

    assert sys.modules[name] is sentinel
    assert sys.modules[name].ChatVertexAI == "REAL_IMPLEMENTATION"


def test_shim_is_idempotent():
    """反复调用不应重复替换或报错。"""
    import sys

    ragas_runner._ensure_ragas_importable()
    first = sys.modules.get("langchain_community.chat_models.vertexai")
    ragas_runner._ensure_ragas_importable()
    assert sys.modules.get("langchain_community.chat_models.vertexai") is first


@requires_ragas
def test_unknown_metric_name_is_rejected():
    with pytest.raises(ValueError, match="没有这个指标"):
        ragas_runner._resolve_metrics(["not_a_real_metric"])


# ------------------------------------------------ 分母诚实（M4 修正）
def test_scored_count_reflects_actually_scored_rows_not_submitted(monkeypatch, samples_and_traces):
    """RAGAS 里单个 job 失败（如裁判 400/超时）时，那一行是 NaN。

    pandas 的 mean() 默认跳过 NaN，于是均值只覆盖成功的行，而"提交了 N 行"
    仍是 N——直接把 N 当分母上报就是编数。每个指标必须报**真正打出分的行数**。
    """
    samples, traces = samples_and_traces

    def fake_evaluate(rows, metric_names=None):
        names = list(metric_names or ragas_runner.DEFAULT_METRICS)
        # 提交 2 行，但 faithfulness 只成功打了 1 行
        scores = {n: 0.8 for n in names}
        counts = {n: len(rows) for n in names}
        counts["faithfulness"] = 1
        return scores, counts

    monkeypatch.setattr(ragas_runner, "_evaluate_rows", fake_evaluate)
    result = ragas_runner.run_ragas(samples, traces)

    assert result["n_submitted"] == 2
    assert result["scored_counts"]["faithfulness"] == 1
    assert result["scored_counts"]["answer_relevancy"] == 2
    assert result["has_incomplete_metric"] is True, "有指标没打满，必须显式标出来"


def test_complete_run_is_not_flagged(monkeypatch, samples_and_traces):
    samples, traces = samples_and_traces

    def fake_evaluate(rows, metric_names=None):
        names = list(metric_names or ragas_runner.DEFAULT_METRICS)
        return {n: 0.9 for n in names}, {n: len(rows) for n in names}

    monkeypatch.setattr(ragas_runner, "_evaluate_rows", fake_evaluate)
    result = ragas_runner.run_ragas(samples, traces)
    assert result["has_incomplete_metric"] is False


def test_judge_llm_bypasses_multi_generation():
    """DeepSeek 只支持 n=1，RAGAS 默认会请求 n=3 做自洽投票并被 400 拒掉。

    必须让 wrapper 走 bypass_n（改成发 n 次单生成请求），否则将近四成的
    评分任务会静默失败，均值只覆盖侥幸成功的那部分。
    """
    pytest.importorskip("langchain_openai")
    if not _HAS_RAGAS:
        pytest.skip("未安装 ragas")
    import os

    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-placeholder-for-construction-only")
    wrapper = ragas_runner.build_judge_llm()
    assert wrapper.bypass_n is True


def test_judge_uses_the_longer_timeout():
    """裁判调用比 Agent 重得多，用 Agent 的 60s 超时会大面积打不完。"""
    pytest.importorskip("langchain_openai")
    if not _HAS_RAGAS:
        pytest.skip("未安装 ragas")
    import os

    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-placeholder-for-construction-only")
    assert config.JUDGE_TIMEOUT_S > config.REQUEST_TIMEOUT_S
    wrapper = ragas_runner.build_judge_llm()
    assert wrapper.langchain_llm.request_timeout == config.JUDGE_TIMEOUT_S


def test_run_config_carries_config_values():
    if not _HAS_RAGAS:
        pytest.skip("未安装 ragas")
    rc = ragas_runner.build_run_config()
    assert rc.timeout == int(config.JUDGE_TIMEOUT_S)
    assert rc.max_workers == config.RAGAS_MAX_WORKERS
