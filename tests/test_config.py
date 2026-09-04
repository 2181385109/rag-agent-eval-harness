"""config 的验收口径：端点/模型名锁定、路径就位、阈值合法、源码里没有 key。"""
from __future__ import annotations

import re
from pathlib import Path

from src import config


def test_endpoint_and_models_locked():
    # CLAUDE.md §2/§3 锁定的值，改动必须是有意为之
    assert config.BASE_URL == "https://api.deepseek.com/v1"
    assert config.MODEL_NAME == "deepseek-chat"
    assert config.JUDGE_MODEL_NAME
    assert config.API_KEY_ENV == "DEEPSEEK_API_KEY"


def test_paths_are_under_project_root_and_exist():
    root = config.PROJECT_ROOT
    for p in (config.CORPUS_DIR, config.DATA_DIR, config.REPORTS_DIR):
        assert p.is_relative_to(root), f"{p} 跑到项目外面去了"
        assert p.is_dir(), f"{p} 不存在"
    assert config.GOLDEN_SET_PATH.parent == config.DATA_DIR
    assert config.HUMAN_LABELS_PATH.parent == config.DATA_DIR


def test_thresholds_are_sane():
    assert 0.0 < config.METRIC_DROP_TOLERANCE < 1.0
    assert config.GATED_METRICS, "回归门禁至少要盯一个指标"
    assert config.RETRIEVE_TOP_K >= 1
    assert config.CONSISTENCY_RUNS >= 2, "多轮一致性至少要跑 2 次才有方差"
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
    assert config.DEFAULT_TEMPERATURE == 0.0, "评测要可复现，默认温度必须是 0"


def test_no_api_key_literal_in_config_source():
    src = Path(config.__file__).read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9]{16,}", src), "config.py 里出现了疑似 API key"


def test_get_api_key_raises_when_env_missing(monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    assert config.has_api_key() is False
    try:
        config.get_api_key()
    except RuntimeError as exc:
        assert config.API_KEY_ENV in str(exc)
    else:
        raise AssertionError("缺 key 时应当抛 RuntimeError，而不是静默继续")


def test_get_api_key_reads_from_env(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "sk-unit-test-value")
    assert config.has_api_key() is True
    assert config.get_api_key() == "sk-unit-test-value"


def test_external_tracing_is_disabled():
    """§1.7：除 DeepSeek 外不外发数据。langsmith 是 langchain-core 的传递依赖，必须默认关闭。"""
    import os

    assert os.environ.get("LANGSMITH_TRACING") == "false"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"
