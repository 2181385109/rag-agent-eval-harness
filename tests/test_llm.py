"""DeepSeek 客户端封装的验收口径：全程打桩，不发真实请求。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import config
from src.agent import llm


def _fake_completion(text: str, prompt_tokens: int = 11, completion_tokens: int = 7):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        model=config.MODEL_NAME,
    )


@pytest.fixture
def recording_client(monkeypatch):
    """替换 get_client，记录发出去的参数。"""
    seen: dict = {}

    def _create(**kwargs):
        seen.update(kwargs)
        return _fake_completion("桩回答")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    monkeypatch.setattr(llm, "get_client", lambda: client)
    return seen


def test_chat_returns_text(recording_client):
    out = llm.chat([{"role": "user", "content": "你好"}])
    assert out == "桩回答"


def test_chat_uses_config_defaults(recording_client):
    llm.chat([{"role": "user", "content": "你好"}])
    assert recording_client["model"] == config.MODEL_NAME
    assert recording_client["temperature"] == config.DEFAULT_TEMPERATURE
    assert recording_client["max_tokens"] == config.DEFAULT_MAX_TOKENS
    assert recording_client["messages"] == [{"role": "user", "content": "你好"}]


def test_chat_overrides_win(recording_client):
    llm.chat([{"role": "user", "content": "x"}], model="deepseek-reasoner", temperature=0.7)
    assert recording_client["model"] == "deepseek-reasoner"
    assert recording_client["temperature"] == 0.7


def test_complete_accepts_plain_prompt(recording_client):
    llm.complete("一句话解释过拟合", system="你是助教")
    msgs = recording_client["messages"]
    assert msgs[0] == {"role": "system", "content": "你是助教"}
    assert msgs[-1]["role"] == "user"


def test_chat_completion_exposes_usage(recording_client):
    resp = llm.chat_completion([{"role": "user", "content": "x"}])
    # M3 之后要按题统计 token 成本，usage 必须原样透出
    assert resp.usage.total_tokens == 18


def test_empty_content_raises(monkeypatch):
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: _fake_completion(None))
        )
    )
    monkeypatch.setattr(llm, "get_client", lambda: client)
    with pytest.raises(RuntimeError):
        llm.chat([{"role": "user", "content": "x"}])


def test_get_client_is_singleton(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "sk-unit-test-value")
    llm.reset_client()
    a = llm.get_client()
    b = llm.get_client()
    assert a is b
    assert str(a.base_url).rstrip("/") == config.BASE_URL.rstrip("/")
    llm.reset_client()


def test_get_client_without_key_raises(monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    llm.reset_client()
    with pytest.raises(RuntimeError):
        llm.get_client()
    llm.reset_client()
