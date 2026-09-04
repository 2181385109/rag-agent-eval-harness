"""DeepSeek 客户端封装。

DeepSeek 的端点与 OpenAI 兼容，因此直接用 openai SDK，把 base_url 指过去即可。
本模块是整个仓库**唯一**对外发请求的地方（CLAUDE.md §1.7），
其它模块一律通过这里调模型，方便统一打桩、统一记 token。
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from src import config

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """返回进程内单例客户端。key 缺失时由 config.get_api_key 抛错。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.get_api_key(),
            base_url=config.BASE_URL,
            timeout=config.REQUEST_TIMEOUT_S,
            max_retries=config.MAX_RETRIES,
        )
    return _client


def reset_client() -> None:
    """丢弃单例（换 key 或测试隔离时用）。"""
    global _client
    _client = None


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
):
    """发一次 chat 请求，返回**原始** ChatCompletion（保留 usage 等元信息）。"""
    return get_client().chat.completions.create(
        model=model or config.MODEL_NAME,
        messages=messages,
        temperature=config.DEFAULT_TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens or config.DEFAULT_MAX_TOKENS,
        **kwargs,
    )


def chat(messages: list[dict[str, Any]], **kwargs: Any) -> str:
    """发一次 chat 请求，返回纯文本回答。"""
    resp = chat_completion(messages, **kwargs)
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("模型返回了空内容（choices[0].message.content 为空）")
    return content


def complete(prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
    """单轮便捷入口：给一段 prompt，拿一段文本。"""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, **kwargs)
