"""共享 fixture。

原则（CLAUDE.md §7）：默认路径下的测试一律不碰真实 API——CI 上没有 key，
而且真实调用既费钱又不确定。凡需要 LLM 的地方都在这里打桩。
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def stub_chat(monkeypatch):
    """把 llm.chat 换成可控桩，返回 (设置回答的函数, 调用记录 list)。"""
    from src.agent import llm

    calls: list[dict] = []
    box = {"reply": "STUB_ANSWER"}

    def _fake_chat(messages, **kwargs):
        calls.append({"messages": messages, "kwargs": kwargs})
        return box["reply"]

    monkeypatch.setattr(llm, "chat", _fake_chat)

    def _set(reply: str) -> None:
        box["reply"] = reply

    return _set, calls


# ===========================================================================
# M2：RAG / ReAct 用的桩
#
# 两条红线（CLAUDE.md §7）：测试不下载 embedding 模型、不发 LLM 请求。
# 所以 embedder 用确定性哈希向量，LLM 用脚本化的假 completion。
# ===========================================================================


class FakeEmbedder:
    """确定性假 embedder：靠字符 n-gram 哈希投影到固定维度。

    不追求语义质量，只需要满足两点：同样文本永远得到同样向量；
    文本共享的字越多，向量越接近——这样 top-k 的相对顺序才有意义，
    检索测试才能断言"含关键词的 chunk 排在前面"。
    """

    dim = 64

    def __init__(self, dim: int | None = None):
        if dim is not None:
            self.dim = dim

    def encode(self, texts, is_query: bool = False):
        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for ch in text:
                # 用 crc32 而不是内置 hash()：后者对字符串逐进程加盐，
                # 会让"同样输入同样输出"这条断言在跨进程时失效。
                out[row, zlib.crc32(ch.encode("utf-8")) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def tiny_corpus(tmp_path):
    """三篇短文档，内容彼此区分度足够高，便于断言检索命中了哪一篇。"""
    docs = {
        "过拟合.md": (
            "# 过拟合\n\n"
            "过拟合指模型把训练集里的噪声也学了进去，训练误差很低但泛化误差很高。\n"
            "常见缓解手段包括正则化、早停、增加数据量。\n"
        ),
        "psi.md": (
            "# 群体稳定性指标\n\n"
            "PSI 用于衡量同一变量在两个样本集上的分布差异，常用于监控模型上线后的漂移。\n"
            "经验阈值：小于零点一认为稳定，大于零点二五认为发生显著漂移。\n"
        ),
        "kfold.md": (
            "# 交叉验证\n\n"
            "GroupKFold 在划分折时保证同一组的样本不会同时出现在训练集和验证集，用来防止数据泄漏。\n"
        ),
    }
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in docs.items():
        (corpus / name).write_text(text, encoding="utf-8")
    return corpus


@pytest.fixture
def scripted_llm(monkeypatch):
    """把 llm.chat_completion 换成按脚本依次返回的桩。

    用法：script(msg1, msg2, ...)，每次调用弹出一条。
    每条用 tests.helpers 里的 assistant_message / tool_call_message 构造。
    """
    from src.agent import llm

    queue: list = []
    calls: list[dict] = []

    def _fake(messages, **kwargs):
        calls.append({"messages": [dict(m) for m in messages], "kwargs": kwargs})
        if not queue:
            raise AssertionError("脚本用光了：Agent 调用 LLM 的次数超出预期")
        return queue.pop(0)

    monkeypatch.setattr(llm, "chat_completion", _fake)

    def _script(*responses):
        queue.extend(responses)

    return _script, calls
