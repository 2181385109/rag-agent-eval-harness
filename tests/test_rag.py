"""RAG 层的验收口径。

最要紧的一条是**确定性**：黄金集里的 expected_doc_ids 是人工标注的，
如果切分或 doc_id 会随运行漂移，整套 recall@k 就没有意义了。
所有测试用 FakeEmbedder，不下载任何模型。
"""
from __future__ import annotations

import pytest

from src import config
from src.agent import rag


# --------------------------------------------------------------- 语料加载
def test_load_corpus_reads_md_and_txt(tiny_corpus):
    docs = rag.load_corpus(tiny_corpus)
    assert {d.source for d in docs} == {"过拟合.md", "psi.md", "kfold.md"}
    assert all(d.text.strip() for d in docs)


def test_load_corpus_is_sorted_by_source(tiny_corpus):
    """按文件名排序，保证 chunk 顺序和 doc_id 分配不受文件系统枚举顺序影响。"""
    docs = rag.load_corpus(tiny_corpus)
    assert [d.source for d in docs] == sorted(d.source for d in docs)


def test_load_corpus_skips_readme(tmp_path):
    """corpus/README.md 是写给人看的说明，不是语料，不能被检索到。"""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "README.md").write_text("这里放语料的说明文字", encoding="utf-8")
    (corpus / "真语料.md").write_text("有效内容", encoding="utf-8")
    assert [d.source for d in rag.load_corpus(corpus)] == ["真语料.md"]


def test_empty_corpus_raises(tmp_path):
    empty = tmp_path / "corpus"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        rag.build_chunks(empty)


# ----------------------------------------------------------------- 切分
def test_chunks_respect_size_limit(tiny_corpus):
    for chunk in rag.build_chunks(tiny_corpus):
        assert len(chunk.text) <= config.CHUNK_SIZE


def test_chunking_is_deterministic(tiny_corpus):
    a = rag.build_chunks(tiny_corpus)
    b = rag.build_chunks(tiny_corpus)
    assert [(c.doc_id, c.text) for c in a] == [(c.doc_id, c.text) for c in b]


def test_doc_ids_are_unique(tiny_corpus):
    ids = [c.doc_id for c in rag.build_chunks(tiny_corpus)]
    assert len(ids) == len(set(ids))


def test_doc_id_survives_adding_another_file(tiny_corpus):
    """新增语料不能改变已有 chunk 的 doc_id，否则旧标注全部作废。"""
    before = {c.doc_id: c.text for c in rag.build_chunks(tiny_corpus)}
    (tiny_corpus / "新增文档.md").write_text("一段全新的内容，和其它文档无关。", encoding="utf-8")
    after = {c.doc_id: c.text for c in rag.build_chunks(tiny_corpus)}
    for doc_id, text in before.items():
        assert doc_id in after, f"新增文件后 {doc_id} 消失了"
        assert after[doc_id] == text, f"{doc_id} 的内容变了"


def test_doc_id_encodes_source_and_index(tiny_corpus):
    for chunk in rag.build_chunks(tiny_corpus):
        assert chunk.doc_id == rag.make_doc_id(chunk.source, chunk.chunk_index)


def test_chunk_carries_source_for_annotation(tiny_corpus):
    """标注黄金集时要靠 source 认出这段来自哪篇文档。"""
    chunks = rag.build_chunks(tiny_corpus)
    assert all(c.source.endswith(".md") for c in chunks)


# ----------------------------------------------------------------- 检索
def test_retriever_returns_top_k(tiny_corpus, fake_embedder):
    r = rag.Retriever.build(tiny_corpus, embedder=fake_embedder)
    hits = r.retrieve("过拟合", k=2)
    assert len(hits) == 2
    assert [h.doc_id for h in hits] == list(dict.fromkeys(h.doc_id for h in hits))


def test_retriever_scores_are_descending(tiny_corpus, fake_embedder):
    hits = rag.Retriever.build(tiny_corpus, embedder=fake_embedder).retrieve("正则化", k=3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_retriever_finds_the_relevant_document(tiny_corpus, fake_embedder):
    """FakeEmbedder 按字符重合度打分，所以整段原文查询必须命中它自己那篇。"""
    r = rag.Retriever.build(tiny_corpus, embedder=fake_embedder)
    hits = r.retrieve("GroupKFold 在划分折时保证同一组的样本不会同时出现在训练集和验证集", k=1)
    assert hits[0].source == "kfold.md"


def test_retrieve_k_larger_than_corpus_is_clamped(tiny_corpus, fake_embedder):
    r = rag.Retriever.build(tiny_corpus, embedder=fake_embedder)
    hits = r.retrieve("任意查询", k=999)
    assert len(hits) == len(r.chunks)


def test_empty_query_rejected(tiny_corpus, fake_embedder):
    r = rag.Retriever.build(tiny_corpus, embedder=fake_embedder)
    with pytest.raises(ValueError):
        r.retrieve("   ")


def test_retrieval_is_deterministic(tiny_corpus, fake_embedder):
    r = rag.Retriever.build(tiny_corpus, embedder=fake_embedder)
    first = [(h.doc_id, round(h.score, 6)) for h in r.retrieve("PSI 漂移", k=3)]
    second = [(h.doc_id, round(h.score, 6)) for h in r.retrieve("PSI 漂移", k=3)]
    assert first == second


def test_query_gets_bge_instruction_prefix(tiny_corpus, monkeypatch):
    """BGE 要求查询侧加指令前缀、文档侧不加——这个区别必须真的传下去。"""
    seen = {"query": [], "doc": []}

    class SpyEmbedder:
        dim = 8

        def encode(self, texts, is_query: bool = False):
            import numpy as np

            seen["query" if is_query else "doc"].extend(texts)
            return np.ones((len(texts), self.dim), dtype="float32") / self.dim**0.5

    r = rag.Retriever.build(tiny_corpus, embedder=SpyEmbedder())
    r.retrieve("什么是 PSI", k=1)
    assert seen["query"] == ["什么是 PSI"]
    assert not any(t.startswith(config.QUERY_INSTRUCTION) for t in seen["doc"])


# ------------------------------------------------------------- 索引落盘
def test_index_save_and_load_round_trip(tiny_corpus, fake_embedder, tmp_path):
    built = rag.Retriever.build(tiny_corpus, embedder=fake_embedder)
    target = tmp_path / "idx"
    built.save(target)

    loaded = rag.Retriever.load(target, embedder=fake_embedder)
    assert [c.doc_id for c in loaded.chunks] == [c.doc_id for c in built.chunks]
    assert [h.doc_id for h in loaded.retrieve("过拟合", k=2)] == [
        h.doc_id for h in built.retrieve("过拟合", k=2)
    ]
