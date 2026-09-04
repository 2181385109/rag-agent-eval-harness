"""RAG 层：加载语料 -> 切分 -> BGE 向量化 -> FAISS 索引 -> 检索。

贯穿本模块的一条纪律是**确定性**。黄金集里的 expected_doc_ids 是人工标注的，
一旦切分边界或 doc_id 随运行漂移，recall@k 就变成了噪声。所以：
  - 语料按文件名排序加载，不依赖文件系统枚举顺序；
  - doc_id 只由 (文件名, 该文件内的第几块) 决定，新增/删除别的文件不影响它；
  - 切分参数集中在 config，改动即视为一次评测口径变更。

命令行：
    python -m src.agent.rag --list          # 列出 doc_id -> 出处，标注黄金集时用
    python -m src.agent.rag --build         # 构建索引并落盘到 config.INDEX_DIR
    python -m src.agent.rag --query "问题"   # 用落盘的索引真实检索一次
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from src import config
from src.agent.trace import RetrievedChunk

CHUNKS_FILENAME = "chunks.jsonl"
INDEX_FILENAME = "faiss.index"


class Document(BaseModel):
    """一篇语料原文。"""

    source: str  # 相对 corpus/ 的文件名
    text: str


class Chunk(BaseModel):
    """切分后的一块，是检索和标注的最小单位。"""

    doc_id: str
    source: str
    chunk_index: int
    text: str


def make_doc_id(source: str, chunk_index: int) -> str:
    """稳定的 doc_id。

    只吃 (文件名, 块序号)：语料目录里增删别的文件不会让已有 id 漂移，
    旧的黄金集标注因此仍然有效。文件名走哈希是为了避开中文/空格/括号，
    真正给人看的出处在 Chunk.source 里，`--list` 会一并打出来。
    """
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{digest}_{chunk_index:04d}"


def load_corpus(corpus_dir: Path | str | None = None) -> list[Document]:
    """读取语料目录下的 .md / .txt，按文件名排序返回。"""
    root = Path(corpus_dir) if corpus_dir is not None else config.CORPUS_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"语料目录不存在：{root}")

    docs: list[Document] = []
    for path in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root))):
        if not path.is_file() or path.suffix.lower() not in config.CORPUS_SUFFIXES:
            continue
        if path.name in config.CORPUS_EXCLUDE:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if text.strip():
            source = str(path.relative_to(root)).replace("\\", "/")
            docs.append(Document(source=source, text=text))

    if not docs:
        raise FileNotFoundError(
            f"{root} 里没有可用语料。请放入 .md / .txt 文档后重试（见 corpus/README.md）。"
        )
    return docs


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=list(config.SPLIT_SEPARATORS),
        length_function=len,
        is_separator_regex=False,
    )


def split_document(doc: Document) -> list[Chunk]:
    """把一篇文档切成块，块序号从 0 开始且只在本文档内计数。"""
    pieces = [p for p in _splitter().split_text(doc.text) if p.strip()]
    return [
        Chunk(
            doc_id=make_doc_id(doc.source, i),
            source=doc.source,
            chunk_index=i,
            text=piece,
        )
        for i, piece in enumerate(pieces)
    ]


def build_chunks(corpus_dir: Path | str | None = None) -> list[Chunk]:
    """加载并切分整个语料目录。"""
    return [chunk for doc in load_corpus(corpus_dir) for chunk in split_document(doc)]


# --------------------------------------------------------------- Embedding
class Embedder(Protocol):
    """检索侧只依赖这个协议，测试才能用假 embedder，不必下载模型。"""

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class BGEEmbedder:
    """sentence-transformers 本地加载 BGE，模型在首次 encode 时才下载。"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        # BGE 官方用法：查询侧加指令前缀，文档侧不加。是否真的有增益由 M3 的 recall@k 决定。
        use_prefix = is_query and config.USE_QUERY_INSTRUCTION
        payload = [config.QUERY_INSTRUCTION + t for t in texts] if use_prefix else list(texts)
        vecs = self._load().encode(payload, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ----------------------------------------------------------------- 检索器
class Retriever:
    """FAISS 内积索引（向量已 L2 归一化，等价于余弦相似度）。"""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, index=None):
        if not chunks:
            raise ValueError("chunks 为空，无法建索引")
        self.chunks = chunks
        self.embedder = embedder
        self.index = index if index is not None else self._build_index(chunks, embedder)

    @staticmethod
    def _build_index(chunks: list[Chunk], embedder: Embedder):
        import faiss

        vecs = _l2_normalize(embedder.encode([c.text for c in chunks], is_query=False))
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return index

    @classmethod
    def build(
        cls, corpus_dir: Path | str | None = None, embedder: Embedder | None = None
    ) -> "Retriever":
        return cls(build_chunks(corpus_dir), embedder or BGEEmbedder())

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        k = min(k or config.RETRIEVE_TOP_K, len(self.chunks))

        qvec = _l2_normalize(self.embedder.encode([query.strip()], is_query=True))
        scores, indices = self.index.search(qvec, k)

        hits: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS 结果不足 k 时用 -1 补位
                continue
            chunk = self.chunks[int(idx)]
            hits.append(
                RetrievedChunk(
                    doc_id=chunk.doc_id,
                    source=chunk.source,
                    score=float(score),
                    text=chunk.text,
                )
            )
        return hits

    # -------------------------------------------------------------- 落盘
    def save(self, directory: Path | str | None = None) -> Path:
        import faiss

        target = Path(directory) if directory is not None else config.INDEX_DIR
        target.mkdir(parents=True, exist_ok=True)
        with (target / CHUNKS_FILENAME).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        faiss.write_index(self.index, str(target / INDEX_FILENAME))
        return target

    @classmethod
    def load(
        cls, directory: Path | str | None = None, embedder: Embedder | None = None
    ) -> "Retriever":
        import faiss

        target = Path(directory) if directory is not None else config.INDEX_DIR
        chunks_path = target / CHUNKS_FILENAME
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"索引不存在：{target}。先跑 python -m src.agent.rag --build"
            )
        chunks = [
            Chunk.model_validate_json(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        index = faiss.read_index(str(target / INDEX_FILENAME))
        return cls(chunks, embedder or BGEEmbedder(), index=index)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="RAG 语料与索引工具")
    parser.add_argument("--list", action="store_true", help="列出 doc_id -> 出处（标注黄金集用）")
    parser.add_argument("--build", action="store_true", help="构建 FAISS 索引并落盘")
    parser.add_argument("--query", type=str, help="用落盘的索引检索一次")
    parser.add_argument("--k", type=int, default=config.RETRIEVE_TOP_K)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    newline = chr(10)

    if args.list:
        chunks = build_chunks()
        sources = {c.source for c in chunks}
        print(f"共 {len(chunks)} 块，来自 {len(sources)} 篇文档")
        print()
        for c in chunks:
            preview = c.text.replace(newline, " ")[:60]
            print(f"{c.doc_id}  {c.source}#{c.chunk_index}  {preview}...")
        return 0

    if args.build:
        retriever = Retriever.build()
        path = retriever.save()
        print(f"已索引 {len(retriever.chunks)} 块 -> {path}")
        return 0

    if args.query:
        for hit in Retriever.load().retrieve(args.query, k=args.k):
            print(f"[{hit.score:.4f}] {hit.doc_id}  {hit.source}")
            print(f"    {hit.text.replace(newline, ' ')[:120]}...")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
