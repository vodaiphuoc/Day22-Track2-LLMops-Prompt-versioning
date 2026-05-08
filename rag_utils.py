from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class Retrieved:
    contexts: List[str]
    context_str: str


def load_kb_text(path: str | Path = "data/knowledge_base.txt") -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Knowledge base not found: {p.resolve()}")
    return p.read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)
    except Exception:
        # Pure-Python fallback (chunk by characters).
        chunks: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            chunks.append(text[i : i + chunk_size])
            i += max(1, chunk_size - chunk_overlap)
        return chunks


def build_faiss_vectorstore(chunks: List[str], embeddings):
    try:
        from langchain_community.vectorstores import FAISS

        return FAISS.from_texts(chunks, embedding=embeddings)
    except Exception:
        return SimpleVectorStore.from_texts(chunks, embeddings)


def retrieve(retriever, question: str) -> Retrieved:
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    return Retrieved(contexts=contexts, context_str="\n\n".join(contexts))


@dataclass(frozen=True)
class Doc:
    page_content: str


class SimpleRetriever:
    def __init__(self, store: "SimpleVectorStore", k: int = 3):
        self._store = store
        self._k = k

    def invoke(self, query: str) -> List[Doc]:
        return self._store.similarity_search(query, k=self._k)


class SimpleVectorStore:
    """
    Minimal vector store + retriever fallback for environments without LangChain/FAISS.
    Uses deterministic embeddings and brute-force cosine similarity.
    """

    def __init__(self, texts: List[str], embeddings):
        self._texts = texts
        self._emb = embeddings
        self._vecs = embeddings.embed_documents(texts)

    @classmethod
    def from_texts(cls, texts: List[str], embeddings) -> "SimpleVectorStore":
        return cls(texts=texts, embeddings=embeddings)

    def as_retriever(self, search_kwargs=None) -> SimpleRetriever:
        search_kwargs = search_kwargs or {}
        k = int(search_kwargs.get("k", 3))
        return SimpleRetriever(self, k=k)

    def similarity_search(self, query: str, k: int = 3) -> List[Doc]:
        qv = self._emb.embed_query(query)
        scored = [(self._cos(qv, v), t) for v, t in zip(self._vecs, self._texts)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [Doc(page_content=t) for _, t in scored[:k]]

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)
