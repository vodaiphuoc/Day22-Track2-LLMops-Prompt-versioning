from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

# ── 1. Environment setup ────────────────────────────────────────────────────
# Load .env if python-dotenv is installed; otherwise continue without it.
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

load_dotenv()

# Use shared config helpers (keeps keys out of code).
from config import DeterministicEmbeddings, configure_langsmith_env, load_config, print_config  # noqa: E402
from qa_pairs import QA_PAIRS  # noqa: E402
from rag_utils import build_faiss_vectorstore, load_kb_text, split_text  # noqa: E402

cfg = load_config()
configure_langsmith_env(cfg)

# ── 2. LangChain + LangSmith imports ────────────────────────────────────────
# LangSmith decorator is optional; provide a no-op fallback.
try:
    from langsmith import traceable
except Exception:
    def traceable(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
except Exception:
    ChatOpenAI = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore
    StrOutputParser = None  # type: ignore
    RunnableLambda = None  # type: ignore
    RunnablePassthrough = None  # type: ignore


# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
def _make_embeddings():
    if cfg.mock_mode or OpenAIEmbeddings is None:
        return DeterministicEmbeddings(dim=384)
    return OpenAIEmbeddings(
        model=cfg.openai_embedding_model,
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
    )


def _make_llm():
    if cfg.mock_mode or ChatOpenAI is None:
        return None
    return ChatOpenAI(
        model=cfg.openai_model,
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
        temperature=0,
    )


# ── 4. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.

    Steps:
      a) Read your dataset
      b) Split text with RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
      c) Build FAISS index
      d) Return the vectorstore
    """
    text = load_kb_text("data/knowledge_base.txt")
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"Split into {len(chunks)} chunks")

    embeddings = _make_embeddings()
    vectorstore = build_faiss_vectorstore(chunks, embeddings)
    return vectorstore


# ── 5. RAG prompt template ──────────────────────────────────────────────────
def _make_prompt():
    if ChatPromptTemplate is None:
        return None
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer using ONLY the provided context.\n"
                "If the context does not contain the answer, say: \"I don't have enough information.\"\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )


# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL (pipe operator).

    Returns: (chain, retriever)
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = _make_prompt()
    llm = _make_llm()

    def format_docs(docs) -> str:
        return "\n\n".join(d.page_content for d in docs)

    # If LangChain isn't available or we're in mock mode, fall back to a simple callable.
    if cfg.mock_mode or prompt is None or llm is None or StrOutputParser is None or RunnablePassthrough is None:
        def chain_invoke(question: str) -> str:
            docs = retriever.invoke(question)
            ctx = "\n\n".join(d.page_content for d in docs)
            first = (ctx.strip().splitlines() or [""])[0].strip()
            return f"(mock) {first}" if first else "(mock) I don't have enough information."
        return chain_invoke, retriever

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 7. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    """
    return chain.invoke(question) if hasattr(chain, "invoke") else chain(question)


# ── 8. Sample questions (50 total) ──────────────────────────────────────────
SAMPLE_QUESTIONS: List[str] = [x["question"] for x in QA_PAIRS]


# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)
    print_config(cfg)

    vectorstore = build_vectorstore()
    chain, _retriever = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {str(answer)[:200]}\n")

    if (not cfg.mock_mode) and os.environ.get("LANGCHAIN_API_KEY"):
        print(f"✅ {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.environ.get('LANGCHAIN_PROJECT')}'")
        print("   Open https://smith.langchain.com to view traces.")
    else:
        print(f"✅ Completed {len(SAMPLE_QUESTIONS)} questions (mock/offline mode).")


if __name__ == "__main__":
    main()

