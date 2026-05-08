from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

load_dotenv()

from config import DeterministicEmbeddings, configure_langsmith_env, load_config, print_config  # noqa: E402
from qa_pairs import QA_PAIRS  # noqa: E402
from rag_utils import build_faiss_vectorstore, load_kb_text, split_text  # noqa: E402

cfg = load_config()
configure_langsmith_env(cfg)

# ── Optional deps (LangChain / LangSmith) ───────────────────────────────────
try:
    from langsmith import Client, traceable
except Exception:
    Client = None  # type: ignore

    def traceable(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    ChatOpenAI = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore
    StrOutputParser = None  # type: ignore


# ── 2. Define two prompt templates ──────────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Use ONLY the provided context.\n"
    "2. Identify key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)

if ChatPromptTemplate is not None:
    PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])
    PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])
else:
    PROMPT_V1 = None
    PROMPT_V2 = None

# Prompt Hub names (stable default; you can override via env PROMPT_V1_NAME / PROMPT_V2_NAME)
PROMPT_V1_NAME = os.getenv("PROMPT_V1_NAME", f"{cfg.langchain_project}-prompt-v1")
PROMPT_V2_NAME = os.getenv("PROMPT_V2_NAME", f"{cfg.langchain_project}-prompt-v2")

EVIDENCE_LOG = Path("evidence/02_ab_routing_log.txt")


# ── 3. Push prompts to LangSmith Prompt Hub ─────────────────────────────────
def push_prompts_to_hub(client):
    if PROMPT_V1 is None or PROMPT_V2 is None:
        print("ℹ️  LangChain not installed; cannot push prompts. Skipping.")
        return

    for name, obj, desc in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 – concise answers"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 – structured answers"),
    ]:
        try:
            url = client.push_prompt(name, object=obj, description=desc)
            print(f"✅ Pushed {name} → {url}")
        except Exception as e:
            print(f"⚠️  Push failed for {name}: {e}")


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    prompts = {PROMPT_V1_NAME: PROMPT_V1, PROMPT_V2_NAME: PROMPT_V2}
    for name in [PROMPT_V1_NAME, PROMPT_V2_NAME]:
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"↓ Pulled '{name}' from Hub")
        except Exception as e:
            print(f"ℹ️  Using local fallback for '{name}' (pull failed: {e})")
    return prompts


# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Build vectorstore (reuse from step 1) ────────────────────────────────
def build_vectorstore():
    text = load_kb_text("data/knowledge_base.txt")
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)

    if cfg.mock_mode or OpenAIEmbeddings is None:
        embeddings = DeterministicEmbeddings(dim=384)
    else:
        embeddings = OpenAIEmbeddings(
            model=cfg.openai_embedding_model,
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
        )

    return build_faiss_vectorstore(chunks, embeddings)


# ── 7. Traced A/B query function ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    if cfg.mock_mode or llm is None or prompt is None or StrOutputParser is None or not hasattr(prompt, "__or__"):
        first = (context.strip().splitlines() or [""])[0].strip()
        answer = f"(mock) {first}" if first else "(mock) I don't have enough information."
    else:
        answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    return {"question": question, "answer": answer, "version": version}


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)
    print_config(cfg)

    client = None
    if (not cfg.mock_mode) and Client is not None and os.environ.get("LANGCHAIN_API_KEY"):
        client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

    if client is not None:
        push_prompts_to_hub(client)
        prompts = pull_prompts_from_hub(client)
    else:
        if not os.environ.get("LANGCHAIN_API_KEY"):
            print("ℹ️  LANGSMITH/LANGCHAIN API key not set; skipping Prompt Hub push/pull.")
        if Client is None:
            print("ℹ️  langsmith not installed; skipping Prompt Hub push/pull.")
        prompts = {PROMPT_V1_NAME: PROMPT_V1, PROMPT_V2_NAME: PROMPT_V2}

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = None
    if (not cfg.mock_mode) and ChatOpenAI is not None:
        llm = ChatOpenAI(
            model=cfg.openai_model,
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            temperature=0,
        )

    EVIDENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    counts = {"v1": 0, "v2": 0}

    questions = [x["question"] for x in QA_PAIRS]
    for i, question in enumerate(questions, 1):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        counts[version_tag] += 1

        prompt = prompts.get(version_key)
        _ = ask_ab(retriever, llm, prompt, question, version_tag)
        line = f"[{i:02d}] [prompt-{version_tag}] request_id={request_id} q={question}"
        print(line)
        lines.append(line)

    summary = f"\nSummary: v1={counts['v1']} v2={counts['v2']}\n"
    print(summary.strip())
    EVIDENCE_LOG.write_text("\n".join(lines) + summary, encoding="utf-8")
    print(f"✅ Wrote A/B routing log to {EVIDENCE_LOG.as_posix()}")


if __name__ == "__main__":
    main()

