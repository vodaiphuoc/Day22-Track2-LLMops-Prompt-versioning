
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

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

# Optional deps
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    ChatOpenAI = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore
    StrOutputParser = None  # type: ignore

REPORT_PATH = Path("data/ragas_report.json")


# ── Prompt templates (same as step 2) ───────────────────────────────────────
SYSTEM_V1 = (
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)

SYSTEM_V2 = (
    "Provide a structured, accurate answer using ONLY the provided context.\n"
    "Answer in 3-5 sentences with a clear structure (definition -> key points -> caveat).\n"
    "If the context lacks information, say so explicitly.\n\n"
    "Context:\n{context}"
)

if ChatPromptTemplate is not None:
    PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])
    PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])
else:
    PROMPT_V1 = None
    PROMPT_V2 = None

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── Build vectorstore (reuse logic from step 1) ─────────────────────────────
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


# ── Run RAG and capture outputs + contexts ──────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    ctx_str = "\n\n".join(contexts)

    if cfg.mock_mode or llm is None or prompt is None or StrOutputParser is None or not hasattr(prompt, "__or__"):
        first = (ctx_str.strip().splitlines() or [""])[0].strip()
        answer = f"(mock) {first}" if first else "(mock) I don't have enough information."
    else:
        answer = (prompt | llm | StrOutputParser()).invoke({"context": ctx_str, "question": question})

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = None
    if (not cfg.mock_mode) and ChatOpenAI is not None:
        llm = ChatOpenAI(
            model=cfg.openai_model,
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            temperature=0,
        )

    prompt = PROMPTS[prompt_version]
    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append(
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": out["answer"],
                "contexts": out["contexts"],  # list[str]
            }
        )
        print(f"  [{i:02d}/50] {qa['question'][:60]}")
    return results


# ── Evaluation helpers ──────────────────────────────────────────────────────
def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def evaluate_mock(rag_results: list) -> dict:
    # Lightweight overlap heuristics to keep offline runs possible.
    def tok(s: str) -> set:
        return {t.strip(".,:;()[]{}\"'").lower() for t in s.split() if t.strip()}

    faith, rel, rec, prec = [], [], [], []
    for r in rag_results:
        q = tok(r["question"])
        a = tok(str(r["answer"]))
        ref = tok(r["reference"])
        ctx = tok(" ".join(r["contexts"]))

        faith.append(len(a & ctx) / max(1, len(a)))
        rel.append(len(q & a) / max(1, len(q)))
        rec.append(len(ref & ctx) / max(1, len(ref)))
        prec.append(len(ctx & (q | ref)) / max(1, len(ctx)))

    return {
        "faithfulness": float(_mean(faith)),
        "answer_relevancy": float(_mean(rel)),
        "context_recall": float(_mean(rec)),
        "context_precision": float(_mean(prec)),
    }


def evaluate_ragas(rag_results: list) -> dict:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=str(r["answer"]),
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    dataset = EvaluationDataset(samples=samples)

    llm_eval = ChatOpenAI(
        model=cfg.openai_model,
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
        temperature=0,
    )
    emb_eval = OpenAIEmbeddings(
        model=cfg.openai_embedding_model,
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    out = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        vals = [v for v in result[key] if v is not None]
        out[key] = float(_mean(vals)) if vals else 0.0
    return out


def run_eval(rag_results: list, version: str) -> dict:
    print(f"\n📐 Running evaluation for prompt {version} ...")
    if cfg.mock_mode or ChatOpenAI is None:
        scores = evaluate_mock(rag_results)
    else:
        scores = evaluate_ragas(rag_results)

    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:18s}: {v:.4f}{star}")
    return scores


def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)
    print_config(cfg)

    vectorstore = build_vectorstore()
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_eval(v1_results, "v1")
    v2_scores = run_eval(v2_results, "v2")

    print("\nComparison (mean scores):")
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "V1" if s1 >= s2 else "V2"
        print(f"  {metric:18s}: V1={s1:.4f}  V2={s2:.4f}  winner={winner}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "mock" if (cfg.mock_mode or ChatOpenAI is None) else "real",
        "langchain_project": cfg.langchain_project,
        "scores": {"v1": v1_scores, "v2": v2_scores},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n✅ Saved report to {REPORT_PATH.as_posix()}")


if __name__ == "__main__":
    main()

