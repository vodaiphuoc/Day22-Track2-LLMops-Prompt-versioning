from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional


def _truthy(s: Optional[str]) -> bool:
    return str(s or "").strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppConfig:
    # LangSmith
    langsmith_api_key: str
    langchain_project: str
    langchain_endpoint: str
    langchain_tracing_v2: bool

    # OpenAI-compatible
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    openai_embedding_model: str

    # Behavior
    mock_mode: bool  # When true, avoid network calls and use local fallbacks.


def load_config() -> AppConfig:
    """
    Loads config from environment variables (optionally via python-dotenv in callers).
    The code is written to run in two modes:
      - Real mode: OPENAI_API_KEY present (and optionally LANGSMITH_API_KEY) -> real calls + traces
      - Mock mode: missing OPENAI_API_KEY -> no network calls, deterministic local fallbacks
    """
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "").strip() or os.getenv("LANGCHAIN_API_KEY", "").strip()
    langchain_project = os.getenv("LANGCHAIN_PROJECT", "").strip() or "day22-langsmith-prompt-versioning"
    langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "").strip() or "https://api.smith.langchain.com"
    langchain_tracing_v2 = _truthy(os.getenv("LANGCHAIN_TRACING_V2", "true"))

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
    openai_model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
    openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip() or "text-embedding-3-small"

    mock_mode = _truthy(os.getenv("MOCK_MODE")) or (openai_api_key == "")

    return AppConfig(
        langsmith_api_key=langsmith_api_key,
        langchain_project=langchain_project,
        langchain_endpoint=langchain_endpoint,
        langchain_tracing_v2=langchain_tracing_v2,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_embedding_model=openai_embedding_model,
        mock_mode=mock_mode,
    )


class DeterministicEmbeddings:
    """
    Minimal embeddings implementation for offline runs.

    FAISS needs an object that implements:
      - embed_documents(list[str]) -> list[list[float]]
      - embed_query(str) -> list[float]
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_to_vec(self, text: str) -> List[float]:
        # Stable, deterministic vector. Not meaningful semantically, but keeps the pipeline runnable.
        h = hashlib.sha256(text.encode("utf-8")).digest()
        out = []
        while len(out) < self.dim:
            h = hashlib.sha256(h).digest()
            out.extend(((b - 128) / 128.0) for b in h)
        return out[: self.dim]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vec(text)


def configure_langsmith_env(cfg: AppConfig) -> None:
    """
    Set LangSmith env vars *before* importing LangChain in the caller.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if cfg.langchain_tracing_v2 else "false"
    if cfg.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = cfg.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = cfg.langchain_endpoint


def print_config(cfg: AppConfig) -> None:
    mode = "MOCK (offline)" if cfg.mock_mode else "REAL (network)"
    print("✅ Config loaded successfully")
    print(f"   Mode            : {mode}")
    print(f"   LangSmith project: {cfg.langchain_project}")
    print(f"   OpenAI endpoint  : {cfg.openai_base_url}")
    print(f"   Default LLM model: {cfg.openai_model}")
    print(f"   Embedding model  : {cfg.openai_embedding_model}")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    cfg = load_config()
    print_config(cfg)


if __name__ == "__main__":
    main()
