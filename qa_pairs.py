"""
50 QA pairs used for this lab.

Each pair contains:
  - question: user question
  - reference: a concise ground-truth answer grounded in data/knowledge_base.txt
"""

QA_PAIRS = [
    {
        "question": "What are the three main types of machine learning?",
        "reference": "The three main types are supervised learning, unsupervised learning, and reinforcement learning.",
    },
    {
        "question": "What is overfitting in machine learning?",
        "reference": "Overfitting is when a model learns the training data (including noise) too well and generalizes poorly to new data.",
    },
    {
        "question": "Explain the bias-variance tradeoff.",
        "reference": "High bias leads to underfitting and high variance leads to overfitting; good models balance both to minimize error.",
    },
    {
        "question": "How does regularization prevent overfitting?",
        "reference": "Regularization (e.g., L1/L2) penalizes model complexity, discouraging overly complex solutions that overfit.",
    },
    {
        "question": "What is cross-validation?",
        "reference": "Cross-validation (such as k-fold) splits data into multiple folds to estimate performance more reliably.",
    },
    {
        "question": "What is backpropagation?",
        "reference": "Backpropagation computes gradients of the loss with respect to parameters using the chain rule.",
    },
    {
        "question": "What are Convolutional Neural Networks primarily used for?",
        "reference": "CNNs are primarily used for grid-like data such as images.",
    },
    {
        "question": "How do LSTM networks address the vanishing gradient problem?",
        "reference": "LSTMs use gating mechanisms to control information flow, helping gradients propagate over long sequences.",
    },
    {
        "question": "What activation functions are commonly used in neural networks?",
        "reference": "Common activation functions include ReLU, sigmoid, and tanh.",
    },
    {
        "question": "What is the role of pooling layers in CNNs?",
        "reference": "Pooling reduces spatial dimensions, lowering compute while retaining important features.",
    },
    {
        "question": "What is the transformer architecture?",
        "reference": "Transformers use self-attention and process tokens in parallel (introduced in 2017).",
    },
    {
        "question": "What are word embeddings?",
        "reference": "Word embeddings represent words as dense vectors where semantically similar words are close in vector space.",
    },
    {
        "question": "What is transfer learning in NLP?",
        "reference": "Transfer learning often means pre-training on a large corpus then fine-tuning on a downstream task.",
    },
    {
        "question": "How does BERT handle language understanding?",
        "reference": "BERT is a bidirectional transformer trained with masked language modeling (and historically next sentence prediction).",
    },
    {
        "question": "What is self-attention in transformers?",
        "reference": "Self-attention lets the model weigh the importance of different tokens relative to each other within a sequence.",
    },
    {
        "question": "What is GPT and how is it trained?",
        "reference": "GPT is trained autoregressively to predict the next token given previous tokens on large text corpora.",
    },
    {
        "question": "What is instruction tuning?",
        "reference": "Instruction tuning fine-tunes models on instruction-following data to improve alignment with human intent.",
    },
    {
        "question": "What is RLHF?",
        "reference": "RLHF uses human preference data to train a reward model and optimize a policy to better align model behavior.",
    },
    {
        "question": "What is chain-of-thought prompting?",
        "reference": "Chain-of-thought prompting encourages step-by-step reasoning and can improve performance on multi-step tasks.",
    },
    {
        "question": "What is Retrieval-Augmented Generation?",
        "reference": "RAG combines an LLM with retrieval from an external knowledge base to produce grounded answers.",
    },
    {
        "question": "What are the main components of a RAG pipeline?",
        "reference": "A basic RAG pipeline includes chunking + embeddings, a vector store, a retriever, a prompt with injected context, and an LLM.",
    },
    {
        "question": "What is dense retrieval?",
        "reference": "Dense retrieval uses embedding similarity (vector distance) to find semantically related passages.",
    },
    {
        "question": "Why is chunking strategy important in RAG?",
        "reference": "Chunk size affects relevance and completeness: large chunks can dilute relevance while small chunks can lose needed context.",
    },
    {
        "question": "What advanced RAG techniques exist beyond basic retrieval?",
        "reference": "Advanced RAG techniques include reranking, query expansion, multi-hop retrieval, hybrid search, and citation/attribution.",
    },
    {
        "question": "What are vector databases used for?",
        "reference": "Vector databases store embeddings and enable similarity search (nearest neighbor retrieval).",
    },
    {
        "question": "What is FAISS?",
        "reference": "FAISS is a library for efficient similarity search over dense vectors.",
    },
    {
        "question": "How do text embeddings capture semantic meaning?",
        "reference": "Embeddings map text to vectors so semantically similar texts have nearby vectors in an embedding space.",
    },
    {
        "question": "What is HNSW?",
        "reference": "HNSW is an approximate nearest neighbor indexing method (Hierarchical Navigable Small World).",
    },
    {
        "question": "What is hybrid search in vector databases?",
        "reference": "Hybrid search combines lexical keyword retrieval (e.g., BM25) with vector similarity retrieval.",
    },
    {
        "question": "What is LangChain?",
        "reference": "LangChain is a framework for building LLM applications with chains, tools, retrievers, and other components.",
    },
    {
        "question": "What is LangChain Expression Language (LCEL)?",
        "reference": "LCEL is a way to compose LangChain runnables into pipelines using a pipe operator.",
    },
    {
        "question": "What is LangGraph?",
        "reference": "LangGraph is a graph-based orchestration framework for building stateful LLM workflows.",
    },
    {
        "question": "What memory types does LangChain support?",
        "reference": "LangChain supports multiple memory types, such as conversation buffer and summary-style memories (among others).",
    },
    {
        "question": "What are LangChain retrievers?",
        "reference": "Retrievers are components that return relevant documents for a query.",
    },
    {
        "question": "What is LangSmith?",
        "reference": "LangSmith is an observability platform for LLM apps, providing tracing, datasets, evals, and monitoring.",
    },
    {
        "question": "What information do LangSmith traces capture?",
        "reference": "Traces can capture inputs, outputs, timings/latency, and intermediate steps like retrieved documents.",
    },
    {
        "question": "What is the LangSmith Prompt Hub?",
        "reference": "The Prompt Hub is where you manage and version prompts (push/pull prompt versions).",
    },
    {
        "question": "How does LangSmith help monitor production LLM applications?",
        "reference": "It helps by recording traces to monitor latency, errors, and quality signals across requests.",
    },
    {
        "question": "What are LangSmith datasets used for?",
        "reference": "Datasets are curated collections of inputs/expected outputs used for evaluation and regression testing.",
    },
    {
        "question": "What is RAGAS?",
        "reference": "RAGAS is a framework for evaluating RAG systems.",
    },
    {
        "question": "How does RAGAS compute faithfulness?",
        "reference": "Faithfulness measures whether the answer is supported by the retrieved context (groundedness).",
    },
    {
        "question": "What is answer relevancy in RAGAS?",
        "reference": "Answer relevancy measures how well the generated answer addresses the user question.",
    },
    {
        "question": "What is context recall in RAGAS?",
        "reference": "Context recall measures whether retrieved contexts contain the information needed for the reference answer.",
    },
    {
        "question": "What is context precision in RAGAS?",
        "reference": "Context precision measures how much of the retrieved context is relevant to the question/answer.",
    },
    {
        "question": "What inputs does RAGAS evaluation require?",
        "reference": "RAGAS evaluation typically needs the question, generated answer, retrieved contexts, and a reference answer.",
    },
    {
        "question": "What is Guardrails AI?",
        "reference": "Guardrails AI helps validate and enforce constraints on LLM outputs such as PII redaction and structured JSON.",
    },
    {
        "question": "What is PII and why is it important to detect in LLM responses?",
        "reference": "PII is personally identifiable information (e.g., email/phone/SSN/credit card); detecting it helps protect privacy.",
    },
    {
        "question": "What does structured output validation ensure?",
        "reference": "It ensures an output conforms to a required format or schema, such as being valid parseable JSON.",
    },
    {
        "question": "What is Constitutional AI?",
        "reference": "Constitutional AI uses a set of guiding principles to steer model behavior toward safer outputs.",
    },
    {
        "question": "What are common AI safety concerns with LLMs?",
        "reference": "Common concerns include hallucinations, privacy/PII leakage, bias, prompt injection, and misuse.",
    },
]
