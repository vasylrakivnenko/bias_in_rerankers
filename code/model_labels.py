"""Short display labels for every reranker, shared by figures and tables.

Full identifiers are long enough to force tables into tiny type, so tables
print these labels and the provenance table maps each one back to its full
identifier (reviewer item 10).
"""

MODEL_LABELS = {
    "BAAI/bge-reranker-v2-m3": "BAAI: BGE v2-m3",
    "BAAI/bge-reranker-base": "BAAI: BGE base",
    "Cohere-rerank-v4.0-fast": "Cohere: v4 fast",
    "Cohere-rerank-v4.0-pro": "Cohere: v4 pro",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "Open: MiniLM-L12-v2",
    "ibm-granite-granite-embedding-reranker-english-r2": "IBM: Granite r2",
    "jinaai/jina-reranker-v2-base-multilingual": "Jina: reranker v2",
    "rerank-1": "Voyage: rerank-1",
    "rerank-2": "Voyage: rerank-2",
    "rerank-2-lite": "Voyage: rerank-2-lite",
    "rerank-2.5": "Voyage: rerank-2.5",
    "rerank-2.5-lite": "Voyage: rerank-2.5-lite",
    "semantic-ranker-default-002": "Google: default-002",
    "semantic-ranker-default-004": "Google: default-004",
    "semantic-ranker-fast-004": "Google: fast-004",
    "rerank-3": "Voyage: rerank-3",
    "rerank-3-lite": "Voyage: rerank-3-lite",
    "cohere/rerank-v3.5": "Cohere: v3.5",
    "qwen/qwen3-reranker-8b": "Qwen: Qwen3 8B",
    "nvidia/llama-nemotron-rerank-vl-1b-v2": "NVIDIA: Nemotron VL 1B",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "Open: MiniLM-L6-v2",
    "BAAI/bge-reranker-large": "BAAI: BGE large",
    "Alibaba-NLP/gte-reranker-modernbert-base": "Alibaba: GTE ModernBERT",
    "Qwen/Qwen3-Reranker-0.6B": "Qwen: Qwen3 0.6B",
    "Qwen/Qwen3-Reranker-4B": "Qwen: Qwen3 4B",
    "Qwen/Qwen3-Reranker-8B": "Qwen: Qwen3 8B (self-hosted)",
    "mixedbread-ai/mxbai-rerank-large-v2": "Mixedbread: large v2",
    "nvidia/llama-nemotron-rerank-1b-v2": "NVIDIA: Nemotron 1B",
    "zeroentropy/zerank-2-reranker": "ZeroEntropy: zerank-2",
    "jinaai/jina-reranker-v3": "Jina: reranker v3",
}


def model_label(model: str) -> str:
    """Short label for `model`; fails loudly so no table silently prints a raw id."""
    return MODEL_LABELS[model]
