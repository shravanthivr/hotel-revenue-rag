"""
Hybrid retrieval for hotel revenue RAG.

Dense retrieval uses the same local embedding model as ingestion so Pinecone
dimensions stay aligned. BM25 runs over the local JSON corpus for exact hotel,
event, date, and pricing matches. RRF merges both rankings before reranking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ingestion.loader import (
    EMBED_MODEL,
    NAMESPACES,
    PINECONE_INDEX,
    build_chunks,
    env_value,
    load_json,
    required_env,
)

RERANK_MODEL = env_value("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2") or (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

ALL_NAMESPACES = list(NAMESPACES.values())
TOP_K_RETRIEVAL = int(env_value("TOP_K_RETRIEVAL", "20") or "20")
TOP_K_FINAL = int(env_value("TOP_K_FINAL", "5") or "5")
CONFIDENCE_FLOOR = float(env_value("CONFIDENCE_FLOOR", "0.25") or "0.25")


@dataclass
class Candidate:
    id: str
    text: str
    collection: str
    source_json: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def get_index():
    return Pinecone(api_key=required_env("PINECONE_API_KEY")).Index(PINECONE_INDEX)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def get_sparse_index(data_path: str = "data/rag_data.json") -> tuple[BM25Okapi, list[Candidate]]:
    chunks = build_chunks(load_json(data_path), strategy="semantic")
    candidates = [
        Candidate(
            id=chunk["id"],
            text=chunk["text"],
            collection=chunk["metadata"]["collection"],
            source_json=chunk["metadata"]["source_json"],
        )
        for chunk in chunks
    ]
    bm25 = BM25Okapi([tokenize(candidate.text) for candidate in candidates])
    return bm25, candidates


def embed_query(query: str) -> list[float]:
    return get_embedder().encode(query).tolist()


def dense_retrieve(
    query_embedding: list[float],
    namespaces: list[str] | None = None,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[Candidate]:
    namespaces = namespaces or ALL_NAMESPACES
    seen: dict[str, Candidate] = {}

    for namespace in namespaces:
        results = get_index().query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        for match in results.matches:
            if match.id in seen:
                continue
            metadata = match.metadata or {}
            seen[match.id] = Candidate(
                id=match.id,
                text=metadata.get("text", ""),
                collection=metadata.get("collection", namespace),
                source_json=metadata.get("source_json", "{}"),
                dense_score=float(match.score or 0.0),
            )
    return list(seen.values())


def sparse_retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[Candidate]:
    bm25, corpus_candidates = get_sparse_index()
    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    results = []
    for index in ranked_indices[:top_k]:
        source = corpus_candidates[index]
        results.append(
            Candidate(
                id=source.id,
                text=source.text,
                collection=source.collection,
                source_json=source.source_json,
                sparse_score=float(scores[index]),
            )
        )
    return results


def rrf_merge(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
    k: int = 60,
) -> list[Candidate]:
    by_id: dict[str, Candidate] = {}
    scores: dict[str, float] = {}

    for rank, candidate in enumerate(dense_candidates):
        by_id[candidate.id] = candidate
        scores[candidate.id] = scores.get(candidate.id, 0.0) + 1 / (k + rank + 1)

    for rank, candidate in enumerate(sparse_candidates):
        existing = by_id.get(candidate.id)
        if existing:
            existing.sparse_score = candidate.sparse_score
        else:
            by_id[candidate.id] = candidate
        scores[candidate.id] = scores.get(candidate.id, 0.0) + 1 / (k + rank + 1)

    for candidate_id, candidate in by_id.items():
        candidate.rrf_score = scores[candidate_id]

    return sorted(by_id.values(), key=lambda candidate: candidate.rrf_score, reverse=True)


def rerank(query: str, candidates: list[Candidate], top_k: int = TOP_K_FINAL) -> list[Candidate]:
    if not candidates:
        return []

    pairs = [(query, candidate.text) for candidate in candidates]
    scores = get_reranker().predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)
    return sorted(candidates, key=lambda candidate: candidate.rerank_score, reverse=True)[:top_k]


def passes_gate(candidates: list[Candidate], floor: float = CONFIDENCE_FLOOR) -> tuple[bool, float]:
    if not candidates:
        return False, 0.0
    best_score = max(candidate.rerank_score for candidate in candidates)
    return best_score >= floor, best_score


def summarize_candidate(candidate: Candidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "collection": candidate.collection,
        "text_preview": candidate.text[:220],
        "dense_score": round(candidate.dense_score, 3),
        "sparse_score": round(candidate.sparse_score, 3),
        "rrf_score": round(candidate.rrf_score, 4),
        "rerank_score": round(candidate.rerank_score, 3),
    }


def retrieve(
    query: str,
    namespaces: list[str] | None = None,
) -> tuple[list[Candidate], dict[str, Any]]:
    query_embedding = embed_query(query)
    dense_candidates = dense_retrieve(query_embedding, namespaces)
    sparse_candidates = sparse_retrieve(query)
    merged = rrf_merge(dense_candidates, sparse_candidates)
    top_candidates = rerank(query, merged[:TOP_K_RETRIEVAL])
    should_answer, best_score = passes_gate(top_candidates)

    meta = {
        "refused": not should_answer,
        "best_score": round(best_score, 3),
        "n_dense": len(dense_candidates),
        "n_sparse": len(sparse_candidates),
        "n_merged": len(merged),
        "top_dense": summarize_candidate(dense_candidates[0] if dense_candidates else None),
        "top_sparse": summarize_candidate(sparse_candidates[0] if sparse_candidates else None),
        "top_merged": summarize_candidate(merged[0] if merged else None),
        "top_reranked": summarize_candidate(top_candidates[0] if top_candidates else None),
        "embedding_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
    }

    if not should_answer:
        return [], meta
    return top_candidates, meta
