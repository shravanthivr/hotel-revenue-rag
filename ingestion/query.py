
from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from ingestion.loader import env_value, required_env
from retrievers.retriever import Candidate, retrieve

NEBIUS_BASE_URL = env_value("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/") or (
    "https://api.studio.nebius.com/v1/"
)
GEN_MODEL = env_value("NEBIUS_GEN_MODEL", "meta-llama/Llama-3.3-70B-Instruct") or (
    "meta-llama/Llama-3.3-70B-Instruct"
)

SYSTEM_PROMPT = """You are a hospitality revenue intelligence assistant.
Answer using ONLY the retrieved context chunks.
If the context does not contain enough information, say:
"I could not find this in our data."

Return a concise, demo-friendly answer with these headings:

Recommended Campaign:
Why This Fits:
Pricing Guardrail:
Supporting Evidence:

Do not invent prices, discounts, event details, or occupancy lift. Cite the hotel,
campaign, event, or persona segment used."""


def get_nebius_client() -> OpenAI:
    return OpenAI(base_url=NEBIUS_BASE_URL, api_key=required_env("NEBIUS_API_KEY"))


def build_prompt(question: str, candidates: list[Candidate]) -> str:
    context_blocks = []
    for index, candidate in enumerate(candidates, 1):
        context_blocks.append(
            f"[Source {index} | {candidate.collection} | score {candidate.rerank_score:.3f}]\n"
            f"{candidate.text}"
        )
    context = "\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate_answer(question: str, candidates: list[Candidate]) -> str:
    response = get_nebius_client().chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, candidates)},
        ],
        temperature=0.1,
        max_tokens=600,
    )
    return response.choices[0].message.content or ""


def serialize_sources(candidates: list[Candidate]) -> list[dict[str, Any]]:
    sources = []
    for candidate in candidates:
        try:
            raw = json.loads(candidate.source_json)
        except json.JSONDecodeError:
            raw = {"source_json": candidate.source_json}
        sources.append(
            {
                "collection": candidate.collection,
                "text": candidate.text,
                "dense_score": round(candidate.dense_score, 3),
                "sparse_score": round(candidate.sparse_score, 3),
                "rrf_score": round(candidate.rrf_score, 4),
                "rerank_score": round(candidate.rerank_score, 3),
                "raw": raw,
            }
        )
    return sources


def first_source_by_collection(sources: list[dict[str, Any]], collection: str) -> dict[str, Any] | None:
    return next((source for source in sources if source["collection"] == collection), None)


def build_campaign_brief(sources: list[dict[str, Any]]) -> dict[str, Any]:
    hotel = first_source_by_collection(sources, "hotels")
    event = first_source_by_collection(sources, "events")
    campaign = first_source_by_collection(sources, "campaign_history")

    hotel_raw = hotel["raw"] if hotel else {}
    event_raw = event["raw"] if event else {}
    campaign_raw = campaign["raw"] if campaign else {}
    hotel_meta = hotel_raw.get("metadata", {})
    campaign_meta = campaign_raw.get("metadata", {})

    return {
        "hotel": hotel_raw.get("hotel_name", "Not found in retrieved sources"),
        "occupancy": (
            f"{hotel_raw.get('current_occupancy_pct')}%"
            if hotel_raw.get("current_occupancy_pct") is not None
            else "Not found in retrieved sources"
        ),
        "available_rooms": hotel_raw.get("available_rooms", "Not found in retrieved sources"),
        "date_range": hotel_meta.get("date_range", "Not found in retrieved sources"),
        "event": event_raw.get("event_name", "Not found in retrieved sources"),
        "campaign": campaign_raw.get("campaign_name", "Not found in retrieved sources"),
        "target_segment": hotel_raw.get(
            "primary_guest_segment",
            campaign_raw.get("hotel_type", "Not found in retrieved sources"),
        ),
        "offer": campaign_raw.get("offer", "Not found in retrieved sources"),
        "expected_lift": (
            f"{campaign_raw.get('occupancy_lift')}%"
            if campaign_raw.get("occupancy_lift") is not None
            else "Not found in retrieved sources"
        ),
        "base_price": (
            f"${hotel_meta.get('base_price_usd')}"
            if hotel_meta.get("base_price_usd") is not None
            else "Not found in retrieved sources"
        ),
        "minimum_rate": (
            f"${hotel_meta.get('recommended_min_rate_usd')}"
            if hotel_meta.get("recommended_min_rate_usd") is not None
            else "Not found in retrieved sources"
        ),
        "max_discount": (
            f"{hotel_meta.get('recommended_max_discount_pct')}%"
            if hotel_meta.get("recommended_max_discount_pct") is not None
            else (
                f"{campaign_meta.get('discount_pct')}%"
                if campaign_meta.get("discount_pct") is not None
                else "Not found in retrieved sources"
            )
        ),
    }


def query_rag(question: str, namespaces: list[str] | None = None, verbose: bool = False) -> dict[str, Any]:
    start = time.time()
    if verbose:
        print(f"Query: {question}")

    candidates, retrieval_meta = retrieve(question, namespaces=namespaces)
    if retrieval_meta["refused"]:
        return {
            "answer": "I could not find this in our data.",
            "campaign_brief": {},
            "sources": [],
            "refused": True,
            "latency": round(time.time() - start, 3),
            "retrieval": retrieval_meta,
            "model": GEN_MODEL,
        }

    answer = generate_answer(question, candidates)
    sources = serialize_sources(candidates)
    return {
        "answer": answer,
        "campaign_brief": build_campaign_brief(sources),
        "sources": sources,
        "refused": False,
        "latency": round(time.time() - start, 3),
        "retrieval": retrieval_meta,
        "model": GEN_MODEL,
    }


if __name__ == "__main__":
    questions = [
        "What campaign should Marriott Tysons Corner run for the Tech Conference?",
        "Which campaign had the highest occupancy lift?",
        "What pricing constraints apply to Westin Copley Place?",
    ]
    for question in questions:
        result = query_rag(question, verbose=True)
        print(f"\nQ: {question}\nA: {result['answer']}\n")
