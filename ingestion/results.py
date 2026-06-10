"""Helpers for shaping RAG candidates into UI-friendly result payloads."""

from __future__ import annotations

import json
from typing import Any

from retrievers.retriever import Candidate

NOT_FOUND = "Not found in retrieved sources"


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


def first_source_by_collection(
    sources: list[dict[str, Any]], collection: str
) -> dict[str, Any] | None:
    return next((source for source in sources if source["collection"] == collection), None)


def percent(value: Any) -> str:
    return f"{value}%" if value is not None else NOT_FOUND


def dollars(value: Any) -> str:
    return f"${value}" if value is not None else NOT_FOUND


def build_campaign_brief(sources: list[dict[str, Any]]) -> dict[str, Any]:
    hotel = first_source_by_collection(sources, "hotels")
    event = first_source_by_collection(sources, "events")
    campaign = first_source_by_collection(sources, "campaign_history")
    playbook = first_source_by_collection(sources, "playbook")

    hotel_raw = hotel["raw"] if hotel else {}
    event_raw = event["raw"] if event else {}
    campaign_raw = campaign["raw"] if campaign else {}
    playbook_raw = playbook["raw"] if playbook else {}
    hotel_meta = hotel_raw.get("metadata", {})
    campaign_meta = campaign_raw.get("metadata", {})

    return {
        "hotel": hotel_raw.get("hotel_name", NOT_FOUND),
        "occupancy": percent(hotel_raw.get("current_occupancy_pct")),
        "available_rooms": hotel_raw.get("available_rooms", NOT_FOUND),
        "date_range": hotel_meta.get("date_range", NOT_FOUND),
        "event": event_raw.get("event_name", NOT_FOUND),
        "campaign": campaign_raw.get("campaign_name", NOT_FOUND),
        "target_segment": hotel_raw.get(
            "primary_guest_segment",
            campaign_raw.get("hotel_type", NOT_FOUND),
        ),
        "offer": campaign_raw.get("offer", NOT_FOUND),
        "expected_lift": percent(campaign_raw.get("occupancy_lift")),
        "base_price": dollars(hotel_meta.get("base_price_usd")),
        "minimum_rate": dollars(hotel_meta.get("recommended_min_rate_usd")),
        "max_discount": (
            percent(hotel_meta.get("recommended_max_discount_pct"))
            if hotel_meta.get("recommended_max_discount_pct") is not None
            else percent(campaign_meta.get("discount_pct"))
        ),
        "playbook_action": "; ".join(playbook_raw.get("recommended_actions", [])) or NOT_FOUND,
    }
