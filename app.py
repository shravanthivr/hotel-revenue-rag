"""Streamlit UI for the hotel revenue RAG assistant."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import streamlit as st

from ingestion.loader import env_value
from ingestion.query import query_rag

DEMO_QUESTIONS = {
    "Tysons conference campaign": "What campaign should Marriott Tysons Corner run for the Tech Conference?",
    "Boston pricing constraints": "What pricing constraints apply to Westin Copley Place?",
    "Highest lift campaign": "Which campaign had the highest occupancy lift?",
    "Chicago sales expo offer": "What should Marriott Westside Chicago offer National Sales Expo attendees?",
    "Business traveler amenities": "What amenities matter most to business travelers?",
}

st.set_page_config(page_title="Hotel Revenue RAG", layout="wide")

SOURCE_LABELS = {
    "hotels": "Hotels",
    "events": "Events",
    "campaign_history": "Historical Campaign Evidence",
    "playbook": "Pricing Rules and Playbooks",
    "personas": "Guest Personas",
}


def value_or_dash(value: Any) -> Any:
    return value if value not in (None, "") else "Not found in retrieved sources"


def render_field_row(label: str, value: Any) -> None:
    label_col, value_col = st.columns([1, 3])
    label_col.markdown(f"**{label}**")
    value_col.write(value_or_dash(value))


def render_kpi_cards(result: dict[str, Any]) -> None:
    retrieval = result.get("retrieval", {})
    latency_col, score_col, merged_col = st.columns(3)
    latency_col.metric("Latency", f"{result.get('latency', 0)}s")
    score_col.metric("Best rerank score", retrieval.get("best_score", "n/a"))
    merged_col.metric("Merged candidates", retrieval.get("n_merged", "n/a"))


def first_source_by_collection(
    sources: list[dict[str, Any]], collection: str
) -> dict[str, Any] | None:
    return next((source for source in sources if source.get("collection") == collection), None)


def render_campaign_recommendation(brief: dict[str, Any], answer: str) -> None:
    with st.container(border=True):
        st.subheader("Campaign Recommendation")
        if not brief:
            st.write(answer)
            return

        render_field_row("Campaign Name", brief.get("campaign"))
        render_field_row("Target Segment", brief.get("target_segment"))
        render_field_row("Offer", brief.get("offer"))
        render_field_row("Expected Occupancy Lift", brief.get("expected_lift"))
        render_field_row(
            "Pricing Guardrails",
            (
                f"Base price: {brief.get('base_price')}; "
                f"minimum rate: {brief.get('minimum_rate')}; "
                f"maximum discount: {brief.get('max_discount')}"
            ),
        )

        st.divider()
        st.markdown("**Recommendation Rationale**")
        st.write(answer)


def render_context_sections(brief: dict[str, Any], sources: list[dict[str, Any]]) -> None:
    if not brief:
        return

    event_source = first_source_by_collection(sources, "events") or {}
    event_raw = event_source.get("raw", {})
    event_meta = event_raw.get("metadata", {}) if isinstance(event_raw, dict) else {}

    hotel_col, event_col = st.columns(2)
    with hotel_col:
        with st.container(border=True):
            st.subheader("Hotel Context")
            render_field_row("Hotel", brief.get("hotel"))
            render_field_row("Occupancy", brief.get("occupancy"))
            render_field_row("Available Rooms", brief.get("available_rooms"))
            render_field_row("Forecast Period", brief.get("date_range"))

    with event_col:
        with st.container(border=True):
            st.subheader("Event Context")
            render_field_row("Relevant Event", brief.get("event"))
            render_field_row("Event Dates", event_meta.get("date_range"))
            render_field_row("Attendance Estimate", event_raw.get("attendance_estimate"))
            render_field_row(
                "Distance From Hotel",
                (
                    f"{event_raw.get('distance_from_hotel_miles')} miles"
                    if event_raw.get("distance_from_hotel_miles") is not None
                    else None
                ),
            )
            render_field_row("Expected Price Pressure", event_meta.get("expected_price_pressure"))

    st.divider()

    with st.container(border=True):
        st.subheader("Pricing Rules")
        price_col, min_col, discount_col = st.columns(3)
        price_col.metric("Base price", brief.get("base_price", "n/a"))
        min_col.metric("Minimum rate", brief.get("minimum_rate", "n/a"))
        discount_col.metric("Max discount", brief.get("max_discount", "n/a"))
        render_field_row("Guardrail summary", brief.get("playbook_action"))


def render_historical_campaign_evidence(sources: list[dict[str, Any]]) -> None:
    with st.container(border=True):
        st.subheader("Historical Campaign Evidence")
        campaign_sources = [
            source for source in sources if source.get("collection") == "campaign_history"
        ]
        if not campaign_sources:
            st.info("No historical campaign sources passed the confidence gate.")
            return

        for index, source in enumerate(campaign_sources, 1):
            raw = source.get("raw", {})
            metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
            with st.expander(
                f"{raw.get('campaign_name', f'Campaign source {index}')} | "
                f"rerank {source.get('rerank_score', 'n/a')}",
                expanded=index == 1,
            ):
                render_field_row("Offer", raw.get("offer"))
                render_field_row("Occupancy Lift", raw.get("occupancy_lift"))
                render_field_row("Discount", metadata.get("discount_pct"))
                st.divider()
                st.write(source.get("text", ""))


def source_type_for(source: dict[str, Any]) -> str:
    raw = source.get("raw", {})
    metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    return metadata.get("source_type") or source.get("collection", "unknown")


def render_sources(sources: list[dict[str, Any]]) -> None:
    with st.container(border=True):
        st.subheader("Retrieved Evidence by Source Type")
        if not sources:
            st.info("No sources passed the confidence gate.")
            return

        grouped_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source in sources:
            grouped_sources[source_type_for(source)].append(source)

        for source_type, group in grouped_sources.items():
            label = SOURCE_LABELS.get(source_type, source_type.replace("_", " ").title())
            with st.expander(f"{label} ({len(group)})", expanded=False):
                for index, source in enumerate(group, 1):
                    st.markdown(
                        f"**Source {index}: {source.get('collection', 'unknown')} | "
                        f"rerank {source.get('rerank_score', 'n/a')}**"
                    )
                    st.write(source.get("text", ""))
                    st.json(source.get("raw", {}), expanded=False)
                    if index < len(group):
                        st.divider()


def render_retrieval_diagnostics(result: dict[str, Any]) -> None:
    retrieval = result.get("retrieval", {})
    with st.expander("Retrieval Diagnostics", expanded=False):
        comparison_rows = []
        for label, key in [
            ("Dense top result", "top_dense"),
            ("BM25 top result", "top_sparse"),
            ("RRF merged top result", "top_merged"),
            ("Final reranked result", "top_reranked"),
        ]:
            item = retrieval.get(key)
            if item:
                comparison_rows.append(
                    {
                        "Stage": label,
                        "Collection": item["collection"],
                        "Dense": item["dense_score"],
                        "BM25": item["sparse_score"],
                        "RRF": item["rrf_score"],
                        "Rerank": item["rerank_score"],
                        "Preview": item["text_preview"],
                    }
                )
        if comparison_rows:
            st.dataframe(comparison_rows, use_container_width=True, hide_index=True)
        st.json(retrieval)


st.title("Hotel Revenue RAG")
st.caption("Ask grounded questions across hotels, events, campaigns, personas, and playbooks.")

missing = [key for key in ("PINECONE_API_KEY", "NEBIUS_API_KEY") if not env_value(key)]
if missing:
    st.warning(f"Missing environment variables: {', '.join(missing)}")

selected_demo = st.selectbox("Demo question", options=list(DEMO_QUESTIONS.keys()))
question = st.text_area("Question", value=DEMO_QUESTIONS[selected_demo], height=100)

namespace_options = {
    "Hotels": "hotels",
    "Campaigns": "campaigns",
    "Events": "events",
    "Personas": "personas",
    "Playbook": "playbook",
}
selected_labels = st.multiselect(
    "Search scope",
    options=list(namespace_options.keys()),
    default=list(namespace_options.keys()),
)
namespaces = [namespace_options[label] for label in selected_labels]

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving, reranking, and generating..."):
        try:
            result = query_rag(question.strip(), namespaces=namespaces or None)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    brief = result.get("campaign_brief")
    sources = result.get("sources", [])

    render_kpi_cards(result)
    st.divider()
    render_campaign_recommendation(brief, result.get("answer", ""))
    st.divider()
    render_context_sections(brief, sources)
    st.divider()
    render_historical_campaign_evidence(sources)
    st.divider()
    render_sources(sources)
    st.divider()
    render_retrieval_diagnostics(result)
