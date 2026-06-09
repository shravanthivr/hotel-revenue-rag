"""Streamlit UI for the hotel revenue RAG assistant."""

from __future__ import annotations

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

    st.subheader("Answer")
    st.write(result["answer"])

    left, right, center = st.columns(3)
    left.metric("Latency", f"{result['latency']}s")
    right.metric("Best rerank score", result["retrieval"]["best_score"])
    center.metric("Merged candidates", result["retrieval"]["n_merged"])

    with st.expander("Retrieval details", expanded=False):
        st.json(result["retrieval"])

    brief = result.get("campaign_brief")
    if brief:
        st.subheader("Campaign Brief")
        top_left, top_mid, top_right = st.columns(3)
        top_left.metric("Hotel", brief["hotel"])
        top_mid.metric("Occupancy", brief["occupancy"])
        top_right.metric("Available rooms", brief["available_rooms"])

        brief_left, brief_right = st.columns(2)
        with brief_left:
            st.write("**Recommended campaign**")
            st.write(brief["campaign"])
            st.write("**Offer**")
            st.write(brief["offer"])
            st.write("**Target segment**")
            st.write(brief["target_segment"])
        with brief_right:
            st.write("**Relevant event**")
            st.write(brief["event"])
            st.write("**Date range**")
            st.write(brief["date_range"])
            st.write("**Expected lift**")
            st.write(brief["expected_lift"])

        price_left, price_mid, price_right = st.columns(3)
        price_left.metric("Base price", brief["base_price"])
        price_mid.metric("Minimum rate", brief["minimum_rate"])
        price_right.metric("Max discount", brief["max_discount"])
        st.write("**Playbook action**")
        st.write(brief["playbook_action"])

    st.subheader("Retrieval Comparison")
    comparison_rows = []
    for label, key in [
        ("Dense top result", "top_dense"),
        ("BM25 top result", "top_sparse"),
        ("RRF merged top result", "top_merged"),
        ("Final reranked result", "top_reranked"),
    ]:
        item = result["retrieval"].get(key)
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

    st.subheader("Sources")
    if not result["sources"]:
        st.info("No sources passed the confidence gate.")
    for index, source in enumerate(result["sources"], 1):
        with st.expander(
            f"Source {index}: {source['collection']} | rerank {source['rerank_score']}",
            expanded=index == 1,
        ):
            st.write(source["text"])
            st.json(source["raw"])
