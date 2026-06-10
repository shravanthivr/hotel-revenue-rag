"""
Load hospitality JSON, convert structured records to retrieval chunks,
embed with a dimension-matched local model, and upsert to Pinecone.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def required_env(name: str) -> str:
    value = env_value(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Add it to your .env file.")
    return value


PINECONE_INDEX = env_value("PINECONE_INDEX", "hospitality-rag") or "hospitality-rag"
PINECONE_REGION = env_value("PINECONE_REGION", "us-east-1") or "us-east-1"

EMBED_MODEL = env_value("EMBED_MODEL", "BAAI/bge-small-en-v1.5") or "BAAI/bge-small-en-v1.5"
EMBED_DIMENSION = int(env_value("EMBED_DIMENSION", "384") or "384")

NAMESPACES = {
    "campaign_history": "campaigns",
    "events": "events",
    "guest_personas": "personas",
    "hotels": "hotels",
}


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=required_env("PINECONE_API_KEY"))


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def ensure_index(pc: Pinecone):
    existing = [index.name for index in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        print(f"Creating index '{PINECONE_INDEX}'...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            print("  waiting...")
            time.sleep(3)
    return pc.Index(PINECONE_INDEX)


def format_metadata(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    parts = [f"{key}: {value}" for key, value in metadata.items()]
    return " Metadata: " + "; ".join(parts) + "."


def record_to_text(collection: str, record: dict[str, Any]) -> str:
    metadata_text = format_metadata(record.get("metadata", {}))

    if collection == "campaign_history":
        return (
            f"Campaign: {record['campaign_name']}. "
            f"Hotel type: {record['hotel_type']}. "
            f"Occupancy lift: {record['occupancy_lift']}%. "
            f"Offer: {record['offer']}."
            f"{metadata_text}"
        )
    if collection == "events":
        return (
            f"Event: {record['event_name']}. "
            f"City: {record['city']}. "
            f"Attendance: {record['attendance']:,}. "
            f"Event dates: {record['event_start']} to {record['event_end']}. "
            f"Distance from hotel: {record['distance_from_hotel_miles']} miles."
            f"{metadata_text}"
        )
    if collection == "guest_personas":
        prefs = ", ".join(record["preferences"])
        return f"Guest segment: {record['segment']}. Preferences: {prefs}.{metadata_text}"
    if collection == "hotels":
        return (
            f"Hotel: {record['hotel_name']}. "
            f"City: {record['city']}. "
            f"Metro area: {record['metro_area']}. "
            f"Hotel type: {record['hotel_type']}. "
            f"Total rooms: {record['total_rooms']}. "
            f"Current occupancy: {record['current_occupancy_pct']}%. "
            f"Occupied rooms: {record['occupied_rooms']}. "
            f"Available rooms: {record['available_rooms']}. "
            f"Forecast period: {record['forecast_period']}. "
            f"Primary guest segment: {record['primary_guest_segment']}. "
            f"Nearby events: {', '.join(record['nearby_events'])}."
            f"{metadata_text}"
        )
    return " ".join(str(value) for value in record.values())


def flatten_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata", {}))
    for key, value in record.items():
        if key == "metadata":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
        elif isinstance(value, list):
            metadata[key] = ", ".join(str(item) for item in value)
    return metadata


def fixed_size_chunks(text: str, chunk_words: int = 90, overlap_words: int = 20) -> list[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = max(chunk_words - overlap_words, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])
        if chunk:
            chunks.append(chunk)
    return chunks


def build_chunks(data: dict[str, Any], strategy: str = "semantic") -> list[dict[str, Any]]:
    chunks = []
    for collection, namespace in NAMESPACES.items():
        for record_index, record in enumerate(data.get(collection, [])):
            text = record_to_text(collection, record)
            texts = [text] if strategy == "semantic" else fixed_size_chunks(text)
            base_metadata = flatten_metadata(record)

            for chunk_index, chunk_text in enumerate(texts):
                chunk_id = hashlib.sha256(
                    f"{collection}:{record_index}:{chunk_index}:{strategy}".encode()
                ).hexdigest()[:16]
                metadata = {
                    **base_metadata,
                    "collection": collection,
                    "record_index": record_index,
                    "chunk_index": chunk_index,
                    "chunk_strategy": strategy,
                    "source_json": json.dumps(record),
                    "text": chunk_text,
                }
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "namespace": namespace,
                        "metadata": metadata,
                    }
                )
    return chunks


def load_json(data_path: str | Path = "data/rag_data.json") -> dict[str, Any]:
    with Path(data_path).open() as file:
        return json.load(file)


def embed_chunks(chunks: list[dict[str, Any]], embedder: SentenceTransformer) -> list[list[float]]:
    texts = [chunk["text"] for chunk in chunks]
    print(f"  Embedding {len(texts)} chunks with {EMBED_MODEL} ({EMBED_DIMENSION} dims)...")
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings.tolist()


def upsert(index, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for chunk, embedding in zip(chunks, embeddings):
        namespace = chunk["namespace"]
        buckets.setdefault(namespace, []).append(
            {"id": chunk["id"], "values": embedding, "metadata": chunk["metadata"]}
        )

    for namespace, vectors in buckets.items():
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"  Upserted {len(vectors)} vectors to namespace '{namespace}'")


def load(data_path: str = "data/rag_data.json", strategy: str = "semantic") -> None:
    data = load_json(data_path)
    pc = get_pinecone_client()

    print("\n-- Step 1: Pinecone index --------------------------------------")
    index = ensure_index(pc)

    print("\n-- Step 2: Build chunks ----------------------------------------")
    chunks = build_chunks(data, strategy=strategy)
    print(f"  {len(chunks)} chunks ready using '{strategy}' chunking")

    print("\n-- Step 3: Embed locally ---------------------------------------")
    embeddings = embed_chunks(chunks, get_embedder())

    print("\n-- Step 4: Upsert to Pinecone ----------------------------------")
    upsert(index, chunks, embeddings)
    print("\nIngestion complete.")


if __name__ == "__main__":
    load()
