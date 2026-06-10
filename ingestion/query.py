
from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from ingestion.loader import env_value, required_env
from ingestion.prompting import backend_system_prompt, build_user_prompt
from ingestion.results import build_campaign_brief, serialize_sources
from retrievers.retriever import Candidate, retrieve

NEBIUS_BASE_URL = env_value("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/") or (
    "https://api.studio.nebius.com/v1/"
)
GEN_MODEL = env_value("NEBIUS_GEN_MODEL", "meta-llama/Llama-3.3-70B-Instruct") or (
    "meta-llama/Llama-3.3-70B-Instruct"
)


def get_nebius_client() -> OpenAI:
    return OpenAI(base_url=NEBIUS_BASE_URL, api_key=required_env("NEBIUS_API_KEY"))


def generate_answer(question: str, candidates: list[Candidate]) -> str:
    response = get_nebius_client().chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": backend_system_prompt()},
            {"role": "user", "content": build_user_prompt(question, candidates)},
        ],
        temperature=0.1,
        max_tokens=600,
    )
    return response.choices[0].message.content or ""


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
