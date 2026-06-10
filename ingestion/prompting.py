"""Prompt loading and formatting for backend generation calls."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from retrievers.retriever import Candidate

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PROMPT = ROOT_DIR / "prompts" / "backend_system_prompt.txt"
DEFAULT_USER_PROMPT = ROOT_DIR / "prompts" / "backend_user_prompt.txt"


@lru_cache(maxsize=16)
def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def format_context(candidates: list[Candidate]) -> str:
    blocks = []
    for index, candidate in enumerate(candidates, 1):
        blocks.append(
            f"[Source {index} | {candidate.collection} | score {candidate.rerank_score:.3f}]\n"
            f"{candidate.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, candidates: list[Candidate]) -> str:
    template = load_prompt(DEFAULT_USER_PROMPT)
    return template.format(context=format_context(candidates), question=question)


def backend_system_prompt() -> str:
    return load_prompt(DEFAULT_SYSTEM_PROMPT)
