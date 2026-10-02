"""Records, episodes, and token heuristics."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Record:
    text: str
    tokens: int = 0

    def __post_init__(self) -> None:
        if self.tokens == 0:
            self.tokens = count_tokens(self.text)


@dataclass
class Question:
    text: str
    answer: str


@dataclass
class Episode:
    records: list[Record]
    questions: list[Question]
    name: str = "episode"


def count_tokens(text: str) -> int:
    """Cheap mixed zh/en token heuristic: ~2 chars per token.

    Deliberately not a real tokenizer (no heavy dependency). Good enough
    for trend-level budget accounting; see README for caveats.
    """
    return max(1, len(text) // 2)


def load_episodes(path: str | Path) -> list[Episode]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    episodes: list[Episode] = []
    for i, ep in enumerate(raw):
        records = [Record(text=r) for r in ep.get("records", [])]
        questions = [
            Question(text=q["text"], answer=q["answer"])
            for q in ep.get("questions", [])
        ]
        episodes.append(Episode(records=records, questions=questions, name=ep.get("name", f"episode_{i}")))
    return episodes