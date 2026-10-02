"""Deterministic fake LLM for tests (no network)."""
from __future__ import annotations

from memcodex.llm import LLM


class FakeLLM(LLM):
    def __init__(self) -> None:
        super().__init__(base_url="http://fake.local/v1", api_key="test", model="fake")

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls = getattr(self, "calls", 0) + 1
        # pretend ~2 prompt tokens / ~8 completion tokens per call
        self.total_prompt_tokens += 2
        self.total_completion_tokens += 8
        if "memory compiler" in system or "Summarize" in system:
            out = "- Alice assigned Bob to the search workstream"
        elif "subject" in system.lower():
            out = "Alice | assigned | Bob"
        else:
            out = "token F1"
        return out